use std::{sync::Arc, time::Duration};

use skw_dialog_service::{
    ASK_QUEUE,
    llm::{LlmClient, LlmConfig},
    models, named_queue,
};
use skw_lib_http_protos::dialog::RpcPayload;
use skw_lib_shared::prelude::{
    APP, AppError,
    iggy::TaskDonePublisher,
    log::*,
    postgres::{PostgresConnected, PostgresPools},
    rabbitmq::{TaskOutcome, run_task_consumer},
    tools::skw_get_back_topic_name,
};
use skw_lib_task_protos::dialog::DialogAskTask;

/// Delay before a failed task is requeued, multiplied by the attempt number.
const RETRY_DELAY: Duration = Duration::from_secs(2);

#[tokio::main]
async fn main() -> Result<(), AppError> {
    let db_pools = APP.postgres.connect().await?;
    let llm_cfg = LlmConfig::from_app_config();
    let llm = Arc::new(LlmClient::new(&llm_cfg));
    let max_attempts = llm_cfg.max_attempts;
    // the final `AskAnswer` reply goes to the gates' back topic, matched there by the Ask signature
    let done = Arc::new(TaskDonePublisher::new(&skw_get_back_topic_name()).await?);

    run_task_consumer(named_queue(ASK_QUEUE), move |task| {
        handle_ask_task(
            task,
            db_pools.clone(),
            llm.clone(),
            done.clone(),
            max_attempts,
        )
    })
    .await
}

async fn handle_ask_task(
    task: DialogAskTask,
    db_pools: PostgresPools<PostgresConnected>,
    llm: Arc<LlmClient>,
    done: Arc<TaskDonePublisher>,
    max_attempts: u32,
) -> TaskOutcome {
    match answer(&task, &db_pools, &llm, max_attempts).await {
        Ok(Answered::Retry) => TaskOutcome::Nack { requeue: true },
        Ok(Answered::Skipped) => TaskOutcome::Ack,
        Ok(Answered::Final) => {
            send_final_reply(&task, &db_pools, &done).await;
            TaskOutcome::Ack
        }
        Err(e) => {
            // infrastructure (db) failure: retry until it comes back
            error!("dialog ask task {task:?} failed: {e}");
            tokio::time::sleep(RETRY_DELAY).await;
            TaskOutcome::Nack { requeue: true }
        }
    }
}

/// What became of one attempt at a Pending answer.
enum Answered {
    /// The answer is stored, Done or Failed: the client gets the final reply.
    Final,
    /// Retryable LLM failure: requeue the task.
    Retry,
    /// The message is no longer Pending (dialog deleted, or answered by another delivery).
    Skipped,
}

/// Sends the stored answer as the final `AskAnswer` reply of the `Ask` request. Best effort: the
/// answer is already stored, so a client that misses this reply still reads it with `GetDialog`.
async fn send_final_reply(task: &DialogAskTask, db_pools: &PostgresPools<PostgresConnected>, done: &TaskDonePublisher) {
    let Some(signature) = task.signature.as_deref() else {
        return;
    };
    let message = match db_pools.master().await {
        Ok(pool) => models::get_message(pool, task.dialog_id, task.message_id).await,
        Err(e) => {
            warn!(
                "message {}: no db for the final reply: {e}",
                task.message_id
            );
            return;
        }
    };
    let payload = match message {
        Ok(Some(message)) => RpcPayload::AskAnswer {
            dialog_id: task.dialog_id,
            message,
        },
        Ok(None) => return, // dialog deleted meanwhile
        Err(e) => {
            warn!(
                "message {}: can't read it for the final reply: {e}",
                task.message_id
            );
            return;
        }
    };
    if let Err(e) = done.publish(signature, &payload, None).await {
        warn!("message {}: final reply not sent: {e}", task.message_id);
    }
}

async fn answer(task: &DialogAskTask, db_pools: &PostgresPools<PostgresConnected>, llm: &LlmClient, max_attempts: u32) -> Result<Answered, AppError> {
    let pool = db_pools.master().await?;

    let Some((model, attempt)) = models::start_attempt(pool, task.message_id).await? else {
        info!(
            "message {} is not pending anymore, skipping",
            task.message_id
        );
        return Ok(Answered::Skipped);
    };

    let history = models::history(pool, task.dialog_id, task.message_id).await?;

    match llm.chat(&model, &history).await {
        Ok(answer) => {
            models::complete_message(pool, task.dialog_id, task.message_id, &answer).await?;
            Ok(Answered::Final)
        }
        Err(e) if e.is_retryable() && attempt < max_attempts => {
            warn!(
                "message {} attempt {attempt}/{max_attempts} failed, retrying: {e}",
                task.message_id
            );
            tokio::time::sleep(RETRY_DELAY * attempt).await;
            Ok(Answered::Retry)
        }
        Err(e) => {
            error!(
                "message {} failed after {attempt} attempt(s): {e}",
                task.message_id
            );
            models::fail_message(pool, task.message_id, e.client_reason()).await?;
            Ok(Answered::Final)
        }
    }
}
