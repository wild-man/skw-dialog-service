use std::{sync::Arc, time::Duration};

use skw_dialog_service::{
    ASK_QUEUE,
    llm::{LlmClient, LlmConfig},
    models, named_queue,
};
use skw_lib_shared::prelude::{
    APP, AppError,
    log::*,
    postgres::{PostgresConnected, PostgresPools},
    rabbitmq::{TaskOutcome, run_task_consumer},
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

    run_task_consumer(named_queue(ASK_QUEUE), move |task| {
        handle_ask_task(task, db_pools.clone(), llm.clone(), max_attempts)
    })
    .await
}

async fn handle_ask_task(task: DialogAskTask, db_pools: PostgresPools<PostgresConnected>, llm: Arc<LlmClient>, max_attempts: u32) -> TaskOutcome {
    match answer(&task, &db_pools, &llm, max_attempts).await {
        Ok(outcome) => outcome,
        Err(e) => {
            // infrastructure (db) failure: retry until it comes back
            error!("dialog ask task {task:?} failed: {e}");
            tokio::time::sleep(RETRY_DELAY).await;
            TaskOutcome::Nack { requeue: true }
        }
    }
}

async fn answer(task: &DialogAskTask, db_pools: &PostgresPools<PostgresConnected>, llm: &LlmClient, max_attempts: u32) -> Result<TaskOutcome, AppError> {
    let pool = db_pools.master().await?;

    let Some((model, attempt)) = models::start_attempt(pool, task.message_id).await? else {
        info!("message {} is not pending anymore, skipping", task.message_id);
        return Ok(TaskOutcome::Ack);
    };

    let history = models::history(pool, task.dialog_id, task.message_id).await?;

    match llm.chat(&model, &history).await {
        Ok(answer) => {
            models::complete_message(pool, task.dialog_id, task.message_id, &answer).await?;
            Ok(TaskOutcome::Ack)
        }
        Err(e) if e.is_retryable() && attempt < max_attempts => {
            warn!("message {} attempt {attempt}/{max_attempts} failed, retrying: {e}", task.message_id);
            tokio::time::sleep(RETRY_DELAY * attempt).await;
            Ok(TaskOutcome::Nack { requeue: true })
        }
        Err(e) => {
            error!("message {} failed after {attempt} attempt(s): {e}", task.message_id);
            models::fail_message(pool, task.message_id, e.client_reason()).await?;
            Ok(TaskOutcome::Ack)
        }
    }
}
