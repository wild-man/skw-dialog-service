use std::sync::Arc;

use skw_dialog_service::{
    ASK_QUEUE,
    llm::LlmConfig,
    models::{self, AskOutcome},
    named_queue,
};
use skw_lib_http_protos::{
    JsonRpcServiceRequest,
    dialog::{RpcPayload, signed::RpcMethod},
};
use skw_lib_shared::prelude::{
    APP, AppError,
    iggy::{RpcRequestMeta, run_service_consumers_all_topic_types},
    jsonrpc::SkwServiceHttpError,
    log::*,
    postgres::{PostgresConnected, PostgresPools},
    rabbitmq::RabbitMqPublisher,
    uuid::*,
};
use skw_lib_task_protos::dialog::DialogAskTask;

const LIST_DIALOGS_DEFAULT_LIMIT: u32 = 20;
const LIST_DIALOGS_MAX_LIMIT: u32 = 100;

#[tokio::main]
async fn main() -> Result<(), AppError> {
    let db_pools = APP.postgres.connect().await?;
    let llm = Arc::new(LlmConfig::from_app_config());
    let ask_publisher = RabbitMqPublisher::new(named_queue(ASK_QUEUE)).await?;

    run_service_consumers_all_topic_types::<JsonRpcServiceRequest<RpcMethod>, _, _, _>(
        move |rpc_method, meta| {
            handle_message(
                rpc_method,
                meta,
                db_pools.clone(),
                llm.clone(),
                ask_publisher.clone(),
            )
        },
        None,
        Some(100),
    )
    .await?;

    Ok(())
}

async fn handle_message(
    rpc_method: RpcMethod,
    meta: RpcRequestMeta,
    db_pools: PostgresPools<PostgresConnected>,
    llm: Arc<LlmConfig>,
    ask_publisher: RabbitMqPublisher,
) -> Result<RpcPayload, AppError> {
    // signed topic: the gate has verified the API-KEY, which is the auth entity id
    let owner_id = meta
        .api_key
        .as_deref()
        .and_then(|k| Uuid::parse_str(k).ok())
        .ok_or(AppError::Service(SkwServiceHttpError::Forbidden))?;

    match rpc_method {
        RpcMethod::CreateDialog {
            model,
            title,
            system_prompt,
        } => {
            let model = model.unwrap_or_else(|| llm.default_model.clone());
            if !llm.models.contains(&model) {
                return Err(SkwServiceHttpError::UnprocessableEntity.into());
            }

            let pool = db_pools.master().await?;
            let dialog = models::create_dialog(pool, owner_id, &model, title, system_prompt).await?;

            Ok(RpcPayload::CreateDialog(dialog))
        }

        RpcMethod::ListDialogs { limit, before } => {
            let limit = limit
                .unwrap_or(LIST_DIALOGS_DEFAULT_LIMIT)
                .clamp(1, LIST_DIALOGS_MAX_LIMIT);

            let pool = db_pools.slave().await?;
            let dialogs = models::list_dialogs(pool, owner_id, limit, before).await?;

            Ok(RpcPayload::ListDialogs(dialogs))
        }

        RpcMethod::GetDialog {
            id,
            after_message_id,
        } => {
            let pool = db_pools.slave().await?;
            let dialog = models::get_dialog(pool, owner_id, id)
                .await?
                .ok_or(AppError::Service(SkwServiceHttpError::NotFound))?;
            let messages = models::list_messages(pool, id, after_message_id).await?;

            Ok(RpcPayload::GetDialog { dialog, messages })
        }

        RpcMethod::Ask { dialog_id, content } => {
            if content.trim().is_empty() {
                return Err(AppError::Service(SkwServiceHttpError::BadRequest));
            }

            let pool = db_pools.master().await?;
            let (user_message_id, assistant_message_id) = match models::ask(pool, owner_id, dialog_id, content).await? {
                AskOutcome::NotFound => return Err(SkwServiceHttpError::NotFound.into()),
                AskOutcome::Busy => return Err(SkwServiceHttpError::UnprocessableEntity.into()),
                AskOutcome::Created {
                    user_message_id,
                    assistant_message_id,
                } => (user_message_id, assistant_message_id),
            };

            let task = DialogAskTask {
                dialog_id,
                message_id: assistant_message_id,
            };

            if let Err(e) = ask_publisher.publish(&task).await {
                // nobody will answer it, don't leave the dialog blocked by a Pending message
                if let Err(e) = models::fail_message(pool, assistant_message_id, "EnqueueFailed").await {
                    error!("can't mark message {assistant_message_id} as failed: {e}");
                }
                return Err(e.into());
            }

            Ok(RpcPayload::Ask {
                user_message_id,
                assistant_message_id,
            })
        }

        RpcMethod::RenameDialog { id, title } => {
            let pool = db_pools.master().await?;
            let dialog = models::rename_dialog(pool, owner_id, id, title)
                .await?
                .ok_or(AppError::Service(SkwServiceHttpError::NotFound))?;

            Ok(RpcPayload::RenameDialog(dialog))
        }

        RpcMethod::DeleteDialog { id } => {
            let pool = db_pools.master().await?;
            if !models::delete_dialog(pool, owner_id, id).await? {
                return Err(AppError::Service(SkwServiceHttpError::NotFound));
            }

            Ok(RpcPayload::DeleteDialog {})
        }
    }
}
