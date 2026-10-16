//! `dialogs` / `messages` tables (schema `skw-dialog-service`).
//!
//! Ids are uuidv7, so `created_at` is derived (`uuid_extract_timestamp(id)`) and
//! `ORDER BY id` is chronological. Dialog access is always scoped by `owner_id`.

use std::str::FromStr;

use skw_lib_http_protos::dialog::{Dialog, Message, MessageRole, MessageStatus};
use skw_lib_shared::prelude::{chrono::*, postgres::SkwPostgresError, uuid::*};
use sqlx::PgPool;

use crate::llm::Answer;

const DIALOG_COLUMNS: &str = "id, title, model, uuid_extract_timestamp(id), updated_at";

type DialogRow = (Uuid, Option<String>, String, DateTime<Utc>, DateTime<Utc>);

fn dialog_from_row((id, title, model, created_at, updated_at): DialogRow) -> Dialog {
    Dialog {
        id,
        title,
        model,
        created_at,
        updated_at,
    }
}

const MESSAGE_COLUMNS: &str = "id, role::text, status::text, content, error, uuid_extract_timestamp(id)";

type MessageRow = (Uuid, String, String, String, Option<String>, DateTime<Utc>);

fn message_from_row((id, role, status, content, error, created_at): MessageRow) -> Result<Message, SkwPostgresError> {
    Ok(Message {
        id,
        role: parse_role(&role)?,
        status: MessageStatus::from_str(&status).map_err(|_| SkwPostgresError::Custom(format!("unknown message_status: {status}")))?,
        content,
        error,
        created_at,
    })
}

fn parse_role(role: &str) -> Result<MessageRole, SkwPostgresError> {
    MessageRole::from_str(role).map_err(|_| SkwPostgresError::Custom(format!("unknown message_role: {role}")))
}

/// Result of [`ask`].
pub enum AskOutcome {
    /// No such dialog for this owner.
    NotFound,
    /// The previous answer of this dialog is still Pending.
    Busy,
    Created {
        user_message_id: Uuid,
        assistant_message_id: Uuid,
    },
}

/// Insert a dialog, plus its `System` message when `system_prompt` is given.
pub async fn create_dialog(
    pool: &PgPool,
    owner_id: Uuid,
    model: &str,
    title: Option<String>,
    system_prompt: Option<String>,
) -> Result<Dialog, SkwPostgresError> {
    let mut tx = pool.begin().await?;

    let row: DialogRow = sqlx::query_as(&format!(
        "INSERT INTO dialogs (owner_id, title, model) VALUES ($1, $2, $3) RETURNING {DIALOG_COLUMNS}"
    ))
    .bind(owner_id)
    .bind(title)
    .bind(model)
    .fetch_one(&mut *tx)
    .await?;

    if let Some(system_prompt) = system_prompt {
        sqlx::query("INSERT INTO messages (dialog_id, role, content) VALUES ($1, 'System', $2)")
            .bind(row.0)
            .bind(system_prompt)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;

    Ok(dialog_from_row(row))
}

/// Owner's dialogs, most recently active first, older than `before` when given.
pub async fn list_dialogs(pool: &PgPool, owner_id: Uuid, limit: u32, before: Option<DateTime<Utc>>) -> Result<Vec<Dialog>, SkwPostgresError> {
    let rows: Vec<DialogRow> = sqlx::query_as(&format!(
        "SELECT {DIALOG_COLUMNS} FROM dialogs WHERE owner_id = $1 AND ($2::timestamptz IS NULL OR updated_at < $2) ORDER BY updated_at DESC LIMIT $3"
    ))
    .bind(owner_id)
    .bind(before)
    .bind(i64::from(limit))
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(dialog_from_row).collect())
}

pub async fn get_dialog(pool: &PgPool, owner_id: Uuid, id: Uuid) -> Result<Option<Dialog>, SkwPostgresError> {
    let row: Option<DialogRow> = sqlx::query_as(&format!(
        "SELECT {DIALOG_COLUMNS} FROM dialogs WHERE id = $1 AND owner_id = $2"
    ))
    .bind(id)
    .bind(owner_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(dialog_from_row))
}

/// Messages of a dialog in chronological order, only those newer than `after` when given.
pub async fn list_messages(pool: &PgPool, dialog_id: Uuid, after: Option<Uuid>) -> Result<Vec<Message>, SkwPostgresError> {
    let rows: Vec<MessageRow> = sqlx::query_as(&format!(
        "SELECT {MESSAGE_COLUMNS} FROM messages WHERE dialog_id = $1 AND ($2::uuid IS NULL OR id > $2) ORDER BY id"
    ))
    .bind(dialog_id)
    .bind(after)
    .fetch_all(pool)
    .await?;

    rows.into_iter().map(message_from_row).collect()
}

pub async fn rename_dialog(pool: &PgPool, owner_id: Uuid, id: Uuid, title: Option<String>) -> Result<Option<Dialog>, SkwPostgresError> {
    let row: Option<DialogRow> = sqlx::query_as(&format!(
        "UPDATE dialogs SET title = $3 WHERE id = $1 AND owner_id = $2 RETURNING {DIALOG_COLUMNS}"
    ))
    .bind(id)
    .bind(owner_id)
    .bind(title)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(dialog_from_row))
}

/// Deletes the dialog with its messages; `false` if there was no such dialog for this owner.
pub async fn delete_dialog(pool: &PgPool, owner_id: Uuid, id: Uuid) -> Result<bool, SkwPostgresError> {
    let result = sqlx::query("DELETE FROM dialogs WHERE id = $1 AND owner_id = $2")
        .bind(id)
        .bind(owner_id)
        .execute(pool)
        .await?;

    Ok(result.rows_affected() > 0)
}

/// Stores the question and a Pending assistant message. The dialog row is locked,
/// so concurrent `Ask`s on one dialog can't both pass the Pending check.
pub async fn ask(pool: &PgPool, owner_id: Uuid, dialog_id: Uuid, content: String) -> Result<AskOutcome, SkwPostgresError> {
    let mut tx = pool.begin().await?;

    let locked: Option<(Uuid,)> = sqlx::query_as("SELECT id FROM dialogs WHERE id = $1 AND owner_id = $2 FOR UPDATE")
        .bind(dialog_id)
        .bind(owner_id)
        .fetch_optional(&mut *tx)
        .await?;

    if locked.is_none() {
        return Ok(AskOutcome::NotFound);
    }

    let pending: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM messages WHERE dialog_id = $1 AND status = 'Pending')")
        .bind(dialog_id)
        .fetch_one(&mut *tx)
        .await?;

    if pending {
        return Ok(AskOutcome::Busy);
    }

    let (user_message_id,): (Uuid,) = sqlx::query_as("INSERT INTO messages (dialog_id, role, content) VALUES ($1, 'User', $2) RETURNING id")
        .bind(dialog_id)
        .bind(content)
        .fetch_one(&mut *tx)
        .await?;

    let (assistant_message_id,): (Uuid,) =
        sqlx::query_as("INSERT INTO messages (dialog_id, role, status) VALUES ($1, 'Assistant', 'Pending') RETURNING id")
            .bind(dialog_id)
            .fetch_one(&mut *tx)
            .await?;

    sqlx::query("UPDATE dialogs SET updated_at = now() WHERE id = $1")
        .bind(dialog_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    Ok(AskOutcome::Created {
        user_message_id,
        assistant_message_id,
    })
}

/// Counts one more LLM attempt for a Pending assistant message.
/// Returns the dialog's model and the attempt number, or `None` if the message
/// is gone (dialog deleted) or no longer Pending (already answered/failed).
pub async fn start_attempt(pool: &PgPool, message_id: Uuid) -> Result<Option<(String, u32)>, SkwPostgresError> {
    let row: Option<(String, i32)> = sqlx::query_as(
        "UPDATE messages m SET attempts = m.attempts + 1 FROM dialogs d WHERE m.id = $1 AND m.status = 'Pending' AND d.id = m.dialog_id RETURNING d.model, m.attempts",
    )
    .bind(message_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|(model, attempts)| (model, attempts.max(0) as u32)))
}

/// LLM context: answered history of the dialog preceding `message_id`.
pub async fn history(pool: &PgPool, dialog_id: Uuid, message_id: Uuid) -> Result<Vec<(MessageRole, String)>, SkwPostgresError> {
    let rows: Vec<(String, String)> =
        sqlx::query_as("SELECT role::text, content FROM messages WHERE dialog_id = $1 AND id < $2 AND status = 'Done' ORDER BY id")
            .bind(dialog_id)
            .bind(message_id)
            .fetch_all(pool)
            .await?;

    rows.into_iter()
        .map(|(role, content)| Ok((parse_role(&role)?, content)))
        .collect()
}

/// Stores the answer of a Pending assistant message and bumps the dialog's activity.
pub async fn complete_message(pool: &PgPool, dialog_id: Uuid, message_id: Uuid, answer: &Answer) -> Result<(), SkwPostgresError> {
    let mut tx = pool.begin().await?;

    sqlx::query("UPDATE messages SET status = 'Done', content = $2, input_tokens = $3, output_tokens = $4 WHERE id = $1 AND status = 'Pending'")
        .bind(message_id)
        .bind(&answer.content)
        .bind(answer.input_tokens)
        .bind(answer.output_tokens)
        .execute(&mut *tx)
        .await?;

    sqlx::query("UPDATE dialogs SET updated_at = now() WHERE id = $1")
        .bind(dialog_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    Ok(())
}

/// Marks a Pending assistant message as Failed with a client-facing `error`.
pub async fn fail_message(pool: &PgPool, message_id: Uuid, error: &str) -> Result<(), SkwPostgresError> {
    sqlx::query("UPDATE messages SET status = 'Failed', error = $2 WHERE id = $1 AND status = 'Pending'")
        .bind(message_id)
        .bind(error)
        .execute(pool)
        .await?;

    Ok(())
}
