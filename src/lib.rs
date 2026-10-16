//! Shared by `dialog-stream-service` (Iggy RPC) and `dialog-mq-service` (LLM worker).

use serde_derive::Deserialize;
use skw_lib_shared::prelude::{APP, consts::RABBITMQ_QUEUES_PATH, rabbitmq::QueueConfig};

pub mod llm;
pub mod models;

/// Base name of the queue `Ask` publishes `DialogAskTask`s to (entry in `/rabbitmq/queues`).
pub const ASK_QUEUE: &str = "dialog-ask";

#[derive(Debug, Deserialize)]
struct QueueSettings {
    name: String,
    prefetch: u16,
}

/// Queue config for `name` from `/rabbitmq/queues`; panics if it is missing (startup only).
pub fn named_queue(name: &str) -> QueueConfig {
    let queues: Vec<QueueSettings> = serde_json::from_value(APP.config.expect_value_cloned(RABBITMQ_QUEUES_PATH)).expect("invalid /rabbitmq/queues config");

    let settings = queues
        .iter()
        .find(|q| q.name == name)
        .unwrap_or_else(|| panic!("queue '{name}' missing from /rabbitmq/queues config"));

    QueueConfig::new(settings.name.as_str()).prefetch(settings.prefetch)
}
