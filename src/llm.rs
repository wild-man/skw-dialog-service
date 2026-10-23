//! OpenAI-compatible chat completions client (`POST {url}/chat/completions`),
//! works with OpenAI, Ollama, vLLM, OpenRouter, LM Studio, ...

use std::time::Duration;

use serde_derive::{Deserialize, Serialize};
use skw_lib_http_protos::dialog::MessageRole;
use skw_lib_shared::prelude::APP;
use thiserror::Error;

const LLM_CONFIG_PATH: &str = "/llm";

/// `/llm` config section.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmConfig {
    /// Base url incl. version prefix, e.g. `http://localhost:11434/v1`.
    pub url: String,
    /// Sent as `Authorization: Bearer`; empty = no auth (local models).
    #[serde(default)]
    pub api_key: String,
    /// Model of a dialog created without `Model`.
    pub default_model: String,
    /// Models a dialog may be created with.
    pub models: Vec<String>,
    pub timeout_secs: u64,
    /// LLM calls per answer before it is marked Failed.
    pub max_attempts: u32,
}

impl LlmConfig {
    /// Panics on a missing/invalid `/llm` section (startup only).
    pub fn from_app_config() -> Self {
        let cfg: LlmConfig = serde_json::from_value(APP.config.expect_value_cloned(LLM_CONFIG_PATH)).expect("invalid /llm config");

        assert!(
            cfg.models.contains(&cfg.default_model),
            "/llm/defaultModel must be one of /llm/models"
        );

        cfg
    }
}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("llm request error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("llm responded {0}: {1}")]
    Status(u16, String),
    #[error("llm returned no answer")]
    EmptyAnswer,
}

impl LlmError {
    /// Rejections (bad request, auth, unknown model) won't succeed on retry; timeouts, rate limits and 5xx may.
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Http(_) => true,
            LlmError::Status(code, _) => *code == 408 || *code == 429 || *code >= 500,
            LlmError::EmptyAnswer => false,
        }
    }

    /// Stored in `messages.error` and returned to the client, so it carries no provider details.
    pub fn client_reason(&self) -> &'static str {
        match self {
            LlmError::Http(e) if e.is_timeout() => "LlmTimeout",
            LlmError::Http(_) => "LlmUnavailable",
            LlmError::Status(429, _) => "LlmRateLimited",
            LlmError::Status(code, _) if *code >= 500 => "LlmUnavailable",
            LlmError::Status(..) => "LlmRejected",
            LlmError::EmptyAnswer => "LlmEmptyAnswer",
        }
    }
}

pub struct Answer {
    pub content: String,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[allow(dead_code)] // MOCK: only read by the commented-out LLM call in `chat`
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[allow(dead_code)] // MOCK: only read by the commented-out LLM call in `chat`
#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[allow(dead_code)] // MOCK: only read by the commented-out LLM call in `chat`
#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[allow(dead_code)] // MOCK: only read by the commented-out LLM call in `chat`
#[derive(Deserialize)]
struct ChatUsage {
    prompt_tokens: Option<i32>,
    completion_tokens: Option<i32>,
}

pub struct LlmClient {
    // own client: APP.reqwest has a 5s timeout, far too short for an LLM answer
    http: reqwest::Client,
    url: String,
    api_key: String,
}

impl LlmClient {
    /// Panics if the http client can't be built (startup only).
    pub fn new(cfg: &LlmConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("cant build llm http client");

        LlmClient {
            http,
            url: format!("{}/chat/completions", cfg.url.trim_end_matches('/')),
            api_key: cfg.api_key.clone(),
        }
    }

    pub async fn chat(&self, model: &str, history: &[(MessageRole, String)]) -> Result<Answer, LlmError> {
        let body = ChatRequest {
            model,
            messages: history
                .iter()
                .map(|(role, content)| ChatMessage {
                    role: match role {
                        MessageRole::System => "system",
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                    },
                    content,
                })
                .collect(),
        };

        // MOCK: no LLM call, answer echoes the last user message after a short delay
        // (enough for clients to see the Pending state); restore the block below for a real LLM
        let _ = (&self.http, &self.url, &self.api_key, &body);
        tokio::time::sleep(Duration::from_secs(1)).await;

        let question = history
            .iter()
            .rev()
            .find(|(role, _)| matches!(role, MessageRole::User))
            .map_or("", |(_, content)| content.as_str());

        Ok(Answer {
            content: format!("[mock {model}] you said: {question}"),
            input_tokens: Some(history.len() as i32),
            output_tokens: Some(1),
        })

        // let mut request = self.http.post(&self.url).json(&body);
        // if !self.api_key.is_empty() {
        //     request = request.bearer_auth(&self.api_key);
        // }
        //
        // let response = request.send().await?;
        // let status = response.status();
        // if !status.is_success() {
        //     let text = response.text().await.unwrap_or_default();
        //     return Err(LlmError::Status(status.as_u16(), text));
        // }
        //
        // let response: ChatResponse = response.json().await?;
        // let content = response
        //     .choices
        //     .into_iter()
        //     .next()
        //     .and_then(|c| c.message.content)
        //     .filter(|c| !c.trim().is_empty())
        //     .ok_or(LlmError::EmptyAnswer)?;
        //
        // Ok(Answer {
        //     content,
        //     input_tokens: response.usage.as_ref().and_then(|u| u.prompt_tokens),
        //     output_tokens: response.usage.as_ref().and_then(|u| u.completion_tokens),
        // })
    }
}
