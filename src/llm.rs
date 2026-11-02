//! OpenAI-compatible chat completions client (`POST {url}/chat/completions`),
//! works with OpenAI, Ollama, vLLM, OpenRouter, LM Studio, ...

use std::{sync::Arc, time::Duration};

use serde_derive::{Deserialize, Serialize};
use serde_json::Value;
use skw_lib_http_protos::dialog::MessageRole;
use skw_lib_shared::prelude::{APP, log::*};
use thiserror::Error;

use crate::{
    code_search::{CodeSearch, CodeSearchConfig, ToolOutput},
    profile::{self, ProfileConfig},
};

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
    /// Sent as `reasoning_effort` on every call, e.g. `"none"` turns off qwen3's thinking
    /// (Ollama ignores the `/no_think` prompt tag); unset = provider default.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Read-only tools over a source tree; unset = plain chat.
    #[serde(default)]
    pub code_search: Option<CodeSearchConfig>,
    /// The site owner's resume, added to the system prompt; unset = none.
    #[serde(default)]
    pub profile: Option<ProfileConfig>,
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
            LlmError::Status(408, _) => "LlmTimeout",
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
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl ChatMessage {
    fn new(role: &'static str, content: impl Into<String>) -> Self {
        ChatMessage {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ToolCall {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default = "function_type")]
    kind: String,
    function: FunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FunctionCall {
    name: String,
    /// JSON object as a string (OpenAI); some providers send the object itself.
    #[serde(deserialize_with = "arguments_as_string")]
    arguments: String,
}

fn arguments_as_string<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    use serde::Deserialize;
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => s,
        Value::Null => String::new(),
        other => other.to_string(),
    })
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

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
    reasoning_effort: Option<String>,
    code_search: Option<Arc<CodeSearch>>,
    tools: Value,
    /// Code-tools instructions and/or the resume, sent first in every chat.
    system_prompt: Option<String>,
}

impl LlmClient {
    /// Panics if the http client can't be built, or `/llm/codeSearch` or `/llm/profile` is invalid (startup only).
    pub fn new(cfg: &LlmConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("cant build llm http client");

        let code_search = cfg
            .code_search
            .as_ref()
            .map(|c| Arc::new(CodeSearch::new(c)));
        let prompts: Vec<String> = cfg
            .profile
            .as_ref()
            .map(profile::load_prompt)
            .into_iter()
            .chain(code_search.iter().map(|c| c.system_prompt().to_string()))
            .collect();
        let system_prompt = (!prompts.is_empty()).then(|| prompts.join("\n\n"));

        let tools = code_search
            .as_ref()
            .map_or(Value::Null, |c| c.tool_definitions());

        LlmClient {
            http,
            url: format!("{}/chat/completions", cfg.url.trim_end_matches('/')),
            api_key: cfg.api_key.clone(),
            reasoning_effort: cfg.reasoning_effort.clone(),
            code_search,
            tools,
            system_prompt,
        }
    }

    /// Answers the last message of `history`. With `/llm/codeSearch` the model may first call
    /// the code tools for up to `maxSteps` rounds; only the final text answer is returned.
    pub async fn chat(&self, model: &str, history: &[(MessageRole, String)]) -> Result<Answer, LlmError> {
        // `/<tool> ...` runs the tool directly, the model is not asked
        if let (Some(code_search), Some((MessageRole::User, last))) = (&self.code_search, history.last())
            && last.trim_start().starts_with('/')
        {
            let code_search = code_search.clone();
            let text = last.clone();
            if let Ok(Some(content)) = tokio::task::spawn_blocking(move || code_search.run_command(&text)).await {
                return Ok(Answer {
                    content,
                    input_tokens: None,
                    output_tokens: None,
                });
            }
        }

        let mut messages = Vec::with_capacity(history.len() + 1);
        if let Some(system_prompt) = &self.system_prompt {
            messages.push(ChatMessage::new("system", system_prompt.as_str()));
        }
        messages.extend(history.iter().map(|(role, content)| {
            ChatMessage::new(
                match role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                },
                content.as_str(),
            )
        }));

        let max_steps = self.code_search.as_ref().map_or(0, |c| c.max_steps());
        let (mut input_tokens, mut output_tokens) = (None, None);
        let mut step = 0;
        let mut nudged = false;
        // files the model showed with `show_file`, appended to the answer verbatim
        let mut shown: Vec<String> = Vec::new();

        loop {
            let tools_allowed = self.code_search.is_some() && step < max_steps;
            if self.code_search.is_some() && !tools_allowed {
                // the last round goes without tools, so the model has to answer
                messages.push(ChatMessage::new("system", FINAL_ROUND_PROMPT));
            }
            let response = self
                .complete(model, &messages, tools_allowed.then_some(&self.tools))
                .await?;

            if let Some(usage) = &response.usage {
                input_tokens = add_tokens(input_tokens, usage.prompt_tokens);
                output_tokens = add_tokens(output_tokens, usage.completion_tokens);
            }

            let message = response
                .choices
                .into_iter()
                .next()
                .map(|c| c.message)
                .ok_or(LlmError::EmptyAnswer)?;
            let mut tool_calls = message.tool_calls.unwrap_or_default();
            // small models sometimes write the call as text instead of a structured tool call
            let (content, text_calls) = split_text_tool_calls(message.content.as_deref().unwrap_or_default());
            if tool_calls.is_empty() {
                tool_calls = text_calls;
            }

            match &self.code_search {
                Some(code_search) if tools_allowed && !tool_calls.is_empty() => {
                    step += 1;
                    let mut assistant = ChatMessage::new("assistant", content);
                    assistant.tool_calls = tool_calls.clone();
                    messages.push(assistant);

                    for (i, call) in tool_calls.into_iter().enumerate() {
                        let id = if call.id.is_empty() {
                            format!("call_{step}_{i}")
                        } else {
                            call.id
                        };
                        info!(
                            "llm tool call {step}/{max_steps}: {}({})",
                            call.function.name, call.function.arguments
                        );

                        let code_search = code_search.clone();
                        let output = tokio::task::spawn_blocking(move || code_search.call(&call.function.name, &call.function.arguments))
                            .await
                            .unwrap_or_else(|e| ToolOutput {
                                for_model: format!("error: tool failed: {e}"),
                                shown: None,
                            });
                        // shown once, even if the model asks for the same file again
                        if let Some(block) = output.shown
                            && !shown.contains(&block)
                        {
                            shown.push(block);
                        }

                        let mut result = ChatMessage::new("tool", output.for_model);
                        result.tool_call_id = Some(id);
                        messages.push(result);
                    }
                    continue;
                }
                Some(_) if tools_allowed && step == 0 && !nudged => {
                    // answered without looking: remind it once of the tools (its first answer is dropped)
                    nudged = true;
                    messages.push(ChatMessage::new("system", NUDGE_PROMPT));
                    continue;
                }
                _ => {}
            }

            let content = [content.trim().to_string()]
                .into_iter()
                .chain(shown)
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            if content.is_empty() {
                return Err(LlmError::EmptyAnswer);
            }
            return Ok(Answer {
                content,
                input_tokens,
                output_tokens,
            });
        }
    }

    async fn complete(&self, model: &str, messages: &[ChatMessage], tools: Option<&Value>) -> Result<ChatResponse, LlmError> {
        let body = ChatRequest {
            model,
            messages,
            tools,
            reasoning_effort: self.reasoning_effort.as_deref(),
        };

        let mut request = self.http.post(&self.url).json(&body);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }

        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::Status(status.as_u16(), text));
        }

        Ok(response.json().await?)
    }
}

const NUDGE_PROMPT: &str = "Reminder: you can read the project's source code with the tools search_text, list_files, read_file and show_file \
(and show its diagrams with diagrams, if offered). \
If the user's last message is about the project or its source code, call the tools now instead of answering from memory. \
Otherwise (small talk, or a question about a person, which the resume in the system prompt answers) just answer the user.";

const FINAL_ROUND_PROMPT: &str = "No more tool calls are possible. Answer the user now, using only what the tools returned so far.";

/// Splits `<tool_call>{"name": .., "arguments": ..}</tool_call>` blocks out of a text answer:
/// returns the remaining text and the calls that parsed.
fn split_text_tool_calls(content: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find(OPEN) {
        text.push_str(&rest[..start]);
        let after = &rest[start + OPEN.len()..];
        let (block, next) = match after.find(CLOSE) {
            Some(end) => (&after[..end], &after[end + CLOSE.len()..]),
            None => (after, ""),
        };
        if let Ok(function) = serde_json::from_str::<FunctionCall>(block.trim()) {
            calls.push(ToolCall {
                id: String::new(),
                kind: function_type(),
                function,
            });
        }
        rest = next;
    }
    text.push_str(rest);
    (text.trim().to_string(), calls)
}

fn add_tokens(total: Option<i32>, more: Option<i32>) -> Option<i32> {
    match (total, more) {
        (None, None) => None,
        (total, more) => Some(total.unwrap_or(0).saturating_add(more.unwrap_or(0))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy() {
        for (code, retry, reason) in [
            (400, false, "LlmRejected"),
            (401, false, "LlmRejected"),
            (404, false, "LlmRejected"),
            (408, true, "LlmTimeout"),
            (429, true, "LlmRateLimited"),
            (500, true, "LlmUnavailable"),
            (503, true, "LlmUnavailable"),
        ] {
            let e = LlmError::Status(code, String::new());
            assert_eq!(
                (e.is_retryable(), e.client_reason()),
                (retry, reason),
                "status {code}"
            );
        }
    }

    #[test]
    fn text_tool_calls_are_split_out() {
        let (text, calls) = split_text_tool_calls(
            "Let me search.\n<tool_call>\n{\"name\": \"search_text\", \"arguments\": {\"query\": \"llm\"}}\n</tool_call>\n<tool_call>broken</tool_call>",
        );
        assert_eq!(text, "Let me search.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_text");
        assert_eq!(calls[0].function.arguments, r#"{"query":"llm"}"#);

        assert_eq!(split_text_tool_calls(" plain answer ").0, "plain answer");
    }

    #[test]
    fn empty_answer_is_final() {
        assert!(!LlmError::EmptyAnswer.is_retryable());
        assert_eq!(LlmError::EmptyAnswer.client_reason(), "LlmEmptyAnswer");
    }
}

/// Manual check against the real LLM of `local.json`:
/// `Q="question" cargo test -p skw-dialog-service live_question -- --ignored --nocapture` (from `services/dialog`).
#[cfg(test)]
mod live {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn live_question() {
        let cfg: LlmConfig =
            serde_json::from_value(serde_json::from_str::<Value>(&std::fs::read_to_string("local.json").unwrap()).unwrap()["llm"].clone()).unwrap();
        let llm = LlmClient::new(&cfg);
        let t = std::time::Instant::now();
        let q = std::env::var("Q").unwrap_or("Which port does the ws gate listen on, and where is the heartbeat interval configured?".into());
        let a = llm
            .chat(&cfg.default_model, &[(MessageRole::User, q)])
            .await
            .unwrap();
        println!(
            "{:?} in={:?} out={:?}\n{}",
            t.elapsed(),
            a.input_tokens,
            a.output_tokens,
            a.content
        );
    }
}
