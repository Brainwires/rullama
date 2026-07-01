//! Continuation Request Handling
//!
//! Continues a chat/TUI conversation after a tool has executed, streaming the
//! follow-up turn through the normal **provider abstraction**.
//!
//! ## History
//!
//! This module previously POSTed to the discontinued hosted Brainwires Studio
//! relay and parsed a Studio-specific SSE stream. That backend is gone.
//! The agent path (`src/agent/process.rs`,
//! `stream_continuation_with_tool_result`) already moved to the provider
//! abstraction; this module mirrors that approach for the chat-mode and TUI
//! tool-continuation paths.
//!
//! ## How it works now
//!
//! The assistant's tool call and the tool's result are appended to the
//! conversation history as proper `Message`s (a `ToolUse` block followed by a
//! `ToolResult` message), and the next turn is re-streamed via
//! `Provider::stream_chat`. If the model emits further (chained) tool calls,
//! they are executed locally and the loop repeats until the model produces a
//! plain text turn. No hosted backend, API key, or `reqwest` involvement.

use anyhow::Result;
use futures::StreamExt;
use std::sync::Arc;

use crate::providers::Provider;
use crate::tools::ToolExecutor;
use crate::types::agent::{AgentContext, PermissionMode};
use crate::types::tool::{ToolContext, ToolContextExt, ToolUse};

/// Logging callback type for tool execution messages
pub type LogCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Boxed future returned by [`send_continuation_request`]: the concatenated
/// assistant text plus the token usage accumulated across the continuation
/// turn(s).
type ContinuationFuture<'a> = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<(String, crate::types::message::Usage)>>
            + Send
            + 'a,
    >,
>;

/// Default logger that writes to stderr (for CLI mode)
pub fn default_logger() -> LogCallback {
    Arc::new(|msg: &str| {
        eprintln!("{}", msg);
    })
}

/// Continue the conversation after a tool executed, routing the follow-up turn
/// through the provider abstraction.
///
/// The assistant tool call (`call_id` / `tool_name` / `tool_parameters`) and its
/// `tool_output` are appended to `context.conversation_history`, then the next
/// turn is streamed via `provider`. Any further chained tool calls the model
/// emits are executed locally and the conversation is re-streamed until the
/// model produces a plain-text turn. Returns the concatenated assistant text
/// together with the token usage accumulated across every continuation turn
/// (so callers can feed it to the cost tracker — see [`Usage`]).
///
/// The `logger` callback emits tool execution status messages. Use
/// `default_logger()` for CLI mode, or a custom callback for TUI mode.
///
/// `model`, `chat_id`, `previous_response_id`, and `accumulated_history` are
/// retained for call-site compatibility but are no longer used now that the
/// provider (constructed with its model) owns the request.
#[allow(clippy::too_many_arguments)]
pub fn send_continuation_request<'a>(
    provider: &'a Arc<dyn Provider>,
    context: &'a AgentContext,
    _model: &'a str,
    _chat_id: Option<String>,
    _previous_response_id: &'a str,
    call_id: &'a str,
    tool_name: &'a str,
    tool_parameters: &'a serde_json::Value,
    tool_output: &'a str,
    _accumulated_history: &'a [serde_json::Value],
    logger: LogCallback,
) -> ContinuationFuture<'a> {
    Box::pin(async move {
        use crate::types::message::{ContentBlock, Message, MessageContent, Role};

        // Append the assistant's tool call and the tool result to the history,
        // then re-stream the next turn through the provider.
        let mut history = context.conversation_history.clone();
        history.push(Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                id: call_id.to_string(),
                name: tool_name.to_string(),
                input: tool_parameters.clone(),
            }]),
            name: None,
            metadata: None,
        });
        history.push(Message::tool_result(
            call_id.to_string(),
            tool_output.to_string(),
        ));

        run_provider_continuation(provider, context, history, logger).await
    })
}

/// Stream the continuation turn(s) through the provider, executing any chained
/// tool calls locally and re-streaming until the model returns plain text.
async fn run_provider_continuation(
    provider: &Arc<dyn Provider>,
    context: &AgentContext,
    mut history: Vec<crate::types::message::Message>,
    logger: LogCallback,
) -> Result<(String, crate::types::message::Usage)> {
    use crate::types::message::{ContentBlock, Message, MessageContent, Role, StreamChunk};
    use crate::types::provider::ChatOptions;

    // Accumulate token usage across every continuation turn so the caller can
    // record it. Providers emit `StreamChunk::Usage` once per stream with that
    // turn's totals; each chained tool call opens a new stream, so we sum.
    let mut acc_prompt: u32 = 0;
    let mut acc_completion: u32 = 0;

    // System prompt carried through every continuation turn.
    let system_prompt = context
        .conversation_history
        .iter()
        .find(|m| m.role == Role::System)
        .and_then(|m| m.text().map(|s| s.to_string()));

    let tool_executor = ToolExecutor::new(PermissionMode::Auto);
    let mut result_text = String::new();

    loop {
        let options = ChatOptions {
            system: system_prompt.clone(),
            ..Default::default()
        };

        let mut stream = provider.stream_chat(&history, Some(&context.tools), &options);

        let mut turn_text = String::new();
        let mut pending_tool: Option<(String, String, serde_json::Value)> = None;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result? {
                StreamChunk::Text(text) => {
                    turn_text.push_str(&text);
                }
                StreamChunk::ToolCall {
                    call_id,
                    tool_name,
                    server,
                    parameters,
                    ..
                } => {
                    // Only execute cli-local tools; ignore anything else.
                    if server != "cli-local" {
                        logger(&format!(
                            "⚠️  Skipping tool from non-local server: {}",
                            server
                        ));
                        continue;
                    }
                    logger(&format!("🔧 Chained tool requested: {}", tool_name));
                    pending_tool = Some((call_id, tool_name, parameters));
                    break;
                }
                StreamChunk::Done => break,
                StreamChunk::Usage(usage) => {
                    acc_prompt = acc_prompt.saturating_add(usage.prompt_tokens);
                    acc_completion = acc_completion.saturating_add(usage.completion_tokens);
                }
                // Other tool formats / compaction events are not needed here.
                _ => {}
            }
        }
        // Drop the stream (it borrows `history`) before we mutate `history`.
        drop(stream);

        result_text.push_str(&turn_text);

        let Some((call_id, tool_name, parameters)) = pending_tool else {
            // No further tool call: the continuation is complete.
            break;
        };

        // Execute the chained tool locally.
        logger(&format!("🔧 Executing chained tool: {}", tool_name));
        let tool_use = ToolUse {
            id: call_id.clone(),
            name: tool_name.clone(),
            input: parameters.clone(),
        };
        let tool_context = ToolContext::from_agent_context(context);
        let result = tool_executor.execute(&tool_use, &tool_context).await?;

        // Limit tool output to prevent context overflow.
        const MAX_TOOL_OUTPUT_CHARS: usize = 10_000;
        let truncated_output = if result.content.len() > MAX_TOOL_OUTPUT_CHARS {
            format!(
                "{}\n\n[Output truncated: {} of {} chars]",
                crate::utils::truncate_on_char_boundary(&result.content, MAX_TOOL_OUTPUT_CHARS),
                MAX_TOOL_OUTPUT_CHARS,
                result.content.len()
            )
        } else {
            result.content.clone()
        };

        logger(&format!(
            "✅ Chained tool {} executed successfully",
            tool_name
        ));

        // Append the chained tool call + result, then loop to re-stream.
        history.push(Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                id: call_id.clone(),
                name: tool_name,
                input: parameters,
            }]),
            name: None,
            metadata: None,
        });
        history.push(Message::tool_result(call_id, truncated_output));
    }

    Ok((
        result_text,
        crate::types::message::Usage::new(acc_prompt, acc_completion),
    ))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    // Integration tests would go here.
}
