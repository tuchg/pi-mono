//! Shared utilities for the OpenAI Completions (Chat) provider.
//!
//! Port of conversion and helper logic from
//! `packages/ai/src/providers/openai-completions.ts`.

use crate::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use crate::types::{
    AssistantContent, AssistantMessage, Content, Context, InputModality, Model,
    StopReason, Tool,
};
use crate::utils::sanitize_unicode::sanitize_surrogates;

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map OpenAI Chat Completion finish reason to our StopReason.
pub fn map_completions_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "tool_calls" | "function_call" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Content filtered by safety system".to_string()),
        ),
        _ => (StopReason::Error, None),
    }
}

// =============================================================================
// Message conversion
// =============================================================================

/// Convert internal messages to OpenAI Chat Completions format.
///
/// Port of message conversion from `packages/ai/src/providers/openai-completions.ts`.
pub fn convert_completions_messages(
    model: &Model,
    context: &Context,
) -> Vec<serde_json::Value> {
    let normalize_tool_call_id: NormalizeToolCallIdFn =
        Box::new(|id: &str, _target_model: &Model, _source: &AssistantMessage| -> String { id.to_string() });

    let transformed = transform_messages(&context.messages, model, Some(&normalize_tool_call_id));
    let supports_image = model.input.contains(&InputModality::Image);
    let mut messages: Vec<serde_json::Value> = Vec::new();

    // System prompt
    if let Some(ref system_prompt) = context.system_prompt {
        let role = if model.reasoning {
            "developer"
        } else {
            "system"
        };
        messages.push(serde_json::json!({
            "role": role,
            "content": sanitize_surrogates(system_prompt),
        }));
    }

    for msg in &transformed {
        match msg {
            crate::types::Message::User(user) => {
                match &user.content {
                    crate::types::UserContent::Text(text) => {
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": sanitize_surrogates(text),
                        }));
                    }
                    crate::types::UserContent::Parts(parts) => {
                        if supports_image {
                            let content: Vec<serde_json::Value> = parts
                                .iter()
                                .map(|p| match p {
                                    crate::types::UserContentPart::Text(t) => {
                                        serde_json::json!({"type": "text", "text": sanitize_surrogates(&t.text)})
                                    }
                                    crate::types::UserContentPart::Image(img) => {
                                        serde_json::json!({
                                            "type": "image_url",
                                            "image_url": {"url": format!("data:{};base64,{}", img.mime_type, img.data)},
                                        })
                                    }
                                })
                                .collect();
                            messages.push(serde_json::json!({"role": "user", "content": content}));
                        } else {
                            let text: String = parts
                                .iter()
                                .filter_map(|p| {
                                    if let crate::types::UserContentPart::Text(t) = p {
                                        Some(t.text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            messages.push(serde_json::json!({"role": "user", "content": sanitize_surrogates(&text)}));
                        }
                    }
                }
            }
            crate::types::Message::Assistant(assistant_msg) => {
                let mut content_text = String::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();
                let mut reasoning_text = String::new();

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Text(t) => {
                            content_text.push_str(&sanitize_surrogates(&t.text));
                        }
                        AssistantContent::Thinking(t) => {
                            if !reasoning_text.is_empty() {
                                reasoning_text.push_str("\n\n");
                            }
                            reasoning_text.push_str(&sanitize_surrogates(&t.thinking));
                        }
                        AssistantContent::ToolCall(tc) => {
                            tool_calls.push(serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                                }
                            }));
                        }
                    }
                }

                let mut msg = serde_json::json!({"role": "assistant"});
                if !content_text.is_empty() || tool_calls.is_empty() {
                    msg["content"] = serde_json::Value::String(content_text);
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = serde_json::json!(tool_calls);
                }
                messages.push(msg);
            }
            crate::types::Message::ToolResult(tr) => {
                let text_result: String = tr
                    .content
                    .iter()
                    .filter_map(|c| {
                        if let Content::Text(t) = c {
                            Some(t.text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let content = if text_result.is_empty() {
                    "(no output)".to_string()
                } else {
                    sanitize_surrogates(&text_result)
                };

                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": content,
                }));
            }
        }
    }

    messages
}

/// Convert tools to OpenAI Chat Completions tool format.
pub fn convert_completions_tools(tools: &[Tool]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect()
}

/// Parse chunk usage from OpenAI streaming response.
pub fn parse_chunk_usage(
    usage: &serde_json::Value,
    model: &Model,
) -> crate::types::Usage {
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
    let total_tokens = usage["total_tokens"]
        .as_u64()
        .unwrap_or(prompt_tokens + completion_tokens);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);

    let mut u = crate::types::Usage {
        input: prompt_tokens.saturating_sub(cached),
        output: completion_tokens,
        cache_read: cached,
        cache_write: 0,
        total_tokens,
        cost: crate::types::UsageCost::default(),
    };
    u.cost = crate::models::ModelRegistry::calculate_cost(model, &u);
    u
}

/// Check if conversation contains tool calls or results.
/// Some providers (e.g., Anthropic via proxy) require the tools param
/// when messages include tool_calls or tool role messages.
pub fn has_tool_history(messages: &[crate::types::Message]) -> bool {
    messages.iter().any(|msg| match msg {
        crate::types::Message::ToolResult(_) => true,
        crate::types::Message::Assistant(a) => {
            a.content
                .iter()
                .any(|b| matches!(b, AssistantContent::ToolCall(_)))
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_stop() {
        assert_eq!(
            map_completions_stop_reason("stop"),
            (StopReason::Stop, None)
        );
    }

    #[test]
    fn stop_reason_length() {
        assert_eq!(
            map_completions_stop_reason("length"),
            (StopReason::Length, None)
        );
    }

    #[test]
    fn stop_reason_tool_calls() {
        assert_eq!(
            map_completions_stop_reason("tool_calls"),
            (StopReason::ToolUse, None)
        );
    }

    #[test]
    fn stop_reason_content_filter() {
        let (reason, msg) = map_completions_stop_reason("content_filter");
        assert_eq!(reason, StopReason::Error);
        assert!(msg.is_some());
    }

    #[test]
    fn has_tool_history_true() {
        let msgs = vec![crate::types::Message::ToolResult(
            crate::types::ToolResultMessage {
                tool_call_id: "tc_1".to_string(),
                tool_name: "test".to_string(),
                content: Vec::new(),
                details: None,
                is_error: false,
                timestamp: 0,
            },
        )];
        assert!(has_tool_history(&msgs));
    }

    #[test]
    fn has_tool_history_false() {
        let msgs = vec![crate::types::Message::User(crate::types::UserMessage {
            content: crate::types::UserContent::Text("hello".to_string()),
            timestamp: 0,
        })];
        assert!(!has_tool_history(&msgs));
    }
}
