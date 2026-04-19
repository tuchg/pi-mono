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
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "tool_calls" | "function_call" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_string()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_string()),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
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

    let mut i = 0;
    while i < transformed.len() {
        let msg = &transformed[i];
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
                let mut thinking_parts: Vec<String> = Vec::new();
                let mut thinking_signature: Option<String> = None;

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Text(t) => {
                            if !t.text.trim().is_empty() {
                                content_text.push_str(&sanitize_surrogates(&t.text));
                            }
                        }
                        AssistantContent::Thinking(t) => {
                            if !t.thinking.trim().is_empty() {
                                thinking_parts.push(sanitize_surrogates(&t.thinking));
                                if thinking_signature.is_none() {
                                    thinking_signature = t.thinking_signature.clone();
                                }
                            }
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

                // Skip assistant messages with no content and no tool calls.
                let has_content = !content_text.is_empty();
                if has_content || !tool_calls.is_empty() {
                    let mut msg = serde_json::json!({"role": "assistant"});
                    if has_content {
                        msg["content"] = serde_json::Value::String(content_text);
                    } else {
                        msg["content"] = serde_json::Value::Null;
                    }
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = serde_json::json!(tool_calls);
                    }

                    // Include thinking/reasoning via the signature field name
                    // (e.g., "reasoning_content", "reasoning") when available.
                    if !thinking_parts.is_empty() {
                        if let Some(ref sig) = thinking_signature {
                            if !sig.is_empty() {
                                msg[sig] = serde_json::Value::String(thinking_parts.join("\n"));
                            }
                        }
                    }

                    messages.push(msg);
                }
            }
            crate::types::Message::ToolResult(_) => {
                // Batch consecutive tool results and collect their images.
                let mut image_parts: Vec<serde_json::Value> = Vec::new();
                let mut j = i;

                while j < transformed.len() {
                    if let crate::types::Message::ToolResult(tr) = &transformed[j] {
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

                        let has_images = tr.content.iter().any(|c| matches!(c, Content::Image(_)));
                        let has_text = !text_result.is_empty();
                        let content_str = if has_text {
                            sanitize_surrogates(&text_result)
                        } else if has_images {
                            "(see attached image)".to_string()
                        } else {
                            "(no output)".to_string()
                        };

                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": content_str,
                        }));

                        if supports_image && has_images {
                            for c in &tr.content {
                                if let Content::Image(img) = c {
                                    image_parts.push(serde_json::json!({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": format!("data:{};base64,{}", img.mime_type, img.data),
                                        },
                                    }));
                                }
                            }
                        }

                        j += 1;
                    } else {
                        break;
                    }
                }

                if !image_parts.is_empty() {
                    let mut content: Vec<serde_json::Value> = vec![serde_json::json!({
                        "type": "text",
                        "text": "Attached image(s) from tool result:",
                    })];
                    content.extend(image_parts);
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }

                i = j;
                continue;
            }
        }
        i += 1;
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
    let reported_cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let cache_write = usage["prompt_tokens_details"]["cache_write_tokens"]
        .as_u64()
        .unwrap_or(0);
    let reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0);

    // Normalize cache semantics: some providers report cached_tokens as
    // (previous hits + current writes). Subtract cacheWrite when present.
    let cache_read = if cache_write > 0 {
        reported_cached.saturating_sub(cache_write)
    } else {
        reported_cached
    };

    let input = prompt_tokens.saturating_sub(cache_read).saturating_sub(cache_write);
    let output = completion_tokens + reasoning_tokens;
    let total_tokens = input + output + cache_read + cache_write;

    let mut u = crate::types::Usage {
        input,
        output,
        cache_read,
        cache_write,
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
    fn stop_reason_end() {
        assert_eq!(
            map_completions_stop_reason("end"),
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
        assert_eq!(msg.as_deref(), Some("Provider finish_reason: content_filter"));
    }

    #[test]
    fn stop_reason_network_error() {
        let (reason, msg) = map_completions_stop_reason("network_error");
        assert_eq!(reason, StopReason::Error);
        assert_eq!(msg.as_deref(), Some("Provider finish_reason: network_error"));
    }

    #[test]
    fn stop_reason_unknown_includes_reason() {
        let (reason, msg) = map_completions_stop_reason("something_weird");
        assert_eq!(reason, StopReason::Error);
        assert_eq!(
            msg.as_deref(),
            Some("Provider finish_reason: something_weird")
        );
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

    #[test]
    fn has_tool_history_with_assistant_tool_call() {
        let msgs = vec![crate::types::Message::Assistant(
            crate::types::AssistantMessage {
                content: vec![crate::types::AssistantContent::ToolCall(
                    crate::types::ToolCall {
                        id: "tc1".to_string(),
                        name: "test".to_string(),
                        arguments: serde_json::json!({}),
                        thought_signature: None,
                    },
                )],
                ..Default::default()
            },
        )];
        assert!(has_tool_history(&msgs));
    }

    #[test]
    fn has_tool_history_empty() {
        assert!(!has_tool_history(&[]));
    }

    #[test]
    fn convert_tools_format() {
        let tools = vec![Tool {
            name: "my_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        }];
        let result = convert_completions_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "function");
        assert_eq!(result[0]["function"]["name"], "my_tool");
        assert_eq!(result[0]["function"]["description"], "A test tool");
        assert!(result[0]["function"]["parameters"]["properties"]["x"]["type"].is_string());
    }

    #[test]
    fn convert_tools_empty() {
        let result = convert_completions_tools(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_chunk_usage_basic() {
        let model = Model {
            cost: crate::types::ModelCost {
                input: 3.0,
                output: 15.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let usage_json = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
        });
        let usage = parse_chunk_usage(&usage_json, &model);
        assert_eq!(usage.input, 100);
        assert_eq!(usage.output, 50);
        assert_eq!(usage.cache_read, 0);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn parse_chunk_usage_with_cache() {
        let model = Model {
            cost: crate::types::ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            ..Default::default()
        };
        let usage_json = serde_json::json!({
            "prompt_tokens": 200,
            "completion_tokens": 50,
            "prompt_tokens_details": {
                "cached_tokens": 120,
                "cache_write_tokens": 30
            }
        });
        let usage = parse_chunk_usage(&usage_json, &model);
        // cache_read = cached_tokens - cache_write = 120 - 30 = 90
        assert_eq!(usage.cache_read, 90);
        assert_eq!(usage.cache_write, 30);
        // input = prompt_tokens - cache_read - cache_write = 200 - 90 - 30 = 80
        assert_eq!(usage.input, 80);
        assert_eq!(usage.output, 50);
    }

    #[test]
    fn parse_chunk_usage_with_reasoning_tokens() {
        let model = Model::default();
        let usage_json = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {
                "reasoning_tokens": 30
            }
        });
        let usage = parse_chunk_usage(&usage_json, &model);
        // output = completion_tokens + reasoning_tokens = 50 + 30 = 80
        assert_eq!(usage.output, 80);
    }
}
