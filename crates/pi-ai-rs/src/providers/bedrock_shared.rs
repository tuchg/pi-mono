//! Shared utilities for the Amazon Bedrock provider.
//!
//! Port of conversion and helper logic from
//! `packages/ai/src/providers/amazon-bedrock.ts`.


use crate::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use crate::types::{
    AssistantContent, AssistantMessage, CacheRetention, Content, Context,
    InputModality, Model, StopReason, TextContent, ThinkingLevel, Tool,
};
use crate::utils::sanitize_unicode::sanitize_surrogates;

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map Bedrock stop reason string to our StopReason.
pub fn map_bedrock_stop_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("end_turn") => StopReason::Stop,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::Length,
        Some("stop_sequence") => StopReason::Stop,
        Some("content_filtered") => StopReason::Error,
        _ => StopReason::Error,
    }
}

// =============================================================================
// Cache control
// =============================================================================

/// Build cache point block for Bedrock.
fn build_cache_point(retention: CacheRetention) -> Option<serde_json::Value> {
    match retention {
        CacheRetention::None => None,
        CacheRetention::Short => Some(serde_json::json!({
            "cachePoint": {"type": "default"}
        })),
        CacheRetention::Long => Some(serde_json::json!({
            "cachePoint": {"type": "default", "ttl": "1h"}
        })),
    }
}

// =============================================================================
// Thinking level utilities
// =============================================================================

/// Default thinking budgets per level.
pub fn default_thinking_budget(level: ThinkingLevel) -> u32 {
    match level {
        ThinkingLevel::Minimal => 1024,
        ThinkingLevel::Low => 4096,
        ThinkingLevel::Medium => 10240,
        ThinkingLevel::High => 32768,
        ThinkingLevel::Xhigh => 65536,
    }
}

// =============================================================================
// Message conversion
// =============================================================================

/// Convert internal messages to Bedrock Converse format.
///
/// Port of `convertMessages()` from `packages/ai/src/providers/amazon-bedrock.ts`.
pub fn convert_bedrock_messages(
    context: &Context,
    model: &Model,
    cache_retention: CacheRetention,
) -> Vec<serde_json::Value> {
    let normalize_fn: NormalizeToolCallIdFn =
        Box::new(|id: &str, _m: &Model, _s: &AssistantMessage| -> String { id.to_string() });
    let transformed = transform_messages(&context.messages, model, Some(&normalize_fn));
    let mut messages: Vec<serde_json::Value> = Vec::new();

    for msg in &transformed {
        match msg {
            crate::types::Message::User(user) => {
                let content = match &user.content {
                    crate::types::UserContent::Text(text) => {
                        vec![serde_json::json!({"text": sanitize_surrogates(text)})]
                    }
                    crate::types::UserContent::Parts(parts) => {
                        let content_blocks: Vec<serde_json::Value> = parts
                            .iter()
                            .filter_map(|p| match p {
                                crate::types::UserContentPart::Text(t) => {
                                    Some(serde_json::json!({"text": sanitize_surrogates(&t.text)}))
                                }
                                crate::types::UserContentPart::Image(img) => {
                                    if model.input.contains(&InputModality::Image) {
                                        let format = mime_to_bedrock_format(&img.mime_type);
                                        Some(serde_json::json!({
                                            "image": {
                                                "format": format,
                                                "source": {"bytes": img.data},
                                            }
                                        }))
                                    } else {
                                        None
                                    }
                                }
                            })
                            .collect();

                        if content_blocks.is_empty() {
                            continue;
                        }
                        content_blocks
                    }
                };

                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content,
                }));
            }
            crate::types::Message::Assistant(assistant_msg) => {
                let is_same_model =
                    assistant_msg.provider == model.provider && assistant_msg.model == model.id;
                let mut content: Vec<serde_json::Value> = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Thinking(t) => {
                            if t.redacted.is_some_and(|v| v) {
                                if is_same_model {
                                    content.push(serde_json::json!({
                                        "reasoningContent": {
                                            "redactedContent": t.thinking_signature.as_deref().unwrap_or(""),
                                        }
                                    }));
                                }
                            } else if is_same_model {
                                let mut reasoning = serde_json::json!({
                                    "reasoningContent": {
                                        "reasoningText": {"text": sanitize_surrogates(&t.thinking)},
                                    }
                                });
                                if let Some(ref sig) = t.thinking_signature {
                                    reasoning["reasoningContent"]["signature"] =
                                        serde_json::Value::String(sig.clone());
                                }
                                content.push(reasoning);
                            } else if !t.thinking.trim().is_empty() {
                                content.push(serde_json::json!({"text": sanitize_surrogates(&t.thinking)}));
                            }
                        }
                        AssistantContent::Text(text_block) => {
                            content.push(serde_json::json!({"text": sanitize_surrogates(&text_block.text)}));
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            content.push(serde_json::json!({
                                "toolUse": {
                                    "toolUseId": tool_call.id,
                                    "name": tool_call.name,
                                    "input": tool_call.arguments,
                                }
                            }));
                        }
                    }
                }

                if !content.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
            }
            crate::types::Message::ToolResult(tr) => {
                let text_content: Vec<&TextContent> = tr
                    .content
                    .iter()
                    .filter_map(|c| {
                        if let Content::Text(t) = c {
                            Some(t)
                        } else {
                            None
                        }
                    })
                    .collect();
                let text_result: String =
                    text_content.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");

                let mut tool_result_content: Vec<serde_json::Value> = Vec::new();
                if !text_result.is_empty() {
                    tool_result_content.push(serde_json::json!({
                        "text": sanitize_surrogates(&text_result),
                    }));
                }

                // Add images if supported
                if model.input.contains(&InputModality::Image) {
                    for block in &tr.content {
                        if let Content::Image(img) = block {
                            let format = mime_to_bedrock_format(&img.mime_type);
                            tool_result_content.push(serde_json::json!({
                                "image": {
                                    "format": format,
                                    "source": {"bytes": img.data},
                                }
                            }));
                        }
                    }
                }

                if tool_result_content.is_empty() {
                    tool_result_content.push(serde_json::json!({"text": "(no output)"}));
                }

                let status = if tr.is_error { "error" } else { "success" };

                // Merge tool results into the last user turn if applicable
                let should_merge = messages
                    .last()
                    .and_then(|m| m["role"].as_str())
                    .map_or(false, |r| r == "user");

                let tool_result = serde_json::json!({
                    "toolResult": {
                        "toolUseId": tr.tool_call_id,
                        "content": tool_result_content,
                        "status": status,
                    }
                });

                if should_merge {
                    if let Some(last) = messages.last_mut() {
                        if let Some(arr) = last["content"].as_array_mut() {
                            arr.push(tool_result);
                        }
                    }
                } else {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": [tool_result],
                    }));
                }
            }
        }
    }

    // Add cache point to the last message if configured
    if let Some(cache_point) = build_cache_point(cache_retention) {
        if let Some(last) = messages.last_mut() {
            if let Some(arr) = last["content"].as_array_mut() {
                arr.push(cache_point);
            }
        }
    }

    messages
}

/// Build system prompt blocks for Bedrock.
pub fn build_bedrock_system_prompt(
    system_prompt: Option<&str>,
    _model: &Model,
    cache_retention: CacheRetention,
) -> Option<Vec<serde_json::Value>> {
    let prompt = system_prompt?;
    let mut blocks = vec![serde_json::json!({"text": sanitize_surrogates(prompt)})];
    if let Some(cache_point) = build_cache_point(cache_retention) {
        blocks.push(cache_point);
    }
    Some(blocks)
}

/// Convert tools to Bedrock tool configuration.
pub fn convert_bedrock_tools(
    tools: Option<&[Tool]>,
    tool_choice: Option<&str>,
) -> Option<serde_json::Value> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }

    let tool_defs: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.parameters},
                }
            })
        })
        .collect();

    let mut config = serde_json::json!({"tools": tool_defs});

    if let Some(choice) = tool_choice {
        match choice {
            "auto" => {
                config["toolChoice"] = serde_json::json!({"auto": {}});
            }
            "any" => {
                config["toolChoice"] = serde_json::json!({"any": {}});
            }
            "none" => {
                // Bedrock doesn't support "none" directly — omit toolChoice
            }
            _ => {}
        }
    }

    Some(config)
}

/// Map MIME type to Bedrock image format string.
fn mime_to_bedrock_format(mime_type: &str) -> &str {
    match mime_type {
        "image/jpeg" | "image/jpg" => "jpeg",
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "png",
    }
}

/// Bedrock error prefixes for human-readable error messages.
pub fn format_bedrock_error_prefix(exception_name: &str) -> &str {
    match exception_name {
        "InternalServerException" => "Internal server error",
        "ModelStreamErrorException" => "Model stream error",
        "ValidationException" => "Validation error",
        "ThrottlingException" => "Throttling error",
        "ServiceUnavailableException" => "Service unavailable",
        _ => exception_name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_stop_reasons() {
        assert_eq!(map_bedrock_stop_reason(Some("end_turn")), StopReason::Stop);
        assert_eq!(
            map_bedrock_stop_reason(Some("tool_use")),
            StopReason::ToolUse
        );
        assert_eq!(
            map_bedrock_stop_reason(Some("max_tokens")),
            StopReason::Length
        );
        assert_eq!(
            map_bedrock_stop_reason(Some("content_filtered")),
            StopReason::Error
        );
    }

    #[test]
    fn thinking_budgets() {
        assert_eq!(default_thinking_budget(ThinkingLevel::Minimal), 1024);
        assert_eq!(default_thinking_budget(ThinkingLevel::High), 32768);
    }

    #[test]
    fn mime_to_format() {
        assert_eq!(mime_to_bedrock_format("image/jpeg"), "jpeg");
        assert_eq!(mime_to_bedrock_format("image/png"), "png");
        assert_eq!(mime_to_bedrock_format("image/gif"), "gif");
        assert_eq!(mime_to_bedrock_format("unknown"), "png");
    }

    #[test]
    fn error_prefix() {
        assert_eq!(
            format_bedrock_error_prefix("ThrottlingException"),
            "Throttling error"
        );
        assert_eq!(
            format_bedrock_error_prefix("CustomError"),
            "CustomError"
        );
    }
}
