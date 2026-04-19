//! Shared utilities for the Anthropic provider.
//!
//! Port of conversion and helper logic from
//! `packages/ai/src/providers/anthropic.ts`.


use crate::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use crate::types::{
    AssistantContent, AssistantMessage, Content, Context, InputModality, Model,
    StopReason, TextContent, ThinkingLevel, Tool,
};
use crate::utils::sanitize_unicode::sanitize_surrogates;

// =============================================================================
// Claude Code stealth-mode tool name mapping
// =============================================================================

/// Claude Code 2.x canonical tool names.
static CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// Convert a tool name to Claude Code canonical casing (case-insensitive match).
pub fn to_claude_code_name(name: &str) -> String {
    let lower = name.to_lowercase();
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|t| t.to_lowercase() == lower)
        .map(|t| t.to_string())
        .unwrap_or_else(|| name.to_string())
}

/// Convert from Claude Code name back to the name used in our tool list.
pub fn from_claude_code_name(name: &str, tools: Option<&[Tool]>) -> String {
    if let Some(tools) = tools {
        if !tools.is_empty() {
            let lower = name.to_lowercase();
            if let Some(matched) = tools.iter().find(|t| t.name.to_lowercase() == lower) {
                return matched.name.clone();
            }
        }
    }
    name.to_string()
}

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map Anthropic stop reason string to our StopReason.
pub fn map_anthropic_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Error,
    }
}

// =============================================================================
// Adaptive thinking utilities
// =============================================================================

/// Anthropic effort levels for adaptive thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl std::fmt::Display for AnthropicEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Xhigh => write!(f, "xhigh"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// Check if a model supports adaptive thinking (Opus 4.6+, Sonnet 4.6).
pub fn supports_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6")
        || model_id.contains("opus-4.6")
        || model_id.contains("opus-4-7")
        || model_id.contains("opus-4.7")
        || model_id.contains("sonnet-4-6")
        || model_id.contains("sonnet-4.6")
}

/// Map ThinkingLevel to Anthropic effort for adaptive thinking.
pub fn map_thinking_level_to_effort(
    level: Option<ThinkingLevel>,
    model_id: &str,
) -> AnthropicEffort {
    match level {
        Some(ThinkingLevel::Minimal) | Some(ThinkingLevel::Low) => AnthropicEffort::Low,
        Some(ThinkingLevel::Medium) => AnthropicEffort::Medium,
        Some(ThinkingLevel::High) => AnthropicEffort::High,
        Some(ThinkingLevel::Xhigh) => {
            if model_id.contains("opus-4-6") || model_id.contains("opus-4.6") {
                AnthropicEffort::Max
            } else if model_id.contains("opus-4-7") || model_id.contains("opus-4.7") {
                AnthropicEffort::Xhigh
            } else {
                AnthropicEffort::High
            }
        }
        None => AnthropicEffort::High,
    }
}

/// Check if API key is an OAuth token.
pub fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

// =============================================================================
// Cache retention
// =============================================================================

/// Get Anthropic cache control based on base URL and retention preference.
pub fn get_anthropic_cache_control(
    base_url: &str,
    cache_retention: Option<crate::types::CacheRetention>,
) -> (
    crate::types::CacheRetention,
    Option<serde_json::Value>,
) {
    let retention = crate::providers::openai_responses_shared::resolve_cache_retention(
        cache_retention,
    );
    if retention == crate::types::CacheRetention::None {
        return (retention, None);
    }

    let ttl = if retention == crate::types::CacheRetention::Long
        && base_url.contains("api.anthropic.com")
    {
        Some("1h")
    } else {
        None
    };

    let cache_control = if let Some(ttl_val) = ttl {
        serde_json::json!({"type": "ephemeral", "ttl": ttl_val})
    } else {
        serde_json::json!({"type": "ephemeral"})
    };

    (retention, Some(cache_control))
}

// =============================================================================
// Message conversion
// =============================================================================

/// Convert internal messages to Anthropic API format.
///
/// Port of the message conversion logic from `buildParams()` in
/// `packages/ai/src/providers/anthropic.ts`.
pub fn convert_anthropic_messages(
    model: &Model,
    context: &Context,
    is_oauth: bool,
) -> (
    Option<Vec<serde_json::Value>>,
    Vec<serde_json::Value>,
) {
    let normalize_fn: NormalizeToolCallIdFn =
        Box::new(|id: &str, _target_model: &Model, _source: &AssistantMessage| -> String {
            id.to_string()
        });

    let transformed_messages = transform_messages(&context.messages, model, Some(&normalize_fn));

    // System prompt as Anthropic system blocks
    let system_blocks = context.system_prompt.as_ref().map(|prompt| {
        vec![serde_json::json!({
            "type": "text",
            "text": sanitize_surrogates(prompt),
        })]
    });

    let mut messages: Vec<serde_json::Value> = Vec::new();

    for msg in &transformed_messages {
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
                        let has_images = parts
                            .iter()
                            .any(|p| matches!(p, crate::types::UserContentPart::Image(_)));
                        if !has_images {
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
                            messages.push(serde_json::json!({
                                "role": "user",
                                "content": sanitize_surrogates(&text),
                            }));
                        } else {
                            let content: Vec<serde_json::Value> = parts
                                .iter()
                                .filter_map(|p| match p {
                                    crate::types::UserContentPart::Text(t) => {
                                        Some(serde_json::json!({
                                            "type": "text",
                                            "text": sanitize_surrogates(&t.text),
                                        }))
                                    }
                                    crate::types::UserContentPart::Image(img) => {
                                        if model.input.contains(&InputModality::Image) {
                                            Some(serde_json::json!({
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": img.mime_type,
                                                    "data": img.data,
                                                }
                                            }))
                                        } else {
                                            None
                                        }
                                    }
                                })
                                .collect();
                            messages.push(serde_json::json!({
                                "role": "user",
                                "content": content,
                            }));
                        }
                    }
                }
            }
            crate::types::Message::Assistant(assistant_msg) => {
                let mut content: Vec<serde_json::Value> = Vec::new();
                let is_same_model =
                    assistant_msg.provider == model.provider && assistant_msg.model == model.id;

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Thinking(t) => {
                            if t.redacted.is_some_and(|v| v) {
                                if is_same_model {
                                    content.push(serde_json::json!({
                                        "type": "redacted_thinking",
                                        "data": t.thinking_signature.as_deref().unwrap_or(""),
                                    }));
                                }
                            } else if is_same_model {
                                let mut block_json = serde_json::json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&t.thinking),
                                });
                                if let Some(ref sig) = t.thinking_signature {
                                    block_json["signature"] = serde_json::Value::String(sig.clone());
                                }
                                content.push(block_json);
                            } else if !t.thinking.trim().is_empty() {
                                content.push(serde_json::json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&t.thinking),
                                }));
                            }
                        }
                        AssistantContent::Text(text_block) => {
                            content.push(serde_json::json!({
                                "type": "text",
                                "text": sanitize_surrogates(&text_block.text),
                            }));
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let name = if is_oauth {
                                to_claude_code_name(&tool_call.name)
                            } else {
                                tool_call.name.clone()
                            };
                            content.push(serde_json::json!({
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": name,
                                "input": tool_call.arguments,
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
                let has_images = tr.content.iter().any(|c| matches!(c, Content::Image(_)));

                let content = if has_images && model.input.contains(&InputModality::Image) {
                    let mut blocks: Vec<serde_json::Value> = Vec::new();
                    if !text_result.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": sanitize_surrogates(&text_result),
                        }));
                    }
                    for block in &tr.content {
                        if let Content::Image(img) = block {
                            blocks.push(serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": img.mime_type,
                                    "data": img.data,
                                }
                            }));
                        }
                    }
                    if blocks.iter().all(|b| b["type"] != "text") {
                        blocks.insert(
                            0,
                            serde_json::json!({"type": "text", "text": "(see attached image)"}),
                        );
                    }
                    serde_json::json!(blocks)
                } else {
                    let text = if text_result.is_empty() {
                        "(no output)".to_string()
                    } else {
                        sanitize_surrogates(&text_result)
                    };
                    serde_json::json!(text)
                };

                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": content,
                        "is_error": tr.is_error,
                    }],
                }));
            }
        }
    }

    (system_blocks, messages)
}

/// Convert internal tools to Anthropic format.
pub fn convert_anthropic_tools(tools: &[Tool], is_oauth: bool) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            let name = if is_oauth {
                to_claude_code_name(&tool.name)
            } else {
                tool.name.clone()
            };
            serde_json::json!({
                "name": name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_code_name_mapping() {
        assert_eq!(to_claude_code_name("read"), "Read");
        assert_eq!(to_claude_code_name("BASH"), "Bash");
        assert_eq!(to_claude_code_name("unknown_tool"), "unknown_tool");
    }

    #[test]
    fn from_claude_code_name_with_tools() {
        let tools = vec![Tool {
            name: "myTool".to_string(),
            description: "desc".to_string(),
            parameters: serde_json::json!({}),
        }];
        assert_eq!(from_claude_code_name("MYTOOL", Some(&tools)), "myTool");
    }

    #[test]
    fn map_stop_reason() {
        assert_eq!(map_anthropic_stop_reason("end_turn"), StopReason::Stop);
        assert_eq!(map_anthropic_stop_reason("max_tokens"), StopReason::Length);
        assert_eq!(map_anthropic_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_anthropic_stop_reason("unknown"), StopReason::Error);
    }

    #[test]
    fn adaptive_thinking_detection() {
        assert!(supports_adaptive_thinking("claude-opus-4.6-20250414"));
        assert!(supports_adaptive_thinking("claude-sonnet-4.6"));
        assert!(supports_adaptive_thinking("claude-opus-4-7"));
        assert!(!supports_adaptive_thinking("claude-sonnet-4-20250514"));
    }

    #[test]
    fn effort_mapping() {
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Low), "claude-opus-4.6"),
            AnthropicEffort::Low
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Xhigh), "claude-opus-4.6-20250414"),
            AnthropicEffort::Max
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Xhigh), "claude-opus-4.7"),
            AnthropicEffort::Xhigh
        );
    }

    #[test]
    fn oauth_token_detection() {
        assert!(is_oauth_token("sk-ant-oat-abc123"));
        assert!(!is_oauth_token("sk-ant-api-abc123"));
    }
}
