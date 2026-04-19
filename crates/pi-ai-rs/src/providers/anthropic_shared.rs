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
        "end_turn" | "stop_sequence" | "pause_turn" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "refusal" | "sensitive" => StopReason::Error,
        other => panic!("Unhandled stop reason: {other}"),
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

/// Normalize tool call IDs to match Anthropic's required pattern and length.
fn normalize_tool_call_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if sanitized.len() > 64 {
        sanitized[..64].to_string()
    } else {
        sanitized
    }
}

/// Convert internal messages to Anthropic API format.
///
/// Port of the message conversion logic from `convertMessages()` in
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
            normalize_tool_call_id(id)
        });

    let transformed_messages = transform_messages(&context.messages, model, Some(&normalize_fn));

    // System prompt as Anthropic system blocks.
    // For OAuth tokens, prepend Claude Code identity.
    let system_blocks = if is_oauth {
        let mut blocks = vec![serde_json::json!({
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
        })];
        if let Some(ref prompt) = context.system_prompt {
            blocks.push(serde_json::json!({
                "type": "text",
                "text": sanitize_surrogates(prompt),
            }));
        }
        Some(blocks)
    } else {
        context.system_prompt.as_ref().map(|prompt| {
            vec![serde_json::json!({
                "type": "text",
                "text": sanitize_surrogates(prompt),
            })]
        })
    };

    let mut messages: Vec<serde_json::Value> = Vec::new();

    let mut i = 0;
    while i < transformed_messages.len() {
        let msg = &transformed_messages[i];
        match msg {
            crate::types::Message::User(user) => {
                match &user.content {
                    crate::types::UserContent::Text(text) => {
                        if !text.trim().is_empty() {
                            messages.push(serde_json::json!({
                                "role": "user",
                                "content": sanitize_surrogates(text),
                            }));
                        }
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
                            if !text.trim().is_empty() {
                                messages.push(serde_json::json!({
                                    "role": "user",
                                    "content": sanitize_surrogates(&text),
                                }));
                            }
                        } else {
                            let mut content: Vec<serde_json::Value> = parts
                                .iter()
                                .filter_map(|p| match p {
                                    crate::types::UserContentPart::Text(t) => {
                                        if t.text.trim().is_empty() {
                                            None
                                        } else {
                                            Some(serde_json::json!({
                                                "type": "text",
                                                "text": sanitize_surrogates(&t.text),
                                            }))
                                        }
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
                            // Filter images when model doesn't support them
                            if !model.input.contains(&InputModality::Image) {
                                content.retain(|b| b["type"] != "image");
                            }
                            if !content.is_empty() {
                                messages.push(serde_json::json!({
                                    "role": "user",
                                    "content": content,
                                }));
                            }
                        }
                    }
                }
            }
            crate::types::Message::Assistant(assistant_msg) => {
                let mut content: Vec<serde_json::Value> = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Thinking(t) => {
                            if t.redacted.is_some_and(|v| v) {
                                // Redacted thinking: pass the opaque payload back
                                content.push(serde_json::json!({
                                    "type": "redacted_thinking",
                                    "data": t.thinking_signature.as_deref().unwrap_or(""),
                                }));
                            } else {
                                if t.thinking.trim().is_empty() {
                                    continue;
                                }
                                // If thinking signature is missing/empty (e.g., from
                                // aborted stream), convert to plain text block to
                                // avoid API rejection.
                                let sig = t.thinking_signature.as_deref().unwrap_or("");
                                if sig.trim().is_empty() {
                                    content.push(serde_json::json!({
                                        "type": "text",
                                        "text": sanitize_surrogates(&t.thinking),
                                    }));
                                } else {
                                    content.push(serde_json::json!({
                                        "type": "thinking",
                                        "thinking": sanitize_surrogates(&t.thinking),
                                        "signature": sig,
                                    }));
                                }
                            }
                        }
                        AssistantContent::Text(text_block) => {
                            if text_block.text.trim().is_empty() {
                                continue;
                            }
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
                // Collect all consecutive toolResult messages into a single
                // user message (needed for z.ai Anthropic endpoint).
                let mut tool_results: Vec<serde_json::Value> = Vec::new();

                tool_results.push(build_tool_result_block(tr, model));

                // Look ahead for consecutive toolResult messages
                while i + 1 < transformed_messages.len() {
                    if let crate::types::Message::ToolResult(next_tr) =
                        &transformed_messages[i + 1]
                    {
                        tool_results.push(build_tool_result_block(next_tr, model));
                        i += 1;
                    } else {
                        break;
                    }
                }

                messages.push(serde_json::json!({
                    "role": "user",
                    "content": tool_results,
                }));
            }
        }
        i += 1;
    }

    (system_blocks, messages)
}

/// Build a single `tool_result` content block from a `ToolResultMessage`.
fn build_tool_result_block(
    tr: &crate::types::ToolResultMessage,
    model: &Model,
) -> serde_json::Value {
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

    serde_json::json!({
        "type": "tool_result",
        "tool_use_id": tr.tool_call_id,
        "content": content,
        "is_error": tr.is_error,
    })
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
    use crate::types::CacheRetention;

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
        assert_eq!(map_anthropic_stop_reason("stop_sequence"), StopReason::Stop);
        assert_eq!(map_anthropic_stop_reason("pause_turn"), StopReason::Stop);
        assert_eq!(map_anthropic_stop_reason("max_tokens"), StopReason::Length);
        assert_eq!(map_anthropic_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_anthropic_stop_reason("refusal"), StopReason::Error);
        assert_eq!(map_anthropic_stop_reason("sensitive"), StopReason::Error);
    }

    #[test]
    fn normalize_tool_call_id_basic() {
        assert_eq!(normalize_tool_call_id("abc-123_def"), "abc-123_def");
        assert_eq!(normalize_tool_call_id("a@b#c"), "a_b_c");
        // Truncation at 64 chars
        let long = "a".repeat(100);
        assert_eq!(normalize_tool_call_id(&long).len(), 64);
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
            map_thinking_level_to_effort(Some(ThinkingLevel::Minimal), "claude-opus-4.6"),
            AnthropicEffort::Low
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Medium), "claude-opus-4.6"),
            AnthropicEffort::Medium
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::High), "claude-opus-4.6"),
            AnthropicEffort::High
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Xhigh), "claude-opus-4.6-20250414"),
            AnthropicEffort::Max
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Xhigh), "claude-opus-4.7"),
            AnthropicEffort::Xhigh
        );
        // Xhigh on non-opus falls to High
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Xhigh), "claude-sonnet-4"),
            AnthropicEffort::High
        );
        assert_eq!(
            map_thinking_level_to_effort(Some(ThinkingLevel::Xhigh), "claude-sonnet-4.6"),
            AnthropicEffort::High
        );
        // None defaults to High
        assert_eq!(
            map_thinking_level_to_effort(None, "claude-opus-4.6"),
            AnthropicEffort::High
        );
    }

    #[test]
    fn effort_display() {
        assert_eq!(AnthropicEffort::Low.to_string(), "low");
        assert_eq!(AnthropicEffort::Medium.to_string(), "medium");
        assert_eq!(AnthropicEffort::High.to_string(), "high");
        assert_eq!(AnthropicEffort::Xhigh.to_string(), "xhigh");
        assert_eq!(AnthropicEffort::Max.to_string(), "max");
    }

    #[test]
    fn adaptive_thinking_sonnet_4_6() {
        assert!(supports_adaptive_thinking("claude-sonnet-4.6"));
        assert!(supports_adaptive_thinking("claude-sonnet-4-6-20250414"));
    }

    #[test]
    fn adaptive_thinking_non_supported() {
        assert!(!supports_adaptive_thinking("claude-sonnet-4-20250514"));
        assert!(!supports_adaptive_thinking("claude-3-5-sonnet"));
        assert!(!supports_adaptive_thinking("claude-haiku-4.6"));
    }

    #[test]
    fn cache_control_none_retention() {
        let (retention, cc) = get_anthropic_cache_control(
            "https://api.anthropic.com",
            Some(CacheRetention::None),
        );
        assert_eq!(retention, CacheRetention::None);
        assert!(cc.is_none());
    }

    #[test]
    fn cache_control_short_on_anthropic_no_ttl() {
        let (retention, cc) = get_anthropic_cache_control(
            "https://api.anthropic.com/v1/messages",
            Some(CacheRetention::Short),
        );
        assert_eq!(retention, CacheRetention::Short);
        let cc = cc.unwrap();
        assert_eq!(cc["type"], "ephemeral");
        assert!(cc.get("ttl").is_none());
    }

    #[test]
    fn cache_control_long_on_anthropic_has_ttl() {
        let (retention, cc) = get_anthropic_cache_control(
            "https://api.anthropic.com/v1/messages",
            Some(CacheRetention::Long),
        );
        assert_eq!(retention, CacheRetention::Long);
        let cc = cc.unwrap();
        assert_eq!(cc["type"], "ephemeral");
        assert_eq!(cc["ttl"], "1h");
    }

    #[test]
    fn cache_control_long_on_non_anthropic_no_ttl() {
        let (retention, cc) = get_anthropic_cache_control(
            "https://bedrock.us-east-1.amazonaws.com",
            Some(CacheRetention::Long),
        );
        assert_eq!(retention, CacheRetention::Long);
        let cc = cc.unwrap();
        assert_eq!(cc["type"], "ephemeral");
        assert!(cc.get("ttl").is_none());
    }

    #[test]
    fn normalize_tool_call_id_preserves_valid() {
        assert_eq!(normalize_tool_call_id("abc-123_def"), "abc-123_def");
        assert_eq!(normalize_tool_call_id("simple"), "simple");
    }

    #[test]
    fn normalize_tool_call_id_replaces_special() {
        assert_eq!(normalize_tool_call_id("a@b#c"), "a_b_c");
        assert_eq!(normalize_tool_call_id("a b c"), "a_b_c");
        assert_eq!(normalize_tool_call_id("a.b.c"), "a_b_c");
    }

    #[test]
    fn normalize_tool_call_id_truncates() {
        let long = "a".repeat(100);
        assert_eq!(normalize_tool_call_id(&long).len(), 64);
    }

    #[test]
    fn oauth_token_detection() {
        assert!(is_oauth_token("sk-ant-oat-abc123"));
        assert!(!is_oauth_token("sk-ant-api-abc123"));
    }
}
