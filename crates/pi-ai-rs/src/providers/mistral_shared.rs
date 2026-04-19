//! Shared utilities for the Mistral provider.
//!
//! Port of conversion and helper logic from
//! `packages/ai/src/providers/mistral.ts`.

use std::collections::HashMap;

use crate::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use crate::types::{
    AssistantContent, AssistantMessage, Content, Context, InputModality, Model,
    StopReason, ThinkingLevel, Tool,
};
use crate::utils::hash::short_hash;
use crate::utils::sanitize_unicode::sanitize_surrogates;

/// Mistral tool call ID length.
const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;

// =============================================================================
// Tool call ID normalization
// =============================================================================

/// Derive a Mistral-compatible tool call ID from an arbitrary input.
pub fn derive_mistral_tool_call_id(id: &str, attempt: u32) -> String {
    let normalized: String = id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let seed_base = if normalized.is_empty() {
        id.to_string()
    } else {
        normalized
    };
    let seed = if attempt == 0 {
        seed_base
    } else {
        format!("{}:{}", seed_base, attempt)
    };
    let hash = short_hash(&seed);
    hash.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(MISTRAL_TOOL_CALL_ID_LENGTH)
        .collect()
}

/// Create a Mistral tool call ID normalizer that ensures uniqueness.
pub struct MistralToolCallIdNormalizer {
    id_map: HashMap<String, String>,
    reverse_map: HashMap<String, String>,
}

impl MistralToolCallIdNormalizer {
    pub fn new() -> Self {
        Self {
            id_map: HashMap::new(),
            reverse_map: HashMap::new(),
        }
    }

    pub fn normalize(&mut self, id: &str) -> String {
        if let Some(existing) = self.id_map.get(id) {
            return existing.clone();
        }

        let mut attempt = 0u32;
        loop {
            let candidate = derive_mistral_tool_call_id(id, attempt);
            let owner = self.reverse_map.get(&candidate);
            if owner.is_none() || owner == Some(&id.to_string()) {
                self.id_map.insert(id.to_string(), candidate.clone());
                self.reverse_map.insert(candidate.clone(), id.to_string());
                return candidate;
            }
            attempt += 1;
        }
    }
}

impl Default for MistralToolCallIdNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map Mistral finish reason to our StopReason.
pub fn map_mistral_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" | "model_length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        "error" => StopReason::Error,
        _ => StopReason::Stop,
    }
}

// =============================================================================
// Reasoning effort mapping
// =============================================================================

/// Mistral reasoning effort levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralReasoningEffort {
    None,
    High,
}

impl std::fmt::Display for MistralReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::High => write!(f, "high"),
        }
    }
}

/// Map ThinkingLevel to Mistral reasoning effort.
///
/// Mistral's `mapReasoningEffort` always returns `"high"` regardless of the
/// requested level — the only question the caller answers is *whether* to
/// enable reasoning at all.
pub fn map_reasoning_effort(_level: Option<ThinkingLevel>) -> MistralReasoningEffort {
    MistralReasoningEffort::High
}

/// Check if model uses prompt mode reasoning.
///
/// Models that support reasoning but do NOT use the `reasoningEffort` parameter
/// use `promptMode: "reasoning"` instead.
pub fn uses_prompt_mode_reasoning(model: &Model) -> bool {
    model.reasoning && !uses_reasoning_effort(model)
}

/// Check if model uses reasoning effort parameter.
pub fn uses_reasoning_effort(model: &Model) -> bool {
    model.id == "mistral-small-2603" || model.id == "mistral-small-latest"
}

// =============================================================================
// Message conversion
// =============================================================================

/// Convert internal messages to Mistral chat format.
///
/// Port of `toChatMessages()` from `packages/ai/src/providers/mistral.ts`.
pub fn convert_mistral_messages(
    model: &Model,
    context: &Context,
) -> Vec<serde_json::Value> {
    let mut normalizer = MistralToolCallIdNormalizer::new();
    let normalize_fn: NormalizeToolCallIdFn = Box::new(|id: &str, _m: &Model, _s: &AssistantMessage| -> String {
        // We do ID normalization after transform
        id.to_string()
    });

    let transformed = transform_messages(&context.messages, model, Some(&normalize_fn));
    let supports_image = model.input.contains(&InputModality::Image);
    let mut messages: Vec<serde_json::Value> = Vec::new();

    // System prompt
    if let Some(ref system_prompt) = context.system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
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
                        let had_images = parts.iter().any(|p| matches!(p, crate::types::UserContentPart::Image(_)));
                        let content: Vec<serde_json::Value> = parts
                            .iter()
                            .filter(|p| matches!(p, crate::types::UserContentPart::Text(_)) || supports_image)
                            .map(|p| match p {
                                crate::types::UserContentPart::Text(t) => {
                                    serde_json::json!({"type": "text", "text": sanitize_surrogates(&t.text)})
                                }
                                crate::types::UserContentPart::Image(img) => {
                                    serde_json::json!({
                                        "type": "image_url",
                                        "imageUrl": format!("data:{};base64,{}", img.mime_type, img.data),
                                    })
                                }
                            })
                            .collect();
                        if !content.is_empty() {
                            messages.push(serde_json::json!({"role": "user", "content": content}));
                        } else if had_images && !supports_image {
                            messages.push(serde_json::json!({"role": "user", "content": "(image omitted: model does not support images)"}));
                        }
                    }
                }
            }
            crate::types::Message::Assistant(assistant_msg) => {
                let mut content_parts: Vec<serde_json::Value> = Vec::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Text(t) => {
                            if t.text.trim().is_empty() {
                                continue;
                            }
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": sanitize_surrogates(&t.text),
                            }));
                        }
                        AssistantContent::Thinking(t) => {
                            if t.thinking.trim().is_empty() {
                                continue;
                            }
                            content_parts.push(serde_json::json!({
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(&t.thinking)}],
                            }));
                        }
                        AssistantContent::ToolCall(tc) => {
                            let normalized_id = normalizer.normalize(&tc.id);
                            tool_calls.push(serde_json::json!({
                                "id": normalized_id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                                }
                            }));
                        }
                    }
                }

                if content_parts.is_empty() && tool_calls.is_empty() {
                    continue;
                }

                let mut msg = serde_json::json!({"role": "assistant"});
                if !content_parts.is_empty() {
                    msg["content"] = serde_json::json!(content_parts);
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

                let has_images = tr.content.iter().any(|c| matches!(c, Content::Image(_)));
                let normalized_id = normalizer.normalize(&tr.tool_call_id);
                let tool_text = build_tool_result_text(&text_result, has_images, supports_image, tr.is_error);

                let mut tool_content: Vec<serde_json::Value> = vec![
                    serde_json::json!({"type": "text", "text": tool_text}),
                ];

                if supports_image {
                    for part in &tr.content {
                        if let Content::Image(img) = part {
                            tool_content.push(serde_json::json!({
                                "type": "image_url",
                                "imageUrl": format!("data:{};base64,{}", img.mime_type, img.data),
                            }));
                        }
                    }
                }

                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": normalized_id,
                    "name": tr.tool_name,
                    "content": tool_content,
                }));
            }
        }
    }

    messages
}

/// Build tool result text matching TS `buildToolResultText`.
fn build_tool_result_text(text: &str, has_images: bool, supports_images: bool, is_error: bool) -> String {
    let trimmed = text.trim();
    let error_prefix = if is_error { "[tool error] " } else { "" };

    if !trimmed.is_empty() {
        let image_suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{}{}{}", error_prefix, trimmed, image_suffix);
    }

    if has_images {
        if supports_images {
            return if is_error {
                "[tool error] (see attached image)".to_string()
            } else {
                "(see attached image)".to_string()
            };
        }
        return if is_error {
            "[tool error] (image omitted: model does not support images)".to_string()
        } else {
            "(image omitted: model does not support images)".to_string()
        };
    }

    if is_error {
        "[tool error] (no tool output)".to_string()
    } else {
        "(no tool output)".to_string()
    }
}

/// Convert tools to Mistral function tool format.
pub fn convert_mistral_tools(tools: &[Tool]) -> Vec<serde_json::Value> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_tool_call_id_exact_length() {
        let id = "abcdefghi"; // exactly 9 alphanumeric
        assert_eq!(derive_mistral_tool_call_id(id, 0), id);
    }

    #[test]
    fn derive_tool_call_id_hash() {
        let id = "some-long-tool-call-id";
        let result = derive_mistral_tool_call_id(id, 0);
        assert_eq!(result.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
        assert!(result.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn normalizer_consistent() {
        let mut norm = MistralToolCallIdNormalizer::new();
        let id1 = norm.normalize("test-id-1");
        let id2 = norm.normalize("test-id-1");
        assert_eq!(id1, id2);
    }

    #[test]
    fn normalizer_unique() {
        let mut norm = MistralToolCallIdNormalizer::new();
        let id1 = norm.normalize("test-id-1");
        let id2 = norm.normalize("test-id-2");
        assert_ne!(id1, id2);
    }

    #[test]
    fn map_stop_reasons() {
        assert_eq!(map_mistral_stop_reason("stop"), StopReason::Stop);
        assert_eq!(map_mistral_stop_reason("length"), StopReason::Length);
        assert_eq!(map_mistral_stop_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(map_mistral_stop_reason("model_length"), StopReason::Length);
    }
}
