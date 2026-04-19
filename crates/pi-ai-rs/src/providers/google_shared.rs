//! Shared utilities for Google Generative AI providers.
//!
//! Port of `packages/ai/src/providers/google-shared.ts`.

use serde::{Deserialize, Serialize};

use crate::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use crate::types::{
    AssistantContent, AssistantMessage, Content, Context, ImageContent, InputModality, Model,
    StopReason, TextContent,
};
use crate::utils::sanitize_unicode::sanitize_surrogates;

// =============================================================================
// Thinking part utilities
// =============================================================================

/// Determines whether a streamed Gemini Part should be treated as "thinking".
///
/// Protocol: `thought: true` is the definitive marker for thinking content.
/// `thoughtSignature` is an encrypted representation used to preserve context.
pub fn is_thinking_part(thought: Option<bool>, _thought_signature: Option<&str>) -> bool {
    thought == Some(true)
}

/// Retain thought signatures during streaming.
///
/// Some backends only send `thoughtSignature` on the first delta; later deltas may omit it.
/// This preserves the last non-empty signature for the current block.
pub fn retain_thought_signature<'a>(
    existing: Option<&'a str>,
    incoming: Option<&'a str>,
) -> Option<&'a str> {
    if let Some(inc) = incoming {
        if !inc.is_empty() {
            return Some(inc);
        }
    }
    existing
}

// Thought signatures must be base64 for Google APIs (TYPE_BYTES).
fn is_valid_thought_signature(signature: Option<&str>) -> bool {
    match signature {
        None => false,
        Some(sig) => {
            if sig.len() % 4 != 0 {
                return false;
            }
            sig.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=')
        }
    }
}

/// Only keep signatures from the same provider/model and with valid base64.
pub fn resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: Option<&str>,
) -> Option<String> {
    if is_same_provider_and_model && is_valid_thought_signature(signature) {
        signature.map(String::from)
    } else {
        None
    }
}

/// Sentinel value that tells the Gemini API to skip thought signature validation.
pub const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

/// Models via Google APIs that require explicit tool call IDs.
pub fn requires_tool_call_id(model_id: &str) -> bool {
    model_id.starts_with("claude-") || model_id.starts_with("gpt-oss-")
}

fn get_gemini_major_version(model_id: &str) -> Option<u32> {
    let lower = model_id.to_lowercase();
    // Match "gemini-N" or "gemini-live-N"
    let stripped = lower
        .strip_prefix("gemini-live-")
        .or_else(|| lower.strip_prefix("gemini-"))?;
    let digits: String = stripped.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn supports_multimodal_function_response(model_id: &str) -> bool {
    match get_gemini_major_version(model_id) {
        Some(v) => v >= 3,
        None => true,
    }
}

// =============================================================================
// Gemini tool choice mapping
// =============================================================================

/// Google FunctionCallingConfigMode equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    Auto,
    None,
    Any,
}

/// Map tool choice string to Gemini FunctionCallingConfigMode.
pub fn map_tool_choice(choice: &str) -> FunctionCallingMode {
    match choice {
        "none" => FunctionCallingMode::None,
        "any" => FunctionCallingMode::Any,
        _ => FunctionCallingMode::Auto,
    }
}

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map Gemini FinishReason enum to our StopReason.
pub fn map_google_stop_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

// =============================================================================
// Gemini Part type (simplified for serialization)
// =============================================================================

/// A simplified Gemini Part for message conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponsePart>,
}

impl GeminiPart {
    pub fn text(s: &str) -> Self {
        Self {
            text: Some(s.to_string()),
            ..Default::default()
        }
    }

    pub fn text_with_signature(s: &str, sig: Option<String>) -> Self {
        Self {
            text: Some(s.to_string()),
            thought_signature: sig,
            ..Default::default()
        }
    }

    pub fn thinking(s: &str, sig: Option<String>) -> Self {
        Self {
            text: Some(s.to_string()),
            thought: Some(true),
            thought_signature: sig,
            ..Default::default()
        }
    }

    pub fn inline_image(mime_type: &str, data: &str) -> Self {
        Self {
            inline_data: Some(InlineData {
                mime_type: mime_type.to_string(),
                data: data.to_string(),
            }),
            ..Default::default()
        }
    }
}

impl Default for GeminiPart {
    fn default() -> Self {
        Self {
            text: None,
            thought: None,
            thought_signature: None,
            inline_data: None,
            function_call: None,
            function_response: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallPart {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponsePart {
    pub name: String,
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<GeminiPart>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Gemini Content message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

// =============================================================================
// Message conversion
// =============================================================================

/// Convert internal messages to Gemini Content[] format.
///
/// Port of `convertMessages()` from `packages/ai/src/providers/google-shared.ts`.
pub fn convert_google_messages(model: &Model, context: &Context) -> Vec<GeminiContent> {
    let model_id = model.id.clone();
    let normalize_tool_call_id: NormalizeToolCallIdFn = Box::new(move |id: &str, _target_model: &Model, _source: &AssistantMessage| -> String {
        if !requires_tool_call_id(&model_id) {
            return id.to_string();
        }
        let sanitized: String = id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if sanitized.len() > 64 {
            sanitized[..64].to_string()
        } else {
            sanitized
        }
    });

    let transformed_messages = transform_messages(&context.messages, model, Some(&normalize_tool_call_id));
    let mut contents: Vec<GeminiContent> = Vec::new();

    for msg in &transformed_messages {
        match msg {
            crate::types::Message::User(user) => match &user.content {
                crate::types::UserContent::Text(text) => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::text(&sanitize_surrogates(text))],
                    });
                }
                crate::types::UserContent::Parts(parts) => {
                    let gemini_parts: Vec<GeminiPart> = parts
                        .iter()
                        .map(|item| match item {
                            crate::types::UserContentPart::Text(t) => {
                                GeminiPart::text(&sanitize_surrogates(&t.text))
                            }
                            crate::types::UserContentPart::Image(img) => {
                                GeminiPart::inline_image(&img.mime_type, &img.data)
                            }
                        })
                        .collect();
                    let filtered: Vec<GeminiPart> = if !model.input.contains(&InputModality::Image) {
                        gemini_parts
                            .into_iter()
                            .filter(|p| p.text.is_some())
                            .collect()
                    } else {
                        gemini_parts
                    };
                    if filtered.is_empty() {
                        continue;
                    }
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: filtered,
                    });
                }
            },
            crate::types::Message::Assistant(assistant_msg) => {
                let is_same_provider_and_model =
                    assistant_msg.provider == model.provider && assistant_msg.model == model.id;
                let mut parts: Vec<GeminiPart> = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Text(text_block) => {
                            if text_block.text.trim().is_empty() {
                                continue;
                            }
                            let thought_sig = resolve_thought_signature(
                                is_same_provider_and_model,
                                text_block.text_signature.as_deref(),
                            );
                            parts.push(GeminiPart::text_with_signature(
                                &sanitize_surrogates(&text_block.text),
                                thought_sig,
                            ));
                        }
                        AssistantContent::Thinking(t) => {
                            if t.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_provider_and_model {
                                let thought_sig = resolve_thought_signature(
                                    is_same_provider_and_model,
                                    t.thinking_signature.as_deref(),
                                );
                                parts.push(GeminiPart::thinking(
                                    &sanitize_surrogates(&t.thinking),
                                    thought_sig,
                                ));
                            } else {
                                parts.push(GeminiPart::text(&sanitize_surrogates(&t.thinking)));
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let thought_sig = resolve_thought_signature(
                                is_same_provider_and_model,
                                tool_call.thought_signature.as_deref(),
                            );
                            let is_gemini3 = model.id.to_lowercase().contains("gemini-3");
                            let effective_sig = thought_sig.or_else(|| {
                                if is_gemini3 {
                                    Some(SKIP_THOUGHT_SIGNATURE.to_string())
                                } else {
                                    None
                                }
                            });
                            let include_id = requires_tool_call_id(&model.id);
                            parts.push(GeminiPart {
                                function_call: Some(FunctionCallPart {
                                    name: tool_call.name.clone(),
                                    args: Some(
                                        tool_call.arguments.clone(),
                                    ),
                                    id: if include_id {
                                        Some(tool_call.id.clone())
                                    } else {
                                        None
                                    },
                                }),
                                thought_signature: effective_sig,
                                ..Default::default()
                            });
                        }
                    }
                }

                if parts.is_empty() {
                    continue;
                }
                contents.push(GeminiContent {
                    role: "model".to_string(),
                    parts,
                });
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

                let image_content: Vec<&ImageContent> = if model.input.contains(&InputModality::Image)
                {
                    tr.content
                        .iter()
                        .filter_map(|c| {
                            if let Content::Image(img) = c {
                                Some(img)
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                let has_text = !text_result.is_empty();
                let has_images = !image_content.is_empty();

                let model_supports_multimodal = supports_multimodal_function_response(&model.id);

                let response_value = sanitize_surrogates(if has_text {
                    &text_result
                } else if has_images {
                    "(see attached image)"
                } else {
                    ""
                });

                let image_parts: Vec<GeminiPart> = image_content
                    .iter()
                    .map(|img| GeminiPart::inline_image(&img.mime_type, &img.data))
                    .collect();

                let include_id = requires_tool_call_id(&model.id);
                let response = if tr.is_error {
                    serde_json::json!({"error": response_value})
                } else {
                    serde_json::json!({"output": response_value})
                };

                let function_response_part = GeminiPart {
                    function_response: Some(FunctionResponsePart {
                        name: tr.tool_name.clone(),
                        response,
                        parts: if has_images && model_supports_multimodal {
                            Some(image_parts.clone())
                        } else {
                            None
                        },
                        id: if include_id {
                            Some(tr.tool_call_id.clone())
                        } else {
                            None
                        },
                    }),
                    ..Default::default()
                };

                // Merge function responses into the last user turn if possible.
                let last_content = contents.last_mut();
                let should_merge = last_content.map_or(false, |c| {
                    c.role == "user"
                        && c.parts
                            .iter()
                            .any(|p| p.function_response.is_some())
                });

                if should_merge {
                    if let Some(last) = contents.last_mut() {
                        last.parts.push(function_response_part);
                    }
                } else {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![function_response_part],
                    });
                }

                // For Gemini < 3, add images in a separate user message.
                if has_images && !model_supports_multimodal {
                    let mut img_parts = vec![GeminiPart::text("Tool result image:")];
                    img_parts.extend(image_parts);
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: img_parts,
                    });
                }
            }
        }
    }

    contents
}

/// Convert tools to Gemini function declarations format.
///
/// Port of `convertTools()` from `packages/ai/src/providers/google-shared.ts`.
pub fn convert_google_tools(
    tools: &[crate::types::Tool],
    use_parameters: bool,
) -> Option<Vec<serde_json::Value>> {
    if tools.is_empty() {
        return None;
    }
    let declarations: Vec<serde_json::Value> = tools
        .iter()
        .map(|tool| {
            let mut decl = serde_json::json!({
                "name": tool.name,
                "description": tool.description,
            });
            if use_parameters {
                decl["parameters"] = tool.parameters.clone();
            } else {
                decl["parametersJsonSchema"] = tool.parameters.clone();
            }
            decl
        })
        .collect();
    Some(vec![serde_json::json!({"functionDeclarations": declarations})])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_thinking_part_true() {
        assert!(is_thinking_part(Some(true), None));
    }

    #[test]
    fn is_thinking_part_false_on_signature_only() {
        assert!(!is_thinking_part(None, Some("some_sig")));
    }

    #[test]
    fn retain_thought_signature_keeps_incoming() {
        assert_eq!(
            retain_thought_signature(Some("old"), Some("new")),
            Some("new")
        );
    }

    #[test]
    fn retain_thought_signature_keeps_existing_when_empty() {
        assert_eq!(
            retain_thought_signature(Some("old"), Some("")),
            Some("old")
        );
    }

    #[test]
    fn retain_thought_signature_keeps_existing_when_none() {
        assert_eq!(retain_thought_signature(Some("old"), None), Some("old"));
    }

    #[test]
    fn requires_tool_call_id_claude() {
        assert!(requires_tool_call_id("claude-4-sonnet"));
    }

    #[test]
    fn requires_tool_call_id_gpt_oss() {
        assert!(requires_tool_call_id("gpt-oss-4o"));
    }

    #[test]
    fn requires_tool_call_id_gemini_false() {
        assert!(!requires_tool_call_id("gemini-2.0-flash"));
    }

    #[test]
    fn map_stop_reason_stop() {
        assert_eq!(map_google_stop_reason("STOP"), StopReason::Stop);
    }

    #[test]
    fn map_stop_reason_max_tokens() {
        assert_eq!(map_google_stop_reason("MAX_TOKENS"), StopReason::Length);
    }

    #[test]
    fn map_stop_reason_unknown() {
        assert_eq!(map_google_stop_reason("SAFETY"), StopReason::Error);
    }

    #[test]
    fn gemini_major_version_2() {
        assert_eq!(get_gemini_major_version("gemini-2.0-flash"), Some(2));
    }

    #[test]
    fn gemini_major_version_3() {
        assert_eq!(get_gemini_major_version("gemini-3-pro-001"), Some(3));
    }

    #[test]
    fn gemini_major_version_none() {
        assert_eq!(get_gemini_major_version("claude-4-sonnet"), None);
    }

    #[test]
    fn valid_thought_signature() {
        assert!(is_valid_thought_signature(Some("YWJjZGVm")));
    }

    #[test]
    fn invalid_thought_signature_bad_chars() {
        assert!(!is_valid_thought_signature(Some("abc!def")));
    }

    #[test]
    fn invalid_thought_signature_bad_length() {
        assert!(!is_valid_thought_signature(Some("abc")));
    }

    #[test]
    fn tool_choice_auto() {
        assert_eq!(map_tool_choice("auto"), FunctionCallingMode::Auto);
    }

    #[test]
    fn tool_choice_none() {
        assert_eq!(map_tool_choice("none"), FunctionCallingMode::None);
    }

    #[test]
    fn tool_choice_any() {
        assert_eq!(map_tool_choice("any"), FunctionCallingMode::Any);
    }

    #[test]
    fn convert_tools_empty() {
        assert!(convert_google_tools(&[], false).is_none());
    }

    #[test]
    fn convert_tools_basic() {
        let tools = vec![crate::types::Tool {
            name: "test".to_string(),
            description: "desc".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let result = convert_google_tools(&tools, false).unwrap();
        assert_eq!(result.len(), 1);
        let decls = &result[0]["functionDeclarations"];
        assert!(decls.is_array());
        assert_eq!(decls[0]["name"], "test");
    }
}
