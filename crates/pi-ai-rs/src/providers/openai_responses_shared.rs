//! Shared utilities for OpenAI Responses API providers.
//!
//! Port of `packages/ai/src/providers/openai-responses-shared.ts`.
//! Contains message conversion, tool conversion, text signature encoding/parsing,
//! and the shared stream processing logic.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use crate::types::{
    AssistantContent, AssistantMessage, Content, Context,
    InputModality, Model, StopReason, TextSignaturePhase, TextSignatureV1, Usage,
};
use crate::utils::hash::short_hash;
use crate::utils::sanitize_unicode::sanitize_surrogates;

// =============================================================================
// Text signature utilities
// =============================================================================

/// Encode a text signature (version 1) as a JSON string.
pub fn encode_text_signature_v1(id: &str, phase: Option<TextSignaturePhase>) -> String {
    let sig = TextSignatureV1 {
        v: 1,
        id: id.to_string(),
        phase,
    };
    serde_json::to_string(&sig).unwrap_or_default()
}

/// Parsed text signature result.
#[derive(Debug, Clone)]
pub struct ParsedTextSignature {
    pub id: String,
    pub phase: Option<TextSignaturePhase>,
}

/// Parse a text signature string — either JSON (v1) or legacy plain string.
pub fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let sig = signature?;
    if sig.is_empty() {
        return None;
    }

    if sig.starts_with('{') {
        // Use a permissive struct so that unknown phase values don't reject the
        // entire payload — mirrors the TS code which parses with `Partial<>` and
        // then validates the phase separately.
        #[derive(Deserialize)]
        struct Permissive {
            v: Option<u32>,
            id: Option<String>,
            phase: Option<String>,
        }
        if let Ok(parsed) = serde_json::from_str::<Permissive>(sig) {
            if parsed.v == Some(1) {
                if let Some(id) = parsed.id {
                    let phase = match parsed.phase.as_deref() {
                        Some("commentary") => Some(TextSignaturePhase::Commentary),
                        Some("final_answer") => Some(TextSignaturePhase::FinalAnswer),
                        _ => None,
                    };
                    return Some(ParsedTextSignature { id, phase });
                }
            }
        }
        // Fall through to legacy plain-string handling
    }

    Some(ParsedTextSignature {
        id: sig.to_string(),
        phase: None,
    })
}

// =============================================================================
// OpenAI Responses input types (documentation types)
//
// These types mirror the OpenAI SDK types for reference but are not directly
// used for serialization/deserialization. The actual message conversion uses
// serde_json::Value for maximum flexibility.
// =============================================================================

/// Input content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputContent {
    InputText { text: String },
    InputImage { detail: String, image_url: String },
}

/// Output text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTextContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

/// Function call output value — either a string or multimodal parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallOutputValue {
    Text(String),
    Parts(Vec<ResponseInputContent>),
}

// =============================================================================
// OpenAI tool type
// =============================================================================

/// OpenAI-format tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Options for tool conversion.
pub struct ConvertResponsesToolsOptions {
    pub strict: Option<bool>,
}

/// Convert internal tools to OpenAI format.
pub fn convert_responses_tools(
    tools: &[crate::types::Tool],
    options: Option<&ConvertResponsesToolsOptions>,
) -> Vec<OpenAITool> {
    let strict = options.and_then(|o| o.strict);
    tools
        .iter()
        .map(|tool| OpenAITool {
            tool_type: "function".to_string(),
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict,
        })
        .collect()
}

// =============================================================================
// Message conversion
// =============================================================================

/// Normalize an ID part to only contain valid characters, max 64 chars.
fn normalize_id_part(part: &str) -> String {
    let sanitized: String = part
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let normalized = if sanitized.len() > 64 {
        &sanitized[..64]
    } else {
        &sanitized
    };
    normalized.trim_end_matches('_').to_string()
}

/// Build a foreign Responses item ID.
fn build_foreign_responses_item_id(item_id: &str) -> String {
    let normalized = format!("fc_{}", short_hash(item_id));
    if normalized.len() > 64 {
        normalized[..64].to_string()
    } else {
        normalized
    }
}

/// Options for message conversion.
pub struct ConvertResponsesMessagesOptions {
    pub include_system_prompt: bool,
}

impl Default for ConvertResponsesMessagesOptions {
    fn default() -> Self {
        Self {
            include_system_prompt: true,
        }
    }
}

/// Convert internal messages to OpenAI Responses API input format.
///
/// Port of `convertResponsesMessages()` from
/// `packages/ai/src/providers/openai-responses-shared.ts`.
pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &HashSet<String>,
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Vec<serde_json::Value> {
    let include_system_prompt = options.map_or(true, |o| o.include_system_prompt);

    let provider_clone = model.provider.clone();
    let api_clone = model.api.clone();
    let allowed_providers = allowed_tool_call_providers.clone();
    let normalize_tool_call_id: NormalizeToolCallIdFn = Box::new(move |id: &str, _target_model: &Model, source: &AssistantMessage| -> String {
        if !allowed_providers.contains(&provider_clone) {
            return normalize_id_part(id);
        }
        if !id.contains('|') {
            return normalize_id_part(id);
        }
        let parts: Vec<&str> = id.splitn(2, '|').collect();
        let call_id = parts[0];
        let item_id = parts.get(1).unwrap_or(&"");
        let normalized_call_id = normalize_id_part(call_id);
        let is_foreign_tool_call = source.provider != provider_clone || source.api != api_clone;
        let mut normalized_item_id = if is_foreign_tool_call {
            build_foreign_responses_item_id(item_id)
        } else {
            normalize_id_part(item_id)
        };
        if !normalized_item_id.starts_with("fc_") {
            normalized_item_id = normalize_id_part(&format!("fc_{}", normalized_item_id));
        }
        format!("{}|{}", normalized_call_id, normalized_item_id)
    });

    let transformed_messages = transform_messages(&context.messages, model, Some(&normalize_tool_call_id));

    let mut messages: Vec<serde_json::Value> = Vec::new();

    if include_system_prompt {
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
    }

    let mut msg_index = 0u64;
    for msg in &transformed_messages {
        match msg {
            crate::types::Message::User(user) => {
                match &user.content {
                    crate::types::UserContent::Text(text) => {
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": [{"type": "input_text", "text": sanitize_surrogates(text)}],
                        }));
                    }
                    crate::types::UserContent::Parts(parts) => {
                        let content: Vec<serde_json::Value> = parts
                            .iter()
                            .map(|item| match item {
                                crate::types::UserContentPart::Text(t) => {
                                    serde_json::json!({
                                        "type": "input_text",
                                        "text": sanitize_surrogates(&t.text),
                                    })
                                }
                                crate::types::UserContentPart::Image(img) => {
                                    serde_json::json!({
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": format!("data:{};base64,{}", img.mime_type, img.data),
                                    })
                                }
                            })
                            .collect();

                        let filtered: Vec<serde_json::Value> = if !model.input.contains(&InputModality::Image) {
                            content
                                .into_iter()
                                .filter(|c| c["type"] != "input_image")
                                .collect()
                        } else {
                            content
                        };
                        if filtered.is_empty() {
                            msg_index += 1;
                            continue;
                        }
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": filtered,
                        }));
                    }
                }
            }
            crate::types::Message::Assistant(assistant_msg) => {
                let is_different_model = assistant_msg.model != model.id
                    && assistant_msg.provider == model.provider
                    && assistant_msg.api == model.api;

                let mut output: Vec<serde_json::Value> = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        AssistantContent::Thinking(t) => {
                            if let Some(ref sig) = t.thinking_signature {
                                if let Ok(reasoning_item) =
                                    serde_json::from_str::<serde_json::Value>(sig)
                                {
                                    output.push(reasoning_item);
                                }
                            }
                        }
                        AssistantContent::Text(text_block) => {
                            let parsed_sig =
                                parse_text_signature(text_block.text_signature.as_deref());
                            let msg_id = match &parsed_sig {
                                Some(ps) if !ps.id.is_empty() => {
                                    if ps.id.len() > 64 {
                                        format!("msg_{}", short_hash(&ps.id))
                                    } else {
                                        ps.id.clone()
                                    }
                                }
                                _ => format!("msg_{}", msg_index),
                            };
                            let phase = parsed_sig.as_ref().and_then(|ps| ps.phase.clone());

                            let mut msg_obj = serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": sanitize_surrogates(&text_block.text), "annotations": []}],
                                "status": "completed",
                                "id": msg_id,
                            });
                            if let Some(p) = phase {
                                msg_obj["phase"] = serde_json::to_value(p).unwrap_or_default();
                            }
                            output.push(msg_obj);
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let parts: Vec<&str> = tool_call.id.splitn(2, '|').collect();
                            let call_id = parts[0];
                            let item_id_raw = parts.get(1).copied();

                            let mut item_id = item_id_raw.map(String::from);

                            // For different-model messages, set id to undefined to avoid pairing validation.
                            if is_different_model {
                                if let Some(ref id) = item_id {
                                    if id.starts_with("fc_") {
                                        item_id = None;
                                    }
                                }
                            }

                            let mut fc = serde_json::json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": tool_call.name,
                                "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_default(),
                            });
                            if let Some(id) = item_id {
                                fc["id"] = serde_json::Value::String(id);
                            }
                            output.push(fc);
                        }
                    }
                }
                if output.is_empty() {
                    msg_index += 1;
                    continue;
                }
                messages.extend(output);
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
                let has_text = !text_result.is_empty();
                let call_id = tr.tool_call_id.split('|').next().unwrap_or(&tr.tool_call_id);

                if has_images && model.input.contains(&InputModality::Image) {
                    let mut content_parts: Vec<serde_json::Value> = Vec::new();
                    if has_text {
                        content_parts.push(serde_json::json!({
                            "type": "input_text",
                            "text": sanitize_surrogates(&text_result),
                        }));
                    }
                    for block in &tr.content {
                        if let Content::Image(img) = block {
                            content_parts.push(serde_json::json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", img.mime_type, img.data),
                            }));
                        }
                    }
                    messages.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": content_parts,
                    }));
                } else {
                    let output_text = sanitize_surrogates(if has_text {
                        &text_result
                    } else {
                        "(see attached image)"
                    });
                    messages.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text,
                    }));
                }
            }
        }
        msg_index += 1;
    }

    messages
}

// =============================================================================
// Stop reason mapping
// =============================================================================

/// Map OpenAI response status to our StopReason.
pub fn map_openai_stop_reason(status: Option<&str>) -> StopReason {
    match status {
        None | Some("completed") => StopReason::Stop,
        Some("incomplete") => StopReason::Length,
        Some("failed") | Some("cancelled") => StopReason::Error,
        Some("in_progress") | Some("queued") => StopReason::Stop,
        _ => StopReason::Error,
    }
}

// =============================================================================
// Service tier pricing
// =============================================================================

/// Get cost multiplier for a service tier.
pub fn get_service_tier_cost_multiplier(service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") => 2.0,
        _ => 1.0,
    }
}

/// Apply service tier pricing to usage costs.
pub fn apply_service_tier_pricing(usage: &mut Usage, service_tier: Option<&str>) {
    let multiplier = get_service_tier_cost_multiplier(service_tier);
    if (multiplier - 1.0).abs() < f64::EPSILON {
        return;
    }
    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

// =============================================================================
// Cache retention utilities
// =============================================================================

/// Resolve cache retention preference.
///
/// Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
pub fn resolve_cache_retention(
    cache_retention: Option<crate::types::CacheRetention>,
) -> crate::types::CacheRetention {
    if let Some(cr) = cache_retention {
        return cr;
    }
    if let Ok(val) = std::env::var("PI_CACHE_RETENTION") {
        if val == "long" {
            return crate::types::CacheRetention::Long;
        }
    }
    crate::types::CacheRetention::Short
}

/// Get prompt cache retention based on cache retention and base URL.
/// Only applies to direct OpenAI API calls (api.openai.com).
pub fn get_prompt_cache_retention(
    base_url: &str,
    cache_retention: crate::types::CacheRetention,
) -> Option<&'static str> {
    if cache_retention != crate::types::CacheRetention::Long {
        return None;
    }
    if base_url.contains("api.openai.com") {
        return Some("24h");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_text_signature_basic() {
        let sig = encode_text_signature_v1("msg_123", None);
        let parsed: TextSignatureV1 = serde_json::from_str(&sig).unwrap();
        assert_eq!(parsed.v, 1);
        assert_eq!(parsed.id, "msg_123");
        assert!(parsed.phase.is_none());
    }

    #[test]
    fn encode_text_signature_with_phase() {
        let sig = encode_text_signature_v1("msg_456", Some(TextSignaturePhase::Commentary));
        let parsed: TextSignatureV1 = serde_json::from_str(&sig).unwrap();
        assert_eq!(parsed.phase, Some(TextSignaturePhase::Commentary));
    }

    #[test]
    fn parse_text_signature_v1() {
        let json = r#"{"v":1,"id":"msg_123","phase":"commentary"}"#;
        let result = parse_text_signature(Some(json)).unwrap();
        assert_eq!(result.id, "msg_123");
        assert_eq!(result.phase, Some(TextSignaturePhase::Commentary));
    }

    #[test]
    fn parse_text_signature_legacy() {
        let result = parse_text_signature(Some("legacy_id")).unwrap();
        assert_eq!(result.id, "legacy_id");
        assert!(result.phase.is_none());
    }

    #[test]
    fn parse_text_signature_none() {
        assert!(parse_text_signature(None).is_none());
    }

    #[test]
    fn parse_text_signature_empty() {
        assert!(parse_text_signature(Some("")).is_none());
    }

    #[test]
    fn normalize_id_part_basic() {
        assert_eq!(normalize_id_part("hello-world_123"), "hello-world_123");
    }

    #[test]
    fn normalize_id_part_special_chars() {
        assert_eq!(normalize_id_part("hello world!@#"), "hello_world");
    }

    #[test]
    fn normalize_id_part_long() {
        let long_id = "a".repeat(100);
        assert_eq!(normalize_id_part(&long_id).len(), 64);
    }

    #[test]
    fn map_stop_reason_completed() {
        assert_eq!(map_openai_stop_reason(Some("completed")), StopReason::Stop);
    }

    #[test]
    fn map_stop_reason_incomplete() {
        assert_eq!(
            map_openai_stop_reason(Some("incomplete")),
            StopReason::Length
        );
    }

    #[test]
    fn map_stop_reason_failed() {
        assert_eq!(map_openai_stop_reason(Some("failed")), StopReason::Error);
    }

    #[test]
    fn service_tier_flex() {
        assert_eq!(get_service_tier_cost_multiplier(Some("flex")), 0.5);
    }

    #[test]
    fn service_tier_priority() {
        assert_eq!(get_service_tier_cost_multiplier(Some("priority")), 2.0);
    }

    #[test]
    fn service_tier_default() {
        assert_eq!(get_service_tier_cost_multiplier(None), 1.0);
    }

    #[test]
    fn build_foreign_id_short() {
        let result = build_foreign_responses_item_id("item_abc");
        assert!(result.starts_with("fc_"));
        assert!(result.len() <= 64);
    }

    #[test]
    fn build_foreign_id_deterministic() {
        let a = build_foreign_responses_item_id("same_id");
        let b = build_foreign_responses_item_id("same_id");
        assert_eq!(a, b);
    }

    #[test]
    fn build_foreign_id_different_for_different_inputs() {
        let a = build_foreign_responses_item_id("id_1");
        let b = build_foreign_responses_item_id("id_2");
        assert_ne!(a, b);
    }

    #[test]
    fn normalize_id_part_trailing_underscore_stripped() {
        assert_eq!(normalize_id_part("hello___"), "hello");
    }

    #[test]
    fn normalize_id_part_empty_input() {
        assert_eq!(normalize_id_part(""), "");
    }

    #[test]
    fn map_stop_reason_cancelled() {
        assert_eq!(map_openai_stop_reason(Some("cancelled")), StopReason::Error);
    }

    #[test]
    fn map_stop_reason_in_progress() {
        assert_eq!(
            map_openai_stop_reason(Some("in_progress")),
            StopReason::Stop
        );
    }

    #[test]
    fn map_stop_reason_queued() {
        assert_eq!(map_openai_stop_reason(Some("queued")), StopReason::Stop);
    }

    #[test]
    fn map_stop_reason_none_default() {
        assert_eq!(map_openai_stop_reason(None), StopReason::Stop);
    }

    #[test]
    fn map_stop_reason_unknown() {
        assert_eq!(
            map_openai_stop_reason(Some("weird_status")),
            StopReason::Error
        );
    }

    #[test]
    fn apply_service_tier_flex_pricing() {
        let mut usage = Usage {
            cost: crate::types::UsageCost {
                input: 10.0,
                output: 20.0,
                cache_read: 5.0,
                cache_write: 3.0,
                total: 38.0,
            },
            ..Default::default()
        };
        apply_service_tier_pricing(&mut usage, Some("flex"));
        assert!((usage.cost.input - 5.0).abs() < f64::EPSILON);
        assert!((usage.cost.output - 10.0).abs() < f64::EPSILON);
        assert!((usage.cost.total - 19.0).abs() < f64::EPSILON);
    }

    #[test]
    fn apply_service_tier_default_no_change() {
        let mut usage = Usage {
            cost: crate::types::UsageCost {
                input: 10.0,
                output: 20.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 30.0,
            },
            ..Default::default()
        };
        apply_service_tier_pricing(&mut usage, None);
        assert!((usage.cost.input - 10.0).abs() < f64::EPSILON);
        assert!((usage.cost.total - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn resolve_cache_retention_default_is_short() {
        // Clean the env var to ensure deterministic behavior.
        let _ = unsafe { std::env::remove_var("PI_CACHE_RETENTION") };
        assert_eq!(
            resolve_cache_retention(None),
            crate::types::CacheRetention::Short
        );
    }

    #[test]
    fn resolve_cache_retention_explicit_overrides() {
        assert_eq!(
            resolve_cache_retention(Some(crate::types::CacheRetention::Long)),
            crate::types::CacheRetention::Long
        );
        assert_eq!(
            resolve_cache_retention(Some(crate::types::CacheRetention::None)),
            crate::types::CacheRetention::None
        );
    }

    #[test]
    fn get_prompt_cache_retention_long_on_openai() {
        assert_eq!(
            get_prompt_cache_retention("https://api.openai.com/v1", crate::types::CacheRetention::Long),
            Some("24h")
        );
    }

    #[test]
    fn get_prompt_cache_retention_short_returns_none() {
        assert_eq!(
            get_prompt_cache_retention("https://api.openai.com/v1", crate::types::CacheRetention::Short),
            None
        );
    }

    #[test]
    fn get_prompt_cache_retention_long_non_openai_returns_none() {
        assert_eq!(
            get_prompt_cache_retention("https://api.other.com/v1", crate::types::CacheRetention::Long),
            None
        );
    }

    #[test]
    fn parse_text_signature_v1_unknown_phase() {
        // Unknown phase values should parse with phase = None (not error).
        let json = r#"{"v":1,"id":"msg_789","phase":"unknown_phase"}"#;
        let result = parse_text_signature(Some(json)).unwrap();
        assert_eq!(result.id, "msg_789");
        assert!(result.phase.is_none());
    }

    #[test]
    fn encode_then_parse_roundtrip() {
        let encoded = encode_text_signature_v1("msg_rt", Some(TextSignaturePhase::Commentary));
        let parsed = parse_text_signature(Some(&encoded)).unwrap();
        assert_eq!(parsed.id, "msg_rt");
        assert_eq!(parsed.phase, Some(TextSignaturePhase::Commentary));
    }
}
