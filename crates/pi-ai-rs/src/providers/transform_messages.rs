use std::collections::{HashMap, HashSet};

use crate::types::{
    AssistantContent, AssistantMessage, Content, Message, Model, StopReason,
    TextContent, ToolCall, ToolResultMessage,
};

/// Optional callback for normalizing tool-call IDs across providers.
pub type NormalizeToolCallIdFn =
    Box<dyn Fn(&str, &Model, &AssistantMessage) -> String + Send + Sync>;

/// Normalize messages for cross-provider compatibility.
///
/// Port of `transformMessages()` from
/// `packages/ai/src/providers/transform-messages.ts`.
///
/// 1. Strips/converts thinking blocks for cross-model scenarios.
/// 2. Normalizes tool-call IDs via an optional callback.
/// 3. Inserts synthetic empty tool results for orphaned tool calls.
/// 4. Skips errored/aborted assistant messages.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<&NormalizeToolCallIdFn>,
) -> Vec<Message> {
    // Build a map of original tool call IDs → normalized IDs.
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();

    // ---------------------------------------------------------------
    // First pass: transform message content
    // ---------------------------------------------------------------
    let transformed: Vec<Message> = messages
        .iter()
        .map(|msg| match msg {
            Message::User(_) => msg.clone(),

            Message::ToolResult(tr) => {
                if let Some(normalized_id) = tool_call_id_map.get(&tr.tool_call_id) {
                    if normalized_id != &tr.tool_call_id {
                        let mut tr2 = tr.clone();
                        tr2.tool_call_id = normalized_id.clone();
                        return Message::ToolResult(tr2);
                    }
                }
                msg.clone()
            }

            Message::Assistant(assistant) => {
                let is_same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;

                let transformed_content: Vec<AssistantContent> = assistant
                    .content
                    .iter()
                    .flat_map(|block| match block {
                        AssistantContent::Thinking(t) => {
                            // Redacted thinking is opaque encrypted content, only valid
                            // for the same model.
                            if t.redacted.is_some_and(|v| v) {
                                if is_same_model {
                                    return vec![block.clone()];
                                }
                                return vec![];
                            }
                            // For same model: keep thinking blocks with signatures.
                            if is_same_model && t.thinking_signature.is_some() {
                                return vec![block.clone()];
                            }
                            // Skip empty thinking blocks; convert others to plain text.
                            let text = t.thinking.trim();
                            if text.is_empty() {
                                return vec![];
                            }
                            if is_same_model {
                                return vec![block.clone()];
                            }
                            vec![AssistantContent::Text(TextContent {
                                text: t.thinking.clone(),
                                text_signature: None,
                            })]
                        }

                        AssistantContent::Text(txt) => {
                            if is_same_model {
                                vec![block.clone()]
                            } else {
                                // Strip text signature for different models.
                                vec![AssistantContent::Text(TextContent {
                                    text: txt.text.clone(),
                                    text_signature: None,
                                })]
                            }
                        }

                        AssistantContent::ToolCall(tc) => {
                            let mut normalized = tc.clone();

                            if !is_same_model && tc.thought_signature.is_some() {
                                normalized.thought_signature = None;
                            }

                            if !is_same_model {
                                if let Some(ref norm_fn) = normalize_tool_call_id {
                                    let new_id = norm_fn(&tc.id, model, assistant);
                                    if new_id != tc.id {
                                        tool_call_id_map.insert(tc.id.clone(), new_id.clone());
                                        normalized.id = new_id;
                                    }
                                }
                            }

                            vec![AssistantContent::ToolCall(normalized)]
                        }
                    })
                    .collect();

                Message::Assistant(AssistantMessage {
                    content: transformed_content,
                    ..assistant.clone()
                })
            }
        })
        .collect();

    // ---------------------------------------------------------------
    // Second pass: insert synthetic tool results for orphaned calls
    // ---------------------------------------------------------------
    let mut result: Vec<Message> = Vec::new();
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: HashSet<String> = HashSet::new();

    let insert_synthetic = |pending: &mut Vec<ToolCall>,
                            existing: &mut HashSet<String>,
                            out: &mut Vec<Message>| {
        for tc in pending.drain(..) {
            if !existing.contains(&tc.id) {
                out.push(Message::ToolResult(ToolResultMessage {
                    tool_call_id: tc.id,
                    tool_name: tc.name,
                    content: vec![Content::Text(TextContent {
                        text: "No result provided".to_string(),
                        text_signature: None,
                    })],
                    details: None,
                    is_error: true,
                    timestamp: 0,
                }));
            }
        }
        existing.clear();
    };

    for msg in &transformed {
        match msg {
            Message::Assistant(assistant) => {
                // Insert synthetic results for any previous orphaned tool calls.
                insert_synthetic(
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                    &mut result,
                );

                // Skip errored/aborted assistant messages entirely.
                if assistant.stop_reason == StopReason::Error
                    || assistant.stop_reason == StopReason::Aborted
                {
                    continue;
                }

                // Track tool calls from this assistant message.
                let tool_calls: Vec<ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        AssistantContent::ToolCall(tc) => Some(tc.clone()),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_result_ids.clear();
                }

                result.push(msg.clone());
            }
            Message::ToolResult(tr) => {
                existing_tool_result_ids.insert(tr.tool_call_id.clone());
                result.push(msg.clone());
            }
            Message::User(_) => {
                // User message interrupts tool flow.
                insert_synthetic(
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                    &mut result,
                );
                result.push(msg.clone());
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Usage, UsageCost, UserContent, UserMessage};

    fn make_model() -> Model {
        Model {
            id: "test-model".to_string(),
            api: "test-api".to_string(),
            provider: "test-provider".to_string(),
            ..Default::default()
        }
    }

    fn user_msg(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_string()),
            timestamp: 0,
        })
    }

    fn assistant_msg(model: &Model, content: Vec<AssistantContent>) -> Message {
        Message::Assistant(AssistantMessage {
            content,
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            stop_reason: StopReason::Stop,
            usage: Usage::default(),
            ..Default::default()
        })
    }

    #[test]
    fn passes_user_messages_through() {
        let model = make_model();
        let msgs = vec![user_msg("hello")];
        let result = transform_messages(&msgs, &model, None);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn strips_errored_assistant_messages() {
        let model = make_model();
        let msgs = vec![
            user_msg("hello"),
            Message::Assistant(AssistantMessage {
                content: vec![AssistantContent::Text(TextContent {
                    text: "partial".to_string(),
                    text_signature: None,
                })],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                stop_reason: StopReason::Error,
                error_message: Some("something failed".to_string()),
                usage: Usage::default(),
                ..Default::default()
            }),
        ];
        let result = transform_messages(&msgs, &model, None);
        // User message kept, errored assistant skipped.
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn inserts_synthetic_tool_results_for_orphaned_calls() {
        let model = make_model();
        let msgs = vec![
            user_msg("do something"),
            assistant_msg(
                &model,
                vec![AssistantContent::ToolCall(ToolCall {
                    id: "tc-1".to_string(),
                    name: "my_tool".to_string(),
                    arguments: serde_json::json!({}),
                    thought_signature: None,
                })],
            ),
            // No tool result follows — orphaned.
            user_msg("next turn"),
        ];
        let result = transform_messages(&msgs, &model, None);
        // user + assistant + synthetic_tool_result + user
        assert_eq!(result.len(), 4);
        match &result[2] {
            Message::ToolResult(tr) => {
                assert_eq!(tr.tool_call_id, "tc-1");
                assert!(tr.is_error);
            }
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn converts_cross_model_thinking_to_text() {
        let model = make_model();
        let other_model = Model {
            id: "other".to_string(),
            api: "other-api".to_string(),
            provider: "other-provider".to_string(),
            ..Default::default()
        };
        let msgs = vec![Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Thinking(
                crate::types::ThinkingContent {
                    thinking: "My reasoning...".to_string(),
                    thinking_signature: None,
                    redacted: None,
                },
            )],
            api: other_model.api.clone(),
            provider: other_model.provider.clone(),
            model: other_model.id.clone(),
            stop_reason: StopReason::Stop,
            usage: Usage::default(),
            ..Default::default()
        })];
        let result = transform_messages(&msgs, &model, None);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Message::Assistant(a) => match &a.content[0] {
                AssistantContent::Text(t) => {
                    assert_eq!(t.text, "My reasoning...");
                }
                other => panic!("expected Text, got {:?}", other),
            },
            other => panic!("expected Assistant, got {:?}", other),
        }
    }
}
