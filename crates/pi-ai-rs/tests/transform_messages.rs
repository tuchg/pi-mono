//! Integration tests for `transform_messages` — cross-provider message normalization.
//!
//! Complements the inline tests in `providers/transform_messages.rs` with
//! scenarios covered in TS `packages/ai/test/`.

use pi_ai_rs::providers::transform_messages::{transform_messages, NormalizeToolCallIdFn};
use pi_ai_rs::{
    AssistantContent, AssistantMessage, Content, Message, Model, StopReason, TextContent,
    ThinkingContent, ToolCall, ToolResultMessage, Usage, UserContent, UserMessage,
};

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: 0,
    })
}

fn tool_result(id: &str, name: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        content: vec![Content::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        details: None,
        is_error: false,
        timestamp: 0,
    })
}

fn assistant_for(model: &Model, content: Vec<AssistantContent>) -> Message {
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

fn assistant_for_with_stop(
    model: &Model,
    content: Vec<AssistantContent>,
    stop: StopReason,
) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        stop_reason: stop,
        usage: Usage::default(),
        ..Default::default()
    })
}

fn model_a() -> Model {
    Model {
        id: "model-a".to_string(),
        api: "api-a".to_string(),
        provider: "provider-a".to_string(),
        ..Default::default()
    }
}

fn model_b() -> Model {
    Model {
        id: "model-b".to_string(),
        api: "api-b".to_string(),
        provider: "provider-b".to_string(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Same-model scenarios
// ---------------------------------------------------------------------------

#[test]
fn same_model_preserves_thinking_with_signature() {
    let a = model_a();
    let msgs = vec![assistant_for(
        &a,
        vec![AssistantContent::Thinking(ThinkingContent {
            thinking: "reason".to_string(),
            thinking_signature: Some("sig-1".to_string()),
            redacted: None,
        })],
    )];
    let out = transform_messages(&msgs, &a, None);
    assert_eq!(out.len(), 1);
    match &out[0] {
        Message::Assistant(m) => match &m.content[0] {
            AssistantContent::Thinking(t) => {
                assert_eq!(t.thinking, "reason");
                assert!(t.thinking_signature.is_some());
            }
            other => panic!("expected Thinking, got {other:?}"),
        },
        other => panic!("expected Assistant, got {other:?}"),
    }
}

#[test]
fn same_model_preserves_redacted_thinking() {
    let a = model_a();
    let msgs = vec![assistant_for(
        &a,
        vec![AssistantContent::Thinking(ThinkingContent {
            thinking: "".to_string(),
            thinking_signature: None,
            redacted: Some(true),
        })],
    )];
    let out = transform_messages(&msgs, &a, None);
    match &out[0] {
        Message::Assistant(m) => {
            assert_eq!(m.content.len(), 1);
            assert!(matches!(&m.content[0], AssistantContent::Thinking(t) if t.redacted == Some(true)));
        }
        _ => panic!("expected Assistant"),
    }
}

#[test]
fn same_model_preserves_text_signature() {
    let a = model_a();
    let msgs = vec![assistant_for(
        &a,
        vec![AssistantContent::Text(TextContent {
            text: "answer".to_string(),
            text_signature: Some("sig".to_string()),
        })],
    )];
    let out = transform_messages(&msgs, &a, None);
    match &out[0] {
        Message::Assistant(m) => match &m.content[0] {
            AssistantContent::Text(t) => assert!(t.text_signature.is_some()),
            _ => panic!("expected Text"),
        },
        _ => panic!("expected Assistant"),
    }
}

// ---------------------------------------------------------------------------
// Cross-model scenarios
// ---------------------------------------------------------------------------

#[test]
fn cross_model_drops_redacted_thinking() {
    let a = model_a();
    let b = model_b();
    let msgs = vec![assistant_for(
        &a,
        vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "".to_string(),
                thinking_signature: None,
                redacted: Some(true),
            }),
            AssistantContent::Text(TextContent {
                text: "visible".to_string(),
                text_signature: None,
            }),
        ],
    )];
    // Transform targeting model-b
    let out = transform_messages(&msgs, &b, None);
    match &out[0] {
        Message::Assistant(m) => {
            // Redacted thinking dropped; only text remains.
            assert_eq!(m.content.len(), 1);
            assert!(matches!(&m.content[0], AssistantContent::Text(t) if t.text == "visible"));
        }
        _ => panic!("expected Assistant"),
    }
}

#[test]
fn cross_model_drops_empty_thinking() {
    let a = model_a();
    let b = model_b();
    let msgs = vec![assistant_for(
        &a,
        vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "   \n  ".to_string(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantContent::Text(TextContent {
                text: "real answer".to_string(),
                text_signature: None,
            }),
        ],
    )];
    let out = transform_messages(&msgs, &b, None);
    match &out[0] {
        Message::Assistant(m) => {
            assert_eq!(m.content.len(), 1);
            assert!(matches!(&m.content[0], AssistantContent::Text(t) if t.text == "real answer"));
        }
        _ => panic!("expected Assistant"),
    }
}

#[test]
fn cross_model_strips_text_signature() {
    let a = model_a();
    let b = model_b();
    let msgs = vec![assistant_for(
        &a,
        vec![AssistantContent::Text(TextContent {
            text: "answer".to_string(),
            text_signature: Some("sig".to_string()),
        })],
    )];
    let out = transform_messages(&msgs, &b, None);
    match &out[0] {
        Message::Assistant(m) => match &m.content[0] {
            AssistantContent::Text(t) => assert!(t.text_signature.is_none()),
            _ => panic!("expected Text"),
        },
        _ => panic!("expected Assistant"),
    }
}

#[test]
fn cross_model_strips_thought_signature_on_tool_call() {
    let a = model_a();
    let b = model_b();
    let msgs = vec![assistant_for(
        &a,
        vec![AssistantContent::ToolCall(ToolCall {
            id: "tc-1".to_string(),
            name: "echo".to_string(),
            arguments: serde_json::json!({}),
            thought_signature: Some("opaque".to_string()),
        })],
    )];
    let out = transform_messages(&msgs, &b, None);
    match &out[0] {
        Message::Assistant(m) => match &m.content[0] {
            AssistantContent::ToolCall(tc) => assert!(tc.thought_signature.is_none()),
            _ => panic!("expected ToolCall"),
        },
        _ => panic!("expected Assistant"),
    }
}

// ---------------------------------------------------------------------------
// Tool call ID normalization
// ---------------------------------------------------------------------------

#[test]
fn normalize_tool_call_id_renames_calls_and_results() {
    let a = model_a();
    let b = model_b();

    let norm: NormalizeToolCallIdFn = Box::new(|id, _model, _asst| format!("norm-{id}"));

    let msgs = vec![
        user("run a tool"),
        assistant_for(
            &a,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "orig-1".to_string(),
                name: "echo".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        tool_result("orig-1", "echo", "result"),
    ];

    let out = transform_messages(&msgs, &b, Some(&norm));

    // Assistant's tool call id is rewritten.
    match &out[1] {
        Message::Assistant(m) => match &m.content[0] {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.id, "norm-orig-1"),
            _ => panic!("expected ToolCall"),
        },
        _ => panic!("expected Assistant"),
    }
    // Tool result's tool_call_id is rewritten to match.
    match &out[2] {
        Message::ToolResult(tr) => assert_eq!(tr.tool_call_id, "norm-orig-1"),
        _ => panic!("expected ToolResult"),
    }
}

#[test]
fn same_model_does_not_normalize_ids() {
    let a = model_a();
    let norm: NormalizeToolCallIdFn = Box::new(|_id, _model, _asst| "SHOULD_NOT_APPLY".to_string());

    let msgs = vec![assistant_for(
        &a,
        vec![AssistantContent::ToolCall(ToolCall {
            id: "orig-1".to_string(),
            name: "echo".to_string(),
            arguments: serde_json::json!({}),
            thought_signature: None,
        })],
    )];
    let out = transform_messages(&msgs, &a, Some(&norm));
    match &out[0] {
        Message::Assistant(m) => match &m.content[0] {
            AssistantContent::ToolCall(tc) => assert_eq!(tc.id, "orig-1"),
            _ => panic!("expected ToolCall"),
        },
        _ => panic!("expected Assistant"),
    }
}

// ---------------------------------------------------------------------------
// Orphaned tool call handling
// ---------------------------------------------------------------------------

#[test]
fn orphaned_tool_call_gets_synthetic_result_at_user_turn() {
    let a = model_a();
    let msgs = vec![
        user("first"),
        assistant_for(
            &a,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "my_tool".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        user("next"),
    ];
    let out = transform_messages(&msgs, &a, None);
    // user + assistant + synthetic_tool_result + user
    assert_eq!(out.len(), 4);
    match &out[2] {
        Message::ToolResult(tr) => {
            assert_eq!(tr.tool_call_id, "tc-1");
            assert!(tr.is_error);
        }
        _ => panic!("expected synthetic ToolResult"),
    }
}

#[test]
fn orphaned_tool_call_gets_synthetic_result_at_next_assistant() {
    let a = model_a();
    let msgs = vec![
        assistant_for(
            &a,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "t".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        assistant_for(
            &a,
            vec![AssistantContent::Text(TextContent {
                text: "second".to_string(),
                text_signature: None,
            })],
        ),
    ];
    let out = transform_messages(&msgs, &a, None);
    // assistant + synthetic + assistant
    assert_eq!(out.len(), 3);
    assert!(matches!(&out[1], Message::ToolResult(_)));
}

#[test]
fn existing_tool_results_suppress_synthetic() {
    let a = model_a();
    let msgs = vec![
        assistant_for(
            &a,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "t".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        tool_result("tc-1", "t", "real result"),
        user("next"),
    ];
    let out = transform_messages(&msgs, &a, None);
    // assistant + real_tool_result + user (no synthetic injected)
    assert_eq!(out.len(), 3);
    match &out[1] {
        Message::ToolResult(tr) => {
            assert!(!tr.is_error);
            assert_eq!(tr.tool_call_id, "tc-1");
        }
        _ => panic!("expected real ToolResult"),
    }
}

// ---------------------------------------------------------------------------
// Error/aborted assistant messages
// ---------------------------------------------------------------------------

#[test]
fn skips_errored_and_aborted_assistant_messages() {
    let a = model_a();
    let msgs = vec![
        user("hi"),
        assistant_for_with_stop(
            &a,
            vec![AssistantContent::Text(TextContent {
                text: "err".to_string(),
                text_signature: None,
            })],
            StopReason::Error,
        ),
        assistant_for_with_stop(
            &a,
            vec![AssistantContent::Text(TextContent {
                text: "abt".to_string(),
                text_signature: None,
            })],
            StopReason::Aborted,
        ),
        user("still there?"),
    ];
    let out = transform_messages(&msgs, &a, None);
    // Only the two user messages survive.
    assert_eq!(out.len(), 2);
    assert!(matches!(&out[0], Message::User(_)));
    assert!(matches!(&out[1], Message::User(_)));
}

// ---------------------------------------------------------------------------
// Empty / passthrough cases
// ---------------------------------------------------------------------------

#[test]
fn empty_messages_returns_empty() {
    let a = model_a();
    let out = transform_messages(&[], &a, None);
    assert!(out.is_empty());
}

#[test]
fn user_only_conversation_is_passthrough() {
    let a = model_a();
    let msgs = vec![user("a"), user("b"), user("c")];
    let out = transform_messages(&msgs, &a, None);
    assert_eq!(out.len(), 3);
}
