//! Integration tests for `convert_google_messages`.
//!
//! Ports scenarios from:
//! - `packages/ai/test/google-shared-gemini3-unsigned-tool-call.test.ts`
//! - `packages/ai/test/google-shared-image-tool-result-routing.test.ts`

use pi_ai_rs::providers::google_shared::{convert_google_messages, SKIP_THOUGHT_SIGNATURE};
use pi_ai_rs::types::{
    AssistantContent, AssistantMessage, Content, Context, ImageContent, InputModality, Message,
    Model, ModelCost, StopReason, TextContent, ToolCall, ToolResultMessage, Usage, UserContent,
    UserMessage,
};

fn base_model(id: &str, provider: &str, api: &str) -> Model {
    Model {
        id: id.to_string(),
        name: id.to_string(),
        api: api.to_string(),
        provider: provider.to_string(),
        base_url: String::new(),
        reasoning: false,
        input: vec![InputModality::Text, InputModality::Image],
        cost: ModelCost { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 },
        context_window: 128_000,
        max_tokens: 4096,
        headers: None,
        compat: None,
        supported_thinking_levels: None,
    }
}

fn google_model(id: &str) -> Model {
    base_model(id, "google", "google-gemini")
}

fn assistant_for(model: &Model, content: Vec<AssistantContent>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        stop_reason: StopReason::ToolUse,
        usage: Usage::default(),
        ..Default::default()
    })
}

fn user(text: &str) -> Message {
    Message::User(UserMessage { content: UserContent::Text(text.to_string()), timestamp: 0 })
}

fn tool_result_with_image(call_id: &str, name: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call_id.to_string(),
        tool_name: name.to_string(),
        content: vec![
            Content::Text(TextContent { text: text.to_string(), text_signature: None }),
            Content::Image(ImageContent {
                data: "ZmFrZQ==".to_string(),
                mime_type: "image/png".to_string(),
            }),
        ],
        details: None,
        is_error: false,
        timestamp: 0,
    })
}

fn tool_result_text_only(call_id: &str, name: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call_id.to_string(),
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

// =============================================================================
// Gemini 3 unsigned tool call tests
// (ports google-shared-gemini3-unsigned-tool-call.test.ts)
// =============================================================================

#[test]
fn gemini3_unsigned_tool_call_gets_skip_sentinel() {
    // Cross-provider (unsigned) tool call on Gemini 3 should get SKIP_THOUGHT_SIGNATURE.
    // TS test uses an assistant from google-antigravity/google-gemini-cli targeting google-generative-ai.
    let target = google_model("gemini-3-pro-preview");
    // Assistant came from a different provider (google-antigravity via google-gemini-cli)
    let cross_provider_asst = Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls -la"}),
            thought_signature: None,
        })],
        api: "google-gemini-cli".to_string(),
        provider: "google-antigravity".to_string(),
        model: "claude-sonnet-4-20250514".to_string(),
        stop_reason: StopReason::Stop,
        usage: Usage::default(),
        ..Default::default()
    });

    let msgs = vec![user("Hi"), cross_provider_asst];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&target, &ctx);

    let model_turn = contents.iter().find(|c| c.role == "model").expect("model turn");
    let fc_part = model_turn.parts.iter().find(|p| p.function_call.is_some()).expect("fc part");

    assert_eq!(fc_part.function_call.as_ref().unwrap().name, "bash");
    assert_eq!(
        fc_part.thought_signature.as_deref(),
        Some(SKIP_THOUGHT_SIGNATURE),
        "unsigned cross-provider tool call should get skip sentinel on Gemini 3"
    );
}

#[test]
fn gemini3_valid_signature_preserved_same_provider() {
    // Same-provider tool call with a valid base64 thought signature should be preserved.
    let model = google_model("gemini-3-flash-preview");
    // Valid base64 signature — 16 bytes = 24 chars, matches TS test's "AAAAAAAAAAAAAAAAAAAAAA=="
    let valid_sig = "AAAAAAAAAAAAAAAAAAAAAA==";

    let msgs = vec![
        user("call the tool"),
        assistant_for(
            &model,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-2".to_string(),
                name: "search".to_string(),
                arguments: serde_json::json!({"q": "hello"}),
                thought_signature: Some(valid_sig.to_string()),
            })],
        ),
    ];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&model, &ctx);

    let model_turn = contents.iter().find(|c| c.role == "model").expect("model turn");

    // Same provider+model → real signature is preserved (not replaced by skip sentinel).
    assert_eq!(
        model_turn.parts[0].thought_signature.as_deref(),
        Some(valid_sig),
        "same-provider valid signature should be preserved"
    );
}

#[test]
fn non_gemini3_unsigned_tool_call_no_sentinel() {
    // Gemini 2.5 should NOT get the SKIP_THOUGHT_SIGNATURE sentinel.
    let target = google_model("gemini-2.5-flash");
    let cross_provider_asst = Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
            thought_signature: None,
        })],
        api: "google-gemini-cli".to_string(),
        provider: "google-antigravity".to_string(),
        model: "claude-sonnet-4-20250514".to_string(),
        stop_reason: StopReason::Stop,
        usage: Usage::default(),
        ..Default::default()
    });

    let msgs = vec![user("Hi"), cross_provider_asst];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&target, &ctx);

    let model_turn = contents.iter().find(|c| c.role == "model").expect("model turn");
    let fc_part = model_turn.parts.iter().find(|p| p.function_call.is_some()).expect("fc part");

    assert!(
        fc_part.thought_signature.is_none(),
        "non-Gemini 3 unsigned tool call should have no thought signature"
    );
}

// =============================================================================
// Image tool result routing tests
// (ports google-shared-image-tool-result-routing.test.ts)
// =============================================================================

#[test]
fn gemini2_google_api_keeps_separate_synthetic_image_turn() {
    // Gemini 2.x does NOT support multimodal function responses.
    // Images should be in a SEPARATE user turn after the function response.
    let model = google_model("gemini-2.5-flash");
    // model's provider matches — same as target

    let msgs = vec![
        user("run the tool"),
        assistant_for(
            &model,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        tool_result_with_image("tc-1", "read", "file contents"),
    ];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&model, &ctx);

    // Expected: user | model | user (function_response) | user (image)
    assert_eq!(contents.len(), 4, "should have 4 turns for Gemini 2.x with image tool result");

    let last = &contents[3];
    assert_eq!(last.role, "user", "last turn should be user");
    assert!(
        last.parts.iter().any(|p| p.inline_data.is_some()),
        "last user turn should contain the image"
    );
}

#[test]
fn gemini3_google_api_nests_image_tool_results() {
    // Gemini 3 supports multimodal function responses.
    // Images should be nested inside the function response turn.
    let model = google_model("gemini-3-flash-preview");

    let msgs = vec![
        user("run the tool"),
        assistant_for(
            &model,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        tool_result_with_image("tc-1", "read", "file contents"),
    ];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&model, &ctx);

    // Expected: user | model | user (function_response with parts)
    assert_eq!(contents.len(), 3, "Gemini 3 should nest image in function response turn");

    let tool_turn = &contents[2];
    assert_eq!(tool_turn.role, "user");
    let fr = tool_turn.parts[0].function_response.as_ref().expect("function_response");
    assert!(
        fr.parts.is_some() && !fr.parts.as_ref().unwrap().is_empty(),
        "image parts should be nested in function_response"
    );
}

#[test]
fn non_gemini_antigravity_nests_image_tool_results() {
    // Non-Gemini models on google-antigravity (e.g., Claude via Cloud Code) support
    // multimodal function responses (same as Gemini 3 routing).
    let model = base_model("claude-opus-4-5-thinking", "google-antigravity", "google-gemini-cli");

    let msgs = vec![
        user("run the tool"),
        assistant_for(
            &model,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        tool_result_with_image("tc-1", "bash", "output"),
    ];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&model, &ctx);

    // Nested: user | model | user (function_response with image parts)
    assert_eq!(contents.len(), 3, "non-Gemini on antigravity should nest image in function response");

    let tool_turn = &contents[2];
    let fr = tool_turn.parts[0].function_response.as_ref().expect("function_response");
    assert!(
        fr.parts.is_some() && !fr.parts.as_ref().unwrap().is_empty(),
        "image parts should be nested"
    );
}

#[test]
fn gemini2_cloud_code_assist_keeps_separate_image_turn() {
    // Gemini 2.x models on google-gemini-cli (Cloud Code Assist) also use separate turns.
    let model = base_model("gemini-2.5-flash", "google-gemini-cli", "google-gemini-cli");

    let msgs = vec![
        user("run the tool"),
        assistant_for(
            &model,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc-1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
        ),
        tool_result_with_image("tc-1", "read", "contents"),
    ];

    let ctx = Context { messages: msgs, system_prompt: None, tools: None };
    let contents = convert_google_messages(&model, &ctx);

    // Separate image turn: user | model | user (function_response) | user (image)
    assert_eq!(contents.len(), 4, "Gemini 2.x on CCA should keep separate image turn");

    let last = &contents[3];
    assert_eq!(last.role, "user");
    assert!(
        last.parts.iter().any(|p| p.inline_data.is_some()),
        "separate user turn should contain the image"
    );
}
