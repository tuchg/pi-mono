//! Integration tests for `convert_responses_messages`.
//!
//! Ports scenarios from:
//! - `packages/ai/test/openai-responses-foreign-toolcall-id.test.ts`

use std::collections::HashSet;

use pi_ai_rs::providers::openai_responses_shared::convert_responses_messages;
use pi_ai_rs::types::{
    AssistantContent, AssistantMessage, Content, Context, InputModality, Message, Model,
    ModelCost, StopReason, TextContent, ToolCall, ToolResultMessage, Usage, UserContent,
    UserMessage,
};
use pi_ai_rs::utils::hash::short_hash;

/// Raw Copilot tool call ID as it appears in cross-provider handoffs.
const COPILOT_RAW_TOOL_CALL_ID: &str = concat!(
    "call_4VnzVawQXPB9MgYib7CiQFEY|",
    "I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLU",
    "G7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD0",
    "70Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04",
    "caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcb",
    "i2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA=="
);

fn codex_model() -> Model {
    Model {
        id: "gpt-5.3-codex".to_string(),
        name: "GPT-5.3 Codex".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai-codex".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        reasoning: true,
        input: vec![InputModality::Text],
        cost: ModelCost { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 },
        context_window: 400_000,
        max_tokens: 128_000,
        headers: None,
        compat: None,
        supported_thinking_levels: None,
    }
}

/// Port of "hashes foreign Copilot tool item IDs into a bounded Codex-safe fc_<hash> shape"
/// from `packages/ai/test/openai-responses-foreign-toolcall-id.test.ts`.
#[test]
fn hashes_foreign_copilot_tool_call_id_into_fc_hash() {
    let model = codex_model();

    let assistant = Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: COPILOT_RAW_TOOL_CALL_ID.to_string(),
            name: "edit".to_string(),
            arguments: serde_json::json!({"path": "src/styles/app.css"}),
            thought_signature: None,
        })],
        api: "openai-responses".to_string(),
        provider: "github-copilot".to_string(),
        model: "gpt-5.3-codex".to_string(),
        stop_reason: StopReason::ToolUse,
        usage: Usage::default(),
        ..Default::default()
    });

    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: COPILOT_RAW_TOOL_CALL_ID.to_string(),
        tool_name: "edit".to_string(),
        content: vec![Content::Text(TextContent { text: "ok".to_string(), text_signature: None })],
        details: None,
        is_error: false,
        timestamp: 0,
    });

    let ctx = Context {
        system_prompt: Some("You are concise.".to_string()),
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Text("Use the tool.".to_string()),
                timestamp: 0,
            }),
            assistant,
            tool_result,
        ],
        tools: None,
    };

    let mut allowed: HashSet<String> = HashSet::new();
    allowed.insert("openai".to_string());
    allowed.insert("openai-codex".to_string());
    allowed.insert("opencode".to_string());

    let input = convert_responses_messages(&model, &ctx, &allowed, None);
    let function_call = input.iter().find(|item| item["type"] == "function_call");

    assert!(function_call.is_some(), "expected function_call item in payload");
    let fc = function_call.unwrap();

    // The item_id (part after |) should be hashed: fc_<short_hash(item_id)>
    let raw_item_id = COPILOT_RAW_TOOL_CALL_ID.splitn(2, '|').nth(1).unwrap();
    let expected_item_id = format!("fc_{}", short_hash(raw_item_id));

    let actual_id = fc["id"].as_str().unwrap_or("");
    assert_eq!(actual_id, expected_item_id);
    assert!(actual_id.len() <= 64, "id must be at most 64 chars");
    assert!(
        actual_id.starts_with("fc_"),
        "id should start with fc_"
    );
}
