//! Integration tests for `convert_completions_messages`.
//!
//! Ports scenarios from:
//! - `packages/ai/test/openai-completions-tool-result-images.test.ts`

use pi_ai_rs::providers::openai_completions_shared::convert_completions_messages;
use pi_ai_rs::types::{
    AssistantContent, AssistantMessage, Content, Context, ImageContent, InputModality, Message,
    Model, ModelCost, StopReason, TextContent, ToolCall, ToolResultMessage, Usage, UserContent,
    UserMessage,
};

fn openai_image_model() -> Model {
    Model {
        id: "gpt-4o-mini".to_string(),
        name: "GPT-4o Mini".to_string(),
        api: "openai-completions".to_string(),
        provider: "openai".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
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

fn user(text: &str) -> Message {
    Message::User(UserMessage { content: UserContent::Text(text.to_string()), timestamp: 0 })
}

fn tool_result_with_image(call_id: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call_id.to_string(),
        tool_name: "read".to_string(),
        content: vec![
            Content::Text(TextContent {
                text: "Read image file [image/png]".to_string(),
                text_signature: None,
            }),
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

/// Port of `openai-completions convertMessages` →
/// "batches tool-result images after consecutive tool results"
#[test]
fn batches_tool_result_images_after_consecutive_tool_results() {
    let model = openai_image_model();

    let assistant = Message::Assistant(AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(ToolCall {
                id: "tool-1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({"path": "img-1.png"}),
                thought_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "tool-2".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({"path": "img-2.png"}),
                thought_signature: None,
            }),
        ],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        stop_reason: StopReason::ToolUse,
        usage: Usage::default(),
        ..Default::default()
    });

    let ctx = Context {
        messages: vec![
            user("Read the images"),
            assistant,
            tool_result_with_image("tool-1"),
            tool_result_with_image("tool-2"),
        ],
        system_prompt: None,
        tools: None,
    };

    let messages = convert_completions_messages(&model, &ctx);
    let roles: Vec<&str> =
        messages.iter().map(|m| m["role"].as_str().unwrap_or("")).collect();

    assert_eq!(roles, vec!["user", "assistant", "tool", "tool", "user"]);

    let image_message = messages.last().expect("last message");
    assert_eq!(image_message["role"], "user");

    let content = image_message["content"].as_array().expect("content array");
    let image_parts: Vec<_> =
        content.iter().filter(|p| p["type"] == "image_url").collect();
    assert_eq!(image_parts.len(), 2, "should batch 2 images from 2 tool results");
}
