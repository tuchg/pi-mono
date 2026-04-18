use pi_ai_rs::types::*;
use serde_json;

#[test]
fn message_user_roundtrip() {
    let msg = Message::User(UserMessage {
        content: UserContent::Text("hello".to_string()),
        timestamp: 1234567890,
    });

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn message_assistant_roundtrip() {
    let msg = Message::Assistant(AssistantMessage {
        content: vec![
            AssistantContent::Text(TextContent {
                text: "hello".to_string(),
                text_signature: None,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "hmm".to_string(),
                thinking_signature: Some("sig".to_string()),
                redacted: None,
            }),
        ],
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "gpt-4o".to_string(),
        response_id: Some("resp-1".to_string()),
        usage: Usage {
            input: 100,
            output: 50,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 150,
            cost: UsageCost {
                input: 0.001,
                output: 0.002,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.003,
            },
        },
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 999,
    });

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn message_tool_result_roundtrip() {
    let msg = Message::ToolResult(ToolResultMessage {
        tool_call_id: "tc-1".to_string(),
        tool_name: "bash".to_string(),
        content: vec![Content::Text(TextContent {
            text: "output".to_string(),
            text_signature: None,
        })],
        details: Some(serde_json::json!({"key": "value"})),
        is_error: false,
        timestamp: 555,
    });

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn api_enum_serde() {
    let api = Api::AnthropicMessages;
    let json = serde_json::to_value(&api).unwrap();
    assert_eq!(json, "anthropic-messages");

    let parsed: Api = serde_json::from_value(json).unwrap();
    assert_eq!(parsed, api);
}

#[test]
fn api_custom_serde() {
    let api = Api::Custom("my-api".to_string());
    let json = serde_json::to_value(&api).unwrap();
    assert_eq!(json, "my-api");

    // Custom API strings that don't match known variants deserialize as Custom
    let parsed: Api = serde_json::from_value(serde_json::json!("my-api")).unwrap();
    assert_eq!(parsed, Api::Custom("my-api".to_string()));
}

#[test]
fn provider_enum_serde() {
    let p = Provider::Anthropic;
    let json = serde_json::to_value(&p).unwrap();
    assert_eq!(json, "anthropic");

    let parsed: Provider = serde_json::from_value(json).unwrap();
    assert_eq!(parsed, p);
}

#[test]
fn stop_reason_serde() {
    let sr = StopReason::ToolUse;
    let json = serde_json::to_value(&sr).unwrap();
    assert_eq!(json, "toolUse");

    let parsed: StopReason = serde_json::from_value(json).unwrap();
    assert_eq!(parsed, sr);
}

#[test]
fn stream_options_default() {
    let opts = StreamOptions::default();
    assert!(opts.temperature.is_none());
    assert!(opts.max_tokens.is_none());
    assert!(opts.api_key.is_none());
}

#[test]
fn simple_stream_options_as_stream_options() {
    let simple = SimpleStreamOptions {
        temperature: Some(0.7),
        max_tokens: Some(1024),
        reasoning: Some(ThinkingLevel::High),
        ..Default::default()
    };

    let base = simple.as_stream_options();
    assert_eq!(base.temperature, Some(0.7));
    assert_eq!(base.max_tokens, Some(1024));
    // reasoning is not part of StreamOptions
}

#[test]
fn tool_serde_roundtrip() {
    let tool = Tool {
        name: "bash".to_string(),
        description: "Run a command".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            }
        }),
    };

    let json = serde_json::to_string(&tool).unwrap();
    let parsed: Tool = serde_json::from_str(&json).unwrap();
    assert_eq!(tool, parsed);
}

#[test]
fn model_default() {
    let model = Model::default();
    assert_eq!(model.id, "unknown");
    assert_eq!(model.reasoning, false);
    assert_eq!(model.context_window, 0);
}

#[test]
fn assistant_message_event_serde() {
    let event = AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "hello".to_string(),
        partial: AssistantMessage::default(),
    };

    let json = serde_json::to_string(&event).unwrap();
    let parsed: AssistantMessageEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, parsed);
}

#[test]
fn user_content_text_serde() {
    let content = UserContent::Text("plain text".to_string());
    let json = serde_json::to_value(&content).unwrap();
    // Untagged — should serialize as a plain string
    assert_eq!(json, "plain text");

    let parsed: UserContent = serde_json::from_value(json).unwrap();
    assert_eq!(content, parsed);
}

#[test]
fn context_serde_roundtrip() {
    let ctx = Context {
        system_prompt: Some("You are helpful.".to_string()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hi".to_string()),
            timestamp: 0,
        })],
        tools: None,
    };

    let json = serde_json::to_string(&ctx).unwrap();
    let parsed: Context = serde_json::from_str(&json).unwrap();
    assert_eq!(ctx, parsed);
}
