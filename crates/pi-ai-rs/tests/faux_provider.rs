use futures::StreamExt;
use pi_ai_rs::{
    faux_assistant_message, faux_assistant_message_with_stop,
    faux_assistant_text, faux_text, faux_thinking, faux_tool_call_with_id,
    register_faux_provider, AssistantContent, AssistantMessageEvent, RegisterFauxProviderOptions,
    StopReason,
};

#[tokio::test]
async fn faux_provider_text_response() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_text("Hello world")]);

    let model = reg.get_model();
    let context = pi_ai_rs::Context {
        system_prompt: Some("test".to_string()),
        messages: vec![],
        tools: None,
    };

    let mut stream = pi_ai_rs::stream_simple(model, context, Default::default()).unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have: Start, TextStart, TextDelta, TextEnd, Done
    assert!(events.len() >= 5);
    assert!(matches!(events.first().unwrap(), AssistantMessageEvent::Start { .. }));
    assert!(matches!(events.last().unwrap(), AssistantMessageEvent::Done { .. }));

    // Verify text delta
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["Hello world"]);

    reg.unregister();
}

#[tokio::test]
async fn faux_provider_response_queue_dequeues_in_order() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![
        faux_assistant_text("first"),
        faux_assistant_text("second"),
    ]);
    assert_eq!(reg.pending_response_count(), 2);

    let model = reg.get_model();
    let ctx = pi_ai_rs::Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    };

    // First call
    let s1 = pi_ai_rs::stream_simple(model, ctx.clone(), Default::default()).unwrap();
    let msg1 = s1.result().await.unwrap();
    let text1 = msg1
        .content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(text1, "first");

    // Second call
    let s2 = pi_ai_rs::stream_simple(model, ctx.clone(), Default::default()).unwrap();
    let msg2 = s2.result().await.unwrap();
    let text2 = msg2
        .content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(text2, "second");

    assert_eq!(reg.pending_response_count(), 0);

    // Third call — no more responses, should get error
    let s3 = pi_ai_rs::stream_simple(model, ctx, Default::default()).unwrap();
    let msg3 = s3.result().await.unwrap();
    assert_eq!(msg3.stop_reason, StopReason::Error);
    assert!(msg3.error_message.is_some());

    reg.unregister();
}

#[tokio::test]
async fn faux_provider_thinking_content() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_message(vec![
        faux_thinking("step by step"),
        faux_text("answer"),
    ])]);

    let model = reg.get_model();
    let ctx = pi_ai_rs::Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    };

    let mut stream = pi_ai_rs::stream_simple(model, ctx, Default::default()).unwrap();

    let mut has_thinking_start = false;
    let mut has_thinking_delta = false;
    let mut has_thinking_end = false;
    let mut has_text_start = false;

    while let Some(event) = stream.next().await {
        match &event {
            AssistantMessageEvent::ThinkingStart { .. } => has_thinking_start = true,
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                assert_eq!(delta, "step by step");
                has_thinking_delta = true;
            }
            AssistantMessageEvent::ThinkingEnd { content, .. } => {
                assert_eq!(content, "step by step");
                has_thinking_end = true;
            }
            AssistantMessageEvent::TextStart { .. } => has_text_start = true,
            _ => {}
        }
    }

    assert!(has_thinking_start);
    assert!(has_thinking_delta);
    assert!(has_thinking_end);
    assert!(has_text_start);

    reg.unregister();
}

#[tokio::test]
async fn faux_provider_tool_call() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_message_with_stop(
        vec![
            faux_text("calling tool"),
            faux_tool_call_with_id("tc-1", "bash", serde_json::json!({"command": "ls"})),
        ],
        StopReason::ToolUse,
    )]);

    let model = reg.get_model();
    let ctx = pi_ai_rs::Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    };

    let stream = pi_ai_rs::stream_simple(model, ctx, Default::default()).unwrap();
    let msg = stream.result().await.unwrap();

    assert_eq!(msg.stop_reason, StopReason::ToolUse);
    assert_eq!(msg.content.len(), 2);
    assert!(matches!(&msg.content[1], AssistantContent::ToolCall(tc) if tc.name == "bash"));

    reg.unregister();
}

#[tokio::test]
async fn faux_provider_append_responses() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_text("a")]);
    reg.append_responses(vec![faux_assistant_text("b")]);
    assert_eq!(reg.pending_response_count(), 2);

    reg.unregister();
}

#[tokio::test]
async fn faux_provider_call_count() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_text("x"), faux_assistant_text("y")]);

    let model = reg.get_model();
    let ctx = pi_ai_rs::Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    };

    assert_eq!(reg.call_count(), 0);

    let _ = pi_ai_rs::stream_simple(model, ctx.clone(), Default::default())
        .unwrap()
        .result()
        .await;
    assert_eq!(reg.call_count(), 1);

    let _ = pi_ai_rs::stream_simple(model, ctx, Default::default())
        .unwrap()
        .result()
        .await;
    assert_eq!(reg.call_count(), 2);

    reg.unregister();
}
