//! Additional integration tests for the faux provider.
//!
//! Complements `faux_provider.rs`, focusing on configuration, registration
//! edge cases, and behaviors covered in TS `packages/ai/test/faux-provider.test.ts`.

use futures::StreamExt;
use pi_ai_rs::{
    complete, faux_assistant_message, faux_assistant_message_with_options,
    faux_assistant_message_with_stop, faux_assistant_text, faux_text, faux_tool_call,
    register_faux_provider, stream, AssistantContent, AssistantMessageEvent,
    FauxAssistantMessageOptions, FauxModelDefinition, RegisterFauxProviderOptions, StopReason,
    TokenSize,
};

fn empty_context() -> pi_ai_rs::Context {
    pi_ai_rs::Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    }
}

#[tokio::test]
async fn custom_api_provider_and_model_names_are_used() {
    let reg = register_faux_provider(RegisterFauxProviderOptions {
        api: Some("faux:test".to_string()),
        provider: Some("faux-provider".to_string()),
        models: Some(vec![FauxModelDefinition {
            id: "faux-model".to_string(),
            ..Default::default()
        }]),
        ..Default::default()
    });

    reg.set_responses(vec![faux_assistant_text("hello")]);

    let msg = complete(reg.get_model(), empty_context(), Default::default())
        .await
        .unwrap();

    assert_eq!(msg.api, "faux:test");
    assert_eq!(msg.provider, "faux-provider");
    assert_eq!(msg.model, "faux-model");

    reg.unregister();
}

#[tokio::test]
async fn multiple_models_with_per_model_reasoning() {
    let reg = register_faux_provider(RegisterFauxProviderOptions {
        models: Some(vec![
            FauxModelDefinition {
                id: "faux-fast".to_string(),
                name: Some("Faux Fast".to_string()),
                reasoning: false,
                ..Default::default()
            },
            FauxModelDefinition {
                id: "faux-thinker".to_string(),
                name: Some("Faux Thinker".to_string()),
                reasoning: true,
                ..Default::default()
            },
        ]),
        ..Default::default()
    });

    assert_eq!(reg.models.len(), 2);
    assert_eq!(reg.models[0].id, "faux-fast");
    assert!(!reg.models[0].reasoning);
    assert_eq!(reg.models[1].id, "faux-thinker");
    assert!(reg.models[1].reasoning);

    assert_eq!(reg.get_model().id, "faux-fast");
    assert_eq!(reg.get_model_by_id("faux-thinker").unwrap().id, "faux-thinker");
    assert!(reg.get_model_by_id("missing").is_none());

    reg.unregister();
}

#[tokio::test]
async fn set_responses_replaces_existing_queue() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![
        faux_assistant_text("a"),
        faux_assistant_text("b"),
        faux_assistant_text("c"),
    ]);
    assert_eq!(reg.pending_response_count(), 3);

    reg.set_responses(vec![faux_assistant_text("x")]);
    assert_eq!(reg.pending_response_count(), 1);

    reg.unregister();
}

#[tokio::test]
async fn append_extends_existing_queue() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_text("a")]);
    reg.append_responses(vec![faux_assistant_text("b"), faux_assistant_text("c")]);
    assert_eq!(reg.pending_response_count(), 3);

    let m1 = complete(reg.get_model(), empty_context(), Default::default()).await.unwrap();
    let m2 = complete(reg.get_model(), empty_context(), Default::default()).await.unwrap();
    let m3 = complete(reg.get_model(), empty_context(), Default::default()).await.unwrap();

    assert!(matches!(&m1.content[0], AssistantContent::Text(t) if t.text == "a"));
    assert!(matches!(&m2.content[0], AssistantContent::Text(t) if t.text == "b"));
    assert!(matches!(&m3.content[0], AssistantContent::Text(t) if t.text == "c"));
    assert_eq!(reg.pending_response_count(), 0);

    reg.unregister();
}

#[tokio::test]
async fn error_when_queue_exhausted() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());
    reg.set_responses(vec![]);

    let msg = complete(reg.get_model(), empty_context(), Default::default())
        .await
        .unwrap();
    assert_eq!(msg.stop_reason, StopReason::Error);
    assert_eq!(msg.error_message.as_deref(), Some("No more faux responses queued"));

    reg.unregister();
}

#[tokio::test]
async fn error_response_emits_error_event() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_message_with_options(
        vec![faux_text("partial")],
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some("boom".to_string()),
            ..Default::default()
        },
    )]);

    let mut s = stream(reg.get_model(), empty_context(), Default::default()).unwrap();
    let mut saw_error = false;
    let mut saw_done = false;
    while let Some(ev) = s.next().await {
        match ev {
            AssistantMessageEvent::Error { error, .. } => {
                saw_error = true;
                assert_eq!(error.error_message.as_deref(), Some("boom"));
            }
            AssistantMessageEvent::Done { .. } => saw_done = true,
            _ => {}
        }
    }

    assert!(saw_error, "should emit Error event for error stop reason");
    assert!(!saw_done, "should not emit Done event for error stop reason");

    reg.unregister();
}

#[tokio::test]
async fn options_variants_set_stop_reason_and_error() {
    let msg = faux_assistant_message_with_options(
        vec![faux_text("hi")],
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            error_message: None,
            response_id: Some("resp-42".to_string()),
            timestamp: Some(123),
        },
    );
    assert_eq!(msg.stop_reason, StopReason::ToolUse);
    assert_eq!(msg.response_id.as_deref(), Some("resp-42"));
    assert_eq!(msg.timestamp, 123);

    let aborted = faux_assistant_message_with_stop(vec![faux_text("bye")], StopReason::Aborted);
    assert_eq!(aborted.stop_reason, StopReason::Aborted);
}

#[tokio::test]
async fn tool_call_content_is_streamed_as_start_and_end() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());

    reg.set_responses(vec![faux_assistant_message_with_stop(
        vec![faux_tool_call("echo", serde_json::json!({"x": 1}))],
        StopReason::ToolUse,
    )]);

    let mut s = stream(reg.get_model(), empty_context(), Default::default()).unwrap();
    let mut saw_tool_start = false;
    let mut saw_tool_end = false;
    while let Some(ev) = s.next().await {
        match ev {
            AssistantMessageEvent::ToolcallStart { .. } => saw_tool_start = true,
            AssistantMessageEvent::ToolcallEnd { .. } => saw_tool_end = true,
            _ => {}
        }
    }
    assert!(saw_tool_start);
    assert!(saw_tool_end);

    reg.unregister();
}

#[tokio::test]
async fn multiple_registrations_do_not_collide() {
    let reg_a = register_faux_provider(RegisterFauxProviderOptions {
        api: Some("faux-a".to_string()),
        ..Default::default()
    });
    let reg_b = register_faux_provider(RegisterFauxProviderOptions {
        api: Some("faux-b".to_string()),
        ..Default::default()
    });

    reg_a.set_responses(vec![faux_assistant_text("from A")]);
    reg_b.set_responses(vec![faux_assistant_text("from B")]);

    let msg_a = complete(reg_a.get_model(), empty_context(), Default::default())
        .await
        .unwrap();
    let msg_b = complete(reg_b.get_model(), empty_context(), Default::default())
        .await
        .unwrap();

    assert!(matches!(&msg_a.content[0], AssistantContent::Text(t) if t.text == "from A"));
    assert!(matches!(&msg_b.content[0], AssistantContent::Text(t) if t.text == "from B"));

    reg_a.unregister();
    reg_b.unregister();
}

#[tokio::test]
async fn unregister_removes_provider_from_registry() {
    let reg = register_faux_provider(RegisterFauxProviderOptions {
        api: Some("faux-ephemeral".to_string()),
        ..Default::default()
    });
    assert!(pi_ai_rs::get_api_provider("faux-ephemeral").is_some());

    reg.unregister();
    assert!(pi_ai_rs::get_api_provider("faux-ephemeral").is_none());
}

#[tokio::test]
async fn custom_token_size_produces_multiple_deltas() {
    let reg = register_faux_provider(RegisterFauxProviderOptions {
        token_size: Some(TokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..Default::default()
    });

    reg.set_responses(vec![faux_assistant_text("abcdefgh")]);

    let mut s = stream(reg.get_model(), empty_context(), Default::default()).unwrap();
    let mut deltas = 0usize;
    while let Some(ev) = s.next().await {
        if let AssistantMessageEvent::TextDelta { .. } = ev {
            deltas += 1;
        }
    }
    // With token_size=1 and text_length=8, chunk_size=1*4=4, so we should see 2 deltas
    assert!(deltas >= 2, "expected multiple deltas, got {deltas}");

    reg.unregister();
}

#[tokio::test]
async fn empty_text_still_emits_start_and_end() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());
    reg.set_responses(vec![faux_assistant_text("")]);

    let mut s = stream(reg.get_model(), empty_context(), Default::default()).unwrap();
    let mut saw_start = false;
    let mut saw_end = false;
    let mut saw_done = false;
    while let Some(ev) = s.next().await {
        match ev {
            AssistantMessageEvent::TextStart { .. } => saw_start = true,
            AssistantMessageEvent::TextEnd { .. } => saw_end = true,
            AssistantMessageEvent::Done { .. } => saw_done = true,
            _ => {}
        }
    }
    assert!(saw_start);
    assert!(saw_end);
    assert!(saw_done);

    reg.unregister();
}

#[tokio::test]
async fn faux_assistant_message_default_stop_reason_is_stop() {
    let msg = faux_assistant_message(vec![faux_text("hi")]);
    assert_eq!(msg.stop_reason, StopReason::Stop);
    assert_eq!(msg.api, "faux");
    assert_eq!(msg.provider, "faux");
}
