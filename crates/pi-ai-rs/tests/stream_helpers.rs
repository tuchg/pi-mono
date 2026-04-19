//! Integration tests for the top-level `stream` / `complete` helpers.

use futures::StreamExt;
use pi_ai_rs::{
    complete, complete_simple, faux_assistant_text, faux_text, faux_assistant_message_with_stop,
    register_faux_provider, stream, stream_simple, AiError, AssistantContent,
    AssistantMessageEvent, Context, Model, RegisterFauxProviderOptions, StopReason,
};

fn empty_ctx() -> Context {
    Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    }
}

#[tokio::test]
async fn stream_returns_no_provider_error_for_unknown_api() {
    let bogus_model = Model {
        id: "ghost".to_string(),
        api: "non-existent-api-xyz".to_string(),
        provider: "nobody".to_string(),
        ..Default::default()
    };

    let result = stream(&bogus_model, empty_ctx(), Default::default());
    assert!(result.is_err());
    match result.err().unwrap() {
        AiError::NoProvider { api } => assert_eq!(api, "non-existent-api-xyz"),
        other => panic!("expected NoProvider, got {other:?}"),
    }
}

#[tokio::test]
async fn complete_returns_no_provider_error_for_unknown_api() {
    let bogus_model = Model {
        id: "ghost".to_string(),
        api: "non-existent-api-xyz".to_string(),
        provider: "nobody".to_string(),
        ..Default::default()
    };
    let err = complete(&bogus_model, empty_ctx(), Default::default())
        .await
        .unwrap_err();
    assert!(matches!(err, AiError::NoProvider { .. }));
}

#[tokio::test]
async fn stream_simple_and_complete_simple_work() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());
    reg.set_responses(vec![faux_assistant_text("hi")]);

    let msg = complete_simple(reg.get_model(), empty_ctx(), Default::default())
        .await
        .unwrap();
    assert!(matches!(&msg.content[0], AssistantContent::Text(t) if t.text == "hi"));
    assert_eq!(msg.stop_reason, StopReason::Stop);

    reg.unregister();
}

#[tokio::test]
async fn stream_and_complete_produce_same_final_message() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());
    reg.set_responses(vec![
        faux_assistant_text("first"),
        faux_assistant_text("first"),
    ]);

    // Via stream, drain events, check partial, and get result.
    let mut s = stream_simple(reg.get_model(), empty_ctx(), Default::default()).unwrap();
    let mut final_text = String::new();
    let mut last_event = None;
    while let Some(ev) = s.next().await {
        if let AssistantMessageEvent::TextEnd { content, .. } = &ev {
            final_text = content.clone();
        }
        last_event = Some(ev);
    }
    assert!(matches!(last_event, Some(AssistantMessageEvent::Done { .. })));
    assert_eq!(final_text, "first");

    // Via complete, should get the same text.
    let msg = complete_simple(reg.get_model(), empty_ctx(), Default::default())
        .await
        .unwrap();
    assert!(matches!(&msg.content[0], AssistantContent::Text(t) if t.text == "first"));

    reg.unregister();
}

#[tokio::test]
async fn error_response_surfaces_through_complete() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());
    // Queue an error-typed response.
    reg.set_responses(vec![faux_assistant_message_with_stop(
        vec![faux_text("x")],
        StopReason::Error,
    )]);

    let msg = complete_simple(reg.get_model(), empty_ctx(), Default::default())
        .await
        .unwrap();
    assert_eq!(msg.stop_reason, StopReason::Error);

    reg.unregister();
}

#[tokio::test]
async fn aborted_response_surfaces_through_complete() {
    let reg = register_faux_provider(RegisterFauxProviderOptions::default());
    reg.set_responses(vec![faux_assistant_message_with_stop(
        vec![faux_text("x")],
        StopReason::Aborted,
    )]);

    let msg = complete_simple(reg.get_model(), empty_ctx(), Default::default())
        .await
        .unwrap();
    assert_eq!(msg.stop_reason, StopReason::Aborted);

    reg.unregister();
}
