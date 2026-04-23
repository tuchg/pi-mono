use pi_ai_rs::registry::{
    clear_api_providers, get_api_provider, register_api_provider,
    unregister_api_providers, ApiProvider,
};
use pi_ai_rs::event_stream::create_assistant_message_event_stream;
use std::sync::Arc;

fn noop_provider(api: &str) -> ApiProvider {
    ApiProvider {
        api: api.to_string(),
        stream: Arc::new(|_model, _ctx, _opts| {
            let (mut s, r) = create_assistant_message_event_stream();
            s.end(Some(pi_ai_rs::AssistantMessage::default()));
            r
        }),
        stream_simple: Arc::new(|_model, _ctx, _opts| {
            let (mut s, r) = create_assistant_message_event_stream();
            s.end(Some(pi_ai_rs::AssistantMessage::default()));
            r
        }),
    }
}

// Registry tests must run serially since they share global state.
// Use unique API names to avoid interference with other test files.

#[test]
fn register_and_lookup() {
    let api = format!("reg-test-1-{}", uuid::Uuid::new_v4());
    register_api_provider(noop_provider(&api), None);
    let found = get_api_provider(&api);
    assert!(found.is_some());
    assert_eq!(found.unwrap().api, api);
}

#[test]
fn lookup_missing_returns_none() {
    assert!(get_api_provider("nonexistent-9999").is_none());
}

#[test]
fn unregister_by_source_id() {
    let source = format!("src-{}", uuid::Uuid::new_v4());
    let api_a = format!("unreg-a-{}", uuid::Uuid::new_v4());
    let api_b = format!("unreg-b-{}", uuid::Uuid::new_v4());
    let api_c = format!("unreg-c-{}", uuid::Uuid::new_v4());

    register_api_provider(noop_provider(&api_a), Some(&source));
    register_api_provider(noop_provider(&api_b), Some(&source));
    register_api_provider(noop_provider(&api_c), Some("other-source"));

    unregister_api_providers(&source);

    assert!(get_api_provider(&api_a).is_none());
    assert!(get_api_provider(&api_b).is_none());
    assert!(get_api_provider(&api_c).is_some());

    // Cleanup
    unregister_api_providers("other-source");
}

#[test]
fn clear_removes_all() {
    // This test uses clear so just verify the function runs.
    // We register unique entries, clear, and verify they're gone.
    let api_x = format!("clear-x-{}", uuid::Uuid::new_v4());
    let api_y = format!("clear-y-{}", uuid::Uuid::new_v4());
    register_api_provider(noop_provider(&api_x), None);
    register_api_provider(noop_provider(&api_y), None);

    // Both should be findable
    assert!(get_api_provider(&api_x).is_some());
    assert!(get_api_provider(&api_y).is_some());

    // After clear, neither should be findable
    clear_api_providers();
    assert!(get_api_provider(&api_x).is_none());
    assert!(get_api_provider(&api_y).is_none());
}
