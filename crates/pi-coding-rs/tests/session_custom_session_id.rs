mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::SessionManager;
use serde_json::json;
use std::fs;
use tempfile::TempDir;

const UUID_V7_RE: &str = r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$";

fn is_uuidv7(s: &str) -> bool {
    let re = regex::Regex::new(UUID_V7_RE).unwrap();
    re.is_match(s)
}

// ---------------------------------------------------------------------------
// Custom session id
// ---------------------------------------------------------------------------

#[test]
fn uses_provided_id_instead_of_generating_one() {
    let mut session = SessionManager::in_memory(None);
    session.new_session(Some(pi_coding_rs::session::NewSessionOptions {
        id: Some("my-custom-id".to_string()),
        parent_session: None,
    }));
    assert_eq!(session.get_session_id(), "my-custom-id");
}

#[test]
fn generates_uuidv7_when_no_id_provided() {
    let mut session = SessionManager::in_memory(None);
    session.new_session(None);
    let id = session.get_session_id().to_string();
    assert!(!id.is_empty());
    assert!(is_uuidv7(&id), "expected UUIDv7 but got: {}", id);
}

#[test]
fn generates_uuidv7_when_options_provided_without_id() {
    let mut session = SessionManager::in_memory(None);
    session.new_session(Some(pi_coding_rs::session::NewSessionOptions {
        id: None,
        parent_session: Some("parent.jsonl".to_string()),
    }));
    let id = session.get_session_id().to_string();
    assert!(!id.is_empty());
    assert!(is_uuidv7(&id), "expected UUIDv7 but got: {}", id);
}

#[test]
fn custom_id_included_in_session_header() {
    let mut session = SessionManager::in_memory(None);
    session.new_session(Some(pi_coding_rs::session::NewSessionOptions {
        id: Some("header-test-id".to_string()),
        parent_session: None,
    }));

    let header = session.get_header().expect("should have header");
    assert_eq!(header.id, "header-test-id");
}

#[test]
fn generates_uuidv7_when_constructed_without_explicit_id() {
    let session = SessionManager::in_memory(None);
    let id = session.get_session_id().to_string();
    assert!(is_uuidv7(&id), "expected UUIDv7 but got: {}", id);
    assert_eq!(session.get_header().unwrap().id, id);
}

#[test]
fn generates_uuidv7_when_creating_branched_session() {
    let mut session = SessionManager::in_memory(None);
    let first_id = session.append_message(user_msg("hello"));

    session.create_branched_session(&first_id).unwrap();

    let id = session.get_session_id().to_string();
    assert!(is_uuidv7(&id), "expected UUIDv7 but got: {}", id);
    assert_eq!(session.get_header().unwrap().id, id);
}

#[test]
fn generates_uuidv7_when_forking_from_another_session_file() {
    let tmp = TempDir::new().unwrap();
    let source_path = tmp.path().join("source.jsonl");

    fs::write(
        &source_path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&json!({
                "type": "session",
                "version": 3,
                "id": "legacy-session-id",
                "timestamp": "2025-01-01T00:00:00Z",
                "cwd": tmp.path().to_string_lossy()
            }))
            .unwrap(),
            serde_json::to_string(&json!({
                "type": "message",
                "id": "entry-1",
                "parentId": null,
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [{ "type": "text", "text": "hello" }],
                    "api": "anthropic-messages",
                    "provider": "anthropic",
                    "model": "test",
                    "usage": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
                    },
                    "stopReason": "stop",
                    "timestamp": 1
                }
            }))
            .unwrap()
        ),
    )
    .unwrap();

    let forked = SessionManager::fork_from(
        &source_path,
        tmp.path().to_str().unwrap(),
        Some(tmp.path()),
    )
    .unwrap();

    let header = forked.get_header().expect("should have header");
    assert!(is_uuidv7(&header.id), "expected UUIDv7 but got: {}", header.id);
    assert_eq!(
        header.parent_session.as_deref(),
        Some(source_path.to_string_lossy().as_ref())
    );
}
