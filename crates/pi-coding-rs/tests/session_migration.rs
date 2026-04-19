mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::{migrate_session_entries, CURRENT_SESSION_VERSION};
use serde_json::{Value, json};

fn entry_id(e: &Value) -> &str {
    e.get("id").and_then(|v| v.as_str()).unwrap_or("")
}

fn entry_parent_id(e: &Value) -> Option<&str> {
    match e.get("parentId") {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// v1 → v2: add id / parentId
// ---------------------------------------------------------------------------

#[test]
fn v1_to_v2_adds_id_and_parent_id() {
    let mut entries = vec![
        json!({
            "type": "session",
            "id": "sess-1",
            "timestamp": "2025-01-01T00:00:00Z",
            "cwd": "/tmp"
        }),
        json!({
            "type": "message",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": { "role": "user", "content": "hi", "timestamp": 1 }
        }),
        json!({
            "type": "message",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "hello" }],
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": "test",
                "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2 },
                "stopReason": "stop",
                "timestamp": 2
            }
        }),
    ];

    migrate_session_entries(&mut entries);

    // Header should be updated to current version
    let header = &entries[0];
    assert_eq!(
        header.get("version").and_then(|v| v.as_u64()),
        Some(CURRENT_SESSION_VERSION as u64)
    );

    let msg1 = &entries[1];
    let id1 = entry_id(msg1);
    assert_eq!(id1.len(), 8, "id should be 8 chars");
    assert_eq!(msg1.get("parentId"), Some(&Value::Null));

    let msg2 = &entries[2];
    let id2 = entry_id(msg2);
    assert_eq!(id2.len(), 8);
    assert_eq!(entry_parent_id(msg2), Some(id1));
}

#[test]
fn v1_migration_is_idempotent() {
    let mut entries = vec![
        json!({
            "type": "session",
            "id": "sess-1",
            "timestamp": "2025-01-01T00:00:00Z",
            "cwd": "/tmp"
        }),
        json!({
            "type": "message",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": { "role": "user", "content": "hi", "timestamp": 1 }
        }),
    ];

    migrate_session_entries(&mut entries);
    let id_after_first = entry_id(&entries[1]).to_string();

    migrate_session_entries(&mut entries);
    let id_after_second = entry_id(&entries[1]).to_string();

    assert_eq!(id_after_first, id_after_second, "ids should not change on second migration");
}

#[test]
fn v1_compaction_firstkeptentryindex_migrated_to_id() {
    let mut entries = vec![
        json!({
            "type": "session",
            "id": "sess-1",
            "timestamp": "2025-01-01T00:00:00Z",
            "cwd": "/tmp"
        }),
        json!({
            "type": "message",
            "timestamp": "2025-01-01T00:00:01Z",
            "message": { "role": "user", "content": "hi", "timestamp": 1 }
        }),
        json!({
            "type": "message",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "response" }],
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": "test",
                "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2 },
                "stopReason": "stop",
                "timestamp": 2
            }
        }),
        json!({
            "type": "compaction",
            "timestamp": "2025-01-01T00:00:03Z",
            "summary": "Compaction summary",
            "firstKeptEntryIndex": 1,  // index 1 = entries[1] (msg1)
            "tokensBefore": 100
        }),
    ];

    migrate_session_entries(&mut entries);

    let msg1_id = entry_id(&entries[1]).to_string();
    let comp = &entries[3];

    assert!(comp.get("firstKeptEntryIndex").is_none(), "old field should be removed");
    assert_eq!(
        comp.get("firstKeptEntryId").and_then(|v| v.as_str()),
        Some(msg1_id.as_str())
    );
}

// ---------------------------------------------------------------------------
// v2 → v3: hookMessage role → custom
// ---------------------------------------------------------------------------

#[test]
fn v2_to_v3_renames_hook_message_role() {
    let mut entries = vec![
        json!({
            "type": "session",
            "version": 2,
            "id": "sess-2",
            "timestamp": "2025-01-01T00:00:00Z",
            "cwd": "/tmp"
        }),
        json!({
            "type": "message",
            "id": "m1",
            "parentId": null,
            "timestamp": "2025-01-01T00:00:01Z",
            "message": { "role": "hookMessage", "content": "hook output", "timestamp": 1 }
        }),
        json!({
            "type": "message",
            "id": "m2",
            "parentId": "m1",
            "timestamp": "2025-01-01T00:00:02Z",
            "message": { "role": "user", "content": "normal user", "timestamp": 2 }
        }),
    ];

    migrate_session_entries(&mut entries);

    let msg1 = &entries[1];
    let role = msg1
        .get("message")
        .and_then(|m| m.get("role"))
        .and_then(|v| v.as_str());
    assert_eq!(role, Some("custom"), "hookMessage should be renamed to custom");

    let msg2 = &entries[2];
    let role2 = msg2
        .get("message")
        .and_then(|m| m.get("role"))
        .and_then(|v| v.as_str());
    assert_eq!(role2, Some("user"), "normal user role should be unchanged");
}

// ---------------------------------------------------------------------------
// Already at current version: no migration needed
// ---------------------------------------------------------------------------

#[test]
fn already_current_version_is_unchanged() {
    let id1 = "existing-id-1";
    let id2 = "existing-id-2";
    let mut entries = vec![
        json!({
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": "sess-3",
            "timestamp": "2025-01-01T00:00:00Z",
            "cwd": "/tmp"
        }),
        json!({
            "type": "message",
            "id": id1,
            "parentId": null,
            "timestamp": "2025-01-01T00:00:01Z",
            "message": { "role": "user", "content": "hi", "timestamp": 1 }
        }),
        json!({
            "type": "message",
            "id": id2,
            "parentId": id1,
            "timestamp": "2025-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "response" }],
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": "test",
                "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2 },
                "stopReason": "stop",
                "timestamp": 2
            }
        }),
    ];

    migrate_session_entries(&mut entries);

    assert_eq!(entry_id(&entries[1]), id1, "existing id should not change");
    assert_eq!(entry_id(&entries[2]), id2, "existing id should not change");
}
