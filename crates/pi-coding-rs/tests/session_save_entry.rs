mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::SessionManager;
use serde_json::json;

// ---------------------------------------------------------------------------
// saveCustomEntry / appendCustomEntry
// ---------------------------------------------------------------------------

#[test]
fn saves_custom_entries_and_includes_them_in_tree_traversal() {
    let mut session = SessionManager::in_memory(None);

    let msg_id = session.append_message(user_msg("hello"));
    let custom_id = session.append_custom_entry("my_data", Some(json!({ "foo": "bar" })));
    let msg2_id = session.append_message(assistant_msg("hi"));

    let entries = session.get_entries();
    assert_eq!(entries.len(), 3);

    let custom_entry = entries
        .iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("custom"))
        .expect("should have custom entry");

    assert_eq!(custom_entry.get("customType").and_then(|v| v.as_str()), Some("my_data"));
    assert_eq!(custom_entry.get("data"), Some(&json!({ "foo": "bar" })));
    assert_eq!(custom_entry.get("id").and_then(|v| v.as_str()), Some(custom_id.as_str()));
    assert_eq!(custom_entry.get("parentId").and_then(|v| v.as_str()), Some(msg_id.as_str()));

    let path = session.get_branch(None);
    assert_eq!(path.len(), 3);
    assert_eq!(path[0].get("id").and_then(|v| v.as_str()), Some(msg_id.as_str()));
    assert_eq!(path[1].get("id").and_then(|v| v.as_str()), Some(custom_id.as_str()));
    assert_eq!(path[2].get("id").and_then(|v| v.as_str()), Some(msg2_id.as_str()));

    // buildSessionContext should skip custom entries (not messages)
    let ctx = session.build_session_context();
    assert_eq!(ctx.messages.len(), 2);
}
