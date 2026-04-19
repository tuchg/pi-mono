mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::SessionManager;
use serde_json::json;

// ---------------------------------------------------------------------------
// append_label_change / get_label
// ---------------------------------------------------------------------------

#[test]
fn labels_can_be_set_on_entries() {
    let mut session = SessionManager::in_memory(None);

    let msg1_id = session.append_message(user_msg("hello"));
    let msg2_id = session.append_message(assistant_msg("hi there"));

    session.append_label_change(&msg1_id, Some("start")).unwrap();
    session.append_label_change(&msg2_id, Some("response")).unwrap();

    assert_eq!(session.get_label(&msg1_id), Some("start"));
    assert_eq!(session.get_label(&msg2_id), Some("response"));
}

#[test]
fn labels_can_be_cleared() {
    let mut session = SessionManager::in_memory(None);

    let id = session.append_message(user_msg("hello"));
    session.append_label_change(&id, Some("mylabel")).unwrap();
    assert_eq!(session.get_label(&id), Some("mylabel"));

    session.append_label_change(&id, None).unwrap();
    assert!(session.get_label(&id).is_none());
}

#[test]
fn last_label_wins_for_same_target() {
    let mut session = SessionManager::in_memory(None);

    let id = session.append_message(user_msg("hello"));
    session.append_label_change(&id, Some("first")).unwrap();
    session.append_label_change(&id, Some("second")).unwrap();

    assert_eq!(session.get_label(&id), Some("second"));
}

#[test]
fn label_entry_errors_for_nonexistent_target() {
    let mut session = SessionManager::in_memory(None);
    session.append_message(user_msg("hello"));

    assert!(session.append_label_change("nonexistent", Some("label")).is_err());
}

#[test]
fn labels_included_in_tree_nodes() {
    let mut session = SessionManager::in_memory(None);

    let msg1_id = session.append_message(user_msg("hello"));
    let msg2_id = session.append_message(assistant_msg("hi"));

    session.append_label_change(&msg1_id, Some("start")).unwrap();
    session.append_label_change(&msg2_id, Some("response")).unwrap();

    let tree = session.get_tree();
    assert_eq!(tree.len(), 1);

    let msg1_node = &tree[0];
    assert_eq!(msg1_node.entry.get("id").and_then(|v| v.as_str()), Some(msg1_id.as_str()));
    assert_eq!(msg1_node.label.as_deref(), Some("start"));

    let msg2_node = &msg1_node.children[0];
    assert_eq!(msg2_node.entry.get("id").and_then(|v| v.as_str()), Some(msg2_id.as_str()));
    assert_eq!(msg2_node.label.as_deref(), Some("response"));
}

#[test]
fn label_preserved_in_create_branched_session() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_message(user_msg("3"));
    let id4 = session.append_message(assistant_msg("4"));

    session.append_label_change(&id2, Some("important")).unwrap();

    // Fork from id3 (path: id1 -> id2 -> id3)
    session.create_branched_session(&id3).unwrap();

    // Label should still be on id2 after branching
    assert_eq!(session.get_label(&id2), Some("important"));
}

#[test]
fn label_not_in_branched_session_when_target_not_in_path() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_message(user_msg("3"));
    let id4 = session.append_message(assistant_msg("4"));

    // Label on id4 (which is NOT in the path to id2)
    session.append_label_change(&id4, Some("not in branch")).unwrap();

    // Fork from id2 (path: id1 -> id2)
    session.create_branched_session(&id2).unwrap();

    // id4 is not in the branched session
    assert!(session.get_entry(&id4).is_none(), "id4 should not be in branched session");
    assert!(session.get_label(&id4).is_none(), "label on id4 should not be in branched session");
}
