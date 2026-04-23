mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::SessionManager;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Append operations
// ---------------------------------------------------------------------------

#[test]
fn append_message_creates_entry_with_correct_parent_chain() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("first"));
    let id2 = session.append_message(assistant_msg("second"));
    let id3 = session.append_message(user_msg("third"));

    let entries = session.get_entries();
    assert_eq!(entries.len(), 3);

    let e0 = &entries[0];
    assert_eq!(e0.get("id").and_then(|v| v.as_str()), Some(id1.as_str()));
    assert_eq!(e0.get("parentId"), Some(&Value::Null));
    assert_eq!(e0.get("type").and_then(|v| v.as_str()), Some("message"));

    let e1 = &entries[1];
    assert_eq!(e1.get("id").and_then(|v| v.as_str()), Some(id2.as_str()));
    assert_eq!(e1.get("parentId").and_then(|v| v.as_str()), Some(id1.as_str()));

    let e2 = &entries[2];
    assert_eq!(e2.get("id").and_then(|v| v.as_str()), Some(id3.as_str()));
    assert_eq!(e2.get("parentId").and_then(|v| v.as_str()), Some(id2.as_str()));
}

#[test]
fn append_thinking_level_change_integrates_into_tree() {
    let mut session = SessionManager::in_memory(None);

    let msg_id = session.append_message(user_msg("hello"));
    let thinking_id = session.append_thinking_level_change("high");
    let _msg2_id = session.append_message(assistant_msg("response"));

    let entries = session.get_entries();
    assert_eq!(entries.len(), 3);

    let thinking_entry = entries
        .iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("thinking_level_change"))
        .expect("should have thinking entry");

    assert_eq!(thinking_entry.get("id").and_then(|v| v.as_str()), Some(thinking_id.as_str()));
    assert_eq!(
        thinking_entry.get("parentId").and_then(|v| v.as_str()),
        Some(msg_id.as_str())
    );

    assert_eq!(entries[2].get("parentId").and_then(|v| v.as_str()), Some(thinking_id.as_str()));
}

#[test]
fn append_model_change_integrates_into_tree() {
    let mut session = SessionManager::in_memory(None);

    let msg_id = session.append_message(user_msg("hello"));
    let model_id = session.append_model_change("openai", "gpt-4");
    let _msg2_id = session.append_message(assistant_msg("response"));

    let entries = session.get_entries();
    let model_entry = entries
        .iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("model_change"))
        .expect("should have model_change entry");

    assert_eq!(model_entry.get("id").and_then(|v| v.as_str()), Some(model_id.as_str()));
    assert_eq!(model_entry.get("parentId").and_then(|v| v.as_str()), Some(msg_id.as_str()));
    assert_eq!(model_entry.get("provider").and_then(|v| v.as_str()), Some("openai"));
    assert_eq!(model_entry.get("modelId").and_then(|v| v.as_str()), Some("gpt-4"));

    assert_eq!(entries[2].get("parentId").and_then(|v| v.as_str()), Some(model_id.as_str()));
}

#[test]
fn append_compaction_integrates_into_tree() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let compaction_id = session.append_compaction("summary", &id1, 1000, None, None);
    let _id3 = session.append_message(user_msg("3"));

    let entries = session.get_entries();
    let comp_entry = entries
        .iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("compaction"))
        .expect("should have compaction entry");

    assert_eq!(comp_entry.get("id").and_then(|v| v.as_str()), Some(compaction_id.as_str()));
    assert_eq!(comp_entry.get("parentId").and_then(|v| v.as_str()), Some(id2.as_str()));
    assert_eq!(comp_entry.get("summary").and_then(|v| v.as_str()), Some("summary"));
    assert_eq!(comp_entry.get("firstKeptEntryId").and_then(|v| v.as_str()), Some(id1.as_str()));
    assert_eq!(comp_entry.get("tokensBefore").and_then(|v| v.as_u64()), Some(1000));

    assert_eq!(entries[3].get("parentId").and_then(|v| v.as_str()), Some(compaction_id.as_str()));
}

#[test]
fn append_custom_entry_integrates_into_tree() {
    let mut session = SessionManager::in_memory(None);

    let msg_id = session.append_message(user_msg("hello"));
    let custom_id = session.append_custom_entry("my_data", Some(json!({ "key": "value" })));
    let _msg2_id = session.append_message(assistant_msg("response"));

    let entries = session.get_entries();
    let custom_entry = entries
        .iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("custom"))
        .expect("should have custom entry");

    assert_eq!(custom_entry.get("id").and_then(|v| v.as_str()), Some(custom_id.as_str()));
    assert_eq!(custom_entry.get("parentId").and_then(|v| v.as_str()), Some(msg_id.as_str()));
    assert_eq!(custom_entry.get("customType").and_then(|v| v.as_str()), Some("my_data"));
    assert_eq!(custom_entry.get("data"), Some(&json!({ "key": "value" })));

    assert_eq!(entries[2].get("parentId").and_then(|v| v.as_str()), Some(custom_id.as_str()));
}

#[test]
fn leaf_pointer_advances_after_each_append() {
    let mut session = SessionManager::in_memory(None);

    assert!(session.get_leaf_id().is_none());

    let id1 = session.append_message(user_msg("1"));
    assert_eq!(session.get_leaf_id(), Some(id1.as_str()));

    let id2 = session.append_message(assistant_msg("2"));
    assert_eq!(session.get_leaf_id(), Some(id2.as_str()));

    let id3 = session.append_thinking_level_change("high");
    assert_eq!(session.get_leaf_id(), Some(id3.as_str()));
}

// ---------------------------------------------------------------------------
// get_branch
// ---------------------------------------------------------------------------

#[test]
fn get_branch_returns_empty_for_empty_session() {
    let session = SessionManager::in_memory(None);
    assert!(session.get_branch(None).is_empty());
}

#[test]
fn get_branch_returns_single_entry() {
    let mut session = SessionManager::in_memory(None);
    let id = session.append_message(user_msg("hello"));

    let path = session.get_branch(None);
    assert_eq!(path.len(), 1);
    assert_eq!(path[0].get("id").and_then(|v| v.as_str()), Some(id.as_str()));
}

#[test]
fn get_branch_returns_full_path_from_root_to_leaf() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_thinking_level_change("high");
    let id4 = session.append_message(user_msg("3"));

    let path = session.get_branch(None);
    assert_eq!(path.len(), 4);
    let ids: Vec<&str> = path.iter().map(|e| e.get("id").and_then(|v| v.as_str()).unwrap_or("")).collect();
    assert_eq!(ids, &[id1.as_str(), id2.as_str(), id3.as_str(), id4.as_str()]);
}

#[test]
fn get_branch_from_specified_entry() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let _id3 = session.append_message(user_msg("3"));
    let _id4 = session.append_message(assistant_msg("4"));

    let path = session.get_branch(Some(&id2));
    assert_eq!(path.len(), 2);
    let ids: Vec<&str> = path.iter().map(|e| e.get("id").and_then(|v| v.as_str()).unwrap_or("")).collect();
    assert_eq!(ids, &[id1.as_str(), id2.as_str()]);
}

// ---------------------------------------------------------------------------
// get_tree
// ---------------------------------------------------------------------------

#[test]
fn get_tree_returns_empty_for_empty_session() {
    let session = SessionManager::in_memory(None);
    assert!(session.get_tree().is_empty());
}

#[test]
fn get_tree_returns_single_root_for_linear_session() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_message(user_msg("3"));

    let tree = session.get_tree();
    assert_eq!(tree.len(), 1);

    let root = &tree[0];
    assert_eq!(root.entry.get("id").and_then(|v| v.as_str()), Some(id1.as_str()));
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].entry.get("id").and_then(|v| v.as_str()), Some(id2.as_str()));
    assert_eq!(root.children[0].children.len(), 1);
    assert_eq!(
        root.children[0].children[0].entry.get("id").and_then(|v| v.as_str()),
        Some(id3.as_str())
    );
    assert!(root.children[0].children[0].children.is_empty());
}

#[test]
fn get_tree_returns_branches_after_branch() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_message(user_msg("3"));

    session.branch(&id2).unwrap();
    let id4 = session.append_message(user_msg("4-branch"));

    let tree = session.get_tree();
    assert_eq!(tree.len(), 1);

    let node2 = &tree[0].children[0];
    assert_eq!(node2.entry.get("id").and_then(|v| v.as_str()), Some(id2.as_str()));
    assert_eq!(node2.children.len(), 2);

    let child_ids: std::collections::HashSet<String> = node2
        .children
        .iter()
        .map(|c| c.entry.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string())
        .collect();
    assert!(child_ids.contains(&id3));
    assert!(child_ids.contains(&id4));
}

#[test]
fn get_tree_three_branches_from_same_node() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id_a = session.append_message(user_msg("branch-A"));

    session.branch(&id2).unwrap();
    let id_b = session.append_message(user_msg("branch-B"));

    session.branch(&id2).unwrap();
    let id_c = session.append_message(user_msg("branch-C"));

    let tree = session.get_tree();
    let node2 = &tree[0].children[0];
    assert_eq!(node2.entry.get("id").and_then(|v| v.as_str()), Some(id2.as_str()));
    assert_eq!(node2.children.len(), 3);

    let child_ids: std::collections::HashSet<String> = node2
        .children
        .iter()
        .map(|c| c.entry.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string())
        .collect();
    assert!(child_ids.contains(&id_a));
    assert!(child_ids.contains(&id_b));
    assert!(child_ids.contains(&id_c));
}

// ---------------------------------------------------------------------------
// branch
// ---------------------------------------------------------------------------

#[test]
fn branch_moves_leaf_pointer() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let _id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_message(user_msg("3"));

    assert_eq!(session.get_leaf_id(), Some(id3.as_str()));

    session.branch(&id1).unwrap();
    assert_eq!(session.get_leaf_id(), Some(id1.as_str()));
}

#[test]
fn branch_errors_for_nonexistent_entry() {
    let mut session = SessionManager::in_memory(None);
    session.append_message(user_msg("hello"));

    assert!(session.branch("nonexistent").is_err());
}

#[test]
fn new_appends_become_children_of_branch_point() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let _id2 = session.append_message(assistant_msg("2"));

    session.branch(&id1).unwrap();
    let id3 = session.append_message(user_msg("branched"));

    let entries = session.get_entries();
    let branched_entry = entries.iter().find(|e| e.get("id").and_then(|v| v.as_str()) == Some(id3.as_str())).unwrap();
    assert_eq!(branched_entry.get("parentId").and_then(|v| v.as_str()), Some(id1.as_str()));
}

// ---------------------------------------------------------------------------
// branch_with_summary
// ---------------------------------------------------------------------------

#[test]
fn branch_with_summary_inserts_entry_and_advances_leaf() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let _id2 = session.append_message(assistant_msg("2"));
    let _id3 = session.append_message(user_msg("3"));

    let summary_id = session.branch_with_summary(Some(&id1), "Summary of abandoned work", None, None).unwrap();

    assert_eq!(session.get_leaf_id(), Some(summary_id.as_str()));

    let entries = session.get_entries();
    let summary_entry = entries
        .iter()
        .find(|e| e.get("type").and_then(|t| t.as_str()) == Some("branch_summary"))
        .expect("should have branch_summary entry");
    assert_eq!(summary_entry.get("parentId").and_then(|v| v.as_str()), Some(id1.as_str()));
    assert_eq!(
        summary_entry.get("summary").and_then(|v| v.as_str()),
        Some("Summary of abandoned work")
    );
}

#[test]
fn branch_with_summary_errors_for_nonexistent_entry() {
    let mut session = SessionManager::in_memory(None);
    session.append_message(user_msg("hello"));

    assert!(session.branch_with_summary(Some("nonexistent"), "summary", None, None).is_err());
}

// ---------------------------------------------------------------------------
// get_leaf_entry + get_entry
// ---------------------------------------------------------------------------

#[test]
fn get_leaf_entry_returns_none_for_empty_session() {
    let session = SessionManager::in_memory(None);
    assert!(session.get_leaf_entry().is_none());
}

#[test]
fn get_leaf_entry_returns_current_leaf() {
    let mut session = SessionManager::in_memory(None);
    session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));

    let leaf = session.get_leaf_entry().expect("should have leaf");
    assert_eq!(leaf.get("id").and_then(|v| v.as_str()), Some(id2.as_str()));
}

#[test]
fn get_entry_returns_none_for_nonexistent_id() {
    let session = SessionManager::in_memory(None);
    assert!(session.get_entry("nonexistent").is_none());
}

#[test]
fn get_entry_returns_entry_by_id() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("first"));
    let id2 = session.append_message(assistant_msg("second"));

    let entry1 = session.get_entry(&id1).expect("should find entry1");
    let msg1 = entry1.get("message").expect("should have message");
    assert_eq!(msg1.get("role").and_then(|v| v.as_str()), Some("user"));
    assert_eq!(msg1.get("content").and_then(|v| v.as_str()), Some("first"));

    let entry2 = session.get_entry(&id2).expect("should find entry2");
    let msg2 = entry2.get("message").expect("should have message");
    assert_eq!(msg2.get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(
        msg2.get("content")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str()),
        Some("second")
    );
}

// ---------------------------------------------------------------------------
// build_session_context on SessionManager
// ---------------------------------------------------------------------------

#[test]
fn build_session_context_returns_messages_from_current_branch() {
    let mut session = SessionManager::in_memory(None);

    session.append_message(user_msg("msg1"));
    let id2 = session.append_message(assistant_msg("msg2"));
    session.append_message(user_msg("msg3"));

    session.branch(&id2).unwrap();
    session.append_message(assistant_msg("msg4-branch"));

    let ctx = session.build_session_context();
    assert_eq!(ctx.messages.len(), 3); // msg1, msg2, msg4-branch

    assert_eq!(
        ctx.messages[0].get("content").and_then(|v| v.as_str()),
        Some("msg1")
    );
    assert_eq!(
        ctx.messages[1]
            .get("content")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str()),
        Some("msg2")
    );
    assert_eq!(
        ctx.messages[2]
            .get("content")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str()),
        Some("msg4-branch")
    );
}

// ---------------------------------------------------------------------------
// create_branched_session
// ---------------------------------------------------------------------------

#[test]
fn create_branched_session_throws_for_nonexistent_entry() {
    let mut session = SessionManager::in_memory(None);
    session.append_message(user_msg("hello"));

    assert!(session.create_branched_session("nonexistent").is_err());
}

#[test]
fn create_branched_session_creates_new_session_with_path_to_leaf_in_memory() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    let id3 = session.append_message(user_msg("3"));
    session.append_message(assistant_msg("4"));

    session.branch(&id3).unwrap();
    let _id5 = session.append_message(user_msg("5"));

    // Create branched session from id2 (should only have 1 -> 2)
    let result = session.create_branched_session(&id2).unwrap();
    assert!(result.is_none()); // in-memory returns None

    let entries = session.get_entries();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].get("id").and_then(|v| v.as_str()), Some(id1.as_str()));
    assert_eq!(entries[1].get("id").and_then(|v| v.as_str()), Some(id2.as_str()));
}

#[test]
fn create_branched_session_extracts_correct_path_from_branched_tree() {
    let mut session = SessionManager::in_memory(None);

    let id1 = session.append_message(user_msg("1"));
    let id2 = session.append_message(assistant_msg("2"));
    session.append_message(user_msg("3"));

    session.branch(&id2).unwrap();
    let id4 = session.append_message(user_msg("4"));
    let id5 = session.append_message(assistant_msg("5"));

    session.create_branched_session(&id5).unwrap();

    let entries = session.get_entries();
    assert_eq!(entries.len(), 4);
    let ids: Vec<&str> = entries.iter().map(|e| e.get("id").and_then(|v| v.as_str()).unwrap_or("")).collect();
    assert_eq!(ids, &[id1.as_str(), id2.as_str(), id4.as_str(), id5.as_str()]);
}
