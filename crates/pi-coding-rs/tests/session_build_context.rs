mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::{build_session_context, SessionContext};
use serde_json::Value;

fn msg_role(m: &Value) -> &str {
    m.get("role").and_then(|v| v.as_str()).unwrap_or("")
}

fn msg_content_str(m: &Value) -> &str {
    m.get("content").and_then(|v| v.as_str()).unwrap_or("")
}

fn msg_content_text(m: &Value) -> &str {
    m.get("content")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|b| b.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("")
}

fn msg_summary(m: &Value) -> &str {
    m.get("summary").and_then(|v| v.as_str()).unwrap_or("")
}

// ---------------------------------------------------------------------------
// Trivial cases
// ---------------------------------------------------------------------------

#[test]
fn empty_entries_returns_empty_context() {
    let ctx = build_session_context(&[], None);
    assert!(ctx.messages.is_empty());
    assert_eq!(ctx.thinking_level, "off");
    assert!(ctx.model.is_none());
}

#[test]
fn single_user_message() {
    let entries = vec![msg_entry("1", None, "user", "hello")];
    let ctx = build_session_context(&entries, None);
    assert_eq!(ctx.messages.len(), 1);
    assert_eq!(msg_role(&ctx.messages[0]), "user");
}

#[test]
fn simple_conversation() {
    let entries = vec![
        msg_entry("1", None, "user", "hello"),
        msg_entry("2", Some("1"), "assistant", "hi there"),
        msg_entry("3", Some("2"), "user", "how are you"),
        msg_entry("4", Some("3"), "assistant", "great"),
    ];
    let ctx = build_session_context(&entries, None);
    assert_eq!(ctx.messages.len(), 4);
    let roles: Vec<&str> = ctx.messages.iter().map(|m| msg_role(m)).collect();
    assert_eq!(roles, &["user", "assistant", "user", "assistant"]);
}

#[test]
fn tracks_thinking_level_changes() {
    let entries = vec![
        msg_entry("1", None, "user", "hello"),
        thinking_level_entry("2", Some("1"), "high"),
        msg_entry("3", Some("2"), "assistant", "thinking hard"),
    ];
    let ctx = build_session_context(&entries, None);
    assert_eq!(ctx.thinking_level, "high");
    assert_eq!(ctx.messages.len(), 2);
}

#[test]
fn tracks_model_from_assistant_message() {
    let entries = vec![
        msg_entry("1", None, "user", "hello"),
        msg_entry("2", Some("1"), "assistant", "hi"),
    ];
    let ctx = build_session_context(&entries, None);
    let model = ctx.model.expect("should have model");
    assert_eq!(model.provider, "anthropic");
    assert_eq!(model.model_id, "claude-test");
}

#[test]
fn tracks_model_from_model_change_entry() {
    let entries = vec![
        msg_entry("1", None, "user", "hello"),
        model_change_entry("2", Some("1"), "openai", "gpt-4"),
        msg_entry("3", Some("2"), "assistant", "hi"),
    ];
    let ctx = build_session_context(&entries, None);
    // Assistant message overwrites model change
    let model = ctx.model.expect("should have model");
    assert_eq!(model.provider, "anthropic");
    assert_eq!(model.model_id, "claude-test");
}

// ---------------------------------------------------------------------------
// With compaction
// ---------------------------------------------------------------------------

#[test]
fn includes_summary_before_kept_messages() {
    let entries = vec![
        msg_entry("1", None, "user", "first"),
        msg_entry("2", Some("1"), "assistant", "response1"),
        msg_entry("3", Some("2"), "user", "second"),
        msg_entry("4", Some("3"), "assistant", "response2"),
        compaction_entry("5", Some("4"), "Summary of first two turns", "3"),
        msg_entry("6", Some("5"), "user", "third"),
        msg_entry("7", Some("6"), "assistant", "response3"),
    ];
    let ctx = build_session_context(&entries, None);

    // summary + kept(3,4) + after(6,7) = 5 messages
    assert_eq!(ctx.messages.len(), 5);
    assert!(msg_summary(&ctx.messages[0]).contains("Summary of first two turns"));
    assert_eq!(msg_content_str(&ctx.messages[1]), "second");
    assert_eq!(msg_content_text(&ctx.messages[2]), "response2");
    assert_eq!(msg_content_str(&ctx.messages[3]), "third");
    assert_eq!(msg_content_text(&ctx.messages[4]), "response3");
}

#[test]
fn handles_compaction_keeping_from_first_message() {
    let entries = vec![
        msg_entry("1", None, "user", "first"),
        msg_entry("2", Some("1"), "assistant", "response"),
        compaction_entry("3", Some("2"), "Empty summary", "1"),
        msg_entry("4", Some("3"), "user", "second"),
    ];
    let ctx = build_session_context(&entries, None);

    // Summary + all messages (1,2,4)
    assert_eq!(ctx.messages.len(), 4);
    assert!(msg_summary(&ctx.messages[0]).contains("Empty summary"));
}

#[test]
fn multiple_compactions_uses_latest() {
    let entries = vec![
        msg_entry("1", None, "user", "a"),
        msg_entry("2", Some("1"), "assistant", "b"),
        compaction_entry("3", Some("2"), "First summary", "1"),
        msg_entry("4", Some("3"), "user", "c"),
        msg_entry("5", Some("4"), "assistant", "d"),
        compaction_entry("6", Some("5"), "Second summary", "4"),
        msg_entry("7", Some("6"), "user", "e"),
    ];
    let ctx = build_session_context(&entries, None);

    // Should use second summary, keep from 4
    assert_eq!(ctx.messages.len(), 4);
    assert!(msg_summary(&ctx.messages[0]).contains("Second summary"));
}

// ---------------------------------------------------------------------------
// With branches
// ---------------------------------------------------------------------------

#[test]
fn follows_path_to_specified_leaf() {
    // 1 -> 2 -> 3 (branch A)
    //       \-> 4 (branch B)
    let entries = vec![
        msg_entry("1", None, "user", "start"),
        msg_entry("2", Some("1"), "assistant", "response"),
        msg_entry("3", Some("2"), "user", "branch A"),
        msg_entry("4", Some("2"), "user", "branch B"),
    ];

    let ctx_a = build_session_context(&entries, Some("3"));
    assert_eq!(ctx_a.messages.len(), 3);
    assert_eq!(msg_content_str(&ctx_a.messages[2]), "branch A");

    let ctx_b = build_session_context(&entries, Some("4"));
    assert_eq!(ctx_b.messages.len(), 3);
    assert_eq!(msg_content_str(&ctx_b.messages[2]), "branch B");
}

#[test]
fn includes_branch_summary_in_path() {
    let entries = vec![
        msg_entry("1", None, "user", "start"),
        msg_entry("2", Some("1"), "assistant", "response"),
        msg_entry("3", Some("2"), "user", "abandoned path"),
        branch_summary_entry("4", Some("2"), "Summary of abandoned work", "3"),
        msg_entry("5", Some("4"), "user", "new direction"),
    ];
    let ctx = build_session_context(&entries, Some("5"));

    assert_eq!(ctx.messages.len(), 4);
    assert!(msg_summary(&ctx.messages[2]).contains("Summary of abandoned work"));
    assert_eq!(msg_content_str(&ctx.messages[3]), "new direction");
}

#[test]
fn complex_tree_with_multiple_branches_and_compaction() {
    let entries = vec![
        msg_entry("1", None, "user", "start"),
        msg_entry("2", Some("1"), "assistant", "r1"),
        msg_entry("3", Some("2"), "user", "q2"),
        msg_entry("4", Some("3"), "assistant", "r2"),
        compaction_entry("5", Some("4"), "Compacted history", "3"),
        msg_entry("6", Some("5"), "user", "q3"),
        msg_entry("7", Some("6"), "assistant", "r3"),
        msg_entry("8", Some("3"), "user", "wrong path"),
        msg_entry("9", Some("8"), "assistant", "wrong response"),
        branch_summary_entry("10", Some("3"), "Tried wrong approach", "9"),
        msg_entry("11", Some("10"), "user", "better approach"),
    ];

    // Main path to 7: summary + kept(3,4) + after(6,7)
    let ctx_main = build_session_context(&entries, Some("7"));
    assert_eq!(ctx_main.messages.len(), 5);
    assert!(msg_summary(&ctx_main.messages[0]).contains("Compacted history"));
    assert_eq!(msg_content_str(&ctx_main.messages[1]), "q2");
    assert_eq!(msg_content_text(&ctx_main.messages[2]), "r2");
    assert_eq!(msg_content_str(&ctx_main.messages[3]), "q3");
    assert_eq!(msg_content_text(&ctx_main.messages[4]), "r3");

    // Branch path to 11: 1,2,3 + branch_summary + 11
    let ctx_branch = build_session_context(&entries, Some("11"));
    assert_eq!(ctx_branch.messages.len(), 5);
    assert_eq!(msg_content_str(&ctx_branch.messages[0]), "start");
    assert_eq!(msg_content_text(&ctx_branch.messages[1]), "r1");
    assert_eq!(msg_content_str(&ctx_branch.messages[2]), "q2");
    assert!(msg_summary(&ctx_branch.messages[3]).contains("Tried wrong approach"));
    assert_eq!(msg_content_str(&ctx_branch.messages[4]), "better approach");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn uses_last_entry_when_leaf_id_not_found() {
    let entries = vec![
        msg_entry("1", None, "user", "hello"),
        msg_entry("2", Some("1"), "assistant", "hi"),
    ];
    let ctx = build_session_context(&entries, Some("nonexistent"));
    assert_eq!(ctx.messages.len(), 2);
}

#[test]
fn handles_orphaned_entries_gracefully() {
    let entries = vec![
        msg_entry("1", None, "user", "hello"),
        msg_entry("2", Some("missing"), "assistant", "orphan"),
    ];
    let ctx = build_session_context(&entries, Some("2"));
    // Only the orphan since parent chain is broken
    assert_eq!(ctx.messages.len(), 1);
}
