mod session_helpers;
use session_helpers::*;

use pi_coding_rs::session::{load_entries_from_file, find_most_recent_session, SessionManager};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn write_valid_session(dir: &Path, filename: &str, id: &str) -> std::path::PathBuf {
    let path = dir.join(filename);
    fs::write(&path, format!(
        "{{\"type\":\"session\",\"id\":\"{}\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}}\n",
        id
    )).unwrap();
    path
}

// ---------------------------------------------------------------------------
// load_entries_from_file
// ---------------------------------------------------------------------------

#[test]
fn load_returns_empty_for_nonexistent_file() {
    let tmp = TempDir::new().unwrap();
    let entries = load_entries_from_file(tmp.path().join("nonexistent.jsonl"));
    assert!(entries.is_empty());
}

#[test]
fn load_returns_empty_for_empty_file() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("empty.jsonl");
    fs::write(&file, "").unwrap();
    assert!(load_entries_from_file(&file).is_empty());
}

#[test]
fn load_returns_empty_for_file_without_valid_session_header() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("no-header.jsonl");
    fs::write(&file, "{\"type\":\"message\",\"id\":\"1\"}\n").unwrap();
    assert!(load_entries_from_file(&file).is_empty());
}

#[test]
fn load_returns_empty_for_malformed_json() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("malformed.jsonl");
    fs::write(&file, "not json\n").unwrap();
    assert!(load_entries_from_file(&file).is_empty());
}

#[test]
fn load_returns_valid_session_entries() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("valid.jsonl");
    fs::write(&file,
        "{\"type\":\"session\",\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n\
         {\"type\":\"message\",\"id\":\"1\",\"parentId\":null,\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n"
    ).unwrap();
    let entries = load_entries_from_file(&file);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].get("type").and_then(|v| v.as_str()), Some("session"));
    assert_eq!(entries[1].get("type").and_then(|v| v.as_str()), Some("message"));
}

#[test]
fn load_skips_malformed_lines_but_keeps_valid_ones() {
    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("mixed.jsonl");
    fs::write(&file,
        "{\"type\":\"session\",\"id\":\"abc\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"cwd\":\"/tmp\"}\n\
         not valid json\n\
         {\"type\":\"message\",\"id\":\"1\",\"parentId\":null,\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n"
    ).unwrap();
    let entries = load_entries_from_file(&file);
    assert_eq!(entries.len(), 2);
}

// ---------------------------------------------------------------------------
// find_most_recent_session
// ---------------------------------------------------------------------------

#[test]
fn find_returns_none_for_empty_directory() {
    let tmp = TempDir::new().unwrap();
    assert!(find_most_recent_session(tmp.path()).is_none());
}

#[test]
fn find_returns_none_for_nonexistent_directory() {
    let tmp = TempDir::new().unwrap();
    assert!(find_most_recent_session(tmp.path().join("nonexistent")).is_none());
}

#[test]
fn find_ignores_non_jsonl_files() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("file.txt"), "hello").unwrap();
    fs::write(tmp.path().join("file.json"), "{}").unwrap();
    assert!(find_most_recent_session(tmp.path()).is_none());
}

#[test]
fn find_ignores_jsonl_without_valid_session_header() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("invalid.jsonl"), "{\"type\":\"message\"}\n").unwrap();
    assert!(find_most_recent_session(tmp.path()).is_none());
}

#[test]
fn find_returns_single_valid_session_file() {
    let tmp = TempDir::new().unwrap();
    let file = write_valid_session(tmp.path(), "session.jsonl", "abc");
    assert_eq!(find_most_recent_session(tmp.path()), Some(file));
}

#[test]
fn find_returns_most_recently_modified_session() {
    let tmp = TempDir::new().unwrap();
    let file1 = write_valid_session(tmp.path(), "older.jsonl", "old");
    // small delay to ensure different mtime
    std::thread::sleep(std::time::Duration::from_millis(20));
    let file2 = write_valid_session(tmp.path(), "newer.jsonl", "new");

    assert_eq!(find_most_recent_session(tmp.path()), Some(file2));
}

#[test]
fn find_skips_invalid_files_and_returns_valid_one() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("invalid.jsonl"), "{\"type\":\"not-session\"}\n").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let valid = write_valid_session(tmp.path(), "valid.jsonl", "abc");

    assert_eq!(find_most_recent_session(tmp.path()), Some(valid));
}

// ---------------------------------------------------------------------------
// SessionManager with corrupted files
// ---------------------------------------------------------------------------

#[test]
fn truncates_and_rewrites_empty_file_with_valid_header() {
    let tmp = TempDir::new().unwrap();
    let empty_file = tmp.path().join("empty.jsonl");
    fs::write(&empty_file, "").unwrap();

    let sm = SessionManager::open(&empty_file, Some(tmp.path()), None);

    assert!(!sm.get_session_id().is_empty());
    assert!(sm.get_header().is_some());

    // File should now have a valid header
    let content = fs::read_to_string(&empty_file).unwrap();
    let lines: Vec<&str> = content.trim().split('\n').filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 1);
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header.get("type").and_then(|v| v.as_str()), Some("session"));
    assert_eq!(header.get("id").and_then(|v| v.as_str()), Some(sm.get_session_id()));
}

#[test]
fn truncates_and_rewrites_file_without_valid_header() {
    let tmp = TempDir::new().unwrap();
    let no_header_file = tmp.path().join("no-header.jsonl");
    fs::write(&no_header_file,
        "{\"type\":\"message\",\"id\":\"abc\",\"parentId\":\"orphaned\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"test\"}}\n"
    ).unwrap();

    let sm = SessionManager::open(&no_header_file, Some(tmp.path()), None);

    assert!(!sm.get_session_id().is_empty());
    assert!(sm.get_header().is_some());

    let content = fs::read_to_string(&no_header_file).unwrap();
    let lines: Vec<&str> = content.trim().split('\n').filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 1);
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header.get("type").and_then(|v| v.as_str()), Some("session"));
    assert_eq!(header.get("id").and_then(|v| v.as_str()), Some(sm.get_session_id()));
}

#[test]
fn preserves_explicit_session_file_path_when_recovering_from_corrupted_file() {
    let tmp = TempDir::new().unwrap();
    let explicit_path = tmp.path().join("my-session.jsonl");
    fs::write(&explicit_path, "").unwrap();

    let sm = SessionManager::open(&explicit_path, Some(tmp.path()), None);

    assert_eq!(sm.get_session_file(), Some(explicit_path.as_path()));
}

#[test]
fn subsequent_loads_of_recovered_file_work_correctly() {
    let tmp = TempDir::new().unwrap();
    let corrupted_file = tmp.path().join("corrupted.jsonl");
    fs::write(&corrupted_file, "garbage content\n").unwrap();

    let sm1 = SessionManager::open(&corrupted_file, Some(tmp.path()), None);
    let session_id = sm1.get_session_id().to_string();

    let sm2 = SessionManager::open(&corrupted_file, Some(tmp.path()), None);
    assert_eq!(sm2.get_session_id(), session_id);
    assert!(sm2.get_header().is_some());
}

// ---------------------------------------------------------------------------
// Deferred write: file only written after first assistant message
// ---------------------------------------------------------------------------

#[test]
fn does_not_write_file_until_assistant_message() {
    let tmp = TempDir::new().unwrap();
    let mut sm = SessionManager::create(tmp.path().to_str().unwrap(), Some(tmp.path()));
    let file = sm.get_session_file().unwrap().to_path_buf();

    sm.append_message(user_msg("first question"));

    // File should not exist yet (no assistant message)
    assert!(!file.exists(), "file should not exist before assistant message");

    sm.append_message(assistant_msg("first answer"));

    // Now it should exist
    assert!(file.exists(), "file should exist after assistant message");
}

#[test]
fn all_buffered_entries_are_written_on_first_flush() {
    let tmp = TempDir::new().unwrap();
    let mut sm = SessionManager::create(tmp.path().to_str().unwrap(), Some(tmp.path()));
    let file = sm.get_session_file().unwrap().to_path_buf();

    let _id1 = sm.append_message(user_msg("question 1"));
    let _id2 = sm.append_message(assistant_msg("answer 1"));

    let content = fs::read_to_string(&file).unwrap();
    let records: Vec<serde_json::Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    // header + msg1 + msg2
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].get("type").and_then(|v| v.as_str()), Some("session"));
}
