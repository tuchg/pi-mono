//! Unit tests for pi-coding-rs tools.
//!
//! Each tool is exercised against the filesystem (via temp dirs) or subprocesses.

use pi_agent_rs::types::{AgentTool, AgentToolResult};
use pi_coding_rs::tools::{BashTool, FindTool, GrepTool, LsTool, ReadTool, ThinkTool, WriteTool};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Extract the text from the first text content block.
fn text_of(result: &AgentToolResult) -> &str {
    result
        .content
        .iter()
        .find_map(|c| match c {
            pi_ai_rs::Content::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .unwrap_or("")
}

// ---------------------------------------------------------------------------
// ThinkTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn think_returns_thought() {
    let tool = ThinkTool;
    assert_eq!(tool.name(), "think");

    let result = tool
        .execute(
            "t1",
            json!({"thought": "step 1: check input"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(text_of(&result), "step 1: check input");
}

#[tokio::test]
async fn think_missing_thought() {
    let tool = ThinkTool;
    let result = tool
        .execute("t2", json!({}), CancellationToken::new(), None)
        .await
        .unwrap();
    assert_eq!(text_of(&result), "(no thought provided)");
}

// ---------------------------------------------------------------------------
// WriteTool + ReadTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn write_then_read_file() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let write = WriteTool { cwd: cwd.clone() };
    let read = ReadTool { cwd: cwd.clone() };

    // Write a file
    let wr = write
        .execute(
            "w1",
            json!({"path": "hello.txt", "content": "line1\nline2\nline3"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
    assert!(text_of(&wr).contains("Successfully wrote"));
    assert!(text_of(&wr).contains("hello.txt"));

    // Read the file
    let rd = read
        .execute(
            "r1",
            json!({"path": "hello.txt"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(text_of(&rd), "line1\nline2\nline3");
}

#[tokio::test]
async fn read_with_offset_and_limit() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let write = WriteTool { cwd: cwd.clone() };
    let read = ReadTool { cwd: cwd.clone() };

    write
        .execute(
            "w1",
            json!({"path": "lines.txt", "content": "a\nb\nc\nd\ne"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    // Read from offset 2 (1-indexed), limit 2 lines
    let rd = read
        .execute(
            "r1",
            json!({"path": "lines.txt", "offset": 2, "limit": 2}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    let text = text_of(&rd);
    assert!(text.starts_with("b\nc"));
    assert!(text.contains("more lines"));
}

#[tokio::test]
async fn read_missing_file_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let read = ReadTool { cwd };

    let result = read
        .execute(
            "r1",
            json!({"path": "nonexistent.txt"}),
            CancellationToken::new(),
            None,
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn write_creates_parent_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let write = WriteTool { cwd: cwd.clone() };

    write
        .execute(
            "w1",
            json!({"path": "a/b/c/deep.txt", "content": "nested"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    let content = std::fs::read_to_string(tmp.path().join("a/b/c/deep.txt")).unwrap();
    assert_eq!(content, "nested");
}

// ---------------------------------------------------------------------------
// LsTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ls_lists_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    // Create some files and a subdirectory
    std::fs::write(tmp.path().join("alpha.txt"), "a").unwrap();
    std::fs::write(tmp.path().join("beta.txt"), "b").unwrap();
    std::fs::create_dir(tmp.path().join("subdir")).unwrap();

    let ls = LsTool { cwd };

    let result = ls
        .execute("l1", json!({}), CancellationToken::new(), None)
        .await
        .unwrap();

    let text = text_of(&result);
    assert!(text.contains("alpha.txt"));
    assert!(text.contains("beta.txt"));
    assert!(text.contains("subdir/"));
}

#[tokio::test]
async fn ls_empty_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let ls = LsTool { cwd };

    let result = ls
        .execute("l1", json!({}), CancellationToken::new(), None)
        .await
        .unwrap();

    assert_eq!(text_of(&result), "(empty directory)");
}

#[tokio::test]
async fn ls_nonexistent_path_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let ls = LsTool { cwd };

    let result = ls
        .execute(
            "l1",
            json!({"path": "does_not_exist"}),
            CancellationToken::new(),
            None,
        )
        .await;

    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// GrepTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grep_finds_pattern() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    std::fs::write(tmp.path().join("file.txt"), "hello world\ngoodbye world\nhello again").unwrap();

    let grep = GrepTool { cwd };

    let result = grep
        .execute(
            "g1",
            json!({"pattern": "hello", "path": "."}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    let text = text_of(&result);
    assert!(text.contains("hello world"));
    assert!(text.contains("hello again"));
    assert!(!text.contains("goodbye"));
}

#[tokio::test]
async fn grep_no_matches() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    std::fs::write(tmp.path().join("file.txt"), "nothing here").unwrap();

    let grep = GrepTool { cwd };

    let result = grep
        .execute(
            "g1",
            json!({"pattern": "NOMATCH", "path": "."}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(text_of(&result), "No matches found");
}

#[tokio::test]
async fn grep_case_insensitive() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    std::fs::write(tmp.path().join("file.txt"), "Hello World\nHELLO again").unwrap();

    let grep = GrepTool { cwd };

    let result = grep
        .execute(
            "g1",
            json!({"pattern": "hello", "path": ".", "ignoreCase": true}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    let text = text_of(&result);
    assert!(text.contains("Hello World"));
    assert!(text.contains("HELLO again"));
}

// ---------------------------------------------------------------------------
// FindTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn find_matches_pattern() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    std::fs::write(tmp.path().join("app.ts"), "").unwrap();
    std::fs::write(tmp.path().join("lib.ts"), "").unwrap();
    std::fs::write(tmp.path().join("readme.md"), "").unwrap();

    let find = FindTool { cwd };

    let result = find
        .execute(
            "f1",
            json!({"pattern": "*.ts"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    let text = text_of(&result);
    assert!(text.contains("app.ts"));
    assert!(text.contains("lib.ts"));
    assert!(!text.contains("readme.md"));
}

#[tokio::test]
async fn find_no_matches() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    std::fs::write(tmp.path().join("readme.md"), "").unwrap();

    let find = FindTool { cwd };

    let result = find
        .execute(
            "f1",
            json!({"pattern": "*.ts"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(text_of(&result), "No files found matching pattern");
}

// ---------------------------------------------------------------------------
// BashTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bash_runs_command() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let bash = BashTool { cwd };

    let result = bash
        .execute(
            "b1",
            json!({"command": "echo hello from bash"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    assert!(text_of(&result).contains("hello from bash"));
}

#[tokio::test]
async fn bash_failing_command_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let bash = BashTool { cwd };

    let result = bash
        .execute(
            "b1",
            json!({"command": "exit 42"}),
            CancellationToken::new(),
            None,
        )
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("42"), "should include exit code: {err}");
}

#[tokio::test]
async fn bash_empty_command_output() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let bash = BashTool { cwd };

    let result = bash
        .execute(
            "b1",
            json!({"command": "true"}),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(text_of(&result), "(no output)");
}

// ---------------------------------------------------------------------------
// Tool definition basics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn all_tools_have_valid_schemas() {
    let tmp = tempfile::tempdir().unwrap();
    let cwd = tmp.path().to_string_lossy().to_string();

    let tools: Vec<Box<dyn AgentTool>> = vec![
        Box::new(ThinkTool),
        Box::new(ReadTool { cwd: cwd.clone() }),
        Box::new(WriteTool { cwd: cwd.clone() }),
        Box::new(LsTool { cwd: cwd.clone() }),
        Box::new(GrepTool { cwd: cwd.clone() }),
        Box::new(FindTool { cwd: cwd.clone() }),
        Box::new(BashTool { cwd }),
    ];

    let expected_names = vec!["think", "read", "write", "ls", "grep", "find", "bash"];

    for (tool, expected_name) in tools.iter().zip(expected_names.iter()) {
        assert_eq!(tool.name(), *expected_name);
        assert!(!tool.description().is_empty(), "{} has empty description", tool.name());

        let schema = tool.parameters_schema();
        assert_eq!(
            schema["type"].as_str(),
            Some("object"),
            "{} schema type should be object",
            tool.name()
        );
        assert!(
            schema["properties"].is_object(),
            "{} schema should have properties",
            tool.name()
        );

        let def = tool.as_tool_definition();
        assert_eq!(def.name, *expected_name);
    }
}
