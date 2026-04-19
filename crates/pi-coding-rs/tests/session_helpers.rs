// This is a shared helper module for session tests.
// Each test file includes it via `mod session_helpers;`
#![allow(dead_code)]

use serde_json::{Value, json};

pub fn user_msg(text: &str) -> Value {
    json!({ "role": "user", "content": text, "timestamp": 1 })
}

pub fn assistant_msg(text: &str) -> Value {
    json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": text }],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "test",
        "usage": {
            "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0,
            "totalTokens": 2,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
        },
        "stopReason": "stop",
        "timestamp": 2
    })
}

pub fn msg_entry(id: &str, parent_id: Option<&str>, role: &str, text: &str) -> Value {
    let parent = match parent_id {
        Some(p) => Value::String(p.to_string()),
        None => Value::Null,
    };
    if role == "user" {
        json!({
            "type": "message",
            "id": id,
            "parentId": parent,
            "timestamp": "2025-01-01T00:00:00Z",
            "message": { "role": "user", "content": text, "timestamp": 1 }
        })
    } else {
        json!({
            "type": "message",
            "id": id,
            "parentId": parent,
            "timestamp": "2025-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": text }],
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": "claude-test",
                "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
                    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
                },
                "stopReason": "stop",
                "timestamp": 1
            }
        })
    }
}

pub fn compaction_entry(
    id: &str,
    parent_id: Option<&str>,
    summary: &str,
    first_kept_entry_id: &str,
) -> Value {
    let parent = match parent_id {
        Some(p) => Value::String(p.to_string()),
        None => Value::Null,
    };
    json!({
        "type": "compaction",
        "id": id,
        "parentId": parent,
        "timestamp": "2025-01-01T00:00:00Z",
        "summary": summary,
        "firstKeptEntryId": first_kept_entry_id,
        "tokensBefore": 1000
    })
}

pub fn branch_summary_entry(
    id: &str,
    parent_id: Option<&str>,
    summary: &str,
    from_id: &str,
) -> Value {
    let parent = match parent_id {
        Some(p) => Value::String(p.to_string()),
        None => Value::Null,
    };
    json!({
        "type": "branch_summary",
        "id": id,
        "parentId": parent,
        "timestamp": "2025-01-01T00:00:00Z",
        "summary": summary,
        "fromId": from_id
    })
}

pub fn thinking_level_entry(id: &str, parent_id: Option<&str>, level: &str) -> Value {
    let parent = match parent_id {
        Some(p) => Value::String(p.to_string()),
        None => Value::Null,
    };
    json!({
        "type": "thinking_level_change",
        "id": id,
        "parentId": parent,
        "timestamp": "2025-01-01T00:00:00Z",
        "thinkingLevel": level
    })
}

pub fn model_change_entry(id: &str, parent_id: Option<&str>, provider: &str, model_id: &str) -> Value {
    let parent = match parent_id {
        Some(p) => Value::String(p.to_string()),
        None => Value::Null,
    };
    json!({
        "type": "model_change",
        "id": id,
        "parentId": parent,
        "timestamp": "2025-01-01T00:00:00Z",
        "provider": provider,
        "modelId": model_id
    })
}
