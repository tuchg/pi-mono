//! Integration tests for pi-agent-rs using the faux provider harness.
//!
//! These tests mirror the TypeScript `packages/agent/test/e2e.test.ts`.

mod harness;

use harness::{get_message_text, Harness};
use pi_ai_rs::{
    faux_assistant_message, faux_assistant_message_with_stop,
    faux_assistant_text, faux_text, faux_thinking, faux_tool_call_with_id, StopReason,
};
use pi_agent_rs::AgentMessage;

#[tokio::test]
async fn basic_text_prompt() {
    let mut harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("4")]);

    let messages = harness.prompt("What is 2+2?").await;

    // Should have user message + assistant message
    assert!(messages.len() >= 2, "expected at least 2 messages, got {}", messages.len());

    // First message is user
    assert_eq!(messages[0].role(), "user");
    assert_eq!(get_message_text(&messages[0]), "What is 2+2?");

    // Second message is assistant
    assert_eq!(messages[1].role(), "assistant");
    assert!(get_message_text(&messages[1]).contains("4"));

    harness.cleanup();
}

#[tokio::test]
async fn emits_lifecycle_events() {
    let mut harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("hello")]);

    harness.prompt("hi").await;

    let types = harness.event_types();

    assert!(types.contains(&"agent_start"), "missing agent_start");
    assert!(types.contains(&"turn_start"), "missing turn_start");
    assert!(types.contains(&"message_start"), "missing message_start");
    assert!(types.contains(&"message_update"), "missing message_update");
    assert!(types.contains(&"message_end"), "missing message_end");
    assert!(types.contains(&"turn_end"), "missing turn_end");
    assert!(types.contains(&"agent_end"), "missing agent_end");

    // Order: agent_start before message_start
    let start_idx = types.iter().position(|t| *t == "agent_start").unwrap();
    let msg_start_idx = types.iter().position(|t| *t == "message_start").unwrap();
    assert!(start_idx < msg_start_idx);

    // Order: message_end before agent_end
    let msg_end_idx = types.iter().rposition(|t| *t == "message_end").unwrap();
    let end_idx = types.iter().rposition(|t| *t == "agent_end").unwrap();
    assert!(msg_end_idx < end_idx);

    harness.cleanup();
}

#[tokio::test]
async fn tool_call_triggers_tool_execution_events() {
    let mut harness = Harness::new();

    // First response: request a tool call
    // Second response: final answer
    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_text("Let me calculate."),
                faux_tool_call_with_id("calc-1", "calculate", serde_json::json!({"expr": "1+1"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("The answer is 2."),
    ]);

    let messages = harness.prompt("What is 1+1?").await;

    // Should have: user, assistant (tool call), tool result, assistant (final)
    assert!(
        messages.len() >= 4,
        "expected at least 4 messages, got {}",
        messages.len()
    );

    // Verify tool execution events were emitted
    let types = harness.event_types();
    assert!(
        types.contains(&"tool_execution_start"),
        "missing tool_execution_start"
    );
    assert!(
        types.contains(&"tool_execution_end"),
        "missing tool_execution_end"
    );

    // The tool result message should exist
    let tool_results: Vec<_> = messages
        .iter()
        .filter(|m| m.role() == "toolResult")
        .collect();
    assert_eq!(tool_results.len(), 1);

    // Final message should be the second assistant response
    let last = messages.last().unwrap();
    assert_eq!(last.role(), "assistant");
    assert!(get_message_text(last).contains("2"));

    harness.cleanup();
}

#[tokio::test]
async fn thinking_content_preserved() {
    let mut harness = Harness::new();

    harness.set_responses(vec![faux_assistant_message(vec![
        faux_thinking("step by step reasoning"),
        faux_text("The answer is 42."),
    ])]);

    let messages = harness.prompt("What is the meaning of life?").await;

    let assistant_msg = messages
        .iter()
        .find(|m| m.role() == "assistant")
        .expect("no assistant message");

    // Verify both thinking and text content are present
    if let AgentMessage::Standard(pi_ai_rs::types::Message::Assistant(am)) = assistant_msg {
        assert_eq!(am.content.len(), 2);
        assert!(matches!(
            &am.content[0],
            pi_ai_rs::AssistantContent::Thinking(t) if t.thinking == "step by step reasoning"
        ));
        assert!(matches!(
            &am.content[1],
            pi_ai_rs::AssistantContent::Text(t) if t.text == "The answer is 42."
        ));
    } else {
        panic!("expected standard assistant message");
    }

    harness.cleanup();
}

#[tokio::test]
async fn no_responses_produces_error() {
    let mut harness = Harness::new();
    // Don't set any responses — the faux provider will error

    let messages = harness.prompt("hello").await;

    // The agent should still produce messages (user + error response)
    assert!(!messages.is_empty());

    // Should have agent_end event
    let types = harness.event_types();
    assert!(types.contains(&"agent_end"));

    harness.cleanup();
}

#[tokio::test]
async fn multiple_tool_calls_in_one_response() {
    let mut harness = Harness::new();

    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_tool_call_with_id("tc-1", "bash", serde_json::json!({"command": "ls"})),
                faux_tool_call_with_id("tc-2", "bash", serde_json::json!({"command": "pwd"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let messages = harness.prompt("run two commands").await;

    // Should have tool results for both tool calls
    let tool_results: Vec<_> = messages
        .iter()
        .filter(|m| m.role() == "toolResult")
        .collect();
    assert_eq!(tool_results.len(), 2);

    // Two tool_execution_start events
    let starts = harness.events_of_type("tool_execution_start");
    assert_eq!(starts.len(), 2);

    harness.cleanup();
}

#[tokio::test]
async fn agent_end_contains_all_messages() {
    let mut harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("response")]);

    harness.prompt("hello").await;

    let end_events = harness.events_of_type("agent_end");
    assert_eq!(end_events.len(), 1);

    if let pi_agent_rs::AgentEvent::AgentEnd { messages } = end_events[0] {
        assert!(messages.len() >= 2);
        assert_eq!(messages[0].role(), "user");
        assert_eq!(messages[1].role(), "assistant");
    } else {
        panic!("expected AgentEnd event");
    }

    harness.cleanup();
}

/// A minimal tool used only for testing that returns a known string.
struct EchoTool;

impl pi_agent_rs::AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn label(&self) -> &str {
        "Echo"
    }
    fn description(&self) -> &str {
        "Echoes a message"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }
    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _on_update: Option<pi_agent_rs::AgentToolUpdateCallback>,
    ) -> pi_agent_rs::BoxFuture<'_, Result<pi_agent_rs::AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let msg = params["message"].as_str().unwrap_or("").to_string();
            Ok(pi_agent_rs::AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("ECHOED:{msg}"),
                    text_signature: None,
                })],
                details: serde_json::json!(null),
            })
        })
    }
}

#[tokio::test]
async fn tool_execution_runs_registered_tool() {
    let mut harness = Harness::new();
    harness.add_tool(std::sync::Arc::new(EchoTool));

    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![faux_tool_call_with_id(
                "echo-1",
                "echo",
                serde_json::json!({"message": "hello world"}),
            )],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let messages = harness.prompt("echo something").await;

    // Find the tool result message
    let tool_result = messages
        .iter()
        .find(|m| m.role() == "toolResult")
        .expect("no tool result message");

    let result_text = get_message_text(tool_result);
    assert_eq!(result_text, "ECHOED:hello world");

    // Verify tool_execution_end event has is_error = false
    let end_events = harness.events_of_type("tool_execution_end");
    assert_eq!(end_events.len(), 1);
    if let pi_agent_rs::AgentEvent::ToolExecutionEnd { is_error, .. } = end_events[0] {
        assert!(!is_error, "expected is_error=false for successful tool execution");
    }

    harness.cleanup();
}

#[tokio::test]
async fn unknown_tool_returns_error_result() {
    let mut harness = Harness::new();
    // Don't register any tools

    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![faux_tool_call_with_id(
                "missing-1",
                "nonexistent_tool",
                serde_json::json!({}),
            )],
            StopReason::ToolUse,
        ),
        faux_assistant_text("ok"),
    ]);

    let messages = harness.prompt("try missing tool").await;

    let tool_result = messages
        .iter()
        .find(|m| m.role() == "toolResult")
        .expect("no tool result message");

    let result_text = get_message_text(tool_result);
    assert!(
        result_text.contains("Unknown tool"),
        "expected 'Unknown tool' in result, got: {result_text}"
    );

    // Verify the tool execution end event has is_error = true
    let end_events = harness.events_of_type("tool_execution_end");
    assert_eq!(end_events.len(), 1);
    if let pi_agent_rs::AgentEvent::ToolExecutionEnd { is_error, .. } = end_events[0] {
        assert!(is_error, "expected is_error=true for unknown tool");
    }

    harness.cleanup();
}
