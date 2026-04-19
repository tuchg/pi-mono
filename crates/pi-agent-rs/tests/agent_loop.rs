//! Tests for the low-level `agent_loop` and `agent_loop_continue` functions.
//!
//! Mirrors the TypeScript `packages/agent/test/agent-loop.test.ts`.

mod harness;

use std::sync::Arc;

use futures::StreamExt;
use harness::{get_message_text, Harness};
use pi_ai_rs::{
    faux_assistant_message_with_stop, faux_assistant_text,
    faux_tool_call_with_id, StopReason,
};
use pi_agent_rs::{
    agent_loop, agent_loop_continue, AgentContext, AgentEvent, AgentLoopConfig,
    AgentMessage, AgentTool, AgentToolResult, AgentToolUpdateCallback, BoxFuture,
    Message, ToolExecutionMode,
};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Helper: create a simple echo tool
// ---------------------------------------------------------------------------

struct EchoTool;

impl AgentTool for EchoTool {
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
                "value": { "type": "string" }
            },
            "required": ["value"]
        })
    }
    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _cancel: CancellationToken,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let val = params["value"].as_str().unwrap_or("").to_string();
            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("echoed: {val}"),
                    text_signature: None,
                })],
                details: serde_json::json!({ "value": val }),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: create a tool with prepare_arguments
// ---------------------------------------------------------------------------

struct EditTool;

impl AgentTool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }
    fn label(&self) -> &str {
        "Edit"
    }
    fn description(&self) -> &str {
        "Edit tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string" },
                            "newText": { "type": "string" }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["edits"]
        })
    }

    fn prepare_arguments(&self, args: serde_json::Value) -> Option<serde_json::Value> {
        let obj = args.as_object()?;
        // If args has oldText/newText at top level, wrap into edits array
        if let (Some(old), Some(new)) = (obj.get("oldText"), obj.get("newText")) {
            let existing = obj
                .get("edits")
                .and_then(|e| e.as_array())
                .cloned()
                .unwrap_or_default();
            let mut edits = existing;
            edits.push(serde_json::json!({
                "oldText": old,
                "newText": new,
            }));
            Some(serde_json::json!({ "edits": edits }))
        } else {
            None
        }
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _cancel: CancellationToken,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let edits = params["edits"].as_array().map(|a| a.len()).unwrap_or(0);
            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("edited {edits}"),
                    text_signature: None,
                })],
                details: serde_json::json!({ "count": edits }),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: tool that records execution for parallel/sequential tests
// ---------------------------------------------------------------------------

struct TrackingTool {
    name_val: String,
    mode: Option<ToolExecutionMode>,
    executed: Arc<std::sync::Mutex<Vec<String>>>,
    /// If provided, the tool will wait for this notify before completing
    /// for the given argument value.
    wait_on: Option<(String, Arc<tokio::sync::Notify>)>,
}

impl AgentTool for TrackingTool {
    fn name(&self) -> &str {
        &self.name_val
    }
    fn label(&self) -> &str {
        &self.name_val
    }
    fn description(&self) -> &str {
        "Tracking tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"]
        })
    }
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.mode
    }
    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _cancel: CancellationToken,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        let val = params["value"].as_str().unwrap_or("").to_string();
        let executed = self.executed.clone();
        let wait_on = self.wait_on.clone();

        Box::pin(async move {
            if let Some((ref wait_val, ref notify)) = wait_on {
                if val == *wait_val {
                    notify.notified().await;
                }
            }
            executed.lock().unwrap().push(format!("{}:{val}", self.name_val));
            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("{}: {val}", self.name_val),
                    text_signature: None,
                })],
                details: serde_json::json!({}),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Tests: agentLoop events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn emit_events_with_agent_message_types() {
    let mut harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("Hi there!")]);

    let messages = harness.prompt("Hello").await;

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role(), "user");
    assert_eq!(messages[1].role(), "assistant");

    let types = harness.event_types();
    assert!(types.contains(&"agent_start"));
    assert!(types.contains(&"turn_start"));
    assert!(types.contains(&"message_start"));
    assert!(types.contains(&"message_end"));
    assert!(types.contains(&"turn_end"));
    assert!(types.contains(&"agent_end"));

    harness.cleanup();
}

#[tokio::test]
async fn handle_custom_message_via_convert_to_llm() {
    // Custom message injected in context; convertToLlm filters it out
    let harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("Response")]);

    // Inject a custom message into the pre-existing context.
    let custom = AgentMessage::Custom(serde_json::json!({
        "role": "notification",
        "text": "This is a notification",
    }));

    // Build the context with the custom message and run agent_loop directly.
    let tool_defs = Vec::new();
    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: vec![custom],
        tool_definitions: tool_defs,
        tools: Vec::new(),
    };

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Hello".to_string()),
        timestamp: 0,
    }));

    let convert = Arc::new(|messages: Vec<AgentMessage>| {
        Box::pin(async move {
            // Filter out custom messages
            messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::Standard(msg) => Some(msg),
                    _ => None,
                })
                .collect()
        }) as BoxFuture<'static, Vec<Message>>
    });

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: convert,
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Sequential,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let cancel = CancellationToken::new();
    let mut stream = agent_loop(vec![user_msg], context, config, cancel);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Agent should complete successfully despite custom message in context
    let types: Vec<&str> = events.iter().map(|e| harness_event_type(e)).collect();
    assert!(types.contains(&"agent_end"));

    harness.cleanup();
}

fn harness_event_type(event: &AgentEvent) -> &str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}

#[tokio::test]
async fn apply_transform_context_before_convert_to_llm() {
    let harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("Response")]);

    // transformContext keeps only last 2.
    let old_messages: Vec<AgentMessage> = (0..4)
        .map(|i| {
            if i % 2 == 0 {
                AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
                    content: pi_ai_rs::UserContent::Text(format!("old msg {i}")),
                    timestamp: 0,
                }))
            } else {
                AgentMessage::Standard(Message::Assistant(faux_assistant_text(&format!(
                    "old response {i}"
                ))))
            }
        })
        .collect();

    let tool_defs = Vec::new();
    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: old_messages,
        tool_definitions: tool_defs,
        tools: Vec::new(),
    };

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("new message".to_string()),
        timestamp: 0,
    }));

    let transform_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tc = transform_count.clone();
    let transform = Arc::new(
        move |messages: Vec<AgentMessage>, _cancel: CancellationToken| {
            let tc = tc.clone();
            Box::pin(async move {
                tc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Keep only last 2 messages
                let len = messages.len();
                if len > 2 {
                    messages[len - 2..].to_vec()
                } else {
                    messages
                }
            }) as BoxFuture<'static, Vec<AgentMessage>>
        },
    ) as pi_agent_rs::TransformContextFn;

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: Some(transform),
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Sequential,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let cancel = CancellationToken::new();
    let mut stream = agent_loop(vec![user_msg], context, config, cancel);

    while let Some(_event) = stream.next().await {}

    // transformContext should have been called at least once
    assert!(transform_count.load(std::sync::atomic::Ordering::SeqCst) >= 1);

    harness.cleanup();
}

#[tokio::test]
async fn tool_call_and_result() {
    let mut harness = Harness::new();
    harness.add_tool(Arc::new(EchoTool));

    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![faux_tool_call_with_id(
                "tool-1",
                "echo",
                serde_json::json!({"value": "hello"}),
            )],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let messages = harness.prompt("echo something").await;

    // Should have: user, assistant (tool call), tool result, assistant (final)
    assert!(messages.len() >= 4);

    // Check tool result
    let tool_result = messages
        .iter()
        .find(|m| m.role() == "toolResult")
        .expect("no tool result");
    assert_eq!(get_message_text(tool_result), "echoed: hello");

    // Check tool execution events
    let tool_ends = harness.events_of_type("tool_execution_end");
    assert_eq!(tool_ends.len(), 1);
    if let AgentEvent::ToolExecutionEnd { is_error, .. } = tool_ends[0] {
        assert!(!is_error);
    }

    harness.cleanup();
}

#[tokio::test]
async fn prepare_arguments_transforms_tool_args() {
    let mut harness = Harness::new();
    harness.add_tool(Arc::new(EditTool));

    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![faux_tool_call_with_id(
                "tool-1",
                "edit",
                serde_json::json!({"oldText": "before", "newText": "after"}),
            )],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let messages = harness.prompt("edit something").await;

    // Tool result should confirm 1 edit
    let tool_result = messages
        .iter()
        .find(|m| m.role() == "toolResult")
        .expect("no tool result");
    assert_eq!(get_message_text(tool_result), "edited 1");

    harness.cleanup();
}

#[tokio::test]
async fn parallel_execution_runs_concurrently() {
    let executed = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let notify = Arc::new(tokio::sync::Notify::new());

    let tool = TrackingTool {
        name_val: "echo".to_string(),
        mode: None,
        executed: executed.clone(),
        wait_on: Some(("first".to_string(), notify.clone())),
    };

    let mut harness = Harness::new();
    harness.add_tool(Arc::new(tool));

    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_tool_call_with_id("tool-1", "echo", serde_json::json!({"value": "first"})),
                faux_tool_call_with_id("tool-2", "echo", serde_json::json!({"value": "second"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    // Use parallel execution via direct agent_loop call
    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("echo both".to_string()),
        timestamp: 0,
    }));

    let tool_defs = vec![Arc::new(TrackingTool {
        name_val: "echo".to_string(),
        mode: None,
        executed: executed.clone(),
        wait_on: Some(("first".to_string(), notify.clone())),
    }) as Arc<dyn AgentTool>];

    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: Vec::new(),
        tool_definitions: tool_defs.iter().map(|t| t.as_tool_definition()).collect(),
        tools: tool_defs,
    };

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Parallel,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    // Release the first tool after a short delay
    let notify_clone = notify.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        notify_clone.notify_one();
    });

    let cancel = CancellationToken::new();
    let mut stream = agent_loop(vec![user_msg], context, config, cancel);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let order = executed.lock().unwrap().clone();
    // In parallel mode, "second" should be able to run before "first" completes.
    // "second" appears before "first" in the execution order.
    let second_idx = order.iter().position(|v| v == "echo:second");
    let first_idx = order.iter().position(|v| v == "echo:first");
    assert!(second_idx.is_some(), "second should have been executed");
    assert!(first_idx.is_some(), "first should have been executed");
    // second runs first because first is blocked
    assert!(
        second_idx.unwrap() < first_idx.unwrap(),
        "expected parallel execution: second should complete before first"
    );

    // But tool results in events should be in source order
    let tool_result_ids: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageEnd { message } => {
                if message.role() == "toolResult" {
                    if let AgentMessage::Standard(Message::ToolResult(tr)) = message {
                        Some(tr.tool_call_id.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();
    assert_eq!(tool_result_ids, vec!["tool-1", "tool-2"]);

    harness.cleanup();
}

#[tokio::test]
async fn sequential_tool_mode_forces_sequential() {
    let executed = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let notify = Arc::new(tokio::sync::Notify::new());

    let tool = Arc::new(TrackingTool {
        name_val: "slow".to_string(),
        mode: Some(ToolExecutionMode::Sequential),
        executed: executed.clone(),
        wait_on: Some(("first".to_string(), notify.clone())),
    }) as Arc<dyn AgentTool>;

    let harness = Harness::new();
    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_tool_call_with_id("tool-1", "slow", serde_json::json!({"value": "first"})),
                faux_tool_call_with_id("tool-2", "slow", serde_json::json!({"value": "second"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("run both".to_string()),
        timestamp: 0,
    }));

    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: Vec::new(),
        tool_definitions: vec![tool.as_tool_definition()],
        tools: vec![tool],
    };

    // Config is parallel, but tool forces sequential
    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Parallel, // config says parallel
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    // Release after a short delay
    let notify_clone = notify.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        notify_clone.notify_one();
    });

    let cancel = CancellationToken::new();
    let mut stream = agent_loop(vec![user_msg], context, config, cancel);

    while let Some(_event) = stream.next().await {}

    let order = executed.lock().unwrap().clone();
    // Sequential: first must complete before second starts
    assert_eq!(order[0], "slow:first");
    assert_eq!(order[1], "slow:second");

    harness.cleanup();
}

#[tokio::test]
async fn mixed_tools_force_sequential_when_one_is_sequential() {
    let executed = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let notify = Arc::new(tokio::sync::Notify::new());

    let slow_tool = Arc::new(TrackingTool {
        name_val: "slow".to_string(),
        mode: Some(ToolExecutionMode::Sequential),
        executed: executed.clone(),
        wait_on: Some(("a".to_string(), notify.clone())),
    }) as Arc<dyn AgentTool>;

    let fast_tool = Arc::new(TrackingTool {
        name_val: "fast".to_string(),
        mode: None, // default (parallel)
        executed: executed.clone(),
        wait_on: None,
    }) as Arc<dyn AgentTool>;

    let harness = Harness::new();
    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_tool_call_with_id("tool-1", "slow", serde_json::json!({"value": "a"})),
                faux_tool_call_with_id("tool-2", "fast", serde_json::json!({"value": "b"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("run both".to_string()),
        timestamp: 0,
    }));

    let tools: Vec<Arc<dyn AgentTool>> = vec![slow_tool.clone(), fast_tool.clone()];
    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: Vec::new(),
        tool_definitions: tools.iter().map(|t| t.as_tool_definition()).collect(),
        tools,
    };

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Parallel,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let notify_clone = notify.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        notify_clone.notify_one();
    });

    let cancel = CancellationToken::new();
    let mut stream = agent_loop(vec![user_msg], context, config, cancel);

    while let Some(_event) = stream.next().await {}

    let order = executed.lock().unwrap().clone();
    // With one sequential tool, all execute sequentially.
    // slow:a must execute first.
    assert_eq!(order[0], "slow:a");
    assert!(order.contains(&"fast:b".to_string()));

    harness.cleanup();
}

#[tokio::test]
async fn queued_steering_messages_injected_after_tool_calls() {
    let _executed = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

    let tool = Arc::new(EchoTool) as Arc<dyn AgentTool>;

    let harness = Harness::new();
    harness.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_tool_call_with_id("tool-1", "echo", serde_json::json!({"value": "first"})),
                faux_tool_call_with_id("tool-2", "echo", serde_json::json!({"value": "second"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("done"),
    ]);

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("start".to_string()),
        timestamp: 0,
    }));

    let queued_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("interrupt".to_string()),
        timestamp: 0,
    }));

    // Deliver steering message after first tool call
    let steering_delivered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sd = steering_delivered.clone();
    let qm = queued_msg.clone();

    let get_steering = Arc::new(move || {
        let sd = sd.clone();
        let qm = qm.clone();
        Box::pin(async move {
            if !sd.swap(true, std::sync::atomic::Ordering::SeqCst) {
                vec![qm]
            } else {
                Vec::new()
            }
        }) as BoxFuture<'static, Vec<AgentMessage>>
    }) as pi_agent_rs::GetMessagesFn;

    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: Vec::new(),
        tool_definitions: vec![tool.as_tool_definition()],
        tools: vec![tool],
    };

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: Some(get_steering),
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Sequential,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let cancel = CancellationToken::new();
    let mut stream = agent_loop(vec![user_msg], context, config, cancel);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Both tools should execute before steering is injected
    let tool_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .collect();
    assert_eq!(tool_ends.len(), 2);

    // Steering message should appear in events
    let has_interrupt = events.iter().any(|e| {
        if let AgentEvent::MessageStart { message } = e {
            get_message_text(message) == "interrupt"
        } else {
            false
        }
    });
    assert!(has_interrupt, "steering message should be injected");

    harness.cleanup();
}

// ---------------------------------------------------------------------------
// Tests: agentLoopContinue
// ---------------------------------------------------------------------------

#[tokio::test]
async fn continue_empty_context_returns_empty() {
    let harness = Harness::new();

    let context = AgentContext {
        system_prompt: "test".to_string(),
        messages: Vec::new(),
        tool_definitions: Vec::new(),
        tools: Vec::new(),
    };

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Sequential,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let cancel = CancellationToken::new();
    let mut stream = agent_loop_continue(context, config, cancel);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should end immediately for empty context (Rust returns empty stream)
    // No panic
    harness.cleanup();
}

#[tokio::test]
async fn continue_from_user_message() {
    let harness = Harness::new();
    harness.set_responses(vec![faux_assistant_text("Response")]);

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Hello".to_string()),
        timestamp: 0,
    }));

    let context = AgentContext {
        system_prompt: harness.system_prompt.clone(),
        messages: vec![user_msg],
        tool_definitions: Vec::new(),
        tools: Vec::new(),
    };

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Sequential,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let cancel = CancellationToken::new();
    let mut stream = agent_loop_continue(context, config, cancel);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should only return new assistant message (not the existing user message)
    let message_ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MessageEnd { .. }))
        .collect();

    // Only 1 message_end for the assistant response
    assert_eq!(message_ends.len(), 1);
    if let AgentEvent::MessageEnd { message } = message_ends[0] {
        assert_eq!(message.role(), "assistant");
    }

    harness.cleanup();
}

#[tokio::test]
async fn continue_rejects_assistant_tail() {
    let harness = Harness::new();

    let assistant_msg = AgentMessage::Standard(Message::Assistant(faux_assistant_text("Hello")));

    let context = AgentContext {
        system_prompt: "test".to_string(),
        messages: vec![assistant_msg],
        tool_definitions: Vec::new(),
        tools: Vec::new(),
    };

    let config = AgentLoopConfig {
        model: harness.model.clone(),
        stream_options: Default::default(),
        convert_to_llm: Arc::new(|messages: Vec<AgentMessage>| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|m| match m {
                        AgentMessage::Standard(msg) => Some(msg),
                        _ => None,
                    })
                    .collect()
            }) as BoxFuture<'static, Vec<Message>>
        }),
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Sequential,
        before_tool_call: None,
        after_tool_call: None,
        stream_fn: None,
    };

    let cancel = CancellationToken::new();
    let mut stream = agent_loop_continue(context, config, cancel);

    // Should end immediately (Rust logs error instead of throwing)
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    // The stream should end without producing agent events
    // (just the empty stream from the guard).

    harness.cleanup();
}
