//! Tests for the stateful `Agent` wrapper.
//!
//! Mirrors the TypeScript `packages/agent/test/agent.test.ts`.

use std::sync::Arc;

use pi_ai_rs::{
    faux_assistant_message, faux_assistant_message_with_stop, faux_assistant_text,
    faux_text, faux_thinking, faux_tool_call_with_id, register_faux_provider,
    FauxModelDefinition, RegisterFauxProviderOptions, StopReason,
};
use pi_agent_rs::{
    Agent, AgentEvent, AgentListenerFn, AgentMessage, AgentOptions, AgentState,
    AgentThinkingLevel, AgentTool, AgentToolResult, AgentToolUpdateCallback, BoxFuture,
    Message,
};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Helper: simple calculate tool (mirrors TS calculateTool)
// ---------------------------------------------------------------------------

struct CalculateTool;

impl AgentTool for CalculateTool {
    fn name(&self) -> &str {
        "calculate"
    }
    fn label(&self) -> &str {
        "Calculator"
    }
    fn description(&self) -> &str {
        "Evaluates a math expression"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            },
            "required": ["expression"]
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
            let expr = params["expression"].as_str().unwrap_or("").to_string();
            // Simple eval for "A * B" patterns
            let result_text = if expr.contains('*') {
                let parts: Vec<&str> = expr.split('*').collect();
                if parts.len() == 2 {
                    let a: i64 = parts[0].trim().parse().unwrap_or(0);
                    let b: i64 = parts[1].trim().parse().unwrap_or(0);
                    format!("{expr} = {}", a * b)
                } else {
                    format!("Cannot evaluate: {expr}")
                }
            } else if expr.contains('+') {
                let parts: Vec<&str> = expr.split('+').collect();
                if parts.len() == 2 {
                    let a: i64 = parts[0].trim().parse().unwrap_or(0);
                    let b: i64 = parts[1].trim().parse().unwrap_or(0);
                    format!("{expr} = {}", a + b)
                } else {
                    format!("Cannot evaluate: {expr}")
                }
            } else {
                format!("Cannot evaluate: {expr}")
            };

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: result_text,
                    text_signature: None,
                })],
                details: serde_json::json!({}),
            })
        })
    }
}

fn get_text_content(msg: &AgentMessage) -> String {
    match msg {
        AgentMessage::Standard(Message::Assistant(am)) => am
            .content
            .iter()
            .filter_map(|c| match c {
                pi_ai_rs::AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        AgentMessage::Standard(Message::ToolResult(tr)) => tr
            .content
            .iter()
            .filter_map(|c| match c {
                pi_ai_rs::Content::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests: Agent default state
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_state() {
    let agent = Agent::new(AgentOptions::default());
    let state = agent.state();

    assert_eq!(state.system_prompt, "");
    assert!(state.tools.is_empty());
    assert!(state.messages.is_empty());
    assert!(!state.is_streaming);
    assert!(state.streaming_message.is_none());
    assert!(state.pending_tool_calls.is_empty());
    assert!(state.error_message.is_none());
    assert_eq!(state.thinking_level, AgentThinkingLevel::Off);
}

#[tokio::test]
async fn custom_initial_state() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    let model = faux.get_model().clone();

    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: model.clone(),
            thinking_level: AgentThinkingLevel::Low,
            ..Default::default()
        }),
        ..Default::default()
    });

    assert_eq!(agent.state().system_prompt, "You are a helpful assistant.");
    assert_eq!(agent.state().model.id, model.id);
    assert_eq!(agent.state().thinking_level, AgentThinkingLevel::Low);

    faux.unregister();
}

// ---------------------------------------------------------------------------
// Tests: subscribe / unsubscribe
// ---------------------------------------------------------------------------

#[tokio::test]
async fn subscribe_and_unsubscribe() {
    let mut agent = Agent::new(AgentOptions::default());

    let event_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ec = event_count.clone();

    let listener: AgentListenerFn = Arc::new(move |_event, _cancel| {
        let ec = ec.clone();
        Box::pin(async move {
            ec.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
    });

    let id = agent.subscribe(listener);

    // No events just from subscribing
    assert_eq!(event_count.load(std::sync::atomic::Ordering::SeqCst), 0);

    // State changes don't emit events
    agent.state_mut().system_prompt = "Test".to_string();
    assert_eq!(event_count.load(std::sync::atomic::Ordering::SeqCst), 0);

    agent.unsubscribe(id);

    agent.state_mut().system_prompt = "Another".to_string();
    assert_eq!(event_count.load(std::sync::atomic::Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------------------
// Tests: state mutators
// ---------------------------------------------------------------------------

#[tokio::test]
async fn state_mutators() {
    let mut agent = Agent::new(AgentOptions::default());

    agent.state_mut().system_prompt = "Custom prompt".to_string();
    assert_eq!(agent.state().system_prompt, "Custom prompt");

    agent.state_mut().thinking_level = AgentThinkingLevel::High;
    assert_eq!(agent.state().thinking_level, AgentThinkingLevel::High);

    // Tools
    let tool = Arc::new(CalculateTool) as Arc<dyn AgentTool>;
    agent.add_tool(tool);
    assert_eq!(agent.state().tools.len(), 1);

    agent.clear_tools();
    assert!(agent.state().tools.is_empty());

    // Messages
    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Hello".to_string()),
        timestamp: 0,
    }));
    agent.state_mut().messages.push(user_msg);
    assert_eq!(agent.state().messages.len(), 1);

    agent.state_mut().messages.clear();
    assert!(agent.state().messages.is_empty());
}

// ---------------------------------------------------------------------------
// Tests: steering and follow-up queues
// ---------------------------------------------------------------------------

#[tokio::test]
async fn steering_queue() {
    let agent = Agent::new(AgentOptions::default());

    let message = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Steering message".to_string()),
        timestamp: 0,
    }));

    agent.steer(message);

    // Message is queued but not in state.messages
    assert!(agent.state().messages.is_empty());
    assert!(agent.has_queued_messages());
}

#[tokio::test]
async fn follow_up_queue() {
    let agent = Agent::new(AgentOptions::default());

    let message = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Follow-up message".to_string()),
        timestamp: 0,
    }));

    agent.follow_up(message);

    assert!(agent.state().messages.is_empty());
    assert!(agent.has_queued_messages());
}

#[tokio::test]
async fn clear_queues() {
    let agent = Agent::new(AgentOptions::default());

    agent.steer(AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("s".to_string()),
        timestamp: 0,
    })));
    agent.follow_up(AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("f".to_string()),
        timestamp: 0,
    })));

    assert!(agent.has_queued_messages());

    agent.clear_all_queues();
    assert!(!agent.has_queued_messages());
}

// ---------------------------------------------------------------------------
// Tests: abort (no-op when idle)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn abort_when_idle_does_not_panic() {
    let mut agent = Agent::new(AgentOptions::default());
    // Should not panic
    agent.abort();
}

// ---------------------------------------------------------------------------
// Tests: integration with faux provider
// ---------------------------------------------------------------------------

#[tokio::test]
async fn basic_text_prompt() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![faux_assistant_text("4")]);

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: faux.get_model().clone(),
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.prompt_text("What is 2+2?", None).await.unwrap();

    assert!(!agent.state().is_streaming);
    assert_eq!(agent.state().messages.len(), 2);
    assert_eq!(agent.state().messages[0].role(), "user");
    assert_eq!(agent.state().messages[1].role(), "assistant");
    assert!(get_text_content(&agent.state().messages[1]).contains('4'));

    faux.unregister();
}

#[tokio::test]
async fn tool_execution_and_pending_tracking() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![
        faux_assistant_message_with_stop(
            vec![
                faux_text("Let me calculate that."),
                faux_tool_call_with_id("calc-1", "calculate", serde_json::json!({"expression": "123 * 456"})),
            ],
            StopReason::ToolUse,
        ),
        faux_assistant_text("The result is 56088."),
    ]);

    let pending_during_events = Arc::new(std::sync::Mutex::new(Vec::<(String, Vec<String>)>::new()));
    let pde = pending_during_events.clone();

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: faux.get_model().clone(),
            tools: vec![Arc::new(CalculateTool) as Arc<dyn AgentTool>],
            ..Default::default()
        }),
        ..Default::default()
    });

    let listener: AgentListenerFn = Arc::new(move |event, _cancel| {
        let pde = pde.clone();
        Box::pin(async move {
            let event_type = match &event {
                AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                _ => return,
            };
            // Can't easily access agent.state() from listener, but we track the event type
            pde.lock().unwrap().push((event_type.to_string(), Vec::new()));
        })
    });
    agent.subscribe(listener);

    agent
        .prompt_text("Calculate 123 * 456 using the calculator tool.", None)
        .await
        .unwrap();

    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.len() >= 4);

    // Check tool result
    let tool_result = agent
        .state()
        .messages
        .iter()
        .find(|m| m.role() == "toolResult")
        .expect("no tool result");
    assert!(get_text_content(tool_result).contains("56088"));

    // Final message should contain the result
    let final_msg = agent.state().messages.last().unwrap();
    assert_eq!(final_msg.role(), "assistant");
    assert!(get_text_content(final_msg).contains("56088"));

    // Pending tool calls should be empty after completion
    assert!(agent.state().pending_tool_calls.is_empty());

    faux.unregister();
}

#[tokio::test]
async fn lifecycle_events() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![faux_assistant_text("1 2 3 4 5")]);

    let events = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let ev = events.clone();

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: faux.get_model().clone(),
            ..Default::default()
        }),
        ..Default::default()
    });

    let listener: AgentListenerFn = Arc::new(move |event, _cancel| {
        let ev = ev.clone();
        Box::pin(async move {
            let name = match event {
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
            };
            ev.lock().unwrap().push(name.to_string());
        })
    });
    agent.subscribe(listener);

    agent.prompt_text("Count from 1 to 5.", None).await.unwrap();

    let ev = events.lock().unwrap();
    assert!(ev.contains(&"agent_start".to_string()));
    assert!(ev.contains(&"turn_start".to_string()));
    assert!(ev.contains(&"message_start".to_string()));
    assert!(ev.contains(&"message_end".to_string()));
    assert!(ev.contains(&"turn_end".to_string()));
    assert!(ev.contains(&"agent_end".to_string()));

    let agent_start_idx = ev.iter().position(|e| e == "agent_start").unwrap();
    let message_start_idx = ev.iter().position(|e| e == "message_start").unwrap();
    let message_end_idx = ev.iter().position(|e| e == "message_end").unwrap();
    let agent_end_idx = ev.iter().rposition(|e| e == "agent_end").unwrap();

    assert!(agent_start_idx < message_start_idx);
    assert!(message_start_idx < message_end_idx);
    assert!(message_end_idx < agent_end_idx);

    assert!(!agent.state().is_streaming);
    assert_eq!(agent.state().messages.len(), 2);

    faux.unregister();
}

#[tokio::test]
async fn multi_turn_conversation() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![
        faux_assistant_text("Nice to meet you, Alice."),
        faux_assistant_text("Your name is Alice."),
    ]);

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: faux.get_model().clone(),
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.prompt_text("My name is Alice.", None).await.unwrap();
    assert_eq!(agent.state().messages.len(), 2);

    agent.prompt_text("What is my name?", None).await.unwrap();
    assert_eq!(agent.state().messages.len(), 4);

    let last = &agent.state().messages[3];
    assert_eq!(last.role(), "assistant");
    assert!(get_text_content(last).to_lowercase().contains("alice"));

    faux.unregister();
}

#[tokio::test]
async fn preserves_thinking_content() {
    let faux = register_faux_provider(RegisterFauxProviderOptions {
        models: Some(vec![FauxModelDefinition {
            id: "faux-reasoning".to_string(),
            reasoning: true,
            ..Default::default()
        }]),
        ..Default::default()
    });
    faux.set_responses(vec![faux_assistant_message(vec![
        faux_thinking("step by step"),
        faux_text("4"),
    ])]);

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: faux.get_model().clone(),
            thinking_level: AgentThinkingLevel::Low,
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.prompt_text("What is 2+2?", None).await.unwrap();

    let assistant = &agent.state().messages[1];
    if let AgentMessage::Standard(Message::Assistant(am)) = assistant {
        assert_eq!(am.content.len(), 2);
        match &am.content[0] {
            pi_ai_rs::AssistantContent::Thinking(t) => {
                assert_eq!(t.thinking, "step by step");
            }
            _ => panic!("Expected thinking content"),
        }
        match &am.content[1] {
            pi_ai_rs::AssistantContent::Text(t) => {
                assert_eq!(t.text, "4");
            }
            _ => panic!("Expected text content"),
        }
    } else {
        panic!("Expected assistant message");
    }

    faux.unregister();
}

// ---------------------------------------------------------------------------
// Tests: continue()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn continue_no_messages_errors() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "Test".to_string(),
            model: faux.get_model().clone(),
            ..Default::default()
        }),
        ..Default::default()
    });

    let result = agent.continue_().await;
    // Should error because there is no message to continue from
    assert!(result.is_err());

    faux.unregister();
}

#[tokio::test]
async fn continue_from_assistant_tail_errors() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "Test".to_string(),
            model: faux.get_model().clone(),
            messages: vec![AgentMessage::Standard(Message::Assistant(
                faux_assistant_text("Hello"),
            ))],
            ..Default::default()
        }),
        ..Default::default()
    });

    let result = agent.continue_().await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("assistant"), "error should mention assistant role: {err_msg}");

    faux.unregister();
}

#[tokio::test]
async fn continue_from_user_message() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![faux_assistant_text("HELLO WORLD")]);

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: faux.get_model().clone(),
            messages: vec![AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
                content: pi_ai_rs::UserContent::Text("Say exactly: HELLO WORLD".to_string()),
                timestamp: 0,
            }))],
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.continue_().await.unwrap();

    assert!(!agent.state().is_streaming);
    assert_eq!(agent.state().messages.len(), 2);
    assert_eq!(agent.state().messages[0].role(), "user");
    assert_eq!(agent.state().messages[1].role(), "assistant");
    assert!(get_text_content(&agent.state().messages[1])
        .to_uppercase()
        .contains("HELLO WORLD"));

    faux.unregister();
}

#[tokio::test]
async fn continue_from_tool_result() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![faux_assistant_text("The answer is 8.")]);

    let model = faux.get_model().clone();

    let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("What is 5 + 3?".to_string()),
        timestamp: 0,
    }));

    let assistant_msg = AgentMessage::Standard(Message::Assistant(
        faux_assistant_message_with_stop(
            vec![
                faux_text("Let me calculate that."),
                faux_tool_call_with_id("calc-1", "calculate", serde_json::json!({"expression": "5 + 3"})),
            ],
            StopReason::ToolUse,
        ),
    ));

    let tool_result = AgentMessage::Standard(Message::ToolResult(pi_ai_rs::ToolResultMessage {
        tool_call_id: "calc-1".to_string(),
        tool_name: "calculate".to_string(),
        content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
            text: "5 + 3 = 8".to_string(),
            text_signature: None,
        })],
        details: None,
        is_error: false,
        timestamp: 0,
    }));

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "You are a helpful assistant.".to_string(),
            model,
            tools: vec![Arc::new(CalculateTool) as Arc<dyn AgentTool>],
            messages: vec![user_msg, assistant_msg, tool_result],
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.continue_().await.unwrap();

    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.len() >= 4);

    let last = agent.state().messages.last().unwrap();
    assert_eq!(last.role(), "assistant");
    assert!(get_text_content(last).contains('8'));

    faux.unregister();
}

#[tokio::test]
async fn continue_drains_follow_up_from_assistant_tail() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![faux_assistant_text("Processed")]);

    let model = faux.get_model().clone();

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "Test".to_string(),
            model,
            messages: vec![
                AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
                    content: pi_ai_rs::UserContent::Text("Initial".to_string()),
                    timestamp: 0,
                })),
                AgentMessage::Standard(Message::Assistant(faux_assistant_text("Initial response"))),
            ],
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.follow_up(AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Queued follow-up".to_string()),
        timestamp: 0,
    })));

    agent.continue_().await.unwrap();

    // The follow-up should have been processed
    let has_follow_up = agent.state().messages.iter().any(|m| {
        if let AgentMessage::Standard(Message::User(um)) = m {
            match &um.content {
                pi_ai_rs::UserContent::Text(t) => t == "Queued follow-up",
                _ => false,
            }
        } else {
            false
        }
    });
    assert!(has_follow_up, "follow-up message should be in messages");

    // Last message should be assistant
    assert_eq!(agent.state().messages.last().unwrap().role(), "assistant");

    faux.unregister();
}

#[tokio::test]
async fn continue_drains_steering_one_at_a_time_from_assistant_tail() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![
        faux_assistant_text("Processed 1"),
        faux_assistant_text("Processed 2"),
    ]);

    let model = faux.get_model().clone();

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "Test".to_string(),
            model,
            messages: vec![
                AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
                    content: pi_ai_rs::UserContent::Text("Initial".to_string()),
                    timestamp: 0,
                })),
                AgentMessage::Standard(Message::Assistant(faux_assistant_text("Initial response"))),
            ],
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.steer(AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Steering 1".to_string()),
        timestamp: 0,
    })));
    agent.steer(AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("Steering 2".to_string()),
        timestamp: 1,
    })));

    agent.continue_().await.unwrap();

    // With one-at-a-time mode, first continue processes Steering 1.
    // The inner loop should pick up Steering 2 via getSteeringMessages.
    let recent_roles: Vec<&str> = agent
        .state()
        .messages
        .iter()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|m| m.role())
        .collect();

    assert_eq!(recent_roles, vec!["user", "assistant", "user", "assistant"]);

    faux.unregister();
}

// ---------------------------------------------------------------------------
// Tests: reset
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reset_clears_state() {
    let faux = register_faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses(vec![faux_assistant_text("Hello")]);

    let mut agent = Agent::new(AgentOptions {
        initial_state: Some(AgentState {
            system_prompt: "Test".to_string(),
            model: faux.get_model().clone(),
            ..Default::default()
        }),
        ..Default::default()
    });

    agent.prompt_text("Hi", None).await.unwrap();
    assert_eq!(agent.state().messages.len(), 2);

    agent.steer(AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
        content: pi_ai_rs::UserContent::Text("s".to_string()),
        timestamp: 0,
    })));

    agent.reset();

    assert!(agent.state().messages.is_empty());
    assert!(!agent.state().is_streaming);
    assert!(agent.state().streaming_message.is_none());
    assert!(agent.state().pending_tool_calls.is_empty());
    assert!(agent.state().error_message.is_none());
    assert!(!agent.has_queued_messages());

    faux.unregister();
}
