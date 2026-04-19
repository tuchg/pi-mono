use futures::StreamExt;
use pi_ai_rs::event_stream::{event_stream, EventStream, EventStreamSender};
use pi_ai_rs::{AssistantContent, AssistantMessage, AssistantMessageEvent, StopReason};
use tokio_util::sync::CancellationToken;

use crate::types::{
    AfterToolCallContext, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage,
    AgentToolResult, BeforeToolCallContext, Message, ToolExecutionMode,
};

/// Sender half for agent event streams.
pub type AgentEventStreamSender = EventStreamSender<AgentEvent, Vec<AgentMessage>>;

/// Receiver half for agent event streams.
pub type AgentEventStream = EventStream<AgentEvent, Vec<AgentMessage>>;

/// Create a sender/receiver pair for agent events.
pub fn create_agent_stream() -> (AgentEventStreamSender, AgentEventStream) {
    event_stream(
        |event: &AgentEvent| event.is_terminal(),
        |event: AgentEvent| match event {
            AgentEvent::AgentEnd { messages } => messages,
            _ => Vec::new(),
        },
    )
}

/// Start an agent loop with new prompt messages.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
) -> AgentEventStream {
    let (mut sender, receiver) = create_agent_stream();

    tokio::spawn(async move {
        let messages = run_agent_loop(prompts, context, config, &mut sender, cancel).await;
        sender.end(Some(messages));
    });

    receiver
}

/// Continue an existing agent loop without adding new messages.
///
/// Validates that the context is non-empty and that the last message is not
/// an assistant message (mirrors the TypeScript `agentLoopContinue` guard).
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
) -> AgentEventStream {
    if context.messages.is_empty() {
        let (mut sender, receiver) = create_agent_stream();
        sender.end(Some(Vec::new()));
        tracing::error!("agent_loop_continue: context has no messages");
        return receiver;
    }

    if context.messages.last().map(|m| m.role()) == Some("assistant") {
        let (mut sender, receiver) = create_agent_stream();
        sender.end(Some(Vec::new()));
        tracing::error!("agent_loop_continue: last message role is 'assistant'");
        return receiver;
    }

    let (mut sender, receiver) = create_agent_stream();

    tokio::spawn(async move {
        let messages = run_agent_loop_continue(context, config, &mut sender, cancel).await;
        sender.end(Some(messages));
    });

    receiver
}

/// Core loop implementation — adds prompts, then runs the inner loop.
async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    mut context: AgentContext,
    config: AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: CancellationToken,
) -> Vec<AgentMessage> {
    let mut new_messages: Vec<AgentMessage> = prompts.clone();
    context.messages.extend(prompts.clone());

    sender.push(AgentEvent::AgentStart);
    sender.push(AgentEvent::TurnStart);

    for prompt in &prompts {
        sender.push(AgentEvent::MessageStart {
            message: prompt.clone(),
        });
        sender.push(AgentEvent::MessageEnd {
            message: prompt.clone(),
        });
    }

    run_loop(&mut context, &mut new_messages, &config, sender, cancel).await;
    new_messages
}

/// Core loop implementation — continues from existing context.
async fn run_agent_loop_continue(
    mut context: AgentContext,
    config: AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: CancellationToken,
) -> Vec<AgentMessage> {
    let mut new_messages: Vec<AgentMessage> = Vec::new();

    sender.push(AgentEvent::AgentStart);
    sender.push(AgentEvent::TurnStart);

    run_loop(&mut context, &mut new_messages, &config, sender, cancel).await;
    new_messages
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Main loop logic shared by `agent_loop` and `agent_loop_continue`.
///
/// Structure mirrors the TypeScript implementation:
///   - Outer loop: continues when follow-up messages arrive
///   - Inner loop: processes tool calls and steering messages
async fn run_loop(
    context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: CancellationToken,
) {
    let mut first_turn = true;
    let mut pending: Vec<AgentMessage> = if let Some(ref get_steering) =
        config.get_steering_messages
    {
        (get_steering)().await
    } else {
        Vec::new()
    };

    loop {
        let mut has_more_tool_calls = true;

        while has_more_tool_calls || !pending.is_empty() {
            if cancel.is_cancelled() {
                sender.push(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                return;
            }

            if !first_turn {
                sender.push(AgentEvent::TurnStart);
            } else {
                first_turn = false;
            }

            // Inject pending messages (steering or follow-up).
            if !pending.is_empty() {
                for msg in pending.drain(..) {
                    sender.push(AgentEvent::MessageStart {
                        message: msg.clone(),
                    });
                    sender.push(AgentEvent::MessageEnd {
                        message: msg.clone(),
                    });
                    context.messages.push(msg.clone());
                    new_messages.push(msg);
                }
            }

            // Stream assistant response.
            let assistant = match stream_assistant_response(context, config, sender, &cancel).await {
                Some(msg) => msg,
                None => {
                    sender.push(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    });
                    return;
                }
            };

            let agent_msg =
                AgentMessage::Standard(Message::Assistant(assistant.clone()));

            new_messages.push(agent_msg.clone());

            // TS: early exit when stopReason is "error" or "aborted".
            if assistant.stop_reason == StopReason::Error
                || assistant.stop_reason == StopReason::Aborted
            {
                sender.push(AgentEvent::TurnEnd {
                    message: agent_msg,
                    tool_results: Vec::new(),
                });
                sender.push(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                return;
            }

            // Collect tool calls.
            let tool_calls: Vec<_> = assistant
                .content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::ToolCall(tc) => Some(tc.clone()),
                    _ => None,
                })
                .collect();

            has_more_tool_calls = !tool_calls.is_empty();

            let tool_results = if has_more_tool_calls {
                let results =
                    execute_tool_calls(context, &assistant, &tool_calls, config, sender, &cancel).await;
                // Add tool result messages to context and new_messages.
                for result in &results {
                    let tr_msg = AgentMessage::Standard(Message::ToolResult(result.clone()));
                    context.messages.push(tr_msg.clone());
                    new_messages.push(tr_msg);
                }
                results
            } else {
                Vec::new()
            };

            sender.push(AgentEvent::TurnEnd {
                message: agent_msg,
                tool_results,
            });

            // Poll for steering messages.
            pending = if let Some(ref get_steering) = config.get_steering_messages {
                (get_steering)().await
            } else {
                Vec::new()
            };
        }

        // Check for follow-up messages.
        let follow_ups = if let Some(ref get_follow_up) = config.get_follow_up_messages {
            (get_follow_up)().await
        } else {
            Vec::new()
        };

        if follow_ups.is_empty() {
            break;
        }
        pending = follow_ups;
    }

    sender.push(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    });
}

// ---------------------------------------------------------------------------
// Assistant streaming
// ---------------------------------------------------------------------------

/// Stream an assistant response and emit `message_start` / `message_update` /
/// `message_end` events matching the TypeScript `streamAssistantResponse`.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: &CancellationToken,
) -> Option<AssistantMessage> {
    // Apply context transform if configured (AgentMessage[] → AgentMessage[]).
    let messages = if let Some(ref transform) = config.transform_context {
        (transform)(context.messages.clone(), cancel.clone()).await
    } else {
        context.messages.clone()
    };

    // Convert to LLM-compatible messages.
    let llm_messages = (config.convert_to_llm)(messages).await;

    // Resolve API key.
    let mut stream_options = config.stream_options.clone();
    if let Some(ref get_api_key) = config.get_api_key {
        if let Some(key) = (get_api_key)(config.model.provider.clone()).await {
            stream_options.api_key = Some(key);
        }
    }

    let ai_context = pi_ai_rs::Context {
        system_prompt: Some(context.system_prompt.clone()),
        messages: llm_messages,
        tools: if context.tool_definitions.is_empty() {
            None
        } else {
            Some(context.tool_definitions.clone())
        },
    };

    // Use custom stream function if provided, otherwise default to stream_simple.
    let mut event_stream = if let Some(ref stream_fn) = config.stream_fn {
        match (stream_fn)(&config.model, ai_context, stream_options) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("custom stream_fn error: {e}");
                return None;
            }
        }
    } else {
        match pi_ai_rs::stream_simple(&config.model, ai_context, stream_options) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("stream_simple error: {e}");
                return None;
            }
        }
    };

    let mut added_partial = false;

    while let Some(event) = event_stream.next().await {
        match event {
            AssistantMessageEvent::Start { partial } => {
                context
                    .messages
                    .push(AgentMessage::Standard(Message::Assistant(partial.clone())));
                added_partial = true;
                sender.push(AgentEvent::MessageStart {
                    message: AgentMessage::Standard(Message::Assistant(partial)),
                });
            }

            AssistantMessageEvent::Done { message, .. } => {
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = AgentMessage::Standard(Message::Assistant(message.clone()));
                    }
                } else {
                    context
                        .messages
                        .push(AgentMessage::Standard(Message::Assistant(message.clone())));
                    sender.push(AgentEvent::MessageStart {
                        message: AgentMessage::Standard(Message::Assistant(message.clone())),
                    });
                }
                sender.push(AgentEvent::MessageEnd {
                    message: AgentMessage::Standard(Message::Assistant(message.clone())),
                });
                return Some(message);
            }

            AssistantMessageEvent::Error { error, .. } => {
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = AgentMessage::Standard(Message::Assistant(error.clone()));
                    }
                } else {
                    context
                        .messages
                        .push(AgentMessage::Standard(Message::Assistant(error.clone())));
                    sender.push(AgentEvent::MessageStart {
                        message: AgentMessage::Standard(Message::Assistant(error.clone())),
                    });
                }
                sender.push(AgentEvent::MessageEnd {
                    message: AgentMessage::Standard(Message::Assistant(error.clone())),
                });
                return Some(error);
            }

            // Delta events — update partial in context and emit message_update.
            delta_event => {
                let partial = partial_from_delta(&delta_event);
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = AgentMessage::Standard(Message::Assistant(partial.clone()));
                    }
                }
                sender.push(AgentEvent::MessageUpdate {
                    message: AgentMessage::Standard(Message::Assistant(partial)),
                    assistant_message_event: delta_event,
                });
            }
        }
    }

    None
}

/// Extract the `partial` field from a non-terminal `AssistantMessageEvent`.
fn partial_from_delta(event: &AssistantMessageEvent) -> AssistantMessage {
    match event {
        AssistantMessageEvent::TextStart { partial, .. }
        | AssistantMessageEvent::TextDelta { partial, .. }
        | AssistantMessageEvent::TextEnd { partial, .. }
        | AssistantMessageEvent::ThinkingStart { partial, .. }
        | AssistantMessageEvent::ThinkingDelta { partial, .. }
        | AssistantMessageEvent::ThinkingEnd { partial, .. }
        | AssistantMessageEvent::ToolcallStart { partial, .. }
        | AssistantMessageEvent::ToolcallDelta { partial, .. }
        | AssistantMessageEvent::ToolcallEnd { partial, .. } => partial.clone(),
        // Start / Done / Error are handled by the caller.
        AssistantMessageEvent::Start { partial } => partial.clone(),
        AssistantMessageEvent::Done { message, .. } => message.clone(),
        AssistantMessageEvent::Error { error, .. } => error.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

async fn execute_tool_calls(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tool_calls: &[pi_ai_rs::ToolCall],
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: &CancellationToken,
) -> Vec<pi_ai_rs::ToolResultMessage> {
    // Determine if any tool requests sequential execution.
    let has_sequential = tool_calls.iter().any(|tc| {
        context
            .tools
            .iter()
            .find(|t| t.name() == tc.name)
            .and_then(|t| t.execution_mode())
            == Some(ToolExecutionMode::Sequential)
    });

    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential {
        execute_tool_calls_sequential(context, assistant, tool_calls, config, sender, cancel).await
    } else {
        execute_tool_calls_parallel(context, assistant, tool_calls, config, sender, cancel).await
    }
}

async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tool_calls: &[pi_ai_rs::ToolCall],
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: &CancellationToken,
) -> Vec<pi_ai_rs::ToolResultMessage> {
    let mut results = Vec::new();

    for tc in tool_calls {
        sender.push(AgentEvent::ToolExecutionStart {
            tool_call_id: tc.id.clone(),
            tool_name: tc.name.clone(),
            args: tc.arguments.clone(),
        });

        let outcome = prepare_and_execute(context, assistant, tc, config, sender, cancel).await;
        let result_msg = emit_tool_outcome(tc, outcome, sender).await;
        results.push(result_msg);
    }

    results
}

async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tool_calls: &[pi_ai_rs::ToolCall],
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: &CancellationToken,
) -> Vec<pi_ai_rs::ToolResultMessage> {
    // Emit tool_execution_start for all calls, prepare them, then run allowed
    // ones concurrently.  Finalize in original order (mirrors TS parallel impl).
    let mut immediate: Vec<(usize, ToolOutcome)> = Vec::new();
    let mut deferred: Vec<(usize, pi_ai_rs::ToolCall, PreparedToolCall)> = Vec::new();

    for (i, tc) in tool_calls.iter().enumerate() {
        sender.push(AgentEvent::ToolExecutionStart {
            tool_call_id: tc.id.clone(),
            tool_name: tc.name.clone(),
            args: tc.arguments.clone(),
        });

        match prepare_tool_call(context, assistant, tc, config, cancel).await {
            PrepareResult::Immediate(outcome) => immediate.push((i, outcome)),
            PrepareResult::Prepared(prepared) => deferred.push((i, tc.clone(), prepared)),
        }
    }

    // Execute deferred calls concurrently.
    // Note: we cannot emit tool_execution_update events from parallel tasks
    // back through the sender (which requires &mut). We collect them post-hoc.
    let deferred_futures: Vec<_> = deferred
        .iter()
        .map(|(_, tc, prepared)| execute_prepared(tc, prepared, cancel))
        .collect();

    let deferred_outcomes: Vec<ToolOutcome> = futures::future::join_all(deferred_futures).await;

    // Reassemble in original order.
    let total = tool_calls.len();
    let mut ordered: Vec<Option<ToolOutcome>> = (0..total).map(|_| None).collect();
    for (i, outcome) in immediate {
        ordered[i] = Some(outcome);
    }
    for ((i, _, _), outcome) in deferred.iter().zip(deferred_outcomes) {
        ordered[*i] = Some(outcome);
    }

    // Finalize with after_tool_call hook and emit events.
    let mut results = Vec::new();
    for (i, outcome_opt) in ordered.into_iter().enumerate() {
        let outcome = outcome_opt.expect("every tool call must have an outcome");
        let tc = &tool_calls[i];
        // Find the validated args from the deferred prepared call if available.
        let validated_args = deferred
            .iter()
            .find(|(idx, _, _)| *idx == i)
            .map(|(_, _, p)| p.args.clone());
        let outcome = apply_after_hook(context, assistant, tc, validated_args, outcome, config, cancel).await;
        let result_msg = emit_tool_outcome(tc, outcome, sender).await;
        results.push(result_msg);
    }

    results
}

// ---------------------------------------------------------------------------
// Tool call preparation and execution helpers
// ---------------------------------------------------------------------------

struct PreparedToolCall {
    tool: std::sync::Arc<dyn crate::types::AgentTool>,
    args: serde_json::Value,
}

struct ToolOutcome {
    result: AgentToolResult,
    is_error: bool,
}

enum PrepareResult {
    Immediate(ToolOutcome),
    Prepared(PreparedToolCall),
}

/// Prepare + execute a tool call (used in sequential mode).
async fn prepare_and_execute(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tc: &pi_ai_rs::ToolCall,
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: &CancellationToken,
) -> ToolOutcome {
    let prepare = prepare_tool_call(context, assistant, tc, config, cancel).await;
    match prepare {
        PrepareResult::Immediate(outcome) => outcome,
        PrepareResult::Prepared(prepared) => {
            let validated_args = prepared.args.clone();
            let outcome = execute_prepared_with_updates(tc, &prepared, sender, cancel).await;
            apply_after_hook(context, assistant, tc, Some(validated_args), outcome, config, cancel).await
        }
    }
}

/// Validate arguments, apply `prepare_arguments`, and run `before_tool_call`.
async fn prepare_tool_call(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tc: &pi_ai_rs::ToolCall,
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
) -> PrepareResult {
    // Find the tool.
    let tool = match context.tools.iter().find(|t| t.name() == tc.name) {
        Some(t) => t.clone(),
        None => {
            return PrepareResult::Immediate(ToolOutcome {
                result: error_result(&format!("Unknown tool: {}", tc.name)),
                is_error: true,
            });
        }
    };

    // Apply prepare_arguments shim.
    let raw_args = tc.arguments.clone();
    let args = tool.prepare_arguments(raw_args.clone()).unwrap_or(raw_args);

    // Validate against schema.
    let tool_def = tool.as_tool_definition();
    let args = match pi_ai_rs::utils::validation::validate_tool_arguments(&tool_def, &args) {
        Ok(validated) => validated,
        Err(e) => {
            return PrepareResult::Immediate(ToolOutcome {
                result: error_result(&e.to_string()),
                is_error: true,
            });
        }
    };

    // before_tool_call hook.
    if let Some(ref before) = config.before_tool_call {
        let before_ctx = BeforeToolCallContext {
            assistant_message: assistant.clone(),
            tool_call: tc.clone(),
            args: args.clone(),
            context: AgentContext {
                system_prompt: context.system_prompt.clone(),
                messages: context.messages.clone(),
                tool_definitions: context.tool_definitions.clone(),
                tools: context.tools.clone(),
            },
        };
        match (before)(before_ctx, cancel.clone()).await {
            Some(result) if result.block == Some(true) => {
                let reason = result
                    .reason
                    .unwrap_or_else(|| "Tool execution was blocked".to_string());
                return PrepareResult::Immediate(ToolOutcome {
                    result: error_result(&reason),
                    is_error: true,
                });
            }
            _ => {}
        }
    }

    PrepareResult::Prepared(PreparedToolCall { tool, args })
}

/// Execute a prepared tool call without update events (used in parallel mode).
async fn execute_prepared(
    tc: &pi_ai_rs::ToolCall,
    prepared: &PreparedToolCall,
    cancel: &CancellationToken,
) -> ToolOutcome {
    match prepared
        .tool
        .execute(&tc.id, prepared.args.clone(), cancel.clone(), None)
        .await
    {
        Ok(result) => ToolOutcome {
            result,
            is_error: false,
        },
        Err(e) => ToolOutcome {
            result: error_result(&e.to_string()),
            is_error: true,
        },
    }
}

/// Execute a prepared tool call with update events (used in sequential mode).
///
/// Mirrors TS `executePreparedToolCall` which passes an `onUpdate` callback
/// that emits `tool_execution_update` events.
async fn execute_prepared_with_updates(
    tc: &pi_ai_rs::ToolCall,
    prepared: &PreparedToolCall,
    sender: &mut AgentEventStreamSender,
    cancel: &CancellationToken,
) -> ToolOutcome {
    let tool_call_id = tc.id.clone();
    let tool_name = tc.name.clone();
    let tool_args = tc.arguments.clone();

    let sender_tool_call_id = tool_call_id.clone();
    let sender_tool_name = tool_name.clone();
    let sender_tool_args = tool_args.clone();

    // Collect update events for emission after execution.
    let updates: std::sync::Arc<std::sync::Mutex<Vec<AgentToolResult>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let updates_clone = updates.clone();

    let on_update: crate::types::AgentToolUpdateCallback = Box::new(move |partial_result| {
        updates_clone
            .lock()
            .expect("updates lock poisoned")
            .push(partial_result.clone());
    });

    let result = prepared
        .tool
        .execute(&tc.id, prepared.args.clone(), cancel.clone(), Some(on_update))
        .await;

    // Emit collected update events.
    let collected_updates = updates.lock().expect("updates lock poisoned").clone();
    for update in collected_updates {
        sender.push(AgentEvent::ToolExecutionUpdate {
            tool_call_id: sender_tool_call_id.clone(),
            tool_name: sender_tool_name.clone(),
            args: sender_tool_args.clone(),
            partial_result: serde_json::to_value(&update).unwrap_or_default(),
        });
    }

    match result {
        Ok(result) => ToolOutcome {
            result,
            is_error: false,
        },
        Err(e) => ToolOutcome {
            result: error_result(&e.to_string()),
            is_error: true,
        },
    }
}

/// Apply the `after_tool_call` hook and merge any overrides.
///
/// `validated_args` should be the args after `prepare_arguments` + schema
/// validation, matching the TS behavior (TS passes `prepared.args`).
async fn apply_after_hook(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tc: &pi_ai_rs::ToolCall,
    validated_args: Option<serde_json::Value>,
    mut outcome: ToolOutcome,
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
) -> ToolOutcome {
    if let Some(ref after) = config.after_tool_call {
        let after_ctx = AfterToolCallContext {
            assistant_message: assistant.clone(),
            tool_call: tc.clone(),
            // Use validated args if available, otherwise fall back to raw args.
            args: validated_args.unwrap_or_else(|| tc.arguments.clone()),
            result: outcome.result.clone(),
            is_error: outcome.is_error,
            context: AgentContext {
                system_prompt: context.system_prompt.clone(),
                messages: context.messages.clone(),
                tool_definitions: context.tool_definitions.clone(),
                tools: context.tools.clone(),
            },
        };
        // TS wraps afterToolCall in try-catch; replicate with catch_unwind-style error handling.
        match (after)(after_ctx, cancel.clone()).await {
            Some(overrides) => {
                if let Some(content) = overrides.content {
                    outcome.result.content = content;
                }
                if let Some(details) = overrides.details {
                    outcome.result.details = details;
                }
                if let Some(err) = overrides.is_error {
                    outcome.is_error = err;
                }
            }
            None => {}
        }
    }
    outcome
}

/// Emit `tool_execution_end` + `message_start` / `message_end` for a tool
/// result (mirrors the TypeScript `emitToolCallOutcome` function).
async fn emit_tool_outcome(
    tc: &pi_ai_rs::ToolCall,
    outcome: ToolOutcome,
    sender: &mut AgentEventStreamSender,
) -> pi_ai_rs::ToolResultMessage {
    sender.push(AgentEvent::ToolExecutionEnd {
        tool_call_id: tc.id.clone(),
        tool_name: tc.name.clone(),
        result: serde_json::to_value(&outcome.result).unwrap_or_default(),
        is_error: outcome.is_error,
    });

    let result_msg = pi_ai_rs::ToolResultMessage {
        tool_call_id: tc.id.clone(),
        tool_name: tc.name.clone(),
        content: outcome.result.content,
        details: Some(outcome.result.details),
        is_error: outcome.is_error,
        timestamp: chrono::Utc::now().timestamp_millis() as u64,
    };

    let agent_result = AgentMessage::Standard(Message::ToolResult(result_msg.clone()));

    sender.push(AgentEvent::MessageStart {
        message: agent_result.clone(),
    });
    sender.push(AgentEvent::MessageEnd {
        message: agent_result,
    });

    result_msg
}

/// Create an error `AgentToolResult` with a single text content item.
fn error_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
            text: message.to_string(),
            text_signature: None,
        })],
        details: serde_json::json!({}),
    }
}
