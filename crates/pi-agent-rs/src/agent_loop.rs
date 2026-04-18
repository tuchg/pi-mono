use futures::StreamExt;
use pi_ai_rs::event_stream::{event_stream, EventStream, EventStreamSender};
use tokio_util::sync::CancellationToken;

use crate::types::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, Message,
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
///
/// The prompts are appended to the context and the loop streams events back
/// through the returned `AgentEventStream`.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
) -> AgentEventStream {
    let (mut sender, receiver) = create_agent_stream();

    tokio::spawn(async move {
        let messages = run_agent_loop(prompts, context, config, &mut sender, cancel).await;
        sender.end(messages);
    });

    receiver
}

/// Continue an existing agent loop without adding new messages.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
) -> AgentEventStream {
    let (mut sender, receiver) = create_agent_stream();

    tokio::spawn(async move {
        let messages =
            run_agent_loop_continue(context, config, &mut sender, cancel).await;
        sender.end(messages);
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

/// Main loop logic shared by `agent_loop` and `agent_loop_continue`.
///
/// Structure mirrors the TypeScript implementation:
///   - Outer loop: continues when follow-up messages arrive
///   - Inner loop: process tool calls and steering messages
async fn run_loop(
    context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    sender: &mut AgentEventStreamSender,
    cancel: CancellationToken,
) {
    let mut first_turn = true;

    // Check for steering messages at start
    let mut pending: Vec<AgentMessage> = if let Some(ref get_steering) = config.get_steering_messages {
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

            // Inject pending steering messages
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

            // Convert context to LLM messages
            let llm_messages = (config.convert_to_llm)(context.messages.clone()).await;

            // Stream the assistant response
            let ai_context = pi_ai_rs::Context {
                system_prompt: Some(context.system_prompt.clone()),
                messages: llm_messages,
                tools: if context.tool_definitions.is_empty() {
                    None
                } else {
                    Some(context.tool_definitions.clone())
                },
            };

            let stream_result =
                pi_ai_rs::stream_simple(&config.model, ai_context, config.stream_options.clone());

            let mut event_stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("stream error: {e}");
                    sender.push(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    });
                    return;
                }
            };

            // Consume events and forward them
            let mut final_message: Option<pi_ai_rs::AssistantMessage> = None;

            while let Some(event) = event_stream.next().await {
                let agent_msg = AgentMessage::Standard(Message::Assistant(
                    match &event {
                        pi_ai_rs::AssistantMessageEvent::Done { message, .. } => message.clone(),
                        pi_ai_rs::AssistantMessageEvent::Error { error, .. } => error.clone(),
                        _ => {
                            // For intermediate events, extract the partial
                            // We emit message_update events
                            pi_ai_rs::AssistantMessage::default()
                        }
                    },
                ));

                sender.push(AgentEvent::MessageUpdate {
                    message: agent_msg,
                    assistant_message_event: event.clone(),
                });

                if let Some(msg) = event.into_final_message() {
                    final_message = Some(msg);
                }
            }

            let assistant = match final_message {
                Some(m) => m,
                None => {
                    sender.push(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    });
                    return;
                }
            };

            let stop = assistant.stop_reason;
            let agent_msg =
                AgentMessage::Standard(Message::Assistant(assistant.clone()));

            sender.push(AgentEvent::MessageEnd {
                message: agent_msg.clone(),
            });

            context.messages.push(agent_msg.clone());
            new_messages.push(agent_msg.clone());

            // Check if there are tool calls to execute
            let tool_calls: Vec<_> = assistant
                .content
                .iter()
                .filter_map(|c| match c {
                    pi_ai_rs::AssistantContent::ToolCall(tc) => Some(tc.clone()),
                    _ => None,
                })
                .collect();

            if tool_calls.is_empty() || stop != pi_ai_rs::StopReason::ToolUse {
                has_more_tool_calls = false;

                sender.push(AgentEvent::TurnEnd {
                    message: agent_msg,
                    tool_results: Vec::new(),
                });
            } else {
                // Execute tool calls (sequential by default in this scaffold)
                let mut tool_results = Vec::new();

                for tc in &tool_calls {
                    sender.push(AgentEvent::ToolExecutionStart {
                        tool_call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        args: tc.arguments.clone(),
                    });

                    // In the full implementation, tools are looked up and
                    // executed here. For now we emit a placeholder result.
                    let result_msg = pi_ai_rs::ToolResultMessage {
                        tool_call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                            text: "tool not implemented".to_string(),
                            text_signature: None,
                        })],
                        details: None,
                        is_error: true,
                        timestamp: 0,
                    };

                    sender.push(AgentEvent::ToolExecutionEnd {
                        tool_call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        result: serde_json::to_value(&result_msg).unwrap_or_default(),
                        is_error: true,
                    });

                    let agent_result =
                        AgentMessage::Standard(Message::ToolResult(result_msg.clone()));
                    context.messages.push(agent_result.clone());
                    new_messages.push(agent_result);
                    tool_results.push(result_msg);
                }

                sender.push(AgentEvent::TurnEnd {
                    message: agent_msg,
                    tool_results,
                });

                has_more_tool_calls = true;
            }

            // Check for new steering messages
            if let Some(ref get_steering) = config.get_steering_messages {
                pending = (get_steering)().await;
            }
        }

        // Check for follow-up messages
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
