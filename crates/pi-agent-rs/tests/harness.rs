//! Test harness for pi-agent-rs, mirroring the TypeScript harness.ts.
//!
//! Provides a `Harness` that wires up a faux provider, agent loop config,
//! and event capture so integration tests can assert on agent behavior
//! without any real LLM calls.

use std::sync::Arc;

use futures::StreamExt;
use pi_ai_rs::{
    register_faux_provider, AssistantMessage, FauxProviderRegistration, Model,
    RegisterFauxProviderOptions,
};
use pi_agent_rs::{
    agent_loop, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool,
    BoxFuture, Message, ToolExecutionMode,
};
use tokio_util::sync::CancellationToken;

/// The test harness — owns the faux provider registration, captures events,
/// and provides helpers for driving the agent loop.
pub struct Harness {
    pub faux: FauxProviderRegistration,
    pub model: Model,
    pub events: Vec<AgentEvent>,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub system_prompt: String,
}

impl Harness {
    /// Create a new harness with default settings.
    pub fn new() -> Self {
        let faux = register_faux_provider(RegisterFauxProviderOptions::default());
        let model = faux.get_model().clone();
        Self {
            faux,
            model,
            events: Vec::new(),
            tools: Vec::new(),
            system_prompt: "You are a test assistant.".to_string(),
        }
    }

    /// Set the responses the faux provider will return.
    pub fn set_responses(&self, responses: Vec<AssistantMessage>) {
        self.faux.set_responses(responses);
    }

    /// Add a tool to the harness.
    pub fn add_tool(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.push(tool);
    }

    /// Run the agent loop with the given prompt and capture all events.
    /// Returns the final messages produced by the loop.
    pub async fn prompt(&mut self, text: &str) -> Vec<AgentMessage> {
        let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
            content: pi_ai_rs::UserContent::Text(text.to_string()),
            timestamp: 0,
        }));

        let tool_defs: Vec<pi_ai_rs::Tool> = self
            .tools
            .iter()
            .map(|t| t.as_tool_definition())
            .collect();

        let context = AgentContext {
            system_prompt: self.system_prompt.clone(),
            messages: Vec::new(),
            tool_definitions: tool_defs,
            tools: self.tools.clone(),
        };

        let model = self.model.clone();
        let config = AgentLoopConfig {
            model,
            stream_options: Default::default(),
            convert_to_llm: std::sync::Arc::new(|messages: Vec<AgentMessage>| {
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
        let mut stream = agent_loop(vec![user_msg], context, config, cancel);

        self.events.clear();
        while let Some(event) = stream.next().await {
            self.events.push(event);
        }

        // Extract messages from AgentEnd event
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::AgentEnd { messages } => Some(messages.clone()),
                _ => None,
            })
            .last()
            .unwrap_or_default()
    }

    /// Filter events by type.
    pub fn events_of_type(&self, type_name: &str) -> Vec<&AgentEvent> {
        self.events
            .iter()
            .filter(|e| event_type_name(e) == type_name)
            .collect()
    }

    /// Get all event type names in order.
    pub fn event_types(&self) -> Vec<&str> {
        self.events.iter().map(event_type_name).collect()
    }

    /// Cleanup — unregister the faux provider.
    pub fn cleanup(self) {
        self.faux.unregister();
    }
}

fn event_type_name(event: &AgentEvent) -> &str {
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

/// Extract text content from an AgentMessage.
pub fn get_message_text(msg: &AgentMessage) -> String {
    match msg {
        AgentMessage::Standard(Message::Assistant(am)) => {
            am.content
                .iter()
                .filter_map(|c| match c {
                    pi_ai_rs::AssistantContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        AgentMessage::Standard(Message::User(um)) => match &um.content {
            pi_ai_rs::UserContent::Text(t) => t.clone(),
            pi_ai_rs::UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    pi_ai_rs::UserContentPart::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        },
        AgentMessage::Standard(Message::ToolResult(tr)) => {
            tr.content
                .iter()
                .filter_map(|c| match c {
                    pi_ai_rs::Content::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        AgentMessage::Custom(_) => String::new(),
    }
}
