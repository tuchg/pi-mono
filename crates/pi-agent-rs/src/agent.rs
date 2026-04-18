use std::sync::Arc;

use pi_ai_rs::{ThinkingBudgets, Transport};
use tokio_util::sync::CancellationToken;

use crate::types::{
    AfterToolCallContext, AfterToolCallResult, AgentMessage, AgentState, AgentTool,
    BeforeToolCallContext, BeforeToolCallResult, BoxFuture, Message,
    QueueMode, ToolExecutionMode,
};

// ---------------------------------------------------------------------------
// Pending message queue
// ---------------------------------------------------------------------------

struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> Self {
        Self {
            messages: Vec::new(),
            mode,
        }
    }

    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    fn drain(&mut self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => std::mem::take(&mut self.messages),
            QueueMode::OneAtATime => {
                if self.messages.is_empty() {
                    Vec::new()
                } else {
                    vec![self.messages.remove(0)]
                }
            }
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

// ---------------------------------------------------------------------------
// Agent options
// ---------------------------------------------------------------------------

/// Configuration for constructing an [`Agent`].
pub struct AgentOptions {
    pub initial_state: Option<AgentState>,
    pub convert_to_llm:
        Option<Box<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync>>,
    pub transform_context:
        Option<Box<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>>,
    pub get_api_key:
        Option<Box<dyn Fn(String) -> BoxFuture<'static, Option<String>> + Send + Sync>>,
    pub before_tool_call: Option<
        Box<
            dyn Fn(BeforeToolCallContext) -> BoxFuture<'static, Option<BeforeToolCallResult>>
                + Send
                + Sync,
        >,
    >,
    pub after_tool_call: Option<
        Box<
            dyn Fn(AfterToolCallContext) -> BoxFuture<'static, Option<AfterToolCallResult>>
                + Send
                + Sync,
        >,
    >,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Option<Transport>,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: ToolExecutionMode,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            get_api_key: None,
            before_tool_call: None,
            after_tool_call: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::All,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            max_retry_delay_ms: None,
            tool_execution: ToolExecutionMode::Sequential,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Stateful wrapper around the low-level agent loop.
///
/// `Agent` owns the current transcript, emits lifecycle events, executes
/// tools, and exposes queueing APIs for steering and follow-up messages.
pub struct Agent {
    state: AgentState,
    tools: Vec<Arc<dyn AgentTool>>,
    steering_queue: PendingMessageQueue,
    follow_up_queue: PendingMessageQueue,
    convert_to_llm:
        Box<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync>,
    session_id: Option<String>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Option<Transport>,
    max_retry_delay_ms: Option<u64>,
    tool_execution: ToolExecutionMode,
    cancel: CancellationToken,
}

impl Agent {
    /// Create a new agent with the given options.
    pub fn new(options: AgentOptions) -> Self {
        let convert_to_llm = options.convert_to_llm.unwrap_or_else(|| {
            Box::new(|messages: Vec<AgentMessage>| {
                Box::pin(async move {
                    messages
                        .iter()
                        .filter_map(|m| match m {
                            AgentMessage::Standard(msg) => Some(msg.clone()),
                            _ => None,
                        })
                        .collect()
                }) as BoxFuture<'static, Vec<Message>>
            })
        });

        Self {
            state: options.initial_state.unwrap_or_default(),
            tools: Vec::new(),
            steering_queue: PendingMessageQueue::new(options.steering_mode),
            follow_up_queue: PendingMessageQueue::new(options.follow_up_mode),
            convert_to_llm,
            session_id: options.session_id,
            thinking_budgets: options.thinking_budgets,
            transport: options.transport,
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options.tool_execution,
            cancel: CancellationToken::new(),
        }
    }

    /// Access the current agent state.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Mutable access to state.
    pub fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    /// Register a tool.
    pub fn add_tool(&mut self, tool: Arc<dyn AgentTool>) {
        self.tools.push(tool);
    }

    /// Queue a steering message (injected before the next assistant turn).
    pub fn steer(&mut self, message: AgentMessage) {
        self.steering_queue.enqueue(message);
    }

    /// Queue a follow-up message (processed after the current loop ends).
    pub fn follow_up(&mut self, message: AgentMessage) {
        self.follow_up_queue.enqueue(message);
    }

    /// Abort the currently running loop.
    pub fn abort(&mut self) {
        self.cancel.cancel();
        self.cancel = CancellationToken::new();
    }

    /// Interrupt the current streaming turn gracefully (issue #3197).
    pub fn interrupt(&self) {
        // In the full implementation this would use a separate
        // CancellationToken that only interrupts the current turn,
        // not the entire loop.
        self.cancel.cancel();
    }
}
