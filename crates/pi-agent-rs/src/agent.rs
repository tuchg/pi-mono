use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use pi_ai_rs::{ThinkingBudgets, Transport};
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{agent_loop, agent_loop_continue};
use crate::types::{
    AfterToolCallContext, AfterToolCallResult, AgentContext, AgentEvent, AgentLoopConfig,
    AgentMessage, AgentState, AgentTool, AfterToolCallFn, BeforeToolCallContext,
    BeforeToolCallResult, BeforeToolCallFn, BoxFuture, ConvertToLlmFn, GetApiKeyFn,
    GetMessagesFn, Message, QueueMode, StreamFn, ToolExecutionMode, TransformContextFn,
};

// ---------------------------------------------------------------------------
// Pending message queue
// ---------------------------------------------------------------------------

struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    pub mode: QueueMode,
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
// Listener type
// ---------------------------------------------------------------------------

/// A subscriber to agent lifecycle events.
///
/// Listeners are called in subscription order after each event.  They receive
/// the event and the cancellation token for the current run.
pub type AgentListenerFn =
    Arc<dyn Fn(AgentEvent, CancellationToken) -> BoxFuture<'static, ()> + Send + Sync>;

// ---------------------------------------------------------------------------
// Agent options
// ---------------------------------------------------------------------------

/// Configuration for constructing an [`Agent`].
pub struct AgentOptions {
    pub initial_state: Option<AgentState>,
    /// Converts [`AgentMessage`]s to LLM-compatible [`Message`]s.
    /// Defaults to filtering out custom messages.
    pub convert_to_llm: Option<ConvertToLlmFn>,
    /// Optional context transform applied before `convert_to_llm`.
    pub transform_context: Option<TransformContextFn>,
    /// Custom stream function, overriding `pi_ai_rs::stream_simple`.
    pub stream_fn: Option<StreamFn>,
    /// Resolves an API key dynamically for each LLM call.
    pub get_api_key: Option<GetApiKeyFn>,
    /// Called before each tool execution.
    pub before_tool_call: Option<BeforeToolCallFn>,
    /// Called after each tool execution.
    pub after_tool_call: Option<AfterToolCallFn>,
    /// How queued steering messages are drained.
    pub steering_mode: QueueMode,
    /// How queued follow-up messages are drained.
    pub follow_up_mode: QueueMode,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Option<Transport>,
    pub max_retry_delay_ms: Option<u64>,
    /// Tool execution strategy (sequential or parallel).
    pub tool_execution: ToolExecutionMode,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            stream_fn: None,
            get_api_key: None,
            before_tool_call: None,
            after_tool_call: None,
            // TS defaults: both queues drain one message at a time.
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            max_retry_delay_ms: None,
            // TS default is "parallel".
            tool_execution: ToolExecutionMode::Parallel,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Stateful wrapper around the low-level agent loop.
///
/// `Agent` owns the current transcript, emits lifecycle events to registered
/// listeners, executes tools, and exposes queueing APIs for steering and
/// follow-up messages.
///
/// Mirrors the TypeScript `Agent` class from `packages/agent/src/agent.ts`.
pub struct Agent {
    state: AgentState,
    steering_queue: Arc<Mutex<PendingMessageQueue>>,
    follow_up_queue: Arc<Mutex<PendingMessageQueue>>,

    // Public fields — mirror TS public fields on `Agent`.
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
    /// Custom stream function, overriding `pi_ai_rs::stream_simple`.
    pub stream_fn: Option<StreamFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Preferred transport forwarded to the stream function.  Default: `"sse"`.
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: ToolExecutionMode,

    // Listener system
    listeners: Vec<(u64, AgentListenerFn)>,
    next_listener_id: u64,

    // Active run cancellation token (None when idle).
    cancel: Option<CancellationToken>,
    // Active run completion notifier.
    idle_notify: Option<Arc<tokio::sync::Notify>>,
}

impl Agent {
    /// Create a new agent with the given options.
    pub fn new(options: AgentOptions) -> Self {
        let initial_state = options.initial_state.unwrap_or_default();

        let convert_to_llm: ConvertToLlmFn = options.convert_to_llm.unwrap_or_else(|| {
            Arc::new(|messages: Vec<AgentMessage>| {
                Box::pin(async move {
                    messages
                        .into_iter()
                        .filter_map(|m| match m {
                            AgentMessage::Standard(msg) => Some(msg),
                            _ => None,
                        })
                        .collect()
                }) as BoxFuture<'static, Vec<Message>>
            })
        });

        Self {
            state: initial_state,
            steering_queue: Arc::new(Mutex::new(PendingMessageQueue::new(
                options.steering_mode,
            ))),
            follow_up_queue: Arc::new(Mutex::new(PendingMessageQueue::new(
                options.follow_up_mode,
            ))),
            convert_to_llm,
            transform_context: options.transform_context,
            stream_fn: options.stream_fn,
            get_api_key: options.get_api_key,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            session_id: options.session_id,
            thinking_budgets: options.thinking_budgets,
            // TS default transport is "sse".
            transport: options.transport.unwrap_or(Transport::Sse),
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options.tool_execution,
            listeners: Vec::new(),
            next_listener_id: 0,
            cancel: None,
            idle_notify: None,
        }
    }

    // -----------------------------------------------------------------------
    // State access
    // -----------------------------------------------------------------------

    /// Access the current agent state.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Mutable access to state.
    pub fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    // -----------------------------------------------------------------------
    // Tool management
    // -----------------------------------------------------------------------

    /// Register a tool.
    pub fn add_tool(&mut self, tool: Arc<dyn AgentTool>) {
        self.state.tools.push(tool);
    }

    /// Remove all registered tools.
    pub fn clear_tools(&mut self) {
        self.state.tools.clear();
    }

    // -----------------------------------------------------------------------
    // Queue accessors
    // -----------------------------------------------------------------------

    /// Controls how queued steering messages are drained.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.lock().expect("steering queue poisoned").mode = mode;
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.lock().expect("steering queue poisoned").mode
    }

    /// Controls how queued follow-up messages are drained.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.lock().expect("follow-up queue poisoned").mode = mode;
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.lock().expect("follow-up queue poisoned").mode
    }

    /// Queue a message to be injected after the current assistant turn finishes.
    pub fn steer(&self, message: AgentMessage) {
        self.steering_queue
            .lock()
            .expect("steering queue poisoned")
            .enqueue(message);
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue
            .lock()
            .expect("follow-up queue poisoned")
            .enqueue(message);
    }

    /// Remove all queued steering messages.
    pub fn clear_steering_queue(&self) {
        self.steering_queue
            .lock()
            .expect("steering queue poisoned")
            .clear();
    }

    /// Remove all queued follow-up messages.
    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue
            .lock()
            .expect("follow-up queue poisoned")
            .clear();
    }

    /// Remove all queued steering and follow-up messages.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Returns `true` when either queue still contains pending messages.
    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue
            .lock()
            .expect("steering queue poisoned")
            .has_items()
            || self
                .follow_up_queue
                .lock()
                .expect("follow-up queue poisoned")
                .has_items()
    }

    // -----------------------------------------------------------------------
    // Cancellation
    // -----------------------------------------------------------------------

    /// Cancellation token for the current run, if one is active.
    pub fn cancel_token(&self) -> Option<&CancellationToken> {
        self.cancel.as_ref()
    }

    /// Abort the current run, if one is active.
    pub fn abort(&mut self) {
        if let Some(ref cancel) = self.cancel {
            cancel.cancel();
        }
    }

    /// Wait for the current run to complete.
    ///
    /// Resolves immediately if no run is active. Mirrors TS `waitForIdle()`.
    pub async fn wait_for_idle(&self) {
        if let Some(ref notify) = self.idle_notify {
            notify.notified().await;
        }
    }

    // -----------------------------------------------------------------------
    // Listener subscription
    // -----------------------------------------------------------------------

    /// Subscribe to agent lifecycle events.
    ///
    /// Returns an unsubscribe ID that can be passed to [`unsubscribe`].
    ///
    /// Listeners are invoked in subscription order after each event and
    /// receive the event plus the active cancellation token.
    pub fn subscribe(&mut self, listener: AgentListenerFn) -> u64 {
        let id = self.next_listener_id;
        self.next_listener_id += 1;
        self.listeners.push((id, listener));
        id
    }

    /// Remove the listener with the given ID (returned by [`subscribe`]).
    pub fn unsubscribe(&mut self, id: u64) {
        self.listeners.retain(|(lid, _)| *lid != id);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Clear transcript state, runtime state, and queued messages.
    pub fn reset(&mut self) {
        self.state.messages = Vec::new();
        self.state.is_streaming = false;
        self.state.streaming_message = None;
        self.state.pending_tool_calls = HashSet::new();
        self.state.error_message = None;
        self.clear_all_queues();
    }

    // -----------------------------------------------------------------------
    // High-level run API
    // -----------------------------------------------------------------------

    /// Start a new run from a single message.
    pub async fn prompt(&mut self, message: AgentMessage) -> Result<(), anyhow::Error> {
        self.run_prompt_messages(vec![message], false).await
    }

    /// Start a new run from a batch of messages.
    pub async fn prompt_many(
        &mut self,
        messages: Vec<AgentMessage>,
    ) -> Result<(), anyhow::Error> {
        self.run_prompt_messages(messages, false).await
    }

    /// Convenience: start a new run from a plain text string.
    ///
    /// Creates a user message with the given text and optional images.
    /// Mirrors the TS `prompt(input: string, images?: ImageContent[])` overload.
    pub async fn prompt_text(
        &mut self,
        text: &str,
        images: Option<Vec<pi_ai_rs::Content>>,
    ) -> Result<(), anyhow::Error> {
        let mut parts: Vec<pi_ai_rs::UserContentPart> = vec![
            pi_ai_rs::UserContentPart::Text(pi_ai_rs::TextContent {
                text: text.to_string(),
                text_signature: None,
            }),
        ];
        if let Some(imgs) = images {
            for img in imgs {
                if let pi_ai_rs::Content::Image(ic) = img {
                    parts.push(pi_ai_rs::UserContentPart::Image(ic));
                }
            }
        }
        let user_msg = AgentMessage::Standard(Message::User(pi_ai_rs::UserMessage {
            content: if parts.len() == 1 {
                // Single text part → use simpler Text variant.
                pi_ai_rs::UserContent::Text(text.to_string())
            } else {
                pi_ai_rs::UserContent::Parts(parts)
            },
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }));
        self.prompt(user_msg).await
    }

    /// Continue from the current transcript.
    ///
    /// The last message in the transcript must not be an assistant message.
    pub async fn continue_(&mut self) -> Result<(), anyhow::Error> {
        if self.state.is_streaming {
            return Err(anyhow::anyhow!(
                "Agent is already processing. Wait for completion before continuing."
            ));
        }

        let last_role = self
            .state
            .messages
            .last()
            .map(|m| m.role())
            .unwrap_or("");

        if last_role == "assistant" {
            // Drain steering/follow-up queues first (mirrors TS `continue()` logic).
            let steering: Vec<AgentMessage> = {
                let mut q = self.steering_queue.lock().expect("queue poisoned");
                q.drain()
            };
            if !steering.is_empty() {
                return self.run_prompt_messages(steering, true).await;
            }

            let follow_ups: Vec<AgentMessage> = {
                let mut q = self.follow_up_queue.lock().expect("queue poisoned");
                q.drain()
            };
            if !follow_ups.is_empty() {
                return self.run_prompt_messages(follow_ups, false).await;
            }

            return Err(anyhow::anyhow!(
                "Cannot continue from message role: assistant"
            ));
        }

        self.run_continuation().await
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    async fn run_prompt_messages(
        &mut self,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
    ) -> Result<(), anyhow::Error> {
        if self.state.is_streaming {
            return Err(anyhow::anyhow!(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue \
                 messages, or wait for completion."
            ));
        }

        let context = self.create_context_snapshot();
        let config = self.create_loop_config(skip_initial_steering_poll);

        self.begin_run();

        let cancel = self.cancel.as_ref().expect("cancel must be set").clone();
        let mut stream = agent_loop(messages, context, config, cancel.clone());

        while let Some(event) = stream.next().await {
            self.process_event(event).await;
        }

        self.finish_run();
        Ok(())
    }

    async fn run_continuation(&mut self) -> Result<(), anyhow::Error> {
        if self.state.is_streaming {
            return Err(anyhow::anyhow!("Agent is already processing."));
        }

        let context = self.create_context_snapshot();
        let config = self.create_loop_config(false);

        self.begin_run();

        let cancel = self.cancel.as_ref().expect("cancel must be set").clone();
        let mut stream = agent_loop_continue(context, config, cancel);

        while let Some(event) = stream.next().await {
            self.process_event(event).await;
        }

        self.finish_run();
        Ok(())
    }

    fn create_context_snapshot(&self) -> AgentContext {
        let tool_definitions = self.state.tools.iter().map(|t| t.as_tool_definition()).collect();
        AgentContext {
            system_prompt: self.state.system_prompt.clone(),
            messages: self.state.messages.clone(),
            tool_definitions,
            tools: self.state.tools.clone(),
        }
    }

    fn create_loop_config(&self, skip_initial_steering_poll: bool) -> AgentLoopConfig {
        // TS: `skipInitialSteeringPoll` is a local `let mut` that flips on first call.
        // Rust: model with an Arc<AtomicBool> so the closure can toggle it.
        use std::sync::atomic::{AtomicBool, Ordering};
        let skip_flag = Arc::new(AtomicBool::new(skip_initial_steering_poll));
        let steering_arc = Arc::clone(&self.steering_queue);
        let get_steering: GetMessagesFn = Arc::new(move || {
            let sq = Arc::clone(&steering_arc);
            let flag = Arc::clone(&skip_flag);
            Box::pin(async move {
                if flag.swap(false, Ordering::SeqCst) {
                    return Vec::new();
                }
                sq.lock().expect("steering queue poisoned").drain()
            }) as BoxFuture<'static, Vec<AgentMessage>>
        });

        let follow_up_arc = Arc::clone(&self.follow_up_queue);
        let get_follow_up: GetMessagesFn = Arc::new(move || {
            let fq = Arc::clone(&follow_up_arc);
            Box::pin(async move {
                fq.lock().expect("follow-up queue poisoned").drain()
            }) as BoxFuture<'static, Vec<AgentMessage>>
        });

        let convert = Arc::clone(&self.convert_to_llm);
        let convert_to_llm: ConvertToLlmFn = Arc::new(move |msgs| (convert)(msgs));

        let transform_context = self.transform_context.as_ref().map(|f| {
            let f = Arc::clone(f);
            Arc::new(move |msgs, cancel| (f)(msgs, cancel)) as TransformContextFn
        });

        let get_api_key = self.get_api_key.as_ref().map(|f| {
            let f = Arc::clone(f);
            Arc::new(move |provider| (f)(provider)) as GetApiKeyFn
        });

        let before_tool_call = self.before_tool_call.as_ref().map(|f| {
            let f = Arc::clone(f);
            Arc::new(move |ctx, cancel| (f)(ctx, cancel)) as BeforeToolCallFn
        });

        let after_tool_call = self.after_tool_call.as_ref().map(|f| {
            let f = Arc::clone(f);
            Arc::new(move |ctx, cancel| (f)(ctx, cancel)) as AfterToolCallFn
        });

        let mut stream_options = pi_ai_rs::SimpleStreamOptions::default();
        stream_options.session_id = self.session_id.clone();
        stream_options.thinking_budgets = self.thinking_budgets.clone();
        stream_options.transport = Some(self.transport.clone());
        stream_options.max_retry_delay_ms = self.max_retry_delay_ms;
        stream_options.reasoning = self
            .state
            .thinking_level
            .to_ai_level();

        AgentLoopConfig {
            model: self.state.model.clone(),
            stream_options,
            convert_to_llm,
            transform_context,
            get_api_key,
            get_steering_messages: Some(get_steering),
            get_follow_up_messages: Some(get_follow_up),
            tool_execution: self.tool_execution,
            before_tool_call,
            after_tool_call,
            stream_fn: self.stream_fn.as_ref().map(Arc::clone),
        }
    }

    fn begin_run(&mut self) {
        self.cancel = Some(CancellationToken::new());
        self.idle_notify = Some(Arc::new(tokio::sync::Notify::new()));
        self.state.is_streaming = true;
        self.state.streaming_message = None;
        self.state.error_message = None;
    }

    fn finish_run(&mut self) {
        self.state.is_streaming = false;
        self.state.streaming_message = None;
        self.state.pending_tool_calls = HashSet::new();
        if let Some(ref notify) = self.idle_notify {
            notify.notify_waiters();
        }
        self.cancel = None;
        self.idle_notify = None;
    }

    /// Update internal state for a loop event, then invoke listeners.
    async fn process_event(&mut self, event: AgentEvent) {
        // Reduce state.
        match &event {
            AgentEvent::MessageStart { message } => {
                self.state.streaming_message = Some(message.clone());
            }
            AgentEvent::MessageUpdate { message, .. } => {
                self.state.streaming_message = Some(message.clone());
            }
            AgentEvent::MessageEnd { message } => {
                self.state.streaming_message = None;
                self.state.messages.push(message.clone());
            }
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                self.state.pending_tool_calls.insert(tool_call_id.clone());
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                self.state.pending_tool_calls.remove(tool_call_id);
            }
            AgentEvent::TurnEnd { message, .. } => {
                if let AgentMessage::Standard(pi_ai_rs::types::Message::Assistant(am)) = message {
                    if am.error_message.is_some() {
                        self.state.error_message = am.error_message.clone();
                    }
                }
            }
            AgentEvent::AgentEnd { .. } => {
                self.state.streaming_message = None;
            }
            _ => {}
        }

        // Notify listeners.
        let cancel = self
            .cancel
            .as_ref()
            .cloned()
            .unwrap_or_else(CancellationToken::new);

        for (_, listener) in &self.listeners {
            (listener)(event.clone(), cancel.clone()).await;
        }
    }
}
