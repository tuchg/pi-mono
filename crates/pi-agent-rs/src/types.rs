use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

use std::sync::Arc;

use pi_ai_rs::{
    AssistantMessage, AssistantMessageEvent,
    Content, Model,
    SimpleStreamOptions, ThinkingLevel as AiThinkingLevel,
    Tool, ToolResultMessage,
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

// Re-export the AI-layer message type.
pub use pi_ai_rs::types::Message;
pub use pi_ai_rs::event_stream::AssistantMessageEventStreamReceiver;

// ---------------------------------------------------------------------------
// StreamFn type
// ---------------------------------------------------------------------------

/// Custom stream function matching the `stream_simple` signature.
///
/// Allows apps to inject custom stream implementations (e.g., proxy streams)
/// instead of using `pi_ai_rs::stream_simple` directly.
pub type StreamFn = Arc<
    dyn Fn(&Model, pi_ai_rs::Context, SimpleStreamOptions) -> Result<AssistantMessageEventStreamReceiver, pi_ai_rs::AiError>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// Agent-level enums
// ---------------------------------------------------------------------------

/// Tool execution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    Sequential,
    /// TS default: "parallel"
    #[default]
    Parallel,
}

/// Agent-level thinking level (adds "off" to the AI-layer enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AgentThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl AgentThinkingLevel {
    /// Convert to the AI-layer `ThinkingLevel`, returning `None` for `Off`.
    pub fn to_ai_level(self) -> Option<AiThinkingLevel> {
        match self {
            Self::Off => None,
            Self::Minimal => Some(AiThinkingLevel::Minimal),
            Self::Low => Some(AiThinkingLevel::Low),
            Self::Medium => Some(AiThinkingLevel::Medium),
            Self::High => Some(AiThinkingLevel::High),
            Self::Xhigh => Some(AiThinkingLevel::Xhigh),
        }
    }
}

/// Queue draining mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QueueMode {
    /// Drain all pending messages at once.
    All,
    /// Process one message at a time.
    #[default]
    OneAtATime,
}

// ---------------------------------------------------------------------------
// Agent messages (extensible via typetag)
// ---------------------------------------------------------------------------

/// Trait for custom agent message types.
///
/// Third-party code can implement this trait (with `#[typetag::serde]`) to
/// register custom message variants that serialize/deserialize automatically.
#[typetag::serde(tag = "type")]
pub trait CustomAgentMessage: std::fmt::Debug + Send + Sync {
    fn role(&self) -> &str;
}

/// An agent-level message — either a standard LLM message or a custom one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentMessage {
    Standard(Message),
    // NOTE: Custom messages are handled via typetag at the serialization
    // boundary. For the initial scaffolding we keep a JSON fallback.
    Custom(serde_json::Value),
}

impl AgentMessage {
    pub fn role(&self) -> &str {
        match self {
            Self::Standard(Message::User(_)) => "user",
            Self::Standard(Message::Assistant(_)) => "assistant",
            Self::Standard(Message::ToolResult(_)) => "toolResult",
            Self::Custom(_) => "custom",
        }
    }
}

// ---------------------------------------------------------------------------
// Tool call hooks
// ---------------------------------------------------------------------------

/// Result of the `before_tool_call` hook.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeforeToolCallResult {
    pub block: Option<bool>,
    pub reason: Option<String>,
}

/// Result of the `after_tool_call` hook.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<Content>>,
    pub details: Option<serde_json::Value>,
    pub is_error: Option<bool>,
}

/// Context passed to the `before_tool_call` hook.
#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: pi_ai_rs::ToolCall,
    pub args: serde_json::Value,
    pub context: AgentContext,
}

/// Context passed to the `after_tool_call` hook.
#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: pi_ai_rs::ToolCall,
    pub args: serde_json::Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

// ---------------------------------------------------------------------------
// Tool result
// ---------------------------------------------------------------------------

/// The result of executing an agent tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolResult {
    pub content: Vec<Content>,
    pub details: serde_json::Value,
}

impl Default for AgentToolResult {
    fn default() -> Self {
        Self {
            content: Vec::new(),
            details: serde_json::Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent tool trait
// ---------------------------------------------------------------------------

/// Callback for incremental tool result updates.
pub type AgentToolUpdateCallback = Box<dyn Fn(&AgentToolResult) + Send + Sync>;

/// Boxed async future (used for trait method returns).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// An executable tool that the agent can call.
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    /// Optional shim to normalize raw tool-call arguments before schema
    /// validation.  Return `Some(normalized)` to replace the arguments, or
    /// `None` to leave them unchanged.
    fn prepare_arguments(&self, _args: serde_json::Value) -> Option<serde_json::Value> {
        None
    }

    /// Execute the tool with the given arguments.
    fn execute(
        &self,
        tool_call_id: &str,
        params: serde_json::Value,
        cancel: CancellationToken,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>>;

    /// Convert to an AI-layer `Tool` definition.
    fn as_tool_definition(&self) -> Tool {
        Tool {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }
}

// ---------------------------------------------------------------------------
// Agent context
// ---------------------------------------------------------------------------

/// The context visible to the agent loop.
#[derive(Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    /// Tool JSON schemas sent to the LLM.
    pub tool_definitions: Vec<Tool>,
    /// Executable tool instances (looked up by name during tool execution).
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl std::fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentContext")
            .field("system_prompt", &self.system_prompt)
            .field("messages", &self.messages)
            .field("tool_definitions", &self.tool_definitions)
            .field("tools", &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

/// Observable agent state.
#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: AgentThinkingLevel,
    /// Available tools. Mirrors TS `AgentState.tools`.
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}

impl std::fmt::Debug for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentState")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tools", &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>())
            .field("messages", &self.messages)
            .field("is_streaming", &self.is_streaming)
            .field("streaming_message", &self.streaming_message)
            .field("pending_tool_calls", &self.pending_tool_calls)
            .field("error_message", &self.error_message)
            .finish()
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: Model::default(),
            thinking_level: AgentThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Agent events
// ---------------------------------------------------------------------------

/// Events emitted by the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: serde_json::Value,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: serde_json::Value,
        is_error: bool,
    },
}

impl AgentEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::AgentEnd { .. })
    }
}

// ---------------------------------------------------------------------------
// Agent loop config
// ---------------------------------------------------------------------------

/// Async callback types used in the loop config and stored on the Agent.
///
/// Using `Arc` instead of `Box` so that callbacks can be cloned when building
/// a new `AgentLoopConfig` for each `prompt()` / `continue_()` call.
pub type ConvertToLlmFn =
    Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync>;
pub type TransformContextFn =
    Arc<dyn Fn(Vec<AgentMessage>, CancellationToken) -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>;
pub type GetApiKeyFn =
    Arc<dyn Fn(String) -> BoxFuture<'static, Option<String>> + Send + Sync>;
pub type GetMessagesFn =
    Arc<dyn Fn() -> BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>;
pub type BeforeToolCallFn = Arc<
    dyn Fn(BeforeToolCallContext, CancellationToken) -> BoxFuture<'static, Option<BeforeToolCallResult>>
        + Send
        + Sync,
>;
pub type AfterToolCallFn = Arc<
    dyn Fn(AfterToolCallContext, CancellationToken) -> BoxFuture<'static, Option<AfterToolCallResult>>
        + Send
        + Sync,
>;

/// Configuration for the agent loop.
pub struct AgentLoopConfig {
    pub model: Model,
    pub stream_options: SimpleStreamOptions,
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub get_steering_messages: Option<GetMessagesFn>,
    pub get_follow_up_messages: Option<GetMessagesFn>,
    pub tool_execution: ToolExecutionMode,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    /// Optional custom stream function, overriding `pi_ai_rs::stream_simple`.
    pub stream_fn: Option<StreamFn>,
}
