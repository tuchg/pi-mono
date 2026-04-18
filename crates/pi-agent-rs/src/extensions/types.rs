//! Extension system types.
//!
//! Extensions can:
//! - Subscribe to agent lifecycle events
//! - Register LLM-callable tools
//! - Register commands and keyboard shortcuts
//!
//! This module mirrors the TypeScript extension system in
//! `packages/coding-agent/src/core/extensions/types.ts`.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_ai_rs::{AssistantMessageEvent, Content, ImageContent, Model, ToolResultMessage};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::types::{AgentMessage, AgentTool, ToolExecutionMode};

// ============================================================================
// Boxed future alias
// ============================================================================

/// Boxed async future used for extension callbacks.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ============================================================================
// Extension Context
// ============================================================================

/// Context usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextUsage {
    /// Estimated context tokens, or None if unknown.
    pub tokens: Option<u64>,
    pub context_window: u64,
    /// Context usage as percentage of context window, or None if tokens unknown.
    pub percent: Option<f64>,
}

/// Options for context compaction.
#[derive(Debug, Clone, Default)]
pub struct CompactOptions {
    pub custom_instructions: Option<String>,
}

/// Read-only session information.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_file: Option<String>,
    pub session_id: Option<String>,
}

// ============================================================================
// Event Bus — cross-extension pub/sub communication
// ============================================================================

/// Shared event bus for inter-extension communication.
///
/// Mirrors the TypeScript `EventBus` from `packages/coding-agent/src/core/event-bus.ts`.
/// Extensions can publish and subscribe to arbitrary string channels.
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<(String, serde_json::Value)>,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Emit data on a named channel.
    pub fn emit(&self, channel: &str, data: serde_json::Value) {
        // Ignore send errors (no active receivers).
        let _ = self.sender.send((channel.to_string(), data));
    }

    /// Subscribe to a named channel. Returns a receiver that yields `(channel, data)` pairs.
    pub fn subscribe(&self) -> broadcast::Receiver<(String, serde_json::Value)> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Context passed to extension event handlers.
pub struct ExtensionContext {
    /// Whether UI is available.
    pub has_ui: bool,
    /// Current working directory.
    pub cwd: String,
    /// Session info (read-only).
    pub session_info: SessionInfo,
    /// Current model (may be None).
    pub model: Option<Model>,
    /// Whether the agent is idle (not streaming).
    pub is_idle: Box<dyn Fn() -> bool + Send + Sync>,
    /// Abort the current agent operation.
    pub abort: Box<dyn Fn() + Send + Sync>,
    /// Whether there are queued messages waiting.
    pub has_pending_messages: Box<dyn Fn() -> bool + Send + Sync>,
    /// Gracefully shutdown.
    pub shutdown: Box<dyn Fn() + Send + Sync>,
    /// Get current context usage.
    pub get_context_usage: Box<dyn Fn() -> Option<ContextUsage> + Send + Sync>,
    /// Trigger compaction without awaiting completion.
    pub compact: Box<dyn Fn(Option<CompactOptions>) + Send + Sync>,
    /// Get the current effective system prompt.
    pub get_system_prompt: Box<dyn Fn() -> String + Send + Sync>,
}

// ============================================================================
// Resource Events
// ============================================================================

/// Reason for resource discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourcesDiscoverReason {
    Startup,
    Reload,
}

/// Fired after session_start to allow extensions to provide additional resource paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesDiscoverEvent {
    pub cwd: String,
    pub reason: ResourcesDiscoverReason,
}

/// Result from resources_discover event handler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourcesDiscoverResult {
    pub skill_paths: Option<Vec<String>>,
    pub prompt_paths: Option<Vec<String>>,
    pub theme_paths: Option<Vec<String>>,
}

// ============================================================================
// Session Events
// ============================================================================

/// Reason for session start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStartReason {
    Startup,
    Reload,
    New,
    Resume,
    Fork,
}

/// Fired when a session is started, loaded, or reloaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStartEvent {
    pub reason: SessionStartReason,
    pub previous_session_file: Option<String>,
}

/// Reason for session switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionSwitchReason {
    New,
    Resume,
}

/// Fired before switching to another session (can be cancelled).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBeforeSwitchEvent {
    pub reason: SessionSwitchReason,
    pub target_session_file: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionBeforeSwitchResult {
    pub cancel: Option<bool>,
}

/// Fired before forking a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBeforeForkEvent {
    pub entry_id: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionBeforeForkResult {
    pub cancel: Option<bool>,
    pub skip_conversation_restore: Option<bool>,
}

/// Fired before context compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBeforeCompactEvent {
    pub custom_instructions: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionBeforeCompactResult {
    pub cancel: Option<bool>,
    /// If provided, use this compaction result instead of the default.
    pub compaction: Option<serde_json::Value>,
}

/// Fired after context compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCompactEvent {
    pub from_extension: bool,
}

/// Fired on graceful process shutdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionShutdownEvent;

/// Fired before navigating in the session tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBeforeTreeEvent {
    pub target_id: String,
    pub custom_instructions: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionBeforeTreeResult {
    pub cancel: Option<bool>,
    /// Override the summary for the tree navigation.
    pub summary: Option<TreeSummary>,
    pub custom_instructions: Option<String>,
    /// Override whether custom_instructions replaces the default prompt.
    pub replace_instructions: Option<bool>,
    pub label: Option<String>,
}

/// Summary override for tree navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSummary {
    pub summary: String,
    pub details: Option<serde_json::Value>,
}

/// Fired after navigating in the session tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTreeEvent {
    pub new_leaf_id: Option<String>,
    pub old_leaf_id: Option<String>,
    pub from_extension: Option<bool>,
}

// ============================================================================
// Agent Events
// ============================================================================

/// Fired before each LLM call. Can modify messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEvent {
    pub messages: Vec<AgentMessage>,
}

/// Result from context event handler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextEventResult {
    pub messages: Option<Vec<AgentMessage>>,
}

/// Fired before a provider request is sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeforeProviderRequestEvent {
    pub payload: serde_json::Value,
}

/// Fired after a provider response is received.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AfterProviderResponseEvent {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

/// Fired after user submits prompt but before agent loop.
#[derive(Debug, Clone)]
pub struct BeforeAgentStartEvent {
    pub prompt: String,
    pub images: Option<Vec<ImageContent>>,
    pub system_prompt: String,
}

/// Result from before_agent_start event handler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeforeAgentStartEventResult {
    /// Optional custom message to inject before the agent starts.
    pub message: Option<CustomMessage>,
    /// Replace the system prompt for this turn. If multiple extensions return this, they are chained.
    pub system_prompt: Option<String>,
}

/// A custom message that extensions can inject into the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMessage {
    pub custom_type: String,
    pub content: Option<serde_json::Value>,
    pub display: Option<String>,
    pub details: Option<serde_json::Value>,
}

/// Fired when an agent loop starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStartEvent;

/// Fired when an agent loop ends.
#[derive(Debug, Clone)]
pub struct AgentEndEvent {
    pub messages: Vec<AgentMessage>,
}

/// Fired at the start of each turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStartEvent {
    pub turn_index: u32,
    pub timestamp: u64,
}

/// Fired at the end of each turn.
#[derive(Debug, Clone)]
pub struct TurnEndEvent {
    pub turn_index: u32,
    pub message: AgentMessage,
    pub tool_results: Vec<ToolResultMessage>,
}

/// Fired when a message starts.
#[derive(Debug, Clone)]
pub struct MessageStartEvent {
    pub message: AgentMessage,
}

/// Fired during assistant message streaming.
#[derive(Debug, Clone)]
pub struct MessageUpdateEvent {
    pub message: AgentMessage,
    pub assistant_message_event: AssistantMessageEvent,
}

/// Fired when a message ends.
#[derive(Debug, Clone)]
pub struct MessageEndEvent {
    pub message: AgentMessage,
}

/// Fired when a tool starts executing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionStartEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
}

/// Fired during tool execution with partial output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionUpdateEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub partial_result: serde_json::Value,
}

/// Fired when a tool finishes executing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionEndEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub result: serde_json::Value,
    pub is_error: bool,
}

// ============================================================================
// Model Events
// ============================================================================

/// Source of model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSelectSource {
    Set,
    Cycle,
    Restore,
}

/// Fired when a new model is selected.
#[derive(Debug, Clone)]
pub struct ModelSelectEvent {
    pub model: Model,
    pub previous_model: Option<Model>,
    pub source: ModelSelectSource,
}

// ============================================================================
// User Bash Events
// ============================================================================

/// Fired when user executes a bash command via ! or !! prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserBashEvent {
    pub command: String,
    pub exclude_from_context: bool,
    pub cwd: String,
}

/// Result from user_bash event handler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserBashEventResult {
    /// Full replacement result if extension handled execution.
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub exit_code: Option<i32>,
}

// ============================================================================
// Input Events
// ============================================================================

/// Source of user input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputSource {
    Interactive,
    Rpc,
    Extension,
}

/// Fired when user input is received, before agent processing.
#[derive(Debug, Clone)]
pub struct InputEvent {
    pub text: String,
    pub images: Option<Vec<ImageContent>>,
    pub source: InputSource,
}

/// Result from input event handler.
#[derive(Debug, Clone)]
pub enum InputEventResult {
    Continue,
    Transform { text: String, images: Option<Vec<ImageContent>> },
    Handled,
}

// ============================================================================
// Tool Call / Tool Result Events
// ============================================================================

/// Fired before a tool executes. Can block.
///
/// `input` is mutable — mutate it to patch tool arguments before execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
}

/// Result from tool_call event handler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallEventResult {
    pub block: Option<bool>,
    pub reason: Option<String>,
}

/// Fired after a tool executes. Can modify result.
#[derive(Debug, Clone)]
pub struct ToolResultEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub content: Vec<Content>,
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
}

/// Result from tool_result event handler.
#[derive(Debug, Clone, Default)]
pub struct ToolResultEventResult {
    pub content: Option<Vec<Content>>,
    pub details: Option<serde_json::Value>,
    pub is_error: Option<bool>,
}

// ============================================================================
// Union Event Type
// ============================================================================

/// Union of all extension event types.
#[derive(Debug, Clone)]
pub enum ExtensionEvent {
    ResourcesDiscover(ResourcesDiscoverEvent),
    SessionStart(SessionStartEvent),
    SessionBeforeSwitch(SessionBeforeSwitchEvent),
    SessionBeforeFork(SessionBeforeForkEvent),
    SessionBeforeCompact(SessionBeforeCompactEvent),
    SessionCompact(SessionCompactEvent),
    SessionShutdown(SessionShutdownEvent),
    SessionBeforeTree(SessionBeforeTreeEvent),
    SessionTree(SessionTreeEvent),
    Context(ContextEvent),
    BeforeProviderRequest(BeforeProviderRequestEvent),
    AfterProviderResponse(AfterProviderResponseEvent),
    BeforeAgentStart(BeforeAgentStartEvent),
    AgentStart(AgentStartEvent),
    AgentEnd(AgentEndEvent),
    TurnStart(TurnStartEvent),
    TurnEnd(TurnEndEvent),
    MessageStart(MessageStartEvent),
    MessageUpdate(MessageUpdateEvent),
    MessageEnd(MessageEndEvent),
    ToolExecutionStart(ToolExecutionStartEvent),
    ToolExecutionUpdate(ToolExecutionUpdateEvent),
    ToolExecutionEnd(ToolExecutionEndEvent),
    ModelSelect(ModelSelectEvent),
    ToolCall(ToolCallEvent),
    ToolResult(ToolResultEvent),
    UserBash(UserBashEvent),
    Input(InputEvent),
}

impl ExtensionEvent {
    /// Returns the string event type name used for handler dispatch.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::ResourcesDiscover(_) => "resources_discover",
            Self::SessionStart(_) => "session_start",
            Self::SessionBeforeSwitch(_) => "session_before_switch",
            Self::SessionBeforeFork(_) => "session_before_fork",
            Self::SessionBeforeCompact(_) => "session_before_compact",
            Self::SessionCompact(_) => "session_compact",
            Self::SessionShutdown(_) => "session_shutdown",
            Self::SessionBeforeTree(_) => "session_before_tree",
            Self::SessionTree(_) => "session_tree",
            Self::Context(_) => "context",
            Self::BeforeProviderRequest(_) => "before_provider_request",
            Self::AfterProviderResponse(_) => "after_provider_response",
            Self::BeforeAgentStart(_) => "before_agent_start",
            Self::AgentStart(_) => "agent_start",
            Self::AgentEnd(_) => "agent_end",
            Self::TurnStart(_) => "turn_start",
            Self::TurnEnd(_) => "turn_end",
            Self::MessageStart(_) => "message_start",
            Self::MessageUpdate(_) => "message_update",
            Self::MessageEnd(_) => "message_end",
            Self::ToolExecutionStart(_) => "tool_execution_start",
            Self::ToolExecutionUpdate(_) => "tool_execution_update",
            Self::ToolExecutionEnd(_) => "tool_execution_end",
            Self::ModelSelect(_) => "model_select",
            Self::ToolCall(_) => "tool_call",
            Self::ToolResult(_) => "tool_result",
            Self::UserBash(_) => "user_bash",
            Self::Input(_) => "input",
        }
    }
}

// ============================================================================
// Tool Definition (for registerTool)
// ============================================================================

/// Tool definition for extension-registered tools.
#[derive(Debug, Clone)]
pub struct ExtensionToolDefinition {
    pub name: String,
    pub label: String,
    pub description: String,
    /// Optional one-line snippet for the system prompt.
    pub prompt_snippet: Option<String>,
    /// Optional guideline bullets for the system prompt.
    pub prompt_guidelines: Option<Vec<String>>,
    /// JSON Schema for parameters.
    pub parameters: serde_json::Value,
    /// Per-tool execution mode override.
    pub execution_mode: Option<ToolExecutionMode>,
}

// ============================================================================
// Command Registration
// ============================================================================

/// A registered extension command.
#[derive(Clone)]
pub struct RegisteredCommand {
    pub name: String,
    pub description: Option<String>,
    pub extension_path: String,
    /// The handler is invoked with (args, ctx).
    pub handler: Arc<
        dyn Fn(String, Arc<ExtensionContext>) -> BoxFuture<'static, ()> + Send + Sync,
    >,
}

impl std::fmt::Debug for RegisteredCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredCommand")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("extension_path", &self.extension_path)
            .finish()
    }
}

// ============================================================================
// Shortcut Registration
// ============================================================================

/// A registered keyboard shortcut.
#[derive(Clone)]
pub struct ExtensionShortcut {
    pub shortcut: String,
    pub description: Option<String>,
    pub extension_path: String,
    pub handler: Arc<
        dyn Fn(Arc<ExtensionContext>) -> BoxFuture<'static, ()> + Send + Sync,
    >,
}

impl std::fmt::Debug for ExtensionShortcut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionShortcut")
            .field("shortcut", &self.shortcut)
            .field("description", &self.description)
            .field("extension_path", &self.extension_path)
            .finish()
    }
}

// ============================================================================
// Extension Flag
// ============================================================================

/// A CLI flag registered by an extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionFlag {
    pub name: String,
    pub description: Option<String>,
    pub flag_type: ExtensionFlagType,
    pub default: Option<serde_json::Value>,
    pub extension_path: String,
}

/// Type of a flag value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtensionFlagType {
    Boolean,
    String,
}

// ============================================================================
// Extension Error
// ============================================================================

/// An error that occurred in an extension handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionError {
    pub extension_path: String,
    pub event: String,
    pub error: String,
    pub stack: Option<String>,
}

// ============================================================================
// Loaded Extension
// ============================================================================

/// Handler function type — receives serialized event and returns serialized result.
pub type HandlerFn = Arc<
    dyn Fn(ExtensionEvent, Arc<ExtensionContext>) -> BoxFuture<'static, Option<serde_json::Value>>
        + Send
        + Sync,
>;

/// Trait for extension registration (the `pi` parameter in factory functions).
///
/// Mirrors the TypeScript `ExtensionAPI` interface. Extensions use this to
/// register handlers, tools, commands, flags, shortcuts, and to invoke
/// runtime actions (sending messages, changing model, etc.).
pub trait ExtensionAPI: Send + Sync {
    // =========================================================================
    // Event Subscription
    // =========================================================================

    /// Subscribe to an extension event.
    fn on(&mut self, event_type: &str, handler: HandlerFn);

    // =========================================================================
    // Tool Registration
    // =========================================================================

    /// Register a tool that the LLM can call.
    fn register_tool(&mut self, tool: ExtensionToolDefinition, execute: Arc<dyn AgentTool>);

    // =========================================================================
    // Command, Shortcut, Flag Registration
    // =========================================================================

    /// Register a custom command.
    fn register_command(&mut self, command: RegisteredCommand);

    /// Register a keyboard shortcut.
    fn register_shortcut(&mut self, shortcut: ExtensionShortcut);

    /// Register a CLI flag.
    fn register_flag(&mut self, flag: ExtensionFlag);

    /// Get the value of a registered CLI flag.
    fn get_flag(&self, name: &str) -> Option<serde_json::Value>;

    // =========================================================================
    // Message Rendering
    // =========================================================================

    /// Register a custom renderer for custom message entries.
    fn register_message_renderer(&mut self, custom_type: &str, renderer: MessageRenderer);

    // =========================================================================
    // Actions (delegated to shared runtime)
    // =========================================================================

    /// Send a custom message to the session.
    fn send_message(&self, message: CustomMessage, options: Option<SendMessageOptions>);

    /// Send a user message to the agent. Always triggers a turn.
    fn send_user_message(&self, content: UserMessageContent, options: Option<SendUserMessageOptions>);

    /// Append a custom entry to the session for state persistence (not sent to LLM).
    fn append_entry(&self, custom_type: &str, data: Option<serde_json::Value>);

    /// Set the session display name.
    fn set_session_name(&self, name: &str);

    /// Get the current session name, if set.
    fn get_session_name(&self) -> Option<String>;

    /// Set or clear a label on an entry.
    fn set_label(&self, entry_id: &str, label: Option<&str>);

    /// Get the list of currently active tool names.
    fn get_active_tools(&self) -> Vec<String>;

    /// Get all configured tools with parameter schema and source metadata.
    fn get_all_tools(&self) -> Vec<ToolInfo>;

    /// Set the active tools by name.
    fn set_active_tools(&self, tool_names: &[String]);

    /// Refresh the tool set.
    fn refresh_tools(&self);

    /// Get available slash commands.
    fn get_commands(&self) -> Vec<SlashCommandInfo>;

    /// Set the current model. Returns false if no API key available.
    fn set_model(&self, model: Model) -> BoxFuture<'static, bool>;

    /// Get current thinking level.
    fn get_thinking_level(&self) -> ThinkingLevel;

    /// Set thinking level (clamped to model capabilities).
    fn set_thinking_level(&self, level: ThinkingLevel);

    // =========================================================================
    // Provider Registration
    // =========================================================================

    /// Register or override a model provider.
    fn register_provider(&mut self, name: &str, config: ProviderConfig);

    /// Unregister a previously registered provider.
    fn unregister_provider(&mut self, name: &str);

    // =========================================================================
    // Event Bus
    // =========================================================================

    /// Get the shared event bus for inter-extension communication.
    fn events(&self) -> &EventBus;
}

// ============================================================================
// Action Types for ExtensionAPI
// ============================================================================

/// Options for `send_message`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SendMessageOptions {
    pub trigger_turn: Option<bool>,
    pub deliver_as: Option<DeliverAs>,
}

/// Options for `send_user_message`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SendUserMessageOptions {
    pub deliver_as: Option<DeliverAs>,
}

/// Message delivery mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DeliverAs {
    Steer,
    FollowUp,
    NextTurn,
}

/// Content for `send_user_message`.
#[derive(Debug, Clone)]
pub enum UserMessageContent {
    Text(String),
    Mixed(Vec<Content>),
}

/// Tool info returned by `get_all_tools`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub source_info: SourceInfo,
}

/// Slash command info returned by `get_commands`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashCommandInfo {
    pub name: String,
    pub description: Option<String>,
}

/// Thinking level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    None,
    Low,
    Medium,
    High,
}

/// Configuration for registering a provider via `register_provider`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Base URL for the API endpoint.
    pub base_url: Option<String>,
    /// API key or environment variable name.
    pub api_key: Option<String>,
    /// API type identifier.
    pub api: Option<String>,
    /// Custom headers to include in requests.
    pub headers: Option<HashMap<String, String>>,
    /// If true, adds Authorization: Bearer header with the resolved API key.
    pub auth_header: Option<bool>,
    /// Models to register. If provided, replaces all existing models for this provider.
    pub models: Option<Vec<ProviderModelConfig>>,
}

/// Configuration for a model within a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderModelConfig {
    pub id: String,
    pub name: String,
    pub api: Option<String>,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    pub headers: Option<HashMap<String, String>>,
}

/// Cost per token for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// A message renderer for custom message types.
#[derive(Clone)]
pub struct MessageRenderer {
    /// Render function: takes custom data and returns display text.
    pub render: Arc<dyn Fn(serde_json::Value) -> String + Send + Sync>,
}

impl std::fmt::Debug for MessageRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageRenderer").finish()
    }
}

// ============================================================================
// Shared Extension Runtime
// ============================================================================

/// Shared runtime state and actions accessible to all extensions.
///
/// Mirrors the TypeScript `ExtensionRuntime` (`ExtensionRuntimeState + ExtensionActions`).
/// Created during extension loading with throwing stubs; actions are wired after
/// the runner calls `bind_core`.
pub struct ExtensionRuntime {
    /// Flag values shared across all extensions.
    pub flag_values: HashMap<String, serde_json::Value>,

    /// Provider registrations queued during extension loading, processed when runner binds.
    pub pending_provider_registrations: std::sync::Mutex<Vec<PendingProviderRegistration>>,

    // =========================================================================
    // Action callbacks — throwing stubs until runner.bind_core() wires them.
    // =========================================================================

    pub send_message: Box<dyn Fn(CustomMessage, Option<SendMessageOptions>) + Send + Sync>,
    pub send_user_message: Box<dyn Fn(UserMessageContent, Option<SendUserMessageOptions>) + Send + Sync>,
    pub append_entry: Box<dyn Fn(&str, Option<serde_json::Value>) + Send + Sync>,
    pub set_session_name: Box<dyn Fn(&str) + Send + Sync>,
    pub get_session_name: Box<dyn Fn() -> Option<String> + Send + Sync>,
    pub set_label: Box<dyn Fn(&str, Option<&str>) + Send + Sync>,
    pub get_active_tools: Box<dyn Fn() -> Vec<String> + Send + Sync>,
    pub get_all_tools: Box<dyn Fn() -> Vec<ToolInfo> + Send + Sync>,
    pub set_active_tools: Box<dyn Fn(&[String]) + Send + Sync>,
    pub refresh_tools: Box<dyn Fn() + Send + Sync>,
    pub get_commands: Box<dyn Fn() -> Vec<SlashCommandInfo> + Send + Sync>,
    pub set_model: Arc<dyn Fn(Model) -> BoxFuture<'static, bool> + Send + Sync>,
    pub get_thinking_level: Box<dyn Fn() -> ThinkingLevel + Send + Sync>,
    pub set_thinking_level: Box<dyn Fn(ThinkingLevel) + Send + Sync>,

    // =========================================================================
    // Provider registration
    // =========================================================================
    pub register_provider: Box<dyn Fn(&str, ProviderConfig, Option<&str>) + Send + Sync>,
    pub unregister_provider: Box<dyn Fn(&str, Option<&str>) + Send + Sync>,
}

/// A provider registration that was queued during extension loading.
#[derive(Debug, Clone)]
pub struct PendingProviderRegistration {
    pub name: String,
    pub config: ProviderConfig,
    pub extension_path: String,
}

impl ExtensionRuntime {
    /// Create a new runtime with throwing/no-op stubs for all actions.
    ///
    /// These stubs are replaced when `ExtensionRunner::bind_core()` is called.
    pub fn new() -> Self {
        Self {
            flag_values: HashMap::new(),
            pending_provider_registrations: std::sync::Mutex::new(Vec::new()),
            send_message: Box::new(|_, _| {
                tracing::warn!("send_message called before runtime was bound");
            }),
            send_user_message: Box::new(|_, _| {
                tracing::warn!("send_user_message called before runtime was bound");
            }),
            append_entry: Box::new(|_, _| {
                tracing::warn!("append_entry called before runtime was bound");
            }),
            set_session_name: Box::new(|_| {
                tracing::warn!("set_session_name called before runtime was bound");
            }),
            get_session_name: Box::new(|| {
                tracing::warn!("get_session_name called before runtime was bound");
                None
            }),
            set_label: Box::new(|_, _| {
                tracing::warn!("set_label called before runtime was bound");
            }),
            get_active_tools: Box::new(|| {
                tracing::warn!("get_active_tools called before runtime was bound");
                Vec::new()
            }),
            get_all_tools: Box::new(|| {
                tracing::warn!("get_all_tools called before runtime was bound");
                Vec::new()
            }),
            set_active_tools: Box::new(|_| {
                tracing::warn!("set_active_tools called before runtime was bound");
            }),
            refresh_tools: Box::new(|| {
                tracing::warn!("refresh_tools called before runtime was bound");
            }),
            get_commands: Box::new(|| {
                tracing::warn!("get_commands called before runtime was bound");
                Vec::new()
            }),
            set_model: Arc::new(|_| {
                tracing::warn!("set_model called before runtime was bound");
                Box::pin(async { false })
            }),
            get_thinking_level: Box::new(|| {
                tracing::warn!("get_thinking_level called before runtime was bound");
                ThinkingLevel::None
            }),
            set_thinking_level: Box::new(|_| {
                tracing::warn!("set_thinking_level called before runtime was bound");
            }),
            register_provider: Box::new(|_, _, _| {
                tracing::warn!("register_provider called before runtime was bound");
            }),
            unregister_provider: Box::new(|_, _| {
                tracing::warn!("unregister_provider called before runtime was bound");
            }),
        }
    }
}

impl Default for ExtensionRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Source information for tracking where an extension was loaded from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub path: String,
    pub source_type: SourceType,
}

/// Type of extension source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    /// Local file/directory.
    Local,
    /// WASM component.
    Wasm,
    /// Built-in extension.
    BuiltIn,
}

/// A loaded extension with all registered items.
#[derive(Clone)]
pub struct Extension {
    pub path: String,
    pub resolved_path: String,
    pub source_info: SourceInfo,
    pub handlers: HashMap<String, Vec<HandlerFn>>,
    pub tools: HashMap<String, (ExtensionToolDefinition, Arc<dyn AgentTool>)>,
    pub message_renderers: HashMap<String, MessageRenderer>,
    pub commands: HashMap<String, RegisteredCommand>,
    pub flags: HashMap<String, ExtensionFlag>,
    pub shortcuts: HashMap<String, ExtensionShortcut>,
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension")
            .field("path", &self.path)
            .field("resolved_path", &self.resolved_path)
            .field("source_info", &self.source_info)
            .field("handlers", &self.handlers.keys().collect::<Vec<_>>())
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("message_renderers", &self.message_renderers.keys().collect::<Vec<_>>())
            .field("commands", &self.commands.keys().collect::<Vec<_>>())
            .field("flags", &self.flags.keys().collect::<Vec<_>>())
            .field("shortcuts", &self.shortcuts.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Result of loading extensions.
pub struct LoadExtensionsResult {
    pub extensions: Vec<Extension>,
    pub errors: Vec<ExtensionError>,
    /// Shared runtime — actions are throwing stubs until `ExtensionRunner::bind_core()`.
    pub runtime: ExtensionRuntime,
}
