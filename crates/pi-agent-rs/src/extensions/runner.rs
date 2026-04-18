//! Extension runner — executes extensions and manages their lifecycle.
//!
//! Mirrors the TypeScript `ExtensionRunner` class from
//! `packages/coding-agent/src/core/extensions/runner.ts`.

use std::collections::HashMap;
use std::sync::Arc;

use futures::FutureExt;
use pi_ai_rs::{Content, ImageContent, Model};

use crate::types::AgentMessage;

use super::types::{
    BeforeAgentStartEvent, BeforeAgentStartEventResult, BeforeProviderRequestEvent,
    CompactOptions, ContextEvent, ContextEventResult, ContextUsage, CustomMessage, Extension,
    ExtensionContext, ExtensionError, ExtensionEvent, ExtensionShortcut,
    ExtensionToolDefinition, InputEvent, InputEventResult,
    RegisteredCommand, SessionInfo, ToolCallEvent, ToolCallEventResult, ToolResultEvent,
    ToolResultEventResult, UserBashEvent, UserBashEventResult,
};
use crate::types::AgentTool;

/// Error listener callback type.
pub type ExtensionErrorListener = Arc<dyn Fn(&ExtensionError) + Send + Sync>;

/// Context action callbacks provided by the host.
///
/// Mirrors the TypeScript `ExtensionContextActions` interface.
pub struct ExtensionContextActions {
    pub get_model: Arc<dyn Fn() -> Option<Model> + Send + Sync>,
    pub is_idle: Arc<dyn Fn() -> bool + Send + Sync>,
    pub abort: Arc<dyn Fn() + Send + Sync>,
    pub has_pending_messages: Arc<dyn Fn() -> bool + Send + Sync>,
    pub shutdown: Arc<dyn Fn() + Send + Sync>,
    pub get_context_usage: Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>,
    pub compact: Arc<dyn Fn(Option<CompactOptions>) + Send + Sync>,
    pub get_system_prompt: Arc<dyn Fn() -> String + Send + Sync>,
}

/// Combined result from all before_agent_start handlers.
#[derive(Debug, Clone, Default)]
pub struct BeforeAgentStartCombinedResult {
    /// Custom messages collected from all handlers.
    pub messages: Option<Vec<CustomMessage>>,
    /// System prompt override (chained from all handlers).
    pub system_prompt: Option<String>,
}

/// The extension runner manages extension lifecycle and event dispatch.
pub struct ExtensionRunner {
    extensions: Vec<Extension>,
    cwd: String,
    session_info: SessionInfo,
    error_listeners: Vec<ExtensionErrorListener>,
    // Context action callbacks, set via bind_core — Arc for shared access in contexts
    get_model: Arc<dyn Fn() -> Option<Model> + Send + Sync>,
    is_idle_fn: Arc<dyn Fn() -> bool + Send + Sync>,
    abort_fn: Arc<dyn Fn() + Send + Sync>,
    has_pending_messages_fn: Arc<dyn Fn() -> bool + Send + Sync>,
    shutdown_fn: Arc<dyn Fn() + Send + Sync>,
    get_context_usage_fn: Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>,
    compact_fn: Arc<dyn Fn(Option<CompactOptions>) + Send + Sync>,
    get_system_prompt_fn: Arc<dyn Fn() -> String + Send + Sync>,
    // Flag values (defaults set during registration, CLI values set after)
    flag_values: HashMap<String, serde_json::Value>,
}

impl ExtensionRunner {
    /// Create a new extension runner with loaded extensions.
    pub fn new(extensions: Vec<Extension>, cwd: String, session_info: SessionInfo) -> Self {
        Self {
            extensions,
            cwd,
            session_info,
            error_listeners: Vec::new(),
            get_model: Arc::new(|| None),
            is_idle_fn: Arc::new(|| true),
            abort_fn: Arc::new(|| {}),
            has_pending_messages_fn: Arc::new(|| false),
            shutdown_fn: Arc::new(|| {}),
            get_context_usage_fn: Arc::new(|| None),
            compact_fn: Arc::new(|_| {}),
            get_system_prompt_fn: Arc::new(|| String::new()),
            flag_values: HashMap::new(),
        }
    }

    /// Bind core context actions — must be called before dispatching events.
    pub fn bind_core(&mut self, actions: ExtensionContextActions) {
        self.get_model = actions.get_model;
        self.is_idle_fn = actions.is_idle;
        self.abort_fn = actions.abort;
        self.has_pending_messages_fn = actions.has_pending_messages;
        self.shutdown_fn = actions.shutdown;
        self.get_context_usage_fn = actions.get_context_usage;
        self.compact_fn = actions.compact;
        self.get_system_prompt_fn = actions.get_system_prompt;
    }

    /// Register an error listener.
    pub fn on_error(&mut self, listener: ExtensionErrorListener) {
        self.error_listeners.push(listener);
    }

    /// Emit an error to all listeners.
    pub fn emit_error(&self, error: ExtensionError) {
        for listener in &self.error_listeners {
            listener(&error);
        }
    }

    /// Check whether any extension has handlers for the given event type.
    pub fn has_handlers(&self, event_type: &str) -> bool {
        for ext in &self.extensions {
            if let Some(handlers) = ext.handlers.get(event_type) {
                if !handlers.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    /// Get all registered tools from all extensions (first registration per name wins).
    pub fn get_all_registered_tools(&self) -> Vec<(&ExtensionToolDefinition, &Arc<dyn AgentTool>)> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for ext in &self.extensions {
            for (name, (def, tool)) in &ext.tools {
                if seen.insert(name.clone()) {
                    result.push((def, tool));
                }
            }
        }
        result
    }

    /// Get a tool definition by name.
    pub fn get_tool_definition(&self, tool_name: &str) -> Option<&ExtensionToolDefinition> {
        for ext in &self.extensions {
            if let Some((def, _)) = ext.tools.get(tool_name) {
                return Some(def);
            }
        }
        None
    }

    /// Get all registered commands.
    pub fn get_registered_commands(&self) -> Vec<&RegisteredCommand> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for ext in &self.extensions {
            for (name, cmd) in &ext.commands {
                if seen.insert(name.clone()) {
                    result.push(cmd);
                }
            }
        }
        result
    }

    /// Get all registered shortcuts.
    pub fn get_shortcuts(&self) -> Vec<&ExtensionShortcut> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for ext in &self.extensions {
            for (key, shortcut) in &ext.shortcuts {
                if seen.insert(key.clone()) {
                    result.push(shortcut);
                }
            }
        }
        result
    }

    /// Get all extension paths.
    pub fn get_extension_paths(&self) -> Vec<&str> {
        self.extensions.iter().map(|e| e.path.as_str()).collect()
    }

    /// Set a flag value.
    pub fn set_flag_value(&mut self, name: String, value: serde_json::Value) {
        self.flag_values.insert(name, value);
    }

    /// Get flag values.
    pub fn get_flag_values(&self) -> &HashMap<String, serde_json::Value> {
        &self.flag_values
    }

    /// Request a graceful shutdown.
    pub fn shutdown(&self) {
        (self.shutdown_fn)();
    }

    /// Create an ExtensionContext for use in event handlers.
    fn create_context(&self) -> Arc<ExtensionContext> {
        let model = (self.get_model)();
        let is_idle = self.is_idle_fn.clone();
        let abort = self.abort_fn.clone();
        let has_pending = self.has_pending_messages_fn.clone();
        let shutdown = self.shutdown_fn.clone();
        let get_usage = self.get_context_usage_fn.clone();
        let compact = self.compact_fn.clone();
        let get_prompt = self.get_system_prompt_fn.clone();

        Arc::new(ExtensionContext {
            has_ui: false,
            cwd: self.cwd.clone(),
            session_info: self.session_info.clone(),
            model,
            is_idle: Box::new(move || is_idle()),
            abort: Box::new(move || abort()),
            has_pending_messages: Box::new(move || has_pending()),
            shutdown: Box::new(move || shutdown()),
            get_context_usage: Box::new(move || get_usage()),
            compact: Box::new(move |opts| compact(opts)),
            get_system_prompt: Box::new(move || get_prompt()),
        })
    }

    // ========================================================================
    // Generic event dispatch
    // ========================================================================

    /// Emit a generic event to all extension handlers.
    ///
    /// For events with dedicated emit methods (tool_call, tool_result, context,
    /// before_provider_request, etc.), use those methods instead for type safety.
    ///
    /// Error isolation: if a handler panics or returns an error, the error is
    /// reported and subsequent handlers continue to execute.
    pub async fn emit(&self, event: ExtensionEvent) {
        let ctx = self.create_context();
        let event_type = event.event_type();

        for ext in &self.extensions {
            let handlers = match ext.handlers.get(event_type) {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                match std::panic::AssertUnwindSafe(handler(event.clone(), ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        // For session_before_* events, check if cancelled
                        if val.get("cancel").and_then(|v| v.as_bool()) == Some(true) {
                            return;
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: event_type.to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }
    }

    // ========================================================================
    // Dedicated emit methods for events with specific return semantics
    // ========================================================================

    /// Emit a tool_call event. Returns block/reason if any handler blocks.
    pub async fn emit_tool_call(&self, event: ToolCallEvent) -> Option<ToolCallEventResult> {
        let ctx = self.create_context();
        let ext_event = ExtensionEvent::ToolCall(event);
        let mut result: Option<ToolCallEventResult> = None;

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("tool_call") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                match std::panic::AssertUnwindSafe(handler(ext_event.clone(), ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        let handler_result: ToolCallEventResult =
                            serde_json::from_value(val).unwrap_or_default();
                        if handler_result.block == Some(true) {
                            return Some(handler_result);
                        }
                        result = Some(handler_result);
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "tool_call".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        result
    }

    /// Emit a tool_result event. Returns merged overrides from all handlers.
    pub async fn emit_tool_result(&self, event: ToolResultEvent) -> Option<ToolResultEventResult> {
        let ctx = self.create_context();
        let mut current_content = event.content.clone();
        let mut current_details = event.details.clone();
        let mut current_is_error = event.is_error;
        let mut modified = false;

        let ext_event = ExtensionEvent::ToolResult(event);

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("tool_result") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                match std::panic::AssertUnwindSafe(handler(ext_event.clone(), ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        // Apply field-level overrides (omit-means-keep)
                        if let Some(content) = val.get("content") {
                            if let Ok(c) = serde_json::from_value::<Vec<Content>>(content.clone()) {
                                current_content = c;
                                modified = true;
                            }
                        }
                        if let Some(details) = val.get("details") {
                            current_details = Some(details.clone());
                            modified = true;
                        }
                        if let Some(is_error) = val.get("is_error").and_then(|v| v.as_bool()) {
                            current_is_error = is_error;
                            modified = true;
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "tool_result".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        if !modified {
            return None;
        }

        Some(ToolResultEventResult {
            content: Some(current_content),
            details: current_details,
            is_error: Some(current_is_error),
        })
    }

    /// Emit a context event. Returns the (potentially modified) messages.
    pub async fn emit_context(&self, messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
        let ctx = self.create_context();
        let mut current_messages = messages;

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("context") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                let event = ExtensionEvent::Context(ContextEvent {
                    messages: current_messages.clone(),
                });

                match std::panic::AssertUnwindSafe(handler(event, ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        if let Ok(result) = serde_json::from_value::<ContextEventResult>(val) {
                            if let Some(msgs) = result.messages {
                                current_messages = msgs;
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "context".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        current_messages
    }

    /// Emit a before_provider_request event. Returns the (potentially modified) payload.
    pub async fn emit_before_provider_request(
        &self,
        payload: serde_json::Value,
    ) -> serde_json::Value {
        let ctx = self.create_context();
        let mut current_payload = payload;

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("before_provider_request") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                let event = ExtensionEvent::BeforeProviderRequest(BeforeProviderRequestEvent {
                    payload: current_payload.clone(),
                });

                match std::panic::AssertUnwindSafe(handler(event, ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        current_payload = val;
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "before_provider_request".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        current_payload
    }

    /// Emit a before_agent_start event. Returns combined result.
    ///
    /// Collects `message` values from all handlers and chains `system_prompt`
    /// overrides, matching the TypeScript `emitBeforeAgentStart` behaviour.
    pub async fn emit_before_agent_start(
        &self,
        prompt: String,
        images: Option<Vec<ImageContent>>,
        system_prompt: String,
    ) -> Option<BeforeAgentStartCombinedResult> {
        let ctx = self.create_context();
        let mut messages: Vec<CustomMessage> = Vec::new();
        let mut current_system_prompt = system_prompt.clone();
        let mut system_prompt_modified = false;

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("before_agent_start") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                let event = ExtensionEvent::BeforeAgentStart(BeforeAgentStartEvent {
                    prompt: prompt.clone(),
                    images: images.clone(),
                    system_prompt: current_system_prompt.clone(),
                });

                match std::panic::AssertUnwindSafe(handler(event, ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        if let Ok(result) =
                            serde_json::from_value::<BeforeAgentStartEventResult>(val)
                        {
                            if let Some(msg) = result.message {
                                messages.push(msg);
                            }
                            if let Some(sp) = result.system_prompt {
                                current_system_prompt = sp;
                                system_prompt_modified = true;
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "before_agent_start".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        if messages.is_empty() && !system_prompt_modified {
            return None;
        }

        Some(BeforeAgentStartCombinedResult {
            messages: if messages.is_empty() { None } else { Some(messages) },
            system_prompt: if system_prompt_modified { Some(current_system_prompt) } else { None },
        })
    }

    /// Emit a user_bash event. Returns the first handler result.
    pub async fn emit_user_bash(&self, event: UserBashEvent) -> Option<UserBashEventResult> {
        let ctx = self.create_context();
        let ext_event = ExtensionEvent::UserBash(event);

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("user_bash") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                match std::panic::AssertUnwindSafe(handler(ext_event.clone(), ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        if let Ok(result) = serde_json::from_value::<UserBashEventResult>(val) {
                            return Some(result);
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "user_bash".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        None
    }

    /// Emit an input event. Transform chain with "handled" short-circuit.
    ///
    /// Mirrors the TypeScript `emitInput` which chains transform results and
    /// short-circuits on "handled".
    pub async fn emit_input(&self, event: InputEvent) -> Option<InputEventResult> {
        let ctx = self.create_context();
        let original_text = event.text.clone();
        let original_images = event.images.clone();
        let mut current_text = event.text.clone();
        let mut current_images = event.images.clone();
        let source = event.source;

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("input") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                let input_event = ExtensionEvent::Input(InputEvent {
                    text: current_text.clone(),
                    images: current_images.clone(),
                    source,
                });

                match std::panic::AssertUnwindSafe(handler(input_event, ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        if let Some(action) = val.get("action").and_then(|v| v.as_str()) {
                            match action {
                                "handled" => return Some(InputEventResult::Handled),
                                "transform" => {
                                    if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
                                        current_text = text.to_string();
                                    }
                                    if let Some(imgs) = val.get("images") {
                                        current_images = serde_json::from_value::<Vec<ImageContent>>(imgs.clone()).ok();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "input".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        // Return transform if text or images changed
        if current_text != original_text || current_images != original_images {
            Some(InputEventResult::Transform {
                text: current_text,
                images: current_images,
            })
        } else {
            Some(InputEventResult::Continue)
        }
    }

    /// Emit a resources_discover event and collect results.
    pub async fn emit_resources_discover(
        &self,
        event: super::types::ResourcesDiscoverEvent,
    ) -> super::types::ResourcesDiscoverResult {
        let ctx = self.create_context();
        let ext_event = ExtensionEvent::ResourcesDiscover(event);
        let mut combined = super::types::ResourcesDiscoverResult::default();

        for ext in &self.extensions {
            let handlers = match ext.handlers.get("resources_discover") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };

            for handler in handlers {
                match std::panic::AssertUnwindSafe(handler(ext_event.clone(), ctx.clone()))
                    .catch_unwind()
                    .await
                {
                    Ok(Some(val)) => {
                        if let Ok(result) =
                            serde_json::from_value::<super::types::ResourcesDiscoverResult>(val)
                        {
                            if let Some(paths) = result.skill_paths {
                                combined
                                    .skill_paths
                                    .get_or_insert_with(Vec::new)
                                    .extend(paths);
                            }
                            if let Some(paths) = result.prompt_paths {
                                combined
                                    .prompt_paths
                                    .get_or_insert_with(Vec::new)
                                    .extend(paths);
                            }
                            if let Some(paths) = result.theme_paths {
                                combined
                                    .theme_paths
                                    .get_or_insert_with(Vec::new)
                                    .extend(paths);
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(_panic) => {
                        self.emit_error(ExtensionError {
                            extension_path: ext.path.clone(),
                            event: "resources_discover".to_string(),
                            error: "handler panicked".to_string(),
                            stack: None,
                        });
                    }
                }
            }
        }

        combined
    }
}

/// Helper function to emit session_shutdown event.
pub async fn emit_session_shutdown_event(runner: Option<&ExtensionRunner>) -> bool {
    if let Some(runner) = runner {
        if runner.has_handlers("session_shutdown") {
            runner
                .emit(ExtensionEvent::SessionShutdown(
                    super::types::SessionShutdownEvent,
                ))
                .await;
            return true;
        }
    }
    false
}
