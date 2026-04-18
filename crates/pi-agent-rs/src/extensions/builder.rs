//! Extension builder — implements the `ExtensionAPI` trait for building extensions.
//!
//! This is the `pi` object passed to extension factory functions, used to
//! register event handlers, tools, commands, shortcuts, and flags.
//!
//! During the load phase the builder also delegates runtime action calls
//! (sendMessage, setModel, etc.) to the shared `ExtensionRuntime` so that
//! extensions can use them even during initialization.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::AgentTool;

use super::types::{
    BoxFuture, CustomMessage, EventBus, Extension, ExtensionAPI, ExtensionFlag,
    ExtensionRuntime, ExtensionShortcut, ExtensionToolDefinition, HandlerFn, MessageRenderer,
    PendingProviderRegistration, ProviderConfig, RegisteredCommand, SendMessageOptions,
    SendUserMessageOptions, SlashCommandInfo, SourceInfo, SourceType, ThinkingLevel, ToolInfo,
    UserMessageContent,
};
use pi_ai_rs::Model;

/// Builder for constructing an [`Extension`] via the [`ExtensionAPI`] trait.
///
/// ```ignore
/// let mut builder = ExtensionBuilder::new("/path/to/extension", runtime, event_bus);
/// // extension factory calls builder.on("tool_call", handler), etc.
/// let extension = builder.build();
/// ```
pub struct ExtensionBuilder {
    path: String,
    resolved_path: String,
    source_type: SourceType,
    handlers: HashMap<String, Vec<HandlerFn>>,
    tools: HashMap<String, (ExtensionToolDefinition, Arc<dyn AgentTool>)>,
    message_renderers: HashMap<String, MessageRenderer>,
    commands: HashMap<String, RegisteredCommand>,
    flags: HashMap<String, ExtensionFlag>,
    shortcuts: HashMap<String, ExtensionShortcut>,
    flag_values: HashMap<String, serde_json::Value>,
    /// Shared runtime for delegating action calls.
    runtime: Arc<ExtensionRuntime>,
    /// Shared event bus for inter-extension communication.
    event_bus: EventBus,
}

impl ExtensionBuilder {
    /// Create a new builder for an extension at the given path.
    pub fn new(path: impl Into<String>, runtime: Arc<ExtensionRuntime>, event_bus: EventBus) -> Self {
        let path = path.into();
        Self {
            resolved_path: path.clone(),
            path,
            source_type: SourceType::Local,
            handlers: HashMap::new(),
            tools: HashMap::new(),
            message_renderers: HashMap::new(),
            commands: HashMap::new(),
            flags: HashMap::new(),
            shortcuts: HashMap::new(),
            flag_values: HashMap::new(),
            runtime,
            event_bus,
        }
    }

    /// Set the source type (Local, Wasm, BuiltIn).
    pub fn with_source_type(mut self, source_type: SourceType) -> Self {
        self.source_type = source_type;
        self
    }

    /// Set the resolved path (e.g., after symlink resolution).
    pub fn with_resolved_path(mut self, resolved_path: impl Into<String>) -> Self {
        self.resolved_path = resolved_path.into();
        self
    }

    /// Set initial flag values.
    pub fn with_flag_values(mut self, values: HashMap<String, serde_json::Value>) -> Self {
        self.flag_values = values;
        self
    }

    /// Build the final [`Extension`].
    pub fn build(self) -> Extension {
        Extension {
            path: self.path.clone(),
            resolved_path: self.resolved_path,
            source_info: SourceInfo {
                path: self.path,
                source_type: self.source_type,
            },
            handlers: self.handlers,
            tools: self.tools,
            message_renderers: self.message_renderers,
            commands: self.commands,
            flags: self.flags,
            shortcuts: self.shortcuts,
        }
    }
}

impl ExtensionAPI for ExtensionBuilder {
    fn on(&mut self, event_type: &str, handler: HandlerFn) {
        self.handlers
            .entry(event_type.to_string())
            .or_default()
            .push(handler);
    }

    fn register_tool(&mut self, tool: ExtensionToolDefinition, execute: Arc<dyn AgentTool>) {
        self.tools.insert(tool.name.clone(), (tool, execute));
    }

    fn register_command(&mut self, command: RegisteredCommand) {
        self.commands.insert(command.name.clone(), command);
    }

    fn register_shortcut(&mut self, shortcut: ExtensionShortcut) {
        self.shortcuts
            .insert(shortcut.shortcut.clone(), shortcut);
    }

    fn register_flag(&mut self, flag: ExtensionFlag) {
        // Set default value if provided
        if let Some(ref default) = flag.default {
            self.flag_values
                .entry(flag.name.clone())
                .or_insert_with(|| default.clone());
        }
        self.flags.insert(flag.name.clone(), flag);
    }

    fn get_flag(&self, name: &str) -> Option<serde_json::Value> {
        self.flag_values.get(name).cloned()
    }

    fn register_message_renderer(&mut self, custom_type: &str, renderer: MessageRenderer) {
        self.message_renderers
            .insert(custom_type.to_string(), renderer);
    }

    // =========================================================================
    // Action delegations to shared runtime
    // =========================================================================

    fn send_message(&self, message: CustomMessage, options: Option<SendMessageOptions>) {
        (self.runtime.send_message)(message, options);
    }

    fn send_user_message(&self, content: UserMessageContent, options: Option<SendUserMessageOptions>) {
        (self.runtime.send_user_message)(content, options);
    }

    fn append_entry(&self, custom_type: &str, data: Option<serde_json::Value>) {
        (self.runtime.append_entry)(custom_type, data);
    }

    fn set_session_name(&self, name: &str) {
        (self.runtime.set_session_name)(name);
    }

    fn get_session_name(&self) -> Option<String> {
        (self.runtime.get_session_name)()
    }

    fn set_label(&self, entry_id: &str, label: Option<&str>) {
        (self.runtime.set_label)(entry_id, label);
    }

    fn get_active_tools(&self) -> Vec<String> {
        (self.runtime.get_active_tools)()
    }

    fn get_all_tools(&self) -> Vec<ToolInfo> {
        (self.runtime.get_all_tools)()
    }

    fn set_active_tools(&self, tool_names: &[String]) {
        (self.runtime.set_active_tools)(tool_names);
    }

    fn refresh_tools(&self) {
        (self.runtime.refresh_tools)();
    }

    fn get_commands(&self) -> Vec<SlashCommandInfo> {
        (self.runtime.get_commands)()
    }

    fn set_model(&self, model: Model) -> BoxFuture<'static, bool> {
        (self.runtime.set_model)(model)
    }

    fn get_thinking_level(&self) -> ThinkingLevel {
        (self.runtime.get_thinking_level)()
    }

    fn set_thinking_level(&self, level: ThinkingLevel) {
        (self.runtime.set_thinking_level)(level);
    }

    fn register_provider(&mut self, name: &str, config: ProviderConfig) {
        // During load phase, queue the registration in the shared runtime
        if let Ok(mut pending) = self.runtime.pending_provider_registrations.lock() {
            pending.push(PendingProviderRegistration {
                name: name.to_string(),
                config,
                extension_path: self.path.clone(),
            });
        }
    }

    fn unregister_provider(&mut self, name: &str) {
        (self.runtime.unregister_provider)(name, Some(&self.path));
    }

    fn events(&self) -> &EventBus {
        &self.event_bus
    }
}
