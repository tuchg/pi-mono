//! Extension builder — implements the `ExtensionAPI` trait for building extensions.
//!
//! This is the `pi` object passed to extension factory functions, used to
//! register event handlers, tools, commands, shortcuts, and flags.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::AgentTool;

use super::types::{
    Extension, ExtensionAPI, ExtensionFlag, ExtensionShortcut, ExtensionToolDefinition,
    HandlerFn, RegisteredCommand, SourceInfo, SourceType,
};

/// Builder for constructing an [`Extension`] via the [`ExtensionAPI`] trait.
///
/// ```ignore
/// let mut builder = ExtensionBuilder::new("/path/to/extension");
/// // extension factory calls builder.on("tool_call", handler), etc.
/// let extension = builder.build();
/// ```
pub struct ExtensionBuilder {
    path: String,
    resolved_path: String,
    source_type: SourceType,
    handlers: HashMap<String, Vec<HandlerFn>>,
    tools: HashMap<String, (ExtensionToolDefinition, Arc<dyn AgentTool>)>,
    commands: HashMap<String, RegisteredCommand>,
    flags: HashMap<String, ExtensionFlag>,
    shortcuts: HashMap<String, ExtensionShortcut>,
    flag_values: HashMap<String, serde_json::Value>,
}

impl ExtensionBuilder {
    /// Create a new builder for an extension at the given path.
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            resolved_path: path.clone(),
            path,
            source_type: SourceType::Local,
            handlers: HashMap::new(),
            tools: HashMap::new(),
            commands: HashMap::new(),
            flags: HashMap::new(),
            shortcuts: HashMap::new(),
            flag_values: HashMap::new(),
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
}
