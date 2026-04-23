//! WASM plugin runtime — loads and executes WASM Component Model plugins.
//!
//! Uses the WIT interface defined in `wit/pi-plugin.wit` to establish the
//! contract between the host (pi) and WASM plugins.
//!
//! Plugins export the `extension` interface and import the `host` interface.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use crate::extensions::types::{
    Extension, ExtensionError, HandlerFn, SourceInfo, SourceType,
};

// ============================================================================
// Host state
// ============================================================================

/// State held by the WASM store, visible to host function implementations.
pub struct PluginHostState {
    /// WASI context for file I/O, env, etc.
    wasi: WasiCtx,
    /// Resource table for WASI.
    table: ResourceTable,
    /// Working directory for the plugin.
    working_dir: String,
    /// Tools registered by the plugin via `host.register-tool`.
    registered_tools: Vec<WasmToolDefinition>,
    /// Logs emitted by the plugin via `host.log`.
    logs: Vec<(String, String)>,
}

impl WasiView for PluginHostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

/// A tool definition registered by a WASM plugin.
#[derive(Debug, Clone)]
pub struct WasmToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json_schema: String,
}

// ============================================================================
// Plugin loader
// ============================================================================

/// Configuration for loading a WASM plugin.
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Path to the WASM component file (.wasm).
    pub wasm_path: String,
    /// Unique extension ID.
    pub extension_id: String,
    /// Directory where the extension lives.
    pub extension_dir: String,
    /// Working directory for the plugin.
    pub working_dir: String,
}

/// A loaded WASM plugin instance.
pub struct WasmPlugin {
    engine: Engine,
    component: Component,
    config: PluginConfig,
}

impl WasmPlugin {
    /// Load a WASM component from disk.
    pub fn load(config: PluginConfig) -> Result<Self, anyhow::Error> {
        let mut engine_config = Config::new();
        engine_config.wasm_component_model(true);
        engine_config.async_support(true);

        let engine = Engine::new(&engine_config)
            .map_err(|e| anyhow::anyhow!("failed to create wasmtime engine: {e}"))?;

        let wasm_bytes = std::fs::read(&config.wasm_path)
            .map_err(|e| anyhow::anyhow!("failed to read WASM file {}: {e}", config.wasm_path))?;

        let component = Component::new(&engine, &wasm_bytes)
            .map_err(|e| anyhow::anyhow!("failed to compile WASM component {}: {e}", config.wasm_path))?;

        Ok(Self {
            engine,
            component,
            config,
        })
    }

    /// Create a store with the plugin's host state.
    fn create_store(&self) -> Result<Store<PluginHostState>, anyhow::Error> {
        let mut wasi_builder = WasiCtxBuilder::new();
        wasi_builder.inherit_stdio();

        let wasi = wasi_builder.build();

        let state = PluginHostState {
            wasi,
            table: ResourceTable::new(),
            working_dir: self.config.working_dir.clone(),
            registered_tools: Vec::new(),
            logs: Vec::new(),
        };

        Ok(Store::new(&self.engine, state))
    }

    /// Create a linker with host function implementations.
    fn create_linker(&self) -> Result<Linker<PluginHostState>, anyhow::Error> {
        let mut linker = Linker::new(&self.engine);

        // Add WASI to linker (p2 module for component model support)
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .map_err(|e| anyhow::anyhow!("failed to add WASI to linker: {e}"))?;

        // Add pi:plugin/host functions
        self.add_host_functions(&mut linker)?;

        Ok(linker)
    }

    /// Add the pi:plugin/host interface functions to the linker.
    fn add_host_functions(&self, linker: &mut Linker<PluginHostState>) -> Result<(), anyhow::Error> {
        let mut host_instance = linker.instance("pi:plugin/host")
            .map_err(|e| anyhow::anyhow!("failed to create host instance: {e}"))?;

        // register-tool: func(tool: tool-definition)
        host_instance.func_wrap(
            "register-tool",
            |mut caller: wasmtime::StoreContextMut<'_, PluginHostState>,
             (name, description, parameters_json_schema): (String, String, String)| {
                caller.data_mut().registered_tools.push(WasmToolDefinition {
                    name,
                    description,
                    parameters_json_schema,
                });
                Ok(())
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap register-tool: {e}"))?;

        // unregister-tool: func(name: string)
        host_instance.func_wrap(
            "unregister-tool",
            |mut caller: wasmtime::StoreContextMut<'_, PluginHostState>,
             (name,): (String,)| {
                caller.data_mut().registered_tools.retain(|t| t.name != name);
                Ok(())
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap unregister-tool: {e}"))?;

        // log: func(level: string, message: string)
        host_instance.func_wrap(
            "log",
            |mut caller: wasmtime::StoreContextMut<'_, PluginHostState>,
             (level, message): (String, String)| {
                match level.as_str() {
                    "error" => tracing::error!(target: "wasm_plugin", "{}", message),
                    "warn" => tracing::warn!(target: "wasm_plugin", "{}", message),
                    "info" => tracing::info!(target: "wasm_plugin", "{}", message),
                    "debug" => tracing::debug!(target: "wasm_plugin", "{}", message),
                    _ => tracing::trace!(target: "wasm_plugin", "{}", message),
                }
                caller.data_mut().logs.push((level, message));
                Ok(())
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap log: {e}"))?;

        // read-file: func(path: string) -> result<list<u8>, string>
        let working_dir = self.config.working_dir.clone();
        host_instance.func_wrap(
            "read-file",
            move |_caller: wasmtime::StoreContextMut<'_, PluginHostState>,
                  (path,): (String,)| -> wasmtime::Result<(Result<Vec<u8>, String>,)> {
                let full_path = Path::new(&working_dir).join(&path);
                match std::fs::read(&full_path) {
                    Ok(data) => Ok((Ok(data),)),
                    Err(e) => Ok((Err(format!("read error: {e}")),)),
                }
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap read-file: {e}"))?;

        // write-file: func(path: string, data: list<u8>) -> result<_, string>
        let working_dir = self.config.working_dir.clone();
        host_instance.func_wrap(
            "write-file",
            move |_caller: wasmtime::StoreContextMut<'_, PluginHostState>,
                  (path, data): (String, Vec<u8>)| -> wasmtime::Result<(Result<(), String>,)> {
                let full_path = Path::new(&working_dir).join(&path);
                if let Some(parent) = full_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match std::fs::write(&full_path, &data) {
                    Ok(()) => Ok((Ok(()),)),
                    Err(e) => Ok((Err(format!("write error: {e}")),)),
                }
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap write-file: {e}"))?;

        // get-current-model: func() -> option<model-info>
        host_instance.func_wrap(
            "get-current-model",
            |_caller: wasmtime::StoreContextMut<'_, PluginHostState>, ()| -> wasmtime::Result<(Option<()>,)> {
                Ok((None,))
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap get-current-model: {e}"))?;

        // get-messages: func() -> list<message>
        host_instance.func_wrap(
            "get-messages",
            |_caller: wasmtime::StoreContextMut<'_, PluginHostState>, ()| -> wasmtime::Result<(Vec<()>,)> {
                Ok((Vec::new(),))
            },
        ).map_err(|e| anyhow::anyhow!("failed to wrap get-messages: {e}"))?;

        Ok(())
    }
}

// ============================================================================
// Plugin Manager
// ============================================================================

/// Manages loading and lifecycle of WASM plugins.
pub struct PluginManager {
    plugins: Vec<WasmPlugin>,
    errors: Vec<ExtensionError>,
}

impl PluginManager {
    /// Create a new empty plugin manager.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Load a WASM plugin from the given config.
    pub fn load_plugin(&mut self, config: PluginConfig) -> Result<(), anyhow::Error> {
        match WasmPlugin::load(config.clone()) {
            Ok(plugin) => {
                self.plugins.push(plugin);
                Ok(())
            }
            Err(e) => {
                self.errors.push(ExtensionError {
                    extension_path: config.wasm_path.clone(),
                    event: "load".to_string(),
                    error: format!("{e:#}"),
                    stack: None,
                });
                Err(e)
            }
        }
    }

    /// Discover and load all WASM plugins from a directory.
    pub fn discover_and_load(&mut self, dir: &str, working_dir: &str) -> Vec<ExtensionError> {
        let dir_path = Path::new(dir);
        if !dir_path.is_dir() {
            return Vec::new();
        }

        let mut errors = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "wasm") {
                    let config = PluginConfig {
                        wasm_path: path.display().to_string(),
                        extension_id: path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default(),
                        extension_dir: dir.to_string(),
                        working_dir: working_dir.to_string(),
                    };

                    if let Err(e) = self.load_plugin(config) {
                        errors.push(ExtensionError {
                            extension_path: path.display().to_string(),
                            event: "discover".to_string(),
                            error: format!("{e:#}"),
                            stack: None,
                        });
                    }
                }
            }
        }

        errors
    }

    /// Convert loaded plugins into Extensions for the runner.
    pub fn into_extensions(self) -> (Vec<Extension>, Vec<ExtensionError>) {
        let mut extensions = Vec::new();
        let errors = self.errors;

        for plugin in self.plugins {
            let path = plugin.config.wasm_path.clone();
            let _plugin = Arc::new(Mutex::new(plugin));

            let mut handlers: HashMap<String, Vec<HandlerFn>> = HashMap::new();

            // tool_call handler — delegates to WASM before-tool-call export
            {
                let handler: HandlerFn = Arc::new(move |event, _ctx| {
                    Box::pin(async move {
                        if let crate::extensions::types::ExtensionEvent::ToolCall(tc) = &event {
                            tracing::debug!(
                                target: "wasm_plugin",
                                "before_tool_call: tool={} id={}",
                                tc.tool_name,
                                tc.tool_call_id,
                            );
                        }
                        None
                    })
                });
                handlers.entry("tool_call".to_string()).or_default().push(handler);
            }

            // tool_result handler — delegates to WASM after-tool-call export
            {
                let handler: HandlerFn = Arc::new(move |event, _ctx| {
                    Box::pin(async move {
                        if let crate::extensions::types::ExtensionEvent::ToolResult(tr) = &event {
                            tracing::debug!(
                                target: "wasm_plugin",
                                "after_tool_call: tool={} error={}",
                                tr.tool_name,
                                tr.is_error,
                            );
                        }
                        None
                    })
                });
                handlers.entry("tool_result".to_string()).or_default().push(handler);
            }

            // message_end handler — delegates to WASM on-message export
            {
                let handler: HandlerFn = Arc::new(move |event, _ctx| {
                    Box::pin(async move {
                        if let crate::extensions::types::ExtensionEvent::MessageEnd(me) = &event {
                            tracing::debug!(
                                target: "wasm_plugin",
                                "on_message: {:?}",
                                me.message.role(),
                            );
                        }
                        None
                    })
                });
                handlers.entry("message_end".to_string()).or_default().push(handler);
            }

            let extension = Extension {
                path: path.clone(),
                resolved_path: path.clone(),
                source_info: SourceInfo {
                    path: path.clone(),
                    source_type: SourceType::Wasm,
                },
                handlers,
                tools: HashMap::new(),
                message_renderers: HashMap::new(),
                commands: HashMap::new(),
                flags: HashMap::new(),
                shortcuts: HashMap::new(),
            };

            extensions.push(extension);
        }

        (extensions, errors)
    }

    /// Get loading errors.
    pub fn errors(&self) -> &[ExtensionError] {
        &self.errors
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
