use std::sync::{Arc, Mutex};

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStreamReceiver,
};
use crate::providers::LlmProvider;
use crate::registry::{register_api_provider, unregister_api_providers, ApiProvider};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, InputModality,
    Model, ModelCost, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingContent, ToolCall, Usage,
};

// ---------------------------------------------------------------------------
// Helper constructors (mirrors TS fauxText / fauxThinking / fauxToolCall / fauxAssistantMessage)
// ---------------------------------------------------------------------------

/// Create a text content block.
pub fn faux_text(text: &str) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: text.to_string(),
        text_signature: None,
    })
}

/// Create a thinking content block.
pub fn faux_thinking(thinking: &str) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: thinking.to_string(),
        thinking_signature: None,
        redacted: None,
    })
}

/// Create a tool call content block.
pub fn faux_tool_call(name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: format!("tool-{}", uuid::Uuid::new_v4()),
        name: name.to_string(),
        arguments,
        thought_signature: None,
    })
}

/// Create a tool call content block with a specific id.
pub fn faux_tool_call_with_id(
    id: &str,
    name: &str,
    arguments: serde_json::Value,
) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
        thought_signature: None,
    })
}

/// Build a complete faux assistant message from content blocks.
pub fn faux_assistant_message(content: Vec<AssistantContent>) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "faux".to_string(),
        provider: "faux".to_string(),
        model: "faux-1".to_string(),
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 0,
    }
}

/// Build a faux assistant message with a specific stop reason.
pub fn faux_assistant_message_with_stop(
    content: Vec<AssistantContent>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        stop_reason,
        ..faux_assistant_message(content)
    }
}

/// Build a simple text-only faux assistant message.
pub fn faux_assistant_text(text: &str) -> AssistantMessage {
    faux_assistant_message(vec![faux_text(text)])
}

// ---------------------------------------------------------------------------
// Model definition for faux provider
// ---------------------------------------------------------------------------

/// Definition used to configure models in the faux provider.
#[derive(Debug, Clone)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: bool,
}

impl Default for FauxModelDefinition {
    fn default() -> Self {
        Self {
            id: "faux-1".to_string(),
            name: None,
            reasoning: false,
        }
    }
}

/// Create a default faux model.
fn make_faux_model(api: &str, provider: &str, def: &FauxModelDefinition) -> Model {
    Model {
        id: def.id.clone(),
        name: def.name.clone().unwrap_or_else(|| def.id.clone()),
        api: api.to_string(),
        provider: provider.to_string(),
        base_url: "http://localhost:0".to_string(),
        reasoning: def.reasoning,
        input: vec![InputModality::Text, InputModality::Image],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        headers: None,
        compat: None,
        supported_thinking_levels: None,
    }
}

// ---------------------------------------------------------------------------
// FauxProvider — response-queue based mock provider
// ---------------------------------------------------------------------------

/// Shared mutable state for the faux provider.
struct FauxState {
    responses: Vec<AssistantMessage>,
    call_count: usize,
}

/// A mock LLM provider that dequeues scripted responses.
///
/// Mirrors the TypeScript `registerFauxProvider` pattern: responses are set
/// via [`FauxProviderRegistration::set_responses`] and consumed in order.
pub struct FauxProvider {
    state: Arc<Mutex<FauxState>>,
}

impl FauxProvider {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(FauxState {
                responses: Vec::new(),
                call_count: 0,
            })),
        }
    }

    fn emit_response(
        &self,
        model: &Model,
        _context: Context,
    ) -> AssistantMessageEventStreamReceiver {
        let (mut sender, receiver) = create_assistant_message_event_stream();
        let mut guard = self.state.lock().expect("faux state poisoned");
        guard.call_count += 1;

        let response = if guard.responses.is_empty() {
            // No more responses queued — emit an error like the TS version.
            AssistantMessage {
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_id: None,
                usage: Usage::default(),
                stop_reason: StopReason::Error,
                error_message: Some("No more faux responses queued".to_string()),
                timestamp: 0,
            }
        } else {
            let mut msg = guard.responses.remove(0);
            msg.api = model.api.clone();
            msg.provider = model.provider.clone();
            msg.model = model.id.clone();
            msg
        };

        // Emit the standard event sequence for a complete response.
        sender.push(AssistantMessageEvent::Start {
            partial: response.clone(),
        });

        for (i, content) in response.content.iter().enumerate() {
            match content {
                AssistantContent::Text(tc) => {
                    sender.push(AssistantMessageEvent::TextStart {
                        content_index: i,
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: i,
                        delta: tc.text.clone(),
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::TextEnd {
                        content_index: i,
                        content: tc.text.clone(),
                        partial: response.clone(),
                    });
                }
                AssistantContent::Thinking(tc) => {
                    sender.push(AssistantMessageEvent::ThinkingStart {
                        content_index: i,
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: i,
                        delta: tc.thinking.clone(),
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: i,
                        content: tc.thinking.clone(),
                        partial: response.clone(),
                    });
                }
                AssistantContent::ToolCall(tc) => {
                    sender.push(AssistantMessageEvent::ToolcallStart {
                        content_index: i,
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: i,
                        tool_call: tc.clone(),
                        partial: response.clone(),
                    });
                }
            }
        }

        if response.stop_reason == StopReason::Error {
            sender.push(AssistantMessageEvent::Error {
                reason: response.stop_reason,
                error: response,
            });
        } else {
            sender.push(AssistantMessageEvent::Done {
                reason: response.stop_reason,
                message: response,
            });
        }

        receiver
    }
}

impl LlmProvider for FauxProvider {
    fn api(&self) -> &str {
        "faux"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStreamReceiver {
        self.emit_response(model, context)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        _options: SimpleStreamOptions,
    ) -> AssistantMessageEventStreamReceiver {
        self.emit_response(model, context)
    }
}

// ---------------------------------------------------------------------------
// FauxProviderRegistration — the public API for test setup
// ---------------------------------------------------------------------------

/// Handle returned by [`register_faux_provider`]. Provides methods to set
/// responses and clean up after the test.
pub struct FauxProviderRegistration {
    /// The API identifier used for this registration.
    pub api: String,
    /// The models registered.
    pub models: Vec<Model>,
    state: Arc<Mutex<FauxState>>,
    source_id: String,
}

impl FauxProviderRegistration {
    /// Replace the pending response queue.
    pub fn set_responses(&self, responses: Vec<AssistantMessage>) {
        let mut guard = self.state.lock().expect("faux state poisoned");
        guard.responses = responses;
    }

    /// Append responses to the end of the queue.
    pub fn append_responses(&self, responses: Vec<AssistantMessage>) {
        let mut guard = self.state.lock().expect("faux state poisoned");
        guard.responses.extend(responses);
    }

    /// How many responses remain in the queue.
    pub fn pending_response_count(&self) -> usize {
        let guard = self.state.lock().expect("faux state poisoned");
        guard.responses.len()
    }

    /// How many times the provider has been called.
    pub fn call_count(&self) -> usize {
        let guard = self.state.lock().expect("faux state poisoned");
        guard.call_count
    }

    /// Get the first (default) model.
    pub fn get_model(&self) -> &Model {
        &self.models[0]
    }

    /// Get a model by ID.
    pub fn get_model_by_id(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Unregister the faux provider from the global registry.
    pub fn unregister(&self) {
        unregister_api_providers(&self.source_id);
    }
}

/// Options for [`register_faux_provider`].
#[derive(Debug, Clone, Default)]
pub struct RegisterFauxProviderOptions {
    pub models: Option<Vec<FauxModelDefinition>>,
}

/// Register a faux provider in the global API registry and return a
/// registration handle.
///
/// This is the Rust equivalent of the TypeScript `registerFauxProvider()`.
pub fn register_faux_provider(
    options: RegisterFauxProviderOptions,
) -> FauxProviderRegistration {
    let source_id = format!("faux-{}", uuid::Uuid::new_v4());
    let api = format!("faux-{}", uuid::Uuid::new_v4());
    let provider_name = "faux";

    let model_defs = options.models.unwrap_or_else(|| vec![FauxModelDefinition::default()]);
    let models: Vec<Model> = model_defs
        .iter()
        .map(|d| make_faux_model(&api, provider_name, d))
        .collect();

    let faux = FauxProvider::new();
    let shared_state = Arc::clone(&faux.state);

    // Wrap the FauxProvider in Arc so we can share it between closures.
    let faux = Arc::new(faux);

    let stream_faux = Arc::clone(&faux);
    let stream_fn: crate::registry::StreamFn = Arc::new(move |model, context, _options| {
        stream_faux.emit_response(model, context)
    });

    let simple_faux = Arc::clone(&faux);
    let stream_simple_fn: crate::registry::StreamSimpleFn =
        Arc::new(move |model, context, _options| {
            simple_faux.emit_response(model, context)
        });

    register_api_provider(
        ApiProvider {
            api: api.clone(),
            stream: stream_fn,
            stream_simple: stream_simple_fn,
        },
        Some(&source_id),
    );

    FauxProviderRegistration {
        api,
        models,
        state: shared_state,
        source_id,
    }
}
