use std::sync::{Arc, Mutex};

use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStreamReceiver,
    AssistantMessageEventStreamSender,
};
use crate::providers::LlmProvider;
use crate::registry::{register_api_provider, unregister_api_providers, ApiProvider};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, InputModality, Model,
    ModelCost, SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent,
    ToolCall, Usage,
};

// ---------------------------------------------------------------------------
// Constants (mirrors TS defaults)
// ---------------------------------------------------------------------------

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
#[allow(dead_code)]
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

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

/// Options for [`faux_assistant_message_with_options`].
#[derive(Debug, Clone, Default)]
pub struct FauxAssistantMessageOptions {
    pub stop_reason: Option<StopReason>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<u64>,
}

/// Build a complete faux assistant message from content blocks.
pub fn faux_assistant_message(content: Vec<AssistantContent>) -> AssistantMessage {
    AssistantMessage {
        content,
        api: DEFAULT_API.to_string(),
        provider: DEFAULT_PROVIDER.to_string(),
        model: DEFAULT_MODEL_ID.to_string(),
        response_id: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 0,
    }
}

/// Build a faux assistant message with full options (mirrors TS `fauxAssistantMessage`).
pub fn faux_assistant_message_with_options(
    content: Vec<AssistantContent>,
    options: FauxAssistantMessageOptions,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: DEFAULT_API.to_string(),
        provider: DEFAULT_PROVIDER.to_string(),
        model: DEFAULT_MODEL_ID.to_string(),
        response_id: options.response_id,
        usage: Usage::default(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        error_message: options.error_message,
        timestamp: options.timestamp.unwrap_or(0),
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
///
/// Port of `FauxModelDefinition` from `packages/ai/src/providers/faux.ts`.
#[derive(Debug, Clone)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: bool,
    pub input: Option<Vec<InputModality>>,
    pub cost: Option<ModelCost>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
}

impl Default for FauxModelDefinition {
    fn default() -> Self {
        Self {
            id: DEFAULT_MODEL_ID.to_string(),
            name: None,
            reasoning: false,
            input: None,
            cost: None,
            context_window: None,
            max_tokens: None,
        }
    }
}

fn make_faux_model(api: &str, provider: &str, def: &FauxModelDefinition) -> Model {
    Model {
        id: def.id.clone(),
        name: def.name.clone().unwrap_or_else(|| def.id.clone()),
        api: api.to_string(),
        provider: provider.to_string(),
        base_url: DEFAULT_BASE_URL.to_string(),
        reasoning: def.reasoning,
        input: def
            .input
            .clone()
            .unwrap_or_else(|| vec![InputModality::Text, InputModality::Image]),
        cost: def.cost.clone().unwrap_or_default(),
        context_window: def.context_window.unwrap_or(128_000),
        max_tokens: def.max_tokens.unwrap_or(16_384),
        headers: None,
        compat: None,
        supported_thinking_levels: None,
    }
}

// ---------------------------------------------------------------------------
// Streaming helpers (port of TS private functions)
// ---------------------------------------------------------------------------

/// Approximate token count from character length (mirrors TS `estimateTokens`).
#[allow(dead_code)]
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Generate a random `usize` using OS entropy.
fn random_usize() -> usize {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("getrandom failed");
    usize::from_le_bytes(buf)
}

/// Split text into variable-sized chunks simulating token-by-token streaming
/// (mirrors TS `splitStringByTokenSize`).
fn split_string_by_token_size(
    text: &str,
    min_token_size: usize,
    max_token_size: usize,
) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        let range = max_token_size - min_token_size + 1;
        let token_size = min_token_size + (random_usize() % range);
        let char_size = token_size.max(1) * 4;
        let end = (index + char_size).min(chars.len());
        chunks.push(chars[index..end].iter().collect());
        index = end;
    }

    chunks
}

/// Emit the full event sequence for a response, using progressive partial
/// accumulation and chunked deltas (mirrors TS `streamWithDeltas`).
fn stream_with_deltas(
    sender: &mut AssistantMessageEventStreamSender,
    message: &AssistantMessage,
    min_token_size: usize,
    max_token_size: usize,
) {
    // Start with empty content (matches TS: `{ ...message, content: [] }`)
    let mut partial = AssistantMessage {
        content: Vec::new(),
        ..message.clone()
    };

    sender.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    for (i, block) in message.content.iter().enumerate() {
        match block {
            AssistantContent::Thinking(tc) => {
                partial.content.push(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: tc.thinking_signature.clone(),
                    redacted: tc.redacted.clone(),
                }));
                sender.push(AssistantMessageEvent::ThinkingStart {
                    content_index: i,
                    partial: partial.clone(),
                });

                for chunk in split_string_by_token_size(&tc.thinking, min_token_size, max_token_size)
                {
                    if let Some(AssistantContent::Thinking(t)) = partial.content.get_mut(i) {
                        t.thinking.push_str(&chunk);
                    }
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: i,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }

                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: i,
                    content: tc.thinking.clone(),
                    partial: partial.clone(),
                });
            }
            AssistantContent::Text(tc) => {
                partial.content.push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: tc.text_signature.clone(),
                }));
                sender.push(AssistantMessageEvent::TextStart {
                    content_index: i,
                    partial: partial.clone(),
                });

                for chunk in split_string_by_token_size(&tc.text, min_token_size, max_token_size) {
                    if let Some(AssistantContent::Text(t)) = partial.content.get_mut(i) {
                        t.text.push_str(&chunk);
                    }
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: i,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }

                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: i,
                    content: tc.text.clone(),
                    partial: partial.clone(),
                });
            }
            AssistantContent::ToolCall(tc) => {
                partial.content.push(AssistantContent::ToolCall(ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: serde_json::Value::Object(serde_json::Map::new()),
                    thought_signature: tc.thought_signature.clone(),
                }));
                sender.push(AssistantMessageEvent::ToolcallStart {
                    content_index: i,
                    partial: partial.clone(),
                });

                let args_json = serde_json::to_string(&tc.arguments).unwrap_or_default();
                for chunk in
                    split_string_by_token_size(&args_json, min_token_size, max_token_size)
                {
                    sender.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: i,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }

                // Set final arguments in partial (mirrors TS line 377)
                if let Some(AssistantContent::ToolCall(t)) = partial.content.get_mut(i) {
                    t.arguments = tc.arguments.clone();
                }
                sender.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: i,
                    tool_call: tc.clone(),
                    partial: partial.clone(),
                });
            }
        }
    }

    // Terminal event — TS emits Error for both "error" and "aborted" stop reasons
    if message.stop_reason == StopReason::Error || message.stop_reason == StopReason::Aborted {
        sender.push(AssistantMessageEvent::Error {
            reason: message.stop_reason,
            error: message.clone(),
        });
    } else {
        sender.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message: message.clone(),
        });
    }
}

// ---------------------------------------------------------------------------
// FauxProvider — response-queue based mock provider
// ---------------------------------------------------------------------------

/// Shared mutable state for the faux provider.
struct FauxState {
    responses: Vec<AssistantMessage>,
    call_count: usize,
    min_token_size: usize,
    max_token_size: usize,
}

/// A mock LLM provider that dequeues scripted responses.
///
/// Mirrors the TypeScript `registerFauxProvider` pattern: responses are set
/// via [`FauxProviderRegistration::set_responses`] and consumed in order.
pub struct FauxProvider {
    state: Arc<Mutex<FauxState>>,
}

impl FauxProvider {
    fn new(min_token_size: usize, max_token_size: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(FauxState {
                responses: Vec::new(),
                call_count: 0,
                min_token_size,
                max_token_size,
            })),
        }
    }

    fn emit_response(
        &self,
        model: &Model,
        _context: Context,
    ) -> AssistantMessageEventStreamReceiver {
        let (mut sender, receiver) = create_assistant_message_event_stream();

        // Extract response and config under lock, then release immediately
        let (response, min_token_size, max_token_size) = {
            let mut guard = self.state.lock().expect("faux state poisoned");
            guard.call_count += 1;

            let response = if guard.responses.is_empty() {
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

            (response, guard.min_token_size, guard.max_token_size)
        };

        // Spawn async task to emit events (mirrors TS queueMicrotask pattern)
        tokio::spawn(async move {
            stream_with_deltas(&mut sender, &response, min_token_size, max_token_size);
        });

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

/// Token size configuration for streaming simulation.
#[derive(Debug, Clone)]
pub struct TokenSize {
    pub min: Option<usize>,
    pub max: Option<usize>,
}

/// Options for [`register_faux_provider`].
///
/// Port of `RegisterFauxProviderOptions` from `packages/ai/src/providers/faux.ts`.
#[derive(Debug, Clone, Default)]
pub struct RegisterFauxProviderOptions {
    pub models: Option<Vec<FauxModelDefinition>>,
    /// Custom API identifier. Defaults to a generated UUID.
    pub api: Option<String>,
    /// Custom provider name. Defaults to `"faux"`.
    pub provider: Option<String>,
    /// Simulated tokens per second (unused in sync mode, reserved for future
    /// async delay support).
    pub tokens_per_second: Option<f64>,
    /// Min/max token size for chunked delta simulation.
    pub token_size: Option<TokenSize>,
}

/// Register a faux provider in the global API registry and return a
/// registration handle.
///
/// This is the Rust equivalent of the TypeScript `registerFauxProvider()`.
pub fn register_faux_provider(options: RegisterFauxProviderOptions) -> FauxProviderRegistration {
    let source_id = format!("faux-{}", uuid::Uuid::new_v4());
    let api = options
        .api
        .unwrap_or_else(|| format!("faux-{}", uuid::Uuid::new_v4()));
    let provider_name = options.provider.as_deref().unwrap_or(DEFAULT_PROVIDER);

    let raw_min = options
        .token_size
        .as_ref()
        .and_then(|ts| ts.min)
        .unwrap_or(DEFAULT_MIN_TOKEN_SIZE);
    let raw_max = options
        .token_size
        .as_ref()
        .and_then(|ts| ts.max)
        .unwrap_or(DEFAULT_MAX_TOKEN_SIZE);
    // Mirrors TS: min = max(1, min(opts.min, opts.max)), max = max(min, opts.max)
    let min_token_size = raw_min.min(raw_max).max(1);
    let max_token_size = raw_max.max(min_token_size);

    let model_defs = options
        .models
        .unwrap_or_else(|| vec![FauxModelDefinition::default()]);
    let models: Vec<Model> = model_defs
        .iter()
        .map(|d| make_faux_model(&api, provider_name, d))
        .collect();

    let faux = FauxProvider::new(min_token_size, max_token_size);
    let shared_state = Arc::clone(&faux.state);

    let faux = Arc::new(faux);

    let stream_faux = Arc::clone(&faux);
    let stream_fn: crate::registry::StreamFn = Arc::new(move |model, context, _options| {
        stream_faux.emit_response(model, context)
    });

    let simple_faux = Arc::clone(&faux);
    let stream_simple_fn: crate::registry::StreamSimpleFn =
        Arc::new(move |model, context, _options| simple_faux.emit_response(model, context));

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
