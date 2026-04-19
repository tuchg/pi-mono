use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::event_stream::AssistantMessageEventStreamReceiver;
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};

// ---------------------------------------------------------------------------
// Provider function types
// ---------------------------------------------------------------------------

/// Low-level streaming function — provider-specific options.
pub type StreamFn =
    Arc<dyn Fn(&Model, Context, StreamOptions) -> AssistantMessageEventStreamReceiver + Send + Sync>;

/// Simplified streaming function — reasoning-aware options.
pub type StreamSimpleFn =
    Arc<dyn Fn(&Model, Context, SimpleStreamOptions) -> AssistantMessageEventStreamReceiver + Send + Sync>;

/// A registered API provider with its stream functions.
pub struct ApiProvider {
    pub api: String,
    pub stream: StreamFn,
    pub stream_simple: StreamSimpleFn,
}

struct RegisteredProvider {
    provider: Arc<ApiProvider>,
    source_id: Option<String>,
}

fn global_registry() -> &'static RwLock<HashMap<String, RegisteredProvider>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, RegisteredProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Register an API provider. If `source_id` is provided it can be used to
/// unregister all providers from that source later.
///
/// The provider's `stream` and `stream_simple` functions are wrapped with a
/// runtime guard that panics when the model's `api` field does not match the
/// provider's registered API (mirrors the TS `wrapStream` / `wrapStreamSimple`
/// behaviour).
pub fn register_api_provider(provider: ApiProvider, source_id: Option<&str>) {
    let api = provider.api.clone();
    let inner_stream = provider.stream;
    let inner_simple = provider.stream_simple;

    let api_for_stream = api.clone();
    let wrapped_stream: StreamFn = Arc::new(move |model, ctx, opts| {
        assert_eq!(
            model.api, api_for_stream,
            "Mismatched api: {} expected {}",
            model.api, api_for_stream
        );
        inner_stream(model, ctx, opts)
    });

    let api_for_simple = api.clone();
    let wrapped_simple: StreamSimpleFn = Arc::new(move |model, ctx, opts| {
        assert_eq!(
            model.api, api_for_simple,
            "Mismatched api: {} expected {}",
            model.api, api_for_simple
        );
        inner_simple(model, ctx, opts)
    });

    let wrapped_provider = ApiProvider {
        api: api.clone(),
        stream: wrapped_stream,
        stream_simple: wrapped_simple,
    };

    let mut registry = global_registry().write().expect("registry poisoned");
    registry.insert(
        api,
        RegisteredProvider {
            provider: Arc::new(wrapped_provider),
            source_id: source_id.map(String::from),
        },
    );
}

/// Look up the provider for a given API identifier.
pub fn get_api_provider(api: &str) -> Option<Arc<ApiProvider>> {
    let registry = global_registry().read().expect("registry poisoned");
    registry.get(api).map(|r| Arc::clone(&r.provider))
}

/// Return all currently registered providers.
pub fn get_api_providers() -> Vec<Arc<ApiProvider>> {
    let registry = global_registry().read().expect("registry poisoned");
    registry.values().map(|r| Arc::clone(&r.provider)).collect()
}

/// Remove all providers that were registered with the given `source_id`.
pub fn unregister_api_providers(source_id: &str) {
    let mut registry = global_registry().write().expect("registry poisoned");
    registry.retain(|_, r| r.source_id.as_deref() != Some(source_id));
}

/// Remove all registered providers.
pub fn clear_api_providers() {
    let mut registry = global_registry().write().expect("registry poisoned");
    registry.clear();
}
