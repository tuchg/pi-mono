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
pub fn register_api_provider(provider: ApiProvider, source_id: Option<&str>) {
    let mut registry = global_registry().write().expect("registry poisoned");
    registry.insert(
        provider.api.clone(),
        RegisteredProvider {
            provider: Arc::new(provider),
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
