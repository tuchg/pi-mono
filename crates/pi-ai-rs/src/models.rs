use std::collections::HashMap;

use crate::types::{Model, Usage, UsageCost};

/// Registry of known models, indexed by provider then model ID.
pub struct ModelRegistry {
    providers: HashMap<String, HashMap<String, Model>>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a model under its provider.
    pub fn register(&mut self, model: Model) {
        self.providers
            .entry(model.provider.clone())
            .or_default()
            .insert(model.id.clone(), model);
    }

    /// Look up a model by provider and model ID.
    pub fn get_model(&self, provider: &str, model_id: &str) -> Option<&Model> {
        self.providers.get(provider)?.get(model_id)
    }

    /// Return all registered provider names.
    pub fn get_providers(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }

    /// Return all models for a given provider.
    pub fn get_models(&self, provider: &str) -> Vec<&Model> {
        self.providers
            .get(provider)
            .map(|m| m.values().collect())
            .unwrap_or_default()
    }

    /// Calculate dollar cost from token usage and model pricing.
    pub fn calculate_cost(model: &Model, usage: &Usage) -> UsageCost {
        let input = (model.cost.input / 1_000_000.0) * usage.input as f64;
        let output = (model.cost.output / 1_000_000.0) * usage.output as f64;
        let cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
        let cache_write = (model.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
        UsageCost {
            input,
            output,
            cache_read,
            cache_write,
            total: input + output + cache_read + cache_write,
        }
    }

    /// Check if a model supports the `xhigh` thinking level.
    pub fn supports_xhigh(model: &Model) -> bool {
        let id = &model.id;
        if id.contains("gpt-5.2") || id.contains("gpt-5.3") || id.contains("gpt-5.4") {
            return true;
        }
        if id.contains("opus-4-6")
            || id.contains("opus-4.6")
            || id.contains("opus-4-7")
            || id.contains("opus-4.7")
        {
            return true;
        }
        false
    }

    /// Check if two models refer to the same model (same id and provider).
    pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
            _ => false,
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
