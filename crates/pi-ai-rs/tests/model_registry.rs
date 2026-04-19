use pi_ai_rs::models::ModelRegistry;
use pi_ai_rs::types::{InputModality, Model, ModelCost, Usage};

fn test_model(id: &str, provider: &str) -> Model {
    Model {
        id: id.to_string(),
        name: id.to_string(),
        api: "test".to_string(),
        provider: provider.to_string(),
        base_url: String::new(),
        reasoning: false,
        input: vec![InputModality::Text],
        cost: ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 128_000,
        max_tokens: 4096,
        headers: None,
        compat: None,
        supported_thinking_levels: None,
    }
}

#[test]
fn register_and_lookup() {
    let mut registry = ModelRegistry::new();
    let model = test_model("gpt-4o", "openai");
    registry.register(model.clone());

    let found = registry.get_model("openai", "gpt-4o");
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, "gpt-4o");
}

#[test]
fn lookup_missing_returns_none() {
    let registry = ModelRegistry::new();
    assert!(registry.get_model("openai", "nonexistent").is_none());
}

#[test]
fn get_providers_lists_all() {
    let mut registry = ModelRegistry::new();
    registry.register(test_model("m1", "provider-a"));
    registry.register(test_model("m2", "provider-b"));

    let mut providers = registry.get_providers();
    providers.sort();
    assert_eq!(providers, vec!["provider-a", "provider-b"]);
}

#[test]
fn get_models_for_provider() {
    let mut registry = ModelRegistry::new();
    registry.register(test_model("m1", "p"));
    registry.register(test_model("m2", "p"));
    registry.register(test_model("m3", "other"));

    let models = registry.get_models("p");
    assert_eq!(models.len(), 2);
}

#[test]
fn calculate_cost() {
    let model = test_model("m1", "p");
    let usage = Usage {
        input: 1_000_000,
        output: 1_000_000,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 2_000_000,
        cost: Default::default(),
    };

    let cost = ModelRegistry::calculate_cost(&model, &usage);
    // input: 3.0 per million * 1M = 3.0
    assert!((cost.input - 3.0).abs() < 1e-6);
    // output: 15.0 per million * 1M = 15.0
    assert!((cost.output - 15.0).abs() < 1e-6);
    assert!((cost.total - 18.0).abs() < 1e-6);
}

#[test]
fn supports_xhigh() {
    let mut model = test_model("gpt-5.2-turbo", "openai");
    assert!(ModelRegistry::supports_xhigh(&model));

    model.id = "gpt-5.3-codex".to_string();
    assert!(ModelRegistry::supports_xhigh(&model));

    model.id = "gpt-5.4".to_string();
    assert!(ModelRegistry::supports_xhigh(&model));

    model.id = "claude-opus-4.6-preview".to_string();
    assert!(ModelRegistry::supports_xhigh(&model));

    model.id = "claude-opus-4-6-20250414".to_string();
    assert!(ModelRegistry::supports_xhigh(&model));

    model.id = "claude-opus-4-7".to_string();
    assert!(ModelRegistry::supports_xhigh(&model));

    model.id = "claude-opus-4.7-20250601".to_string();
    assert!(ModelRegistry::supports_xhigh(&model));

    // Non-xhigh models
    model.id = "gpt-4o".to_string();
    assert!(!ModelRegistry::supports_xhigh(&model));

    model.id = "gpt-5.1-codex".to_string();
    assert!(!ModelRegistry::supports_xhigh(&model));

    model.id = "claude-sonnet-4.6".to_string();
    assert!(!ModelRegistry::supports_xhigh(&model));

    model.id = "claude-sonnet-4-20250514".to_string();
    assert!(!ModelRegistry::supports_xhigh(&model));
}

#[test]
fn models_are_equal() {
    let a = test_model("m1", "p1");
    let b = test_model("m1", "p1");
    let c = test_model("m1", "p2");

    assert!(ModelRegistry::models_are_equal(Some(&a), Some(&b)));
    assert!(!ModelRegistry::models_are_equal(Some(&a), Some(&c)));
    assert!(!ModelRegistry::models_are_equal(Some(&a), None));
    assert!(!ModelRegistry::models_are_equal(None, None));
}

#[test]
fn calculate_cost_with_cache() {
    let mut model = test_model("m1", "p");
    model.cost.cache_read = 1.5;
    model.cost.cache_write = 6.0;

    let usage = Usage {
        input: 500_000,
        output: 500_000,
        cache_read: 1_000_000,
        cache_write: 200_000,
        total_tokens: 2_200_000,
        cost: Default::default(),
    };

    let cost = ModelRegistry::calculate_cost(&model, &usage);
    // input: 3.0/M * 500K = 1.5
    assert!((cost.input - 1.5).abs() < 1e-6);
    // output: 15.0/M * 500K = 7.5
    assert!((cost.output - 7.5).abs() < 1e-6);
    // cache_read: 1.5/M * 1M = 1.5
    assert!((cost.cache_read - 1.5).abs() < 1e-6);
    // cache_write: 6.0/M * 200K = 1.2
    assert!((cost.cache_write - 1.2).abs() < 1e-6);
    assert!((cost.total - 11.7).abs() < 1e-6);
}

#[test]
fn calculate_cost_zero_usage() {
    let model = test_model("m1", "p");
    let usage = Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 0,
        cost: Default::default(),
    };
    let cost = ModelRegistry::calculate_cost(&model, &usage);
    assert!((cost.total - 0.0).abs() < 1e-10);
}

#[test]
fn register_overwrites_same_model() {
    let mut registry = ModelRegistry::new();
    let m1 = test_model("m1", "p");
    let mut m1_v2 = test_model("m1", "p");
    m1_v2.name = "updated".to_string();

    registry.register(m1);
    registry.register(m1_v2);

    let models = registry.get_models("p");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "updated");
}

#[test]
fn get_models_empty_provider() {
    let registry = ModelRegistry::new();
    let models = registry.get_models("nonexistent");
    assert!(models.is_empty());
}

#[test]
fn default_creates_empty_registry() {
    let registry = ModelRegistry::default();
    assert!(registry.get_providers().is_empty());
}
