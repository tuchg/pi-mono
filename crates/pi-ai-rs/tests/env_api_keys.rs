//! Integration tests for environment-based API key resolution.
//!
//! Port of behaviors covered in TS `packages/ai/test/github-copilot-*.test.ts`
//! and the provider registration suite. Because env vars are process-global
//! these tests run on a single mutex-guarded thread.

use pi_ai_rs::get_env_api_key;
use std::sync::Mutex;

// All env-var mutating tests share this mutex to avoid interference.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Snapshot/restore helper for env vars.
struct EnvGuard {
    saved: Vec<(String, Option<String>)>,
}

impl EnvGuard {
    fn new(vars: &[&str]) -> Self {
        let saved = vars
            .iter()
            .map(|v| (v.to_string(), std::env::var(v).ok()))
            .collect();
        // Clear all listed vars before the test.
        for v in vars {
            unsafe { std::env::remove_var(v) };
        }
        Self { saved }
    }

    fn set(&self, key: &str, value: &str) {
        unsafe { std::env::set_var(key, value) };
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, v) in self.saved.drain(..) {
            match v {
                Some(val) => unsafe { std::env::set_var(&k, val) },
                None => unsafe { std::env::remove_var(&k) },
            }
        }
    }
}

#[test]
fn unknown_provider_returns_none() {
    let _lock = ENV_LOCK.lock().unwrap();
    assert!(get_env_api_key("does-not-exist").is_none());
}

#[test]
fn openai_reads_openai_api_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&["OPENAI_API_KEY"]);
    guard.set("OPENAI_API_KEY", "sk-test");
    assert_eq!(get_env_api_key("openai").as_deref(), Some("sk-test"));
}

#[test]
fn empty_env_var_is_treated_as_absent() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&["MISTRAL_API_KEY"]);
    guard.set("MISTRAL_API_KEY", "");
    assert!(get_env_api_key("mistral").is_none());
}

#[test]
fn anthropic_prefers_oauth_token_over_api_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]);
    guard.set("ANTHROPIC_OAUTH_TOKEN", "oauth-value");
    guard.set("ANTHROPIC_API_KEY", "apikey-value");
    assert_eq!(get_env_api_key("anthropic").as_deref(), Some("oauth-value"));
}

#[test]
fn anthropic_falls_back_to_api_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]);
    guard.set("ANTHROPIC_API_KEY", "apikey-value");
    assert_eq!(get_env_api_key("anthropic").as_deref(), Some("apikey-value"));
}

#[test]
fn github_copilot_priority_order() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]);
    guard.set("GITHUB_TOKEN", "github-level");
    assert_eq!(get_env_api_key("github-copilot").as_deref(), Some("github-level"));

    guard.set("GH_TOKEN", "gh-level");
    assert_eq!(get_env_api_key("github-copilot").as_deref(), Some("gh-level"));

    guard.set("COPILOT_GITHUB_TOKEN", "copilot-level");
    assert_eq!(get_env_api_key("github-copilot").as_deref(), Some("copilot-level"));
}

#[test]
fn bedrock_authenticated_via_access_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&[
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ]);
    // No credentials set yet.
    assert!(get_env_api_key("amazon-bedrock").is_none());

    guard.set("AWS_ACCESS_KEY_ID", "AKIA...");
    guard.set("AWS_SECRET_ACCESS_KEY", "secret");
    assert_eq!(
        get_env_api_key("amazon-bedrock").as_deref(),
        Some("<authenticated>")
    );
}

#[test]
fn bedrock_authenticated_via_profile() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&[
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ]);
    guard.set("AWS_PROFILE", "my-profile");
    assert_eq!(
        get_env_api_key("amazon-bedrock").as_deref(),
        Some("<authenticated>")
    );
}

#[test]
fn bedrock_authenticated_via_bearer_token() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&[
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ]);
    guard.set("AWS_BEARER_TOKEN_BEDROCK", "bearer-value");
    assert_eq!(
        get_env_api_key("amazon-bedrock").as_deref(),
        Some("<authenticated>")
    );
}

#[test]
fn google_vertex_requires_project_location_and_credentials() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]);
    // Nothing set → no auth.
    assert!(get_env_api_key("google-vertex").is_none());

    // With only project but no location/credentials → still no auth.
    guard.set("GOOGLE_CLOUD_PROJECT", "my-proj");
    assert!(get_env_api_key("google-vertex").is_none());
}

#[test]
fn google_vertex_with_api_key_returns_it_directly() {
    let _lock = ENV_LOCK.lock().unwrap();
    let guard = EnvGuard::new(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]);
    guard.set("GOOGLE_CLOUD_API_KEY", "vertex-api-key");
    assert_eq!(get_env_api_key("google-vertex").as_deref(), Some("vertex-api-key"));
}

#[test]
fn known_providers_each_have_their_env_var() {
    let _lock = ENV_LOCK.lock().unwrap();
    let cases = [
        ("openai", "OPENAI_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("cerebras", "CEREBRAS_API_KEY"),
        ("xai", "XAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("huggingface", "HF_TOKEN"),
    ];

    for (provider, env_var) in cases {
        let guard = EnvGuard::new(&[env_var]);
        guard.set(env_var, "token-value");
        assert_eq!(
            get_env_api_key(provider).as_deref(),
            Some("token-value"),
            "{} should read from {}",
            provider,
            env_var
        );
        // guard drops here and restores previous state
    }
}
