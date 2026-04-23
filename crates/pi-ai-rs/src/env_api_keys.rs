use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Read an env var, treating empty strings as absent (matching JS `||` semantics).
fn env_var_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Get API key for a provider from known environment variables.
///
/// Port of `getEnvApiKey()` from `packages/ai/src/env-api-keys.ts`.
pub fn get_env_api_key(provider: &str) -> Option<String> {
    match provider {
        "github-copilot" => env_var_nonempty("COPILOT_GITHUB_TOKEN")
            .or_else(|| env_var_nonempty("GH_TOKEN"))
            .or_else(|| env_var_nonempty("GITHUB_TOKEN")),

        "anthropic" => env_var_nonempty("ANTHROPIC_OAUTH_TOKEN")
            .or_else(|| env_var_nonempty("ANTHROPIC_API_KEY")),

        "google-vertex" => {
            if let Some(key) = env_var_nonempty("GOOGLE_CLOUD_API_KEY") {
                return Some(key);
            }
            let has_credentials = has_vertex_adc_credentials();
            let has_project = env_var_nonempty("GOOGLE_CLOUD_PROJECT").is_some()
                || env_var_nonempty("GCLOUD_PROJECT").is_some();
            let has_location = env_var_nonempty("GOOGLE_CLOUD_LOCATION").is_some();
            if has_credentials && has_project && has_location {
                Some("<authenticated>".to_string())
            } else {
                None
            }
        }

        "amazon-bedrock" => {
            if env_var_nonempty("AWS_PROFILE").is_some()
                || (env_var_nonempty("AWS_ACCESS_KEY_ID").is_some()
                    && env_var_nonempty("AWS_SECRET_ACCESS_KEY").is_some())
                || env_var_nonempty("AWS_BEARER_TOKEN_BEDROCK").is_some()
                || env_var_nonempty("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
                || env_var_nonempty("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some()
                || env_var_nonempty("AWS_WEB_IDENTITY_TOKEN_FILE").is_some()
            {
                Some("<authenticated>".to_string())
            } else {
                None
            }
        }

        _ => {
            let env_map = env_var_map();
            env_map
                .get(provider)
                .and_then(|var| env_var_nonempty(var))
        }
    }
}

/// Static map of provider name → environment variable name.
fn env_var_map() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&str, &str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("openai", "OPENAI_API_KEY");
        m.insert("azure-openai-responses", "AZURE_OPENAI_API_KEY");
        m.insert("google", "GEMINI_API_KEY");
        m.insert("groq", "GROQ_API_KEY");
        m.insert("cerebras", "CEREBRAS_API_KEY");
        m.insert("xai", "XAI_API_KEY");
        m.insert("openrouter", "OPENROUTER_API_KEY");
        m.insert("vercel-ai-gateway", "AI_GATEWAY_API_KEY");
        m.insert("zai", "ZAI_API_KEY");
        m.insert("mistral", "MISTRAL_API_KEY");
        m.insert("minimax", "MINIMAX_API_KEY");
        m.insert("minimax-cn", "MINIMAX_CN_API_KEY");
        m.insert("huggingface", "HF_TOKEN");
        m.insert("opencode", "OPENCODE_API_KEY");
        m.insert("opencode-go", "OPENCODE_API_KEY");
        m.insert("kimi-coding", "KIMI_API_KEY");
        m
    })
}

/// Check if Google Vertex AI Application Default Credentials exist.
fn has_vertex_adc_credentials() -> bool {
    // Check GOOGLE_APPLICATION_CREDENTIALS env var first (skip empty strings like TS)
    if let Some(gac_path) = env_var_nonempty("GOOGLE_APPLICATION_CREDENTIALS") {
        return std::path::Path::new(&gac_path).exists();
    }

    // Fall back to default ADC path
    if let Some(home) = dirs_home() {
        let adc_path = home
            .join(".config")
            .join("gcloud")
            .join("application_default_credentials.json");
        return adc_path.exists();
    }

    false
}

/// Get user home directory.
fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_provider_returns_none() {
        assert!(get_env_api_key("unknown-provider-xyz").is_none());
    }

    #[test]
    fn env_map_contains_openai() {
        let map = env_var_map();
        assert_eq!(map.get("openai"), Some(&"OPENAI_API_KEY"));
    }
}
