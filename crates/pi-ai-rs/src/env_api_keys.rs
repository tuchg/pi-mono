use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Get API key for a provider from known environment variables.
///
/// Port of `getEnvApiKey()` from `packages/ai/src/env-api-keys.ts`.
pub fn get_env_api_key(provider: &str) -> Option<String> {
    match provider {
        "github-copilot" => std::env::var("COPILOT_GITHUB_TOKEN")
            .or_else(|_| std::env::var("GH_TOKEN"))
            .or_else(|_| std::env::var("GITHUB_TOKEN"))
            .ok(),

        "anthropic" => std::env::var("ANTHROPIC_OAUTH_TOKEN")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .ok(),

        "google-vertex" => {
            if let Ok(key) = std::env::var("GOOGLE_CLOUD_API_KEY") {
                return Some(key);
            }
            let has_credentials = has_vertex_adc_credentials();
            let has_project = std::env::var("GOOGLE_CLOUD_PROJECT").is_ok()
                || std::env::var("GCLOUD_PROJECT").is_ok();
            let has_location = std::env::var("GOOGLE_CLOUD_LOCATION").is_ok();
            if has_credentials && has_project && has_location {
                Some("<authenticated>".to_string())
            } else {
                None
            }
        }

        "amazon-bedrock" => {
            if std::env::var("AWS_PROFILE").is_ok()
                || (std::env::var("AWS_ACCESS_KEY_ID").is_ok()
                    && std::env::var("AWS_SECRET_ACCESS_KEY").is_ok())
                || std::env::var("AWS_BEARER_TOKEN_BEDROCK").is_ok()
                || std::env::var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_ok()
                || std::env::var("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_ok()
                || std::env::var("AWS_WEB_IDENTITY_TOKEN_FILE").is_ok()
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
                .and_then(|var| std::env::var(var).ok())
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
    // Check GOOGLE_APPLICATION_CREDENTIALS env var first
    if let Ok(gac_path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
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
