//! OAuth types for provider authentication.
//!
//! Port of `packages/ai/src/utils/oauth/types.ts`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Persisted OAuth credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub refresh: String,
    pub access: String,
    /// Expiry timestamp (seconds since epoch).
    pub expires: u64,
    /// Extra provider-specific fields.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// OAuth provider identifier.
pub type OAuthProviderId = String;

/// Prompt displayed to the user during OAuth login.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthPrompt {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_empty: Option<bool>,
}

/// Authorization URL and instructions for OAuth login.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthAuthInfo {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Trait for OAuth provider implementations.
///
/// Port of `OAuthProviderInterface` from `packages/ai/src/utils/oauth/types.ts`.
pub trait OAuthProvider: Send + Sync {
    /// Provider identifier string.
    fn id(&self) -> &str;
    /// Human-readable provider name.
    fn name(&self) -> &str;

    /// Whether login uses a local callback server and supports manual code input.
    fn uses_callback_server(&self) -> bool {
        false
    }

    /// Convert credentials to API key string for the provider.
    fn get_api_key(&self, credentials: &OAuthCredentials) -> String;
}
