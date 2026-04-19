use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// String-based enums
// ---------------------------------------------------------------------------

/// Known LLM API protocol identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Api {
    #[serde(rename = "openai-completions")]
    OpenAiCompletions,
    #[serde(rename = "mistral-conversations")]
    MistralConversations,
    #[serde(rename = "openai-responses")]
    OpenAiResponses,
    #[serde(rename = "azure-openai-responses")]
    AzureOpenAiResponses,
    #[serde(rename = "openai-codex-responses")]
    OpenAiCodexResponses,
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
    #[serde(rename = "bedrock-converse-stream")]
    BedrockConverseStream,
    #[serde(rename = "google-generative-ai")]
    GoogleGenerativeAi,
    #[serde(rename = "google-gemini-cli")]
    GoogleGeminiCli,
    #[serde(rename = "google-vertex")]
    GoogleVertex,
    /// Extension-defined API identifier.
    #[serde(untagged)]
    Custom(String),
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenAiCompletions => write!(f, "openai-completions"),
            Self::MistralConversations => write!(f, "mistral-conversations"),
            Self::OpenAiResponses => write!(f, "openai-responses"),
            Self::AzureOpenAiResponses => write!(f, "azure-openai-responses"),
            Self::OpenAiCodexResponses => write!(f, "openai-codex-responses"),
            Self::AnthropicMessages => write!(f, "anthropic-messages"),
            Self::BedrockConverseStream => write!(f, "bedrock-converse-stream"),
            Self::GoogleGenerativeAi => write!(f, "google-generative-ai"),
            Self::GoogleGeminiCli => write!(f, "google-gemini-cli"),
            Self::GoogleVertex => write!(f, "google-vertex"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Known LLM provider names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    #[serde(rename = "amazon-bedrock")]
    AmazonBedrock,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "google")]
    Google,
    #[serde(rename = "google-gemini-cli")]
    GoogleGeminiCli,
    #[serde(rename = "google-antigravity")]
    GoogleAntigravity,
    #[serde(rename = "google-vertex")]
    GoogleVertex,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "azure-openai-responses")]
    AzureOpenAiResponses,
    #[serde(rename = "openai-codex")]
    OpenAiCodex,
    #[serde(rename = "github-copilot")]
    GithubCopilot,
    #[serde(rename = "xai")]
    Xai,
    #[serde(rename = "groq")]
    Groq,
    #[serde(rename = "cerebras")]
    Cerebras,
    #[serde(rename = "openrouter")]
    OpenRouter,
    #[serde(rename = "vercel-ai-gateway")]
    VercelAiGateway,
    #[serde(rename = "zai")]
    Zai,
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "minimax")]
    Minimax,
    #[serde(rename = "minimax-cn")]
    MinimaxCn,
    #[serde(rename = "huggingface")]
    HuggingFace,
    #[serde(rename = "opencode")]
    OpenCode,
    #[serde(rename = "opencode-go")]
    OpenCodeGo,
    #[serde(rename = "kimi-coding")]
    KimiCoding,
    /// Extension-defined provider.
    #[serde(untagged)]
    Custom(String),
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_str().map(String::from));
        match s {
            Some(name) => write!(f, "{name}"),
            None => write!(f, "{self:?}"),
        }
    }
}

/// Thinking/reasoning level for models that support extended reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

/// Cache retention hint for providers that support prompt caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

/// Transport protocol for streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    Sse,
    Websocket,
    Auto,
}

/// Reason the assistant stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    /// Graceful interrupt (from Agent.interrupt()).
    Interrupted,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ToolUse => write!(f, "toolUse"),
            Self::Error => write!(f, "error"),
            Self::Aborted => write!(f, "aborted"),
            Self::Interrupted => write!(f, "interrupted"),
        }
    }
}

/// Input modality supported by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputModality {
    Text,
    Image,
    Video,
    Audio,
}

// ---------------------------------------------------------------------------
// Content types
// ---------------------------------------------------------------------------

/// Decoded text signature (version 1).
///
/// Port of `TextSignatureV1` from `packages/ai/src/types.ts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSignatureV1 {
    pub v: u32,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<TextSignaturePhase>,
}

/// Possible phase values inside a [`TextSignatureV1`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSignaturePhase {
    Commentary,
    FinalAnswer,
}

/// Text content within a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

/// Extended thinking content from reasoning models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

/// Base64-encoded image content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data.
    pub data: String,
    pub mime_type: String,
}

/// Video content (multimodal extension — issue #3200).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoContent {
    pub data: String,
    pub mime_type: String,
}

/// Audio content (multimodal extension — issue #3200).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
    pub data: String,
    pub mime_type: String,
}

/// A tool (function) call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

/// Any content type that can appear in a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Content {
    Text(TextContent),
    Thinking(ThinkingContent),
    Image(ImageContent),
    Video(VideoContent),
    Audio(AudioContent),
    ToolCall(ToolCall),
}

/// Content types that can appear in an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantContent {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

/// User message content — either a plain string or structured parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<UserContentPart>),
}

/// A single part within structured user content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserContentPart {
    Text(TextContent),
    Image(ImageContent),
}

// ---------------------------------------------------------------------------
// Usage / cost tracking
// ---------------------------------------------------------------------------

/// Cost breakdown in dollars.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// Token usage and cost for an assistant response.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A user message in the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// An assistant (LLM) response message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

impl Default for AssistantMessage {
    fn default() -> Self {
        Self {
            content: Vec::new(),
            api: String::new(),
            provider: String::new(),
            model: String::new(),
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }
    }
}

/// A tool execution result message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Union of all conversation message types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

/// Events emitted during assistant message streaming.
///
/// This is the core streaming protocol — each variant maps 1:1 to the
/// TypeScript `AssistantMessageEvent` discriminated union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ToolcallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ToolcallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ToolcallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Returns `true` when the event terminates the stream.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error { .. })
    }

    /// Extract the final `AssistantMessage` from a terminal event.
    pub fn into_final_message(self) -> Option<AssistantMessage> {
        match self {
            Self::Done { message, .. } => Some(message),
            Self::Error { error, .. } => Some(error),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-level token budgets for extended thinking.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingBudgets {
    pub minimal: Option<u32>,
    pub low: Option<u32>,
    pub medium: Option<u32>,
    pub high: Option<u32>,
}

/// Options for raw provider-level streaming.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub api_key: Option<String>,
    pub transport: Option<Transport>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub max_retry_delay_ms: Option<u64>,
    pub metadata: Option<serde_json::Value>,
}

/// Simplified streaming options with reasoning support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimpleStreamOptions {
    // Inherited from StreamOptions
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub api_key: Option<String>,
    pub transport: Option<Transport>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub max_retry_delay_ms: Option<u64>,
    pub metadata: Option<serde_json::Value>,
    // SimpleStreamOptions extensions
    pub reasoning: Option<ThinkingLevel>,
    pub thinking_budgets: Option<ThinkingBudgets>,
}

impl SimpleStreamOptions {
    /// Extract the base `StreamOptions` fields.
    pub fn as_stream_options(&self) -> StreamOptions {
        StreamOptions {
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            api_key: self.api_key.clone(),
            transport: self.transport,
            cache_retention: self.cache_retention,
            session_id: self.session_id.clone(),
            headers: self.headers.clone(),
            max_retry_delay_ms: self.max_retry_delay_ms,
            metadata: self.metadata.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tools & Context
// ---------------------------------------------------------------------------

/// A tool definition for LLM function calling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: serde_json::Value,
}

/// Conversation context sent to the LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Cost per million tokens.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// LLM model metadata and capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<InputModality>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Provider-specific compatibility flags (opaque JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<serde_json::Value>,
    /// Per-model thinking levels (issue #3208).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_thinking_levels: Option<Vec<ThinkingLevel>>,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            id: "unknown".to_string(),
            name: "unknown".to_string(),
            api: "unknown".to_string(),
            provider: "unknown".to_string(),
            base_url: String::new(),
            reasoning: false,
            input: Vec::new(),
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
            headers: None,
            compat: None,
            supported_thinking_levels: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during streaming or provider operations.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("no API provider registered for api: {api}")]
    NoProvider { api: String },

    #[error("provider error: {message}")]
    Provider { message: String },

    #[error("validation error: {message}")]
    Validation { message: String },

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("stream aborted")]
    Aborted,

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Provider HTTP response metadata (for `onResponse` hooks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
}
