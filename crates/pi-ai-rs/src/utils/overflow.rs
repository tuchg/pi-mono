use regex::Regex;
use std::sync::LazyLock;

use crate::types::{AssistantMessage, StopReason};

/// Compiled overflow detection patterns.
///
/// Port of OVERFLOW_PATTERNS from `packages/ai/src/utils/overflow.ts`.
static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)prompt is too long",                                // Anthropic token overflow
        r"(?i)request_too_large",                                 // Anthropic request byte-size overflow (HTTP 413)
        r"(?i)input is too long for requested model",             // Amazon Bedrock
        r"(?i)exceeds the context window",                        // OpenAI (Completions & Responses API)
        r"(?i)input token count.*exceeds the maximum",            // Google (Gemini)
        r"(?i)maximum prompt length is \d+",                      // xAI (Grok)
        r"(?i)reduce the length of the messages",                 // Groq
        r"(?i)maximum context length is \d+ tokens",              // OpenRouter (all backends)
        r"(?i)exceeds the limit of \d+",                          // GitHub Copilot
        r"(?i)exceeds the available context size",                // llama.cpp server
        r"(?i)greater than the context length",                   // LM Studio
        r"(?i)context window exceeds limit",                      // MiniMax
        r"(?i)exceeded model token limit",                        // Kimi For Coding
        r"(?i)too large for model with \d+ maximum context length", // Mistral
        r"(?i)model_context_window_exceeded",                     // z.ai
        r"(?i)prompt too long; exceeded (?:max )?context length", // Ollama explicit overflow error
        r"(?i)context[_ ]length[_ ]exceeded",                     // Generic fallback
        r"(?i)too many tokens",                                   // Generic fallback
        r"(?i)token limit exceeded",                              // Generic fallback
        r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)",     // Cerebras: 400/413 with no body
    ]
    .iter()
    .map(|p| Regex::new(p).expect("invalid overflow pattern"))
    .collect()
});

/// Patterns that indicate non-overflow errors (e.g. rate limiting).
///
/// Error messages matching any of these are excluded from overflow detection
/// even if they also match an OVERFLOW_PATTERN.
static NON_OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)^(Throttling error|Service unavailable):", // AWS Bedrock non-overflow
        r"(?i)rate limit",                                // Generic rate limiting
        r"(?i)too many requests",                         // Generic HTTP 429 style
    ]
    .iter()
    .map(|p| Regex::new(p).expect("invalid non-overflow pattern"))
    .collect()
});

/// Check if an assistant message represents a context overflow error.
///
/// Handles two cases:
/// 1. Error-based overflow: error message matches known overflow patterns.
/// 2. Silent overflow: successful response where `usage.input` exceeds the
///    context window (z.ai style).
///
/// Port of `isContextOverflow()` from `packages/ai/src/utils/overflow.ts`.
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    // Case 1: Check error message patterns
    if message.stop_reason == StopReason::Error {
        if let Some(ref err_msg) = message.error_message {
            let is_non_overflow = NON_OVERFLOW_PATTERNS.iter().any(|p| p.is_match(err_msg));
            if !is_non_overflow && OVERFLOW_PATTERNS.iter().any(|p| p.is_match(err_msg)) {
                return true;
            }
        }
    }

    // Case 2: Silent overflow — successful but usage exceeds context
    if let Some(cw) = context_window {
        if message.stop_reason == StopReason::Stop {
            let input_tokens = message.usage.input + message.usage.cache_read;
            if input_tokens > cw {
                return true;
            }
        }
    }

    false
}

/// Return clones of the compiled overflow patterns (for testing).
pub fn get_overflow_patterns() -> Vec<Regex> {
    OVERFLOW_PATTERNS.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMessage, StopReason, Usage, UsageCost};

    fn make_msg(stop: StopReason, err: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            stop_reason: stop,
            error_message: err.map(|s| s.to_string()),
            usage: Usage::default(),
            ..Default::default()
        }
    }

    #[test]
    fn detects_anthropic_overflow() {
        let msg = make_msg(
            StopReason::Error,
            Some("prompt is too long: 213462 tokens > 200000 maximum"),
        );
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn detects_openai_overflow() {
        let msg = make_msg(
            StopReason::Error,
            Some("Your input exceeds the context window of this model"),
        );
        assert!(is_context_overflow(&msg, None));
    }

    #[test]
    fn excludes_throttling() {
        let msg = make_msg(
            StopReason::Error,
            Some("Throttling error: Too many tokens, please wait before trying again."),
        );
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn excludes_rate_limit() {
        let msg = make_msg(StopReason::Error, Some("rate limit exceeded"));
        assert!(!is_context_overflow(&msg, None));
    }

    #[test]
    fn detects_silent_overflow() {
        let mut msg = make_msg(StopReason::Stop, None);
        msg.usage.input = 250_000;
        msg.usage.cache_read = 0;
        assert!(is_context_overflow(&msg, Some(200_000)));
    }

    #[test]
    fn no_overflow_when_within_context() {
        let mut msg = make_msg(StopReason::Stop, None);
        msg.usage.input = 100_000;
        assert!(!is_context_overflow(&msg, Some(200_000)));
    }
}
