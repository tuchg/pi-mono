//! Additional integration tests for context overflow detection.
//!
//! Port of behaviors covered in TS `packages/ai/test/overflow.test.ts` and
//! `context-overflow.test.ts` — specifically provider-specific error message
//! patterns and rate-limit exclusions.

use pi_ai_rs::utils::overflow::is_context_overflow;
use pi_ai_rs::{AssistantMessage, StopReason, Usage};

fn err_msg(text: &str) -> AssistantMessage {
    AssistantMessage {
        stop_reason: StopReason::Error,
        error_message: Some(text.to_string()),
        usage: Usage::default(),
        ..Default::default()
    }
}

fn stop_msg_with_input(input: u64, cache_read: u64) -> AssistantMessage {
    let mut msg = AssistantMessage {
        stop_reason: StopReason::Stop,
        error_message: None,
        usage: Usage::default(),
        ..Default::default()
    };
    msg.usage.input = input;
    msg.usage.cache_read = cache_read;
    msg
}

// ---------------------------------------------------------------------------
// Provider-specific overflow patterns
// ---------------------------------------------------------------------------

#[test]
fn ollama_prompt_too_long() {
    let m = err_msg("400 `prompt too long; exceeded max context length by 100918 tokens`");
    assert!(is_context_overflow(&m, Some(32768)));
}

#[test]
fn ollama_prompt_too_long_without_max_keyword() {
    let m = err_msg("400 `prompt too long; exceeded context length by 5000 tokens`");
    assert!(is_context_overflow(&m, Some(32768)));
}

#[test]
fn anthropic_prompt_is_too_long() {
    let m = err_msg("prompt is too long: 213462 tokens > 200000 maximum");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn anthropic_request_too_large() {
    let m = err_msg("413 request_too_large: The request body is too large.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn bedrock_input_is_too_long() {
    let m = err_msg("Input is too long for requested model.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn openai_exceeds_context_window() {
    let m = err_msg("This model's maximum context length is 128000 tokens. Your input exceeds the context window of this model.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn google_input_token_count_exceeds_max() {
    let m = err_msg("The input token count of 210000 exceeds the maximum number of tokens allowed: 200000.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn xai_maximum_prompt_length() {
    let m = err_msg("The maximum prompt length is 131072 tokens, but received ...");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn groq_reduce_length() {
    let m = err_msg("Please reduce the length of the messages or completion.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn openrouter_maximum_context_length() {
    let m = err_msg("This model's maximum context length is 128000 tokens.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn github_copilot_exceeds_limit() {
    let m = err_msg("Your request exceeds the limit of 90000 tokens per minute.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn llamacpp_exceeds_available_context_size() {
    let m = err_msg("Input exceeds the available context size");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn lmstudio_greater_than_context_length() {
    let m = err_msg("Prompt length is greater than the context length");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn mistral_too_large_for_model_maximum_context() {
    let m = err_msg("Input is too large for model with 32768 maximum context length.");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn zai_model_context_window_exceeded() {
    let m = err_msg("Error: model_context_window_exceeded");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn generic_context_length_exceeded() {
    let m = err_msg("context_length_exceeded");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn generic_context_length_exceeded_spaced() {
    let m = err_msg("context length exceeded");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn generic_too_many_tokens() {
    let m = err_msg("too many tokens in the conversation");
    assert!(is_context_overflow(&m, None));
}

#[test]
fn cerebras_400_no_body() {
    let m = err_msg("400 (no body)");
    assert!(is_context_overflow(&m, None));
    let m2 = err_msg("413 status code (no body)");
    assert!(is_context_overflow(&m2, None));
}

// ---------------------------------------------------------------------------
// Non-overflow / rate-limit exclusions
// ---------------------------------------------------------------------------

#[test]
fn bedrock_throttling_is_not_overflow() {
    let m = err_msg("Throttling error: Too many tokens, please wait before trying again.");
    assert!(!is_context_overflow(&m, Some(200_000)));
}

#[test]
fn bedrock_service_unavailable_is_not_overflow() {
    let m = err_msg("Service unavailable: The service is temporarily unavailable.");
    assert!(!is_context_overflow(&m, Some(200_000)));
}

#[test]
fn generic_rate_limit_is_not_overflow() {
    let m = err_msg("Rate limit exceeded, please retry after 30 seconds.");
    assert!(!is_context_overflow(&m, Some(200_000)));
}

#[test]
fn http_429_too_many_requests_is_not_overflow() {
    let m = err_msg("429 Too Many Requests. Please slow down.");
    assert!(!is_context_overflow(&m, Some(200_000)));
}

#[test]
fn unrelated_error_is_not_overflow() {
    let m = err_msg("500 `model runner crashed unexpectedly`");
    assert!(!is_context_overflow(&m, Some(32_768)));
}

// ---------------------------------------------------------------------------
// Silent (successful-but-over-budget) overflow
// ---------------------------------------------------------------------------

#[test]
fn silent_overflow_with_input_over_budget() {
    let m = stop_msg_with_input(250_000, 0);
    assert!(is_context_overflow(&m, Some(200_000)));
}

#[test]
fn silent_overflow_counts_cache_read_tokens() {
    // input 100k + cache_read 150k = 250k which exceeds 200k budget.
    let m = stop_msg_with_input(100_000, 150_000);
    assert!(is_context_overflow(&m, Some(200_000)));
}

#[test]
fn no_overflow_when_context_window_omitted() {
    let m = stop_msg_with_input(1_000_000, 0);
    assert!(!is_context_overflow(&m, None));
}

#[test]
fn no_overflow_when_within_context() {
    let m = stop_msg_with_input(50_000, 0);
    assert!(!is_context_overflow(&m, Some(200_000)));
}

#[test]
fn no_overflow_when_no_error_and_usage_at_boundary() {
    // Exactly at the limit is not overflow (TS uses strict >).
    let m = stop_msg_with_input(200_000, 0);
    assert!(!is_context_overflow(&m, Some(200_000)));
}

// ---------------------------------------------------------------------------
// Case sensitivity
// ---------------------------------------------------------------------------

#[test]
fn patterns_are_case_insensitive() {
    let m = err_msg("PROMPT IS TOO LONG: exceeded max");
    assert!(is_context_overflow(&m, None));
}
