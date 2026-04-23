use crate::types::{Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel};

/// Build base `StreamOptions` from a model and optional `SimpleStreamOptions`.
///
/// Port of `buildBaseOptions()` from `packages/ai/src/providers/simple-options.ts`.
pub fn build_base_options(
    model: &Model,
    options: Option<&SimpleStreamOptions>,
    api_key: Option<&str>,
) -> StreamOptions {
    let max_tokens = options
        .and_then(|o| o.max_tokens)
        .unwrap_or_else(|| std::cmp::min(model.max_tokens as u32, 32_000));

    StreamOptions {
        temperature: options.and_then(|o| o.temperature),
        max_tokens: Some(max_tokens),
        api_key: api_key
            .map(|k| k.to_string())
            .or_else(|| options.and_then(|o| o.api_key.clone())),
        cache_retention: options.and_then(|o| o.cache_retention),
        session_id: options.and_then(|o| o.session_id.clone()),
        headers: options.and_then(|o| o.headers.clone()),
        max_retry_delay_ms: options.and_then(|o| o.max_retry_delay_ms),
        metadata: options.and_then(|o| o.metadata.clone()),
        transport: options.and_then(|o| o.transport),
    }
}

/// Clamp `xhigh` reasoning down to `high`.
///
/// Port of `clampReasoning()` from `packages/ai/src/providers/simple-options.ts`.
pub fn clamp_reasoning(effort: Option<ThinkingLevel>) -> Option<ThinkingLevel> {
    match effort {
        Some(ThinkingLevel::Xhigh) => Some(ThinkingLevel::High),
        other => other,
    }
}

/// Adjust max tokens and compute a thinking budget for the given reasoning
/// level.
///
/// Port of `adjustMaxTokensForThinking()` from
/// `packages/ai/src/providers/simple-options.ts`.
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: u32,
    model_max_tokens: u64,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> AdjustedTokens {
    let default_budgets = ThinkingBudgets {
        minimal: Some(1024),
        low: Some(2048),
        medium: Some(8192),
        high: Some(16384),
    };

    let level = clamp_reasoning(Some(reasoning_level)).unwrap();
    let budget_for_level = match level {
        ThinkingLevel::Minimal => custom_budgets
            .and_then(|b| b.minimal)
            .or(default_budgets.minimal)
            .unwrap(),
        ThinkingLevel::Low => custom_budgets
            .and_then(|b| b.low)
            .or(default_budgets.low)
            .unwrap(),
        ThinkingLevel::Medium => custom_budgets
            .and_then(|b| b.medium)
            .or(default_budgets.medium)
            .unwrap(),
        ThinkingLevel::High | ThinkingLevel::Xhigh => custom_budgets
            .and_then(|b| b.high)
            .or(default_budgets.high)
            .unwrap(),
    };

    let min_output_tokens: u32 = 1024;
    let mut thinking_budget = budget_for_level;
    let max_tokens = std::cmp::min(
        (base_max_tokens as u64) + (thinking_budget as u64),
        model_max_tokens,
    ) as u32;

    if max_tokens <= thinking_budget {
        thinking_budget = max_tokens.saturating_sub(min_output_tokens);
    }

    AdjustedTokens {
        max_tokens,
        thinking_budget,
    }
}

/// Result of [`adjust_max_tokens_for_thinking`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdjustedTokens {
    pub max_tokens: u32,
    pub thinking_budget: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_xhigh_to_high() {
        assert_eq!(
            clamp_reasoning(Some(ThinkingLevel::Xhigh)),
            Some(ThinkingLevel::High)
        );
    }

    #[test]
    fn clamp_keeps_other_levels() {
        assert_eq!(
            clamp_reasoning(Some(ThinkingLevel::Medium)),
            Some(ThinkingLevel::Medium)
        );
        assert_eq!(clamp_reasoning(None), None);
    }

    #[test]
    fn adjust_tokens_basic() {
        let result = adjust_max_tokens_for_thinking(32_000, 128_000, ThinkingLevel::High, None);
        assert_eq!(result.max_tokens, 32_000 + 16_384);
        assert_eq!(result.thinking_budget, 16_384);
    }

    #[test]
    fn adjust_tokens_clamped_to_model_max() {
        let result = adjust_max_tokens_for_thinking(32_000, 33_000, ThinkingLevel::High, None);
        assert_eq!(result.max_tokens, 33_000);
        assert_eq!(result.thinking_budget, 16_384);
    }

    #[test]
    fn adjust_tokens_budget_exceeds_max() {
        let result = adjust_max_tokens_for_thinking(500, 2000, ThinkingLevel::High, None);
        // max_tokens = min(500+16384, 2000) = 2000
        // since 2000 > 16384 is false: max_tokens(2000) <= thinking_budget(16384) is true
        // so: thinking_budget = max(0, 2000-1024) = 976
        assert_eq!(result.max_tokens, 2000);
        assert_eq!(result.thinking_budget, 976);
    }

    #[test]
    fn build_base_options_defaults() {
        let model = Model {
            max_tokens: 100_000,
            ..Default::default()
        };
        let opts = build_base_options(&model, None, None);
        assert_eq!(opts.max_tokens, Some(32_000));
    }

    #[test]
    fn build_base_options_with_api_key_priority() {
        let model = Model::default();
        let simple = SimpleStreamOptions {
            api_key: Some("from-options".to_string()),
            ..Default::default()
        };
        // Explicit api_key parameter takes priority.
        let opts = build_base_options(&model, Some(&simple), Some("explicit-key"));
        assert_eq!(opts.api_key.as_deref(), Some("explicit-key"));

        // Falls back to options.
        let opts2 = build_base_options(&model, Some(&simple), None);
        assert_eq!(opts2.api_key.as_deref(), Some("from-options"));
    }
}
