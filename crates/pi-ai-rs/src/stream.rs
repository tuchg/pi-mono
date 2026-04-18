use crate::event_stream::AssistantMessageEventStreamReceiver;
use crate::registry::get_api_provider;
use crate::types::{AiError, AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};

/// Stream assistant responses using the raw provider-level options.
pub fn stream(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> Result<AssistantMessageEventStreamReceiver, AiError> {
    let provider =
        get_api_provider(&model.api).ok_or_else(|| AiError::NoProvider {
            api: model.api.clone(),
        })?;
    Ok((provider.stream)(model, context, options))
}

/// Stream and collect the final assistant message (raw options).
pub async fn complete(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> Result<AssistantMessage, AiError> {
    let s = stream(model, context, options)?;
    s.result()
        .await
        .ok_or_else(|| AiError::Provider {
            message: "stream ended without a result".to_string(),
        })
}

/// Stream assistant responses with simplified reasoning-aware options.
pub fn stream_simple(
    model: &Model,
    context: Context,
    options: SimpleStreamOptions,
) -> Result<AssistantMessageEventStreamReceiver, AiError> {
    let provider =
        get_api_provider(&model.api).ok_or_else(|| AiError::NoProvider {
            api: model.api.clone(),
        })?;
    Ok((provider.stream_simple)(model, context, options))
}

/// Stream and collect the final assistant message (simplified options).
pub async fn complete_simple(
    model: &Model,
    context: Context,
    options: SimpleStreamOptions,
) -> Result<AssistantMessage, AiError> {
    let s = stream_simple(model, context, options)?;
    s.result()
        .await
        .ok_or_else(|| AiError::Provider {
            message: "stream ended without a result".to_string(),
        })
}
