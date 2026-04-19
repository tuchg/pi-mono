pub mod faux;
pub mod simple_options;
pub mod transform_messages;

use crate::event_stream::AssistantMessageEventStreamReceiver;
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};

/// Trait that all LLM providers must implement.
///
/// Each provider knows how to turn a `(Model, Context, Options)` triple into
/// an async event stream of `AssistantMessageEvent`s.
pub trait LlmProvider: Send + Sync {
    /// The API protocol identifier this provider handles (e.g. `"anthropic-messages"`).
    fn api(&self) -> &str;

    /// Stream with raw provider-specific options.
    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStreamReceiver;

    /// Stream with simplified reasoning-aware options.
    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStreamReceiver;
}
