use crate::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStreamReceiver,
};
use crate::providers::LlmProvider;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, Model,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, Usage,
};

/// A mock LLM provider that returns scripted responses.
///
/// Useful for testing agent loops and tool calling without hitting real APIs.
pub struct FauxProvider {
    responses: Vec<AssistantMessage>,
}

impl FauxProvider {
    /// Create a faux provider that cycles through the given responses.
    pub fn new(responses: Vec<AssistantMessage>) -> Self {
        Self { responses }
    }

    /// Convenience: create a provider that always returns a single text response.
    pub fn with_text(text: &str) -> Self {
        let msg = AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: text.to_string(),
                text_signature: None,
            })],
            api: "faux".to_string(),
            provider: "faux".to_string(),
            model: "faux".to_string(),
            response_id: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        };
        Self::new(vec![msg])
    }

    fn emit_response(
        &self,
        _model: &Model,
        _context: Context,
    ) -> AssistantMessageEventStreamReceiver {
        let (mut sender, receiver) = create_assistant_message_event_stream();
        let response = self
            .responses
            .first()
            .cloned()
            .unwrap_or_default();

        // Emit the standard event sequence for a complete response.
        sender.push(AssistantMessageEvent::Start {
            partial: response.clone(),
        });

        for (i, content) in response.content.iter().enumerate() {
            match content {
                AssistantContent::Text(tc) => {
                    sender.push(AssistantMessageEvent::TextStart {
                        content_index: i,
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: i,
                        delta: tc.text.clone(),
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::TextEnd {
                        content_index: i,
                        content: tc.text.clone(),
                        partial: response.clone(),
                    });
                }
                AssistantContent::Thinking(tc) => {
                    sender.push(AssistantMessageEvent::ThinkingStart {
                        content_index: i,
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: i,
                        delta: tc.thinking.clone(),
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: i,
                        content: tc.thinking.clone(),
                        partial: response.clone(),
                    });
                }
                AssistantContent::ToolCall(tc) => {
                    sender.push(AssistantMessageEvent::ToolcallStart {
                        content_index: i,
                        partial: response.clone(),
                    });
                    sender.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: i,
                        tool_call: tc.clone(),
                        partial: response.clone(),
                    });
                }
            }
        }

        sender.push(AssistantMessageEvent::Done {
            reason: response.stop_reason,
            message: response,
        });

        receiver
    }
}

impl LlmProvider for FauxProvider {
    fn api(&self) -> &str {
        "faux"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStreamReceiver {
        self.emit_response(model, context)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        _options: SimpleStreamOptions,
    ) -> AssistantMessageEventStreamReceiver {
        self.emit_response(model, context)
    }
}
