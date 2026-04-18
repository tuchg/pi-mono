use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::{mpsc, oneshot};

use crate::types::{AssistantMessage, AssistantMessageEvent};

// ---------------------------------------------------------------------------
// Generic EventStream
// ---------------------------------------------------------------------------

/// Sender half — used by the producer (provider / agent loop) to push events.
pub struct EventStreamSender<T: Send + 'static, R: Send + 'static = T> {
    tx: mpsc::UnboundedSender<T>,
    result_tx: Option<oneshot::Sender<R>>,
    is_complete: Box<dyn Fn(&T) -> bool + Send>,
    extract_result: Box<dyn Fn(T) -> R + Send>,
}

impl<T: Clone + Send + 'static, R: Send + 'static> EventStreamSender<T, R> {
    /// Push an event to consumers. If the event is terminal the stream is
    /// closed automatically and the final result is forwarded.
    pub fn push(&mut self, event: T) {
        if (self.is_complete)(&event) {
            let result = (self.extract_result)(event.clone());
            let _ = self.tx.send(event);
            if let Some(tx) = self.result_tx.take() {
                let _ = tx.send(result);
            }
        } else {
            let _ = self.tx.send(event);
        }
    }

    /// Explicitly end the stream with a result (e.g. after an external loop
    /// finishes). If a terminal event already triggered, this is a no-op.
    pub fn end(&mut self, result: R) {
        if let Some(tx) = self.result_tx.take() {
            let _ = tx.send(result);
        }
    }
}

/// Receiver half — implements `futures::Stream` for async iteration.
pub struct EventStream<T: Send + 'static, R: Send + 'static = T> {
    rx: mpsc::UnboundedReceiver<T>,
    result_rx: Option<oneshot::Receiver<R>>,
}

impl<T: Send + 'static, R: Send + 'static> EventStream<T, R> {
    /// Wait for the final result (available after the stream completes).
    pub async fn result(mut self) -> Option<R> {
        if let Some(rx) = self.result_rx.take() {
            rx.await.ok()
        } else {
            None
        }
    }

    /// Consume this stream into the `oneshot::Receiver` for the final result
    /// without draining events. Useful when you only care about the outcome.
    pub fn into_result_receiver(mut self) -> Option<oneshot::Receiver<R>> {
        self.result_rx.take()
    }
}

impl<T: Send + 'static, R: Send + 'static> Stream for EventStream<T, R> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// Create a linked sender/receiver pair.
pub fn event_stream<T, R>(
    is_complete: impl Fn(&T) -> bool + Send + 'static,
    extract_result: impl Fn(T) -> R + Send + 'static,
) -> (EventStreamSender<T, R>, EventStream<T, R>)
where
    T: Clone + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel();

    let sender = EventStreamSender {
        tx,
        result_tx: Some(result_tx),
        is_complete: Box::new(is_complete),
        extract_result: Box::new(extract_result),
    };

    let receiver = EventStream {
        rx,
        result_rx: Some(result_rx),
    };

    (sender, receiver)
}

// ---------------------------------------------------------------------------
// Specialised aliases for the LLM streaming protocol
// ---------------------------------------------------------------------------

/// Sender half for assistant message event streams.
pub type AssistantMessageEventStreamSender =
    EventStreamSender<AssistantMessageEvent, AssistantMessage>;

/// Receiver half for assistant message event streams.
pub type AssistantMessageEventStreamReceiver =
    EventStream<AssistantMessageEvent, AssistantMessage>;

/// Create a sender/receiver pair for assistant message streaming.
pub fn create_assistant_message_event_stream() -> (
    AssistantMessageEventStreamSender,
    AssistantMessageEventStreamReceiver,
) {
    event_stream(
        |event: &AssistantMessageEvent| event.is_terminal(),
        |event: AssistantMessageEvent| {
            event
                .into_final_message()
                .unwrap_or_default()
        },
    )
}
