use futures::StreamExt;
use pi_ai_rs::event_stream::{create_assistant_message_event_stream, event_stream};
use pi_ai_rs::{
    AssistantMessage, AssistantMessageEvent, StopReason,
};

#[tokio::test]
async fn event_stream_sends_and_receives() {
    let (mut sender, mut receiver) = event_stream::<String, String>(
        |s| s == "done",
        |s| s.clone(),
    );

    sender.push("hello".to_string());
    sender.push("world".to_string());
    sender.push("done".to_string());

    // Drop sender so receiver stream ends
    drop(sender);

    let mut items = Vec::new();
    while let Some(item) = receiver.next().await {
        items.push(item);
    }

    assert_eq!(items, vec!["hello", "world", "done"]);
}

#[tokio::test]
async fn event_stream_result_available_after_terminal() {
    let (mut sender, receiver) = event_stream::<String, String>(
        |s| s == "done",
        |s| s.clone(),
    );

    sender.push("a".to_string());
    sender.push("done".to_string());

    let result = receiver.result().await;
    assert_eq!(result, Some("done".to_string()));
}

#[tokio::test]
async fn event_stream_end_sends_result() {
    let (mut sender, receiver) = event_stream::<String, String>(
        |_| false,
        |s| s.clone(),
    );

    sender.push("a".to_string());
    sender.end(Some("final".to_string()));

    // Drop the sender explicitly to close the channel
    drop(sender);

    let result = receiver.result().await;
    assert_eq!(result, Some("final".to_string()));
}

#[tokio::test]
async fn assistant_message_event_stream_lifecycle() {
    let (mut sender, mut receiver) = create_assistant_message_event_stream();

    let msg = AssistantMessage::default();

    sender.push(AssistantMessageEvent::Start {
        partial: msg.clone(),
    });
    sender.push(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: msg.clone(),
    });
    sender.push(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "hello".to_string(),
        partial: msg.clone(),
    });
    sender.push(AssistantMessageEvent::TextEnd {
        content_index: 0,
        content: "hello".to_string(),
        partial: msg.clone(),
    });
    sender.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg.clone(),
    });

    // Drop sender so the stream closes
    drop(sender);

    let mut events = Vec::new();
    while let Some(event) = receiver.next().await {
        events.push(event);
    }

    assert_eq!(events.len(), 5);
    assert!(!events[0].is_terminal());
    assert!(!events[1].is_terminal());
    assert!(events[4].is_terminal());
}

#[tokio::test]
async fn into_final_message_returns_message_for_done() {
    let msg = AssistantMessage::default();
    let event = AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg.clone(),
    };
    assert_eq!(event.into_final_message(), Some(msg));
}

#[tokio::test]
async fn into_final_message_returns_none_for_non_terminal() {
    let msg = AssistantMessage::default();
    let event = AssistantMessageEvent::Start {
        partial: msg,
    };
    assert_eq!(event.into_final_message(), None);
}
