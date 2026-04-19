use std::collections::HashMap;

use crate::types::Message;

/// Infer whether the request is user-initiated or agent-initiated
/// based on the last message role.
///
/// Port of `inferCopilotInitiator()` from
/// `packages/ai/src/providers/github-copilot-headers.ts`.
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(Message::User(_)) | None => "user",
        _ => "agent",
    }
}

/// Check if any message contains image content (for Copilot-Vision-Request header).
///
/// Port of `hasCopilotVisionInput()` from
/// `packages/ai/src/providers/github-copilot-headers.ts`.
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|msg| match msg {
        Message::User(user) => {
            if let crate::types::UserContent::Parts(parts) = &user.content {
                parts
                    .iter()
                    .any(|p| matches!(p, crate::types::UserContentPart::Image(_)))
            } else {
                false
            }
        }
        Message::ToolResult(tr) => tr
            .content
            .iter()
            .any(|c| matches!(c, crate::types::Content::Image(_))),
        _ => false,
    })
}

/// Build dynamic headers for GitHub Copilot requests.
///
/// Port of `buildCopilotDynamicHeaders()` from
/// `packages/ai/src/providers/github-copilot-headers.ts`.
pub fn build_copilot_dynamic_headers(
    messages: &[Message],
    has_images: bool,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert(
        "X-Initiator".to_string(),
        infer_copilot_initiator(messages).to_string(),
    );
    headers.insert(
        "Openai-Intent".to_string(),
        "conversation-edits".to_string(),
    );

    if has_images {
        headers.insert("Copilot-Vision-Request".to_string(), "true".to_string());
    }

    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{UserContent, UserMessage};

    #[test]
    fn initiator_user_when_last_is_user() {
        let msgs = vec![Message::User(UserMessage {
            content: UserContent::Text("hello".to_string()),
            timestamp: 0,
        })];
        assert_eq!(infer_copilot_initiator(&msgs), "user");
    }

    #[test]
    fn initiator_agent_when_last_is_assistant() {
        let msgs = vec![Message::Assistant(crate::types::AssistantMessage::default())];
        assert_eq!(infer_copilot_initiator(&msgs), "agent");
    }

    #[test]
    fn initiator_user_when_empty() {
        assert_eq!(infer_copilot_initiator(&[]), "user");
    }

    #[test]
    fn copilot_headers_include_vision_when_images() {
        let headers = build_copilot_dynamic_headers(&[], true);
        assert_eq!(
            headers.get("Copilot-Vision-Request"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn copilot_headers_no_vision_when_no_images() {
        let headers = build_copilot_dynamic_headers(&[], false);
        assert!(!headers.contains_key("Copilot-Vision-Request"));
    }
}
