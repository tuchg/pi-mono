use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;

/// A "thinking" tool that lets the agent reason step-by-step.
pub struct ThinkTool;

impl AgentTool for ThinkTool {
    fn name(&self) -> &str {
        "think"
    }

    fn label(&self) -> &str {
        "Think"
    }

    fn description(&self) -> &str {
        "Use this tool to think through a problem step-by-step before acting."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your step-by-step reasoning"
                }
            },
            "required": ["thought"]
        })
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _on_update: Option<pi_agent_rs::types::AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let thought = params["thought"]
                .as_str()
                .unwrap_or("(no thought provided)")
                .to_string();

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: thought,
                    text_signature: None,
                })],
                details: json!(null),
            })
        })
    }
}
