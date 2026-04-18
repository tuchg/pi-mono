use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;

/// Search for files matching a glob pattern.
pub struct GlobTool;

impl AgentTool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn label(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g. '**/*.rs')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search from"
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _on_update: Option<pi_agent_rs::types::AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let pattern = params["pattern"].as_str().unwrap_or("*");
            let _path = params["path"].as_str().unwrap_or(".");

            // Placeholder — full implementation would use the `glob` crate
            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("glob tool: pattern={pattern} (not yet implemented)"),
                    text_signature: None,
                })],
                details: json!(null),
            })
        })
    }
}
