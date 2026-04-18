use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;

/// Search file contents using regex patterns.
pub struct GrepTool;

impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn label(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        "Search for a regex pattern in file contents."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files"
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
            let pattern = params["pattern"].as_str().unwrap_or("");
            let _path = params["path"].as_str().unwrap_or(".");

            // Placeholder — full implementation would use the `grep` crate
            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("grep tool: pattern={pattern} (not yet implemented)"),
                    text_signature: None,
                })],
                details: json!(null),
            })
        })
    }
}
