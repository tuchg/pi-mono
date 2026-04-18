use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;

/// Read the contents of a file.
pub struct ReadFileTool;

impl AgentTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn label(&self) -> &str {
        "Read File"
    }

    fn description(&self) -> &str {
        "Read the full contents of a file at the given path."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                },
                "startLine": {
                    "type": "integer",
                    "description": "Start line (1-indexed, inclusive)"
                },
                "endLine": {
                    "type": "integer",
                    "description": "End line (1-indexed, inclusive)"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _on_update: Option<pi_agent_rs::types::AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let path = params["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'path' parameter"))?;

            let content = tokio::fs::read_to_string(path).await?;

            // Apply line range if specified
            let start = params["startLine"].as_u64().map(|n| n as usize);
            let end = params["endLine"].as_u64().map(|n| n as usize);

            let text = match (start, end) {
                (Some(s), Some(e)) => {
                    content
                        .lines()
                        .enumerate()
                        .filter(|(i, _)| *i + 1 >= s && *i + 1 <= e)
                        .map(|(i, line)| format!("{}. {}", i + 1, line))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
                (Some(s), None) => {
                    content
                        .lines()
                        .enumerate()
                        .filter(|(i, _)| *i + 1 >= s)
                        .map(|(i, line)| format!("{}. {}", i + 1, line))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
                _ => content,
            };

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text,
                    text_signature: None,
                })],
                details: json!(null),
            })
        })
    }
}
