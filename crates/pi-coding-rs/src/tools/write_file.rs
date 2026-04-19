use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Write content to a file (equivalent to TS `write` tool).
pub struct WriteTool {
    pub cwd: String,
}

impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn label(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        _cancel: CancellationToken,
        _on_update: Option<pi_agent_rs::types::AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let path_str = params["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'path' parameter"))?;
            let content = params["content"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'content' parameter"))?;

            let absolute_path = if std::path::Path::new(path_str).is_absolute() {
                path_str.to_string()
            } else {
                format!("{}/{path_str}", self.cwd)
            };

            // Ensure parent directory exists
            if let Some(parent) = std::path::Path::new(&absolute_path).parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            tokio::fs::write(&absolute_path, content).await?;

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text: format!("Successfully wrote {} bytes to {path_str}", content.len()),
                    text_signature: None,
                })],
                details: json!({
                    "path": path_str,
                    "bytes": content.len()
                }),
            })
        })
    }
}
