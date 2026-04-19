use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// List directory contents (equivalent to TS `ls` tool).
pub struct LsTool {
    pub cwd: String,
}

impl AgentTool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn label(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of entries to return (default: 500)"
                }
            },
            "required": []
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
            let path_str = params["path"].as_str().unwrap_or(".");
            let limit = params["limit"].as_u64().unwrap_or(500) as usize;

            let dir_path = if std::path::Path::new(path_str).is_absolute() {
                path_str.to_string()
            } else {
                format!("{}/{path_str}", self.cwd)
            };

            if !tokio::fs::try_exists(&dir_path).await.unwrap_or(false) {
                return Err(anyhow::anyhow!("Path not found: {dir_path}"));
            }

            let metadata = tokio::fs::metadata(&dir_path).await?;
            if !metadata.is_dir() {
                return Err(anyhow::anyhow!("Not a directory: {dir_path}"));
            }

            let mut entries = Vec::new();
            let mut dir = tokio::fs::read_dir(&dir_path).await?;

            while let Some(entry) = dir.next_entry().await? {
                if entries.len() >= limit {
                    break;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                let file_type = entry.file_type().await?;
                if file_type.is_dir() {
                    entries.push(format!("{name}/"));
                } else {
                    entries.push(name);
                }
            }

            // Sort case-insensitively to match TS behavior
            entries.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));

            let text = if entries.is_empty() {
                "(empty directory)".to_string()
            } else {
                entries.join("\n")
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
