use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Read the contents of a file (equivalent to TS `read` tool).
pub struct ReadTool {
    pub cwd: String,
}

impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn label(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports text files. Use offset/limit for large files. When you need the full file, continue with offset until complete."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
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

            let absolute_path = if std::path::Path::new(path_str).is_absolute() {
                path_str.to_string()
            } else {
                format!("{}/{path_str}", self.cwd)
            };

            let content = tokio::fs::read_to_string(&absolute_path).await?;

            let offset = params["offset"].as_u64().map(|n| n as usize);
            let limit = params["limit"].as_u64().map(|n| n as usize);

            let all_lines: Vec<&str> = content.lines().collect();
            let total_lines = all_lines.len();

            // Apply offset (1-indexed to 0-indexed)
            let start = offset.map(|o| o.saturating_sub(1).min(total_lines)).unwrap_or(0);

            if start >= total_lines && total_lines > 0 {
                return Err(anyhow::anyhow!(
                    "Offset {} is beyond end of file ({} lines total)",
                    offset.unwrap_or(1),
                    total_lines
                ));
            }

            let end = if let Some(lim) = limit {
                (start + lim).min(total_lines)
            } else {
                total_lines
            };

            let selected: Vec<&str> = all_lines[start..end].to_vec();
            let mut text = selected.join("\n");

            // Add continuation notice if there's more content
            if end < total_lines {
                let remaining = total_lines - end;
                let next_offset = end + 1;
                text += &format!("\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]");
            }

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
