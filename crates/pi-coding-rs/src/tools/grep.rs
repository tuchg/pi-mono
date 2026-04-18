use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;

/// Search file contents using regex patterns via `grep -rn`.
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
                    "description": "Glob pattern to filter files (e.g. '*.rs')"
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
            let pattern = params["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'pattern' parameter"))?;
            let path = params["path"].as_str().unwrap_or(".");
            let include = params["include"].as_str();

            let mut cmd = tokio::process::Command::new("grep");
            cmd.arg("-rn").arg("--color=never");

            if let Some(glob) = include {
                cmd.arg("--include").arg(glob);
            }

            cmd.arg("--").arg(pattern).arg(path);

            let output = cmd.output().await?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            // grep exits 1 when no matches found — not an error
            let exit_code = output.status.code().unwrap_or(-1);
            if exit_code > 1 || (exit_code != 0 && !stderr.is_empty()) {
                return Err(anyhow::anyhow!("grep failed (exit {exit_code}): {stderr}"));
            }

            let text = if stdout.is_empty() {
                "No matches found.".to_string()
            } else {
                stdout
            };

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text,
                    text_signature: None,
                })],
                details: json!({ "exitCode": exit_code }),
            })
        })
    }
}
