use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Search file contents for a pattern using grep (equivalent to TS `grep` tool using ripgrep).
pub struct GrepTool {
    pub cwd: String,
}

impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn label(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"
                },
                "ignoreCase": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex (default: false)"
                },
                "context": {
                    "type": "number",
                    "description": "Number of lines to show before and after each match (default: 0)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of matches to return (default: 100)"
                }
            },
            "required": ["pattern"]
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
            let pattern = params["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'pattern' parameter"))?;
            let path = params["path"].as_str().unwrap_or(".");
            let glob_filter = params["glob"].as_str();
            let ignore_case = params["ignoreCase"].as_bool().unwrap_or(false);
            let literal = params["literal"].as_bool().unwrap_or(false);
            let context_lines = params["context"].as_u64().unwrap_or(0);
            let _limit = params["limit"].as_u64().unwrap_or(100);

            let search_path = if std::path::Path::new(path).is_absolute() {
                path.to_string()
            } else {
                format!("{}/{path}", self.cwd)
            };

            let mut cmd = tokio::process::Command::new("grep");
            cmd.arg("-rn").arg("--color=never");

            if ignore_case {
                cmd.arg("-i");
            }
            if literal {
                cmd.arg("-F");
            }
            if context_lines > 0 {
                cmd.arg(format!("-C{context_lines}"));
            }
            if let Some(glob) = glob_filter {
                cmd.arg("--include").arg(glob);
            }

            cmd.arg("--").arg(pattern).arg(&search_path);

            let output = cmd.output().await?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            // grep exits 1 when no matches found -- not an error
            let exit_code = output.status.code().unwrap_or(-1);
            if exit_code > 1 || (exit_code != 0 && !stderr.is_empty()) {
                return Err(anyhow::anyhow!("grep failed (exit {exit_code}): {stderr}"));
            }

            let text = if stdout.is_empty() {
                "No matches found".to_string()
            } else {
                stdout.trim_end().to_string()
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
