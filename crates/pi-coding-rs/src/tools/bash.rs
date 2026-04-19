use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Execute a bash command in the current working directory.
pub struct BashTool {
    pub cwd: String,
}

impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn label(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the current working directory. Returns stdout and stderr. Optionally provide a timeout in seconds."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(
        &self,
        _tool_call_id: &str,
        params: serde_json::Value,
        cancel: CancellationToken,
        _on_update: Option<pi_agent_rs::types::AgentToolUpdateCallback>,
    ) -> BoxFuture<'_, Result<AgentToolResult, anyhow::Error>> {
        Box::pin(async move {
            let command = params["command"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let timeout_secs = params["timeout"].as_f64();

            let child = tokio::process::Command::new("bash")
                .arg("-c")
                .arg(&command)
                .current_dir(&self.cwd)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;

            let output_future = child.wait_with_output();

            let result = if let Some(secs) = timeout_secs {
                let duration = std::time::Duration::from_secs_f64(secs);
                tokio::select! {
                    res = output_future => Ok(res?),
                    _ = tokio::time::sleep(duration) => {
                        Err(anyhow::anyhow!("Command timed out after {secs} seconds"))
                    }
                    _ = cancel.cancelled() => {
                        Err(anyhow::anyhow!("Command aborted"))
                    }
                }
            } else {
                tokio::select! {
                    res = output_future => Ok(res?),
                    _ = cancel.cancelled() => {
                        Err(anyhow::anyhow!("Command aborted"))
                    }
                }
            };

            let output = result?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            let exit_code = output.status.code().unwrap_or(-1);
            let mut text = if stdout.is_empty() && stderr.is_empty() {
                "(no output)".to_string()
            } else if stderr.is_empty() {
                stdout
            } else if stdout.is_empty() {
                stderr
            } else {
                format!("{stdout}{stderr}")
            };

            if exit_code != 0 {
                text += &format!("\n\nCommand exited with code {exit_code}");
                return Err(anyhow::anyhow!("{text}"));
            }

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text,
                    text_signature: None,
                })],
                details: json!({
                    "exitCode": exit_code
                }),
            })
        })
    }
}
