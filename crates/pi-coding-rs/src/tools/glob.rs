use std::path::Path;

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
            let pattern = params["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'pattern' parameter"))?;
            let base = params["path"].as_str().unwrap_or(".");

            // Combine base path with the glob pattern.
            let full_pattern = if Path::new(pattern).is_absolute() {
                pattern.to_string()
            } else {
                format!("{base}/{pattern}")
            };

            // glob::glob is synchronous; run it on a blocking thread.
            let entries = tokio::task::spawn_blocking(move || -> Result<Vec<String>, anyhow::Error> {
                let paths = glob::glob(&full_pattern)
                    .map_err(|e| anyhow::anyhow!("invalid glob pattern: {e}"))?;
                let mut results = Vec::new();
                for entry in paths {
                    match entry {
                        Ok(path) => results.push(path.display().to_string()),
                        Err(e) => {
                            tracing::warn!("glob entry error: {e}");
                        }
                    }
                }
                results.sort();
                Ok(results)
            })
            .await??;

            let text = if entries.is_empty() {
                "No files found.".to_string()
            } else {
                entries.join("\n")
            };

            Ok(AgentToolResult {
                content: vec![pi_ai_rs::Content::Text(pi_ai_rs::TextContent {
                    text,
                    text_signature: None,
                })],
                details: json!({ "count": entries.len() }),
            })
        })
    }
}
