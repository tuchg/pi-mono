use std::path::Path;

use pi_agent_rs::types::{AgentTool, AgentToolResult, BoxFuture};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Search for files matching a glob pattern (equivalent to TS `find` tool).
pub struct FindTool {
    pub cwd: String,
}

impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn label(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of results (default: 1000)"
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
            let search_dir = params["path"].as_str().unwrap_or(".");
            let limit = params["limit"].as_u64().unwrap_or(1000) as usize;

            let base = if Path::new(search_dir).is_absolute() {
                search_dir.to_string()
            } else {
                format!("{}/{search_dir}", self.cwd)
            };

            let full_pattern = if Path::new(pattern).is_absolute() {
                pattern.to_string()
            } else {
                format!("{base}/{pattern}")
            };

            let base_clone = base.clone();
            let entries = tokio::task::spawn_blocking(move || -> Result<Vec<String>, anyhow::Error> {
                let paths = glob::glob(&full_pattern)
                    .map_err(|e| anyhow::anyhow!("invalid glob pattern: {e}"))?;
                let mut results = Vec::new();
                for entry in paths {
                    if results.len() >= limit {
                        break;
                    }
                    match entry {
                        Ok(path) => {
                            let path_str = path.display().to_string();
                            // Return relative paths
                            let relative = if path_str.starts_with(&base_clone) {
                                path_str[base_clone.len()..].trim_start_matches('/').to_string()
                            } else {
                                path_str
                            };
                            results.push(relative);
                        }
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
                "No files found matching pattern".to_string()
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
