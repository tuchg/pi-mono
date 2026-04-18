use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Metadata for a saved session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMeta {
    pub id: String,
    pub created_at: String,
    pub updated_at: String,
    pub title: Option<String>,
    pub model: String,
    pub message_count: usize,
}

/// Manages JSONL-based session persistence.
pub struct SessionManager {
    sessions_dir: PathBuf,
}

impl SessionManager {
    pub fn new(sessions_dir: impl AsRef<Path>) -> Self {
        Self {
            sessions_dir: sessions_dir.as_ref().to_path_buf(),
        }
    }

    /// Create a new session and return its ID.
    pub fn create_session(&self) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let session_dir = self.sessions_dir.join(&id);
        std::fs::create_dir_all(&session_dir)?;

        let meta = SessionMeta {
            id: id.clone(),
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
            title: None,
            model: String::new(),
            message_count: 0,
        };

        let meta_path = session_dir.join("meta.json");
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

        Ok(id)
    }

    /// List all session metadata, sorted by most recent first.
    pub fn list_sessions(&self) -> Result<Vec<SessionMeta>> {
        let mut sessions = Vec::new();

        if !self.sessions_dir.exists() {
            return Ok(sessions);
        }

        for entry in std::fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let meta_path = entry.path().join("meta.json");
            if meta_path.exists() {
                let data = std::fs::read_to_string(&meta_path)?;
                if let Ok(meta) = serde_json::from_str::<SessionMeta>(&data) {
                    sessions.push(meta);
                }
            }
        }

        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    /// Append a JSONL record to a session's message log.
    pub fn append_message(
        &self,
        session_id: &str,
        message: &serde_json::Value,
    ) -> Result<()> {
        let log_path = self
            .sessions_dir
            .join(session_id)
            .join("messages.jsonl");

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        writeln!(file, "{}", serde_json::to_string(message)?)?;
        Ok(())
    }
}
