use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CURRENT_SESSION_VERSION: u32 = 3;

// ---------------------------------------------------------------------------
// Public typed structures (output / option types)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct NewSessionOptions {
    pub id: Option<String>,
    pub parent_session: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionContextModel {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone)]
pub struct SessionContext {
    pub messages: Vec<Value>,
    pub thinking_level: String,
    pub model: Option<SessionContextModel>,
}

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub path: String,
    pub id: String,
    pub cwd: String,
    pub name: Option<String>,
    pub parent_session_path: Option<String>,
    pub created: chrono::DateTime<chrono::Utc>,
    pub modified: chrono::DateTime<chrono::Utc>,
    pub message_count: usize,
    pub first_message: String,
    pub all_messages_text: String,
}

#[derive(Debug, Clone)]
pub struct SessionTreeNode {
    pub entry: Value,
    pub children: Vec<SessionTreeNode>,
    pub label: Option<String>,
    pub label_timestamp: Option<String>,
}

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

fn create_session_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Generate a unique 8-char hex ID (first 8 chars of a UUID v4 string).
fn generate_id(contains: impl Fn(&str) -> bool) -> String {
    for _ in 0..100 {
        let full = uuid::Uuid::new_v4().to_string();
        let id: String = full.chars().take(8).collect();
        if !contains(&id) {
            return id;
        }
    }
    uuid::Uuid::new_v4().to_string()
}

// ---------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------

fn entry_type(e: &Value) -> &str {
    e.get("type").and_then(|v| v.as_str()).unwrap_or("")
}

fn entry_id(e: &Value) -> &str {
    e.get("id").and_then(|v| v.as_str()).unwrap_or("")
}

fn entry_parent_id(e: &Value) -> Option<&str> {
    match e.get("parentId") {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn entry_timestamp(e: &Value) -> &str {
    e.get("timestamp").and_then(|v| v.as_str()).unwrap_or("")
}

fn is_session_entry(e: &Value) -> bool {
    entry_type(e) != "session"
}

fn now_iso() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

// ---------------------------------------------------------------------------
// Migration
// ---------------------------------------------------------------------------

/// Migrate v1 → v2: add id/parentId tree structure. Mutates entries in place.
fn migrate_v1_to_v2(entries: &mut Vec<Value>) {
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut prev_id: Option<String> = None;

    for entry in entries.iter_mut() {
        if entry_type(entry) == "session" {
            if let Some(obj) = entry.as_object_mut() {
                obj.insert("version".to_string(), Value::Number(2.into()));
            }
            continue;
        }

        let new_id = generate_id(|id| ids.contains(id));
        ids.insert(new_id.clone());

        if let Some(obj) = entry.as_object_mut() {
            obj.insert("id".to_string(), Value::String(new_id.clone()));
            obj.insert(
                "parentId".to_string(),
                match &prev_id {
                    Some(p) => Value::String(p.clone()),
                    None => Value::Null,
                },
            );
        }
        prev_id = Some(new_id);
    }

    // Second pass: convert firstKeptEntryIndex → firstKeptEntryId for compaction entries
    // We need to collect the index→id mapping first, then update
    let id_by_idx: Vec<Option<String>> = entries
        .iter()
        .map(|e| {
            if entry_type(e) == "session" {
                None
            } else {
                e.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
            }
        })
        .collect();

    for entry in entries.iter_mut() {
        if entry_type(entry) != "compaction" {
            continue;
        }
        let idx = entry
            .get("firstKeptEntryIndex")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        if let Some(idx) = idx {
            let target_id = id_by_idx.get(idx).and_then(|o| o.as_deref());
            if let Some(tid) = target_id {
                if let Some(obj) = entry.as_object_mut() {
                    obj.insert("firstKeptEntryId".to_string(), Value::String(tid.to_string()));
                    obj.remove("firstKeptEntryIndex");
                }
            }
        }
    }
}

/// Migrate v2 → v3: rename hookMessage role to custom.
fn migrate_v2_to_v3(entries: &mut Vec<Value>) {
    for entry in entries.iter_mut() {
        if entry_type(entry) == "session" {
            if let Some(obj) = entry.as_object_mut() {
                obj.insert("version".to_string(), Value::Number(3.into()));
            }
            continue;
        }

        if entry_type(entry) == "message" {
            if let Some(role) = entry
                .get("message")
                .and_then(|m| m.get("role"))
                .and_then(|r| r.as_str())
            {
                if role == "hookMessage" {
                    if let Some(msg) = entry.get_mut("message") {
                        if let Some(obj) = msg.as_object_mut() {
                            obj.insert("role".to_string(), Value::String("custom".to_string()));
                        }
                    }
                }
            }
        }
    }
}

fn migrate_to_current_version(entries: &mut Vec<Value>) -> bool {
    let version = entries
        .iter()
        .find(|e| entry_type(e) == "session")
        .and_then(|e| e.get("version"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;

    if version >= CURRENT_SESSION_VERSION {
        return false;
    }

    if version < 2 {
        migrate_v1_to_v2(entries);
    }
    if version < 3 {
        migrate_v2_to_v3(entries);
    }

    true
}

/// Exported for testing.
pub fn migrate_session_entries(entries: &mut Vec<Value>) {
    migrate_to_current_version(entries);
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

pub fn parse_session_entries(content: &str) -> Vec<Value> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Context building
// ---------------------------------------------------------------------------

pub fn get_latest_compaction_entry(entries: &[Value]) -> Option<&Value> {
    entries.iter().rev().find(|e| entry_type(e) == "compaction")
}

/// Build session context from entries, optionally anchored at a specific leaf.
///
/// `leaf_id = None`  → use last entry
/// `leaf_id = Some(id)` → walk from that entry (if not found, falls back to last)
pub fn build_session_context(entries: &[Value], leaf_id: Option<&str>) -> SessionContext {
    build_session_context_with_index(entries, leaf_id, None)
}

fn build_session_context_with_index(
    entries: &[Value],
    leaf_id: Option<&str>,
    by_id_opt: Option<&HashMap<String, Value>>,
) -> SessionContext {
    let owned_map: HashMap<String, Value>;
    let by_id: &HashMap<String, Value> = if let Some(m) = by_id_opt {
        m
    } else {
        owned_map = entries
            .iter()
            .filter(|e| is_session_entry(e))
            .map(|e| (entry_id(e).to_string(), e.clone()))
            .collect();
        &owned_map
    };

    // Find leaf entry
    let leaf: &Value = if let Some(id) = leaf_id {
        by_id
            .get(id)
            .or_else(|| entries.iter().filter(|e| is_session_entry(e)).last())
    } else {
        entries.iter().filter(|e| is_session_entry(e)).last()
    }
    .unwrap_or_else(|| {
        return &Value::Null;
    });

    if leaf.is_null() {
        return SessionContext {
            messages: vec![],
            thinking_level: "off".to_string(),
            model: None,
        };
    }

    // Walk from leaf to root
    let mut path: Vec<&Value> = vec![];
    let mut current = Some(leaf);
    while let Some(entry) = current {
        if entry.is_null() {
            break;
        }
        path.insert(0, entry);
        current = entry_parent_id(entry).and_then(|pid| by_id.get(pid));
    }

    // Extract settings along path
    let mut thinking_level = "off".to_string();
    let mut model: Option<SessionContextModel> = None;
    let mut compaction: Option<&Value> = None;

    for entry in &path {
        match entry_type(entry) {
            "thinking_level_change" => {
                if let Some(level) = entry.get("thinkingLevel").and_then(|v| v.as_str()) {
                    thinking_level = level.to_string();
                }
            }
            "model_change" => {
                if let (Some(provider), Some(mid)) = (
                    entry.get("provider").and_then(|v| v.as_str()),
                    entry.get("modelId").and_then(|v| v.as_str()),
                ) {
                    model = Some(SessionContextModel {
                        provider: provider.to_string(),
                        model_id: mid.to_string(),
                    });
                }
            }
            "message" => {
                if let Some(msg) = entry.get("message") {
                    if msg.get("role").and_then(|v| v.as_str()) == Some("assistant") {
                        if let (Some(provider), Some(mid)) = (
                            msg.get("provider").and_then(|v| v.as_str()),
                            msg.get("model").and_then(|v| v.as_str()),
                        ) {
                            model = Some(SessionContextModel {
                                provider: provider.to_string(),
                                model_id: mid.to_string(),
                            });
                        }
                    }
                }
            }
            "compaction" => {
                compaction = Some(entry);
            }
            _ => {}
        }
    }

    // Build messages list
    let mut messages: Vec<Value> = vec![];

    let to_message = |entry: &Value| -> Option<Value> {
        match entry_type(entry) {
            "message" => entry.get("message").cloned(),
            "custom_message" => {
                let ts = entry
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.timestamp_millis())
                    .unwrap_or(0);
                Some(serde_json::json!({
                    "role": "custom",
                    "customType": entry.get("customType"),
                    "content": entry.get("content"),
                    "display": entry.get("display"),
                    "details": entry.get("details"),
                    "timestamp": ts
                }))
            }
            "branch_summary" => {
                let summary = entry.get("summary").and_then(|v| v.as_str()).unwrap_or("");
                if summary.is_empty() {
                    return None;
                }
                let ts = entry
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.timestamp_millis())
                    .unwrap_or(0);
                Some(serde_json::json!({
                    "role": "branchSummary",
                    "summary": summary,
                    "fromId": entry.get("fromId"),
                    "timestamp": ts
                }))
            }
            _ => None,
        }
    };

    if let Some(comp) = compaction {
        let comp_summary = comp
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let comp_tokens = comp.get("tokensBefore").and_then(|v| v.as_u64()).unwrap_or(0);
        let comp_id = entry_id(comp).to_string();
        let first_kept_id = comp
            .get("firstKeptEntryId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        messages.push(serde_json::json!({
            "role": "compactionSummary",
            "summary": comp_summary,
            "tokensBefore": comp_tokens,
        }));

        let comp_idx = path
            .iter()
            .position(|e| entry_type(e) == "compaction" && entry_id(e) == comp_id)
            .unwrap_or(path.len());

        let mut found_first_kept = false;
        for i in 0..comp_idx {
            let e = path[i];
            if entry_id(e) == first_kept_id {
                found_first_kept = true;
            }
            if found_first_kept {
                if let Some(msg) = to_message(e) {
                    messages.push(msg);
                }
            }
        }

        for i in (comp_idx + 1)..path.len() {
            if let Some(msg) = to_message(path[i]) {
                messages.push(msg);
            }
        }
    } else {
        for entry in &path {
            if let Some(msg) = to_message(entry) {
                messages.push(msg);
            }
        }
    }

    SessionContext {
        messages,
        thinking_level,
        model,
    }
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------

/// Read and parse a .jsonl session file. Returns empty vec if file doesn't exist or is invalid.
pub fn load_entries_from_file(file_path: impl AsRef<Path>) -> Vec<Value> {
    let path = file_path.as_ref();
    if !path.exists() {
        return vec![];
    }
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let entries: Vec<Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if entries.is_empty() {
        return vec![];
    }
    // Validate session header
    let header = &entries[0];
    if entry_type(header) != "session" {
        return vec![];
    }
    if header.get("id").and_then(|v| v.as_str()).is_none() {
        return vec![];
    }
    entries
}

fn is_valid_session_file(path: impl AsRef<Path>) -> bool {
    use std::io::Read;
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; 512];
    let n = match f.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let s = std::str::from_utf8(&buf[..n]).unwrap_or("");
    let first_line = s.lines().next().unwrap_or("");
    if first_line.is_empty() {
        return false;
    }
    match serde_json::from_str::<Value>(first_line) {
        Ok(v) => entry_type(&v) == "session" && v.get("id").and_then(|v| v.as_str()).is_some(),
        Err(_) => false,
    }
}

pub fn find_most_recent_session(session_dir: impl AsRef<Path>) -> Option<PathBuf> {
    let dir = session_dir.as_ref();
    let read = std::fs::read_dir(dir).ok()?;
    let mut candidates: Vec<(PathBuf, std::time::SystemTime)> = read
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .filter(|p| is_valid_session_file(p))
        .filter_map(|p| {
            let mtime = std::fs::metadata(&p).ok()?.modified().ok()?;
            Some((p, mtime))
        })
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.into_iter().next().map(|(p, _)| p)
}

pub fn get_default_session_dir(cwd: &str) -> PathBuf {
    let safe = format!(
        "--{}--",
        cwd.trim_start_matches(['/', '\\'])
            .replace(['/', '\\', ':'], "-")
    );
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    let base = home.join(".pi").join("agent").join("sessions").join(safe);
    if !base.exists() {
        let _ = std::fs::create_dir_all(&base);
    }
    base
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

pub struct SessionManager {
    session_id: String,
    session_file: Option<PathBuf>,
    session_dir: PathBuf,
    cwd: String,
    persist: bool,
    flushed: bool,
    // All file entries including header
    file_entries: Vec<Value>,
    // Index of non-header entries by id
    by_id: HashMap<String, Value>,
    labels_by_id: HashMap<String, String>,
    label_timestamps_by_id: HashMap<String, String>,
    leaf_id: Option<String>,
}

impl SessionManager {
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    fn new_internal(
        cwd: String,
        session_dir: PathBuf,
        session_file: Option<PathBuf>,
        persist: bool,
    ) -> Self {
        let mut sm = Self {
            session_id: String::new(),
            session_file: None,
            session_dir: session_dir.clone(),
            cwd,
            persist,
            flushed: false,
            file_entries: vec![],
            by_id: HashMap::new(),
            labels_by_id: HashMap::new(),
            label_timestamps_by_id: HashMap::new(),
            leaf_id: None,
        };

        if persist && !session_dir.as_os_str().is_empty() && !session_dir.exists() {
            let _ = std::fs::create_dir_all(&session_dir);
        }

        if let Some(sf) = session_file {
            sm.set_session_file(sf);
        } else {
            sm.new_session(None);
        }

        sm
    }

    /// Create a new in-memory session (no file persistence).
    pub fn in_memory(cwd: Option<&str>) -> Self {
        let cwd = cwd.unwrap_or(".").to_string();
        Self::new_internal(cwd, PathBuf::new(), None, false)
    }

    /// Create a new persisted session in the given directory.
    pub fn create(cwd: &str, session_dir: Option<&Path>) -> Self {
        let dir = session_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| get_default_session_dir(cwd));
        Self::new_internal(cwd.to_string(), dir, None, true)
    }

    /// Open a specific session file.
    pub fn open(path: impl AsRef<Path>, session_dir: Option<&Path>, cwd_override: Option<&str>) -> Self {
        let path = path.as_ref().to_path_buf();
        let entries = load_entries_from_file(&path);
        let header_cwd = entries
            .iter()
            .find(|e| entry_type(e) == "session")
            .and_then(|e| e.get("cwd"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let cwd = cwd_override.unwrap_or(&header_cwd).to_string();
        let dir = session_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| path.parent().unwrap_or(Path::new(".")).to_path_buf());
        Self::new_internal(cwd, dir, Some(path), true)
    }

    /// Continue the most recent session, or create a new one.
    pub fn continue_recent(cwd: &str, session_dir: Option<&Path>) -> Self {
        let dir = session_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| get_default_session_dir(cwd));
        let most_recent = find_most_recent_session(&dir);
        if let Some(path) = most_recent {
            Self::new_internal(cwd.to_string(), dir, Some(path), true)
        } else {
            Self::new_internal(cwd.to_string(), dir, None, true)
        }
    }

    /// Fork a session from another project's session file.
    pub fn fork_from(source_path: impl AsRef<Path>, target_cwd: &str, session_dir: Option<&Path>) -> Result<Self> {
        let source_path = source_path.as_ref();
        let source_entries = load_entries_from_file(source_path);
        if source_entries.is_empty() {
            return Err(anyhow!(
                "Cannot fork: source session file is empty or invalid: {}",
                source_path.display()
            ));
        }
        if source_entries.iter().find(|e| entry_type(e) == "session").is_none() {
            return Err(anyhow!(
                "Cannot fork: source session has no header: {}",
                source_path.display()
            ));
        }

        let dir = session_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| get_default_session_dir(target_cwd));
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }

        let new_id = create_session_id();
        let timestamp = now_iso();
        let file_ts = timestamp.replace([':', '.'], "-");
        let new_file = dir.join(format!("{}_{}.jsonl", file_ts, new_id));

        let new_header = serde_json::json!({
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": new_id,
            "timestamp": timestamp,
            "cwd": target_cwd,
            "parentSession": source_path.to_string_lossy()
        });

        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_file)?;
        writeln!(f, "{}", serde_json::to_string(&new_header)?)?;

        for entry in &source_entries {
            if entry_type(entry) != "session" {
                writeln!(f, "{}", serde_json::to_string(entry)?)?;
            }
        }

        Ok(Self::new_internal(
            target_cwd.to_string(),
            dir,
            Some(new_file),
            true,
        ))
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    fn set_session_file(&mut self, session_file: PathBuf) {
        let abs = if session_file.is_absolute() {
            session_file.clone()
        } else {
            std::env::current_dir()
                .unwrap_or_default()
                .join(&session_file)
        };
        self.session_file = Some(abs.clone());

        if abs.exists() {
            self.file_entries = load_entries_from_file(&abs);

            if self.file_entries.is_empty() {
                // Corrupted or empty – start fresh, preserve path
                let explicit = self.session_file.clone();
                self.new_session(None);
                self.session_file = explicit;
                self.rewrite_file();
                self.flushed = true;
                return;
            }

            let header_id = self
                .file_entries
                .iter()
                .find(|e| entry_type(e) == "session")
                .and_then(|e| e.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.session_id = if header_id.is_empty() {
                create_session_id()
            } else {
                header_id
            };

            if migrate_to_current_version(&mut self.file_entries) {
                self.rewrite_file();
            }

            self.build_index();
            self.flushed = true;
        } else {
            let explicit = self.session_file.clone();
            self.new_session(None);
            self.session_file = explicit;
        }
    }

    /// Start a new session, returning the new session file path if persisting.
    pub fn new_session(&mut self, options: Option<NewSessionOptions>) -> Option<PathBuf> {
        self.session_id = options
            .as_ref()
            .and_then(|o| o.id.clone())
            .unwrap_or_else(create_session_id);
        let timestamp = now_iso();
        let parent_session = options.and_then(|o| o.parent_session);

        let mut header = serde_json::json!({
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self.session_id,
            "timestamp": timestamp,
            "cwd": self.cwd,
        });
        if let Some(ref ps) = parent_session {
            header["parentSession"] = Value::String(ps.clone());
        }

        self.file_entries = vec![header];
        self.by_id.clear();
        self.labels_by_id.clear();
        self.label_timestamps_by_id.clear();
        self.leaf_id = None;
        self.flushed = false;

        if self.persist {
            let file_ts = timestamp.replace([':', '.'], "-");
            let path = self
                .session_dir
                .join(format!("{}_{}.jsonl", file_ts, self.session_id));
            self.session_file = Some(path);
        }
        self.session_file.clone()
    }

    fn build_index(&mut self) {
        self.by_id.clear();
        self.labels_by_id.clear();
        self.label_timestamps_by_id.clear();
        self.leaf_id = None;

        for entry in &self.file_entries {
            if entry_type(entry) == "session" {
                continue;
            }
            let id = entry_id(entry).to_string();
            self.by_id.insert(id.clone(), entry.clone());
            self.leaf_id = Some(id.clone());

            if entry_type(entry) == "label" {
                let target_id = entry
                    .get("targetId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let label = entry.get("label").and_then(|v| v.as_str());
                let ts = entry_timestamp(entry).to_string();
                if let Some(lbl) = label {
                    if !lbl.is_empty() {
                        self.labels_by_id.insert(target_id.clone(), lbl.to_string());
                        self.label_timestamps_by_id.insert(target_id, ts);
                    } else {
                        self.labels_by_id.remove(&target_id);
                        self.label_timestamps_by_id.remove(&target_id);
                    }
                } else {
                    // null label → clear
                    self.labels_by_id.remove(&target_id);
                    self.label_timestamps_by_id.remove(&target_id);
                }
            }
        }
    }

    fn rewrite_file(&self) {
        if !self.persist {
            return;
        }
        let Some(ref path) = self.session_file else {
            return;
        };
        let content: String = self
            .file_entries
            .iter()
            .map(|e| serde_json::to_string(e).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let _ = std::fs::write(path, content);
    }

    fn has_assistant(&self) -> bool {
        self.file_entries.iter().any(|e| {
            entry_type(e) == "message"
                && e.get("message")
                    .and_then(|m| m.get("role"))
                    .and_then(|r| r.as_str())
                    == Some("assistant")
        })
    }

    fn persist_entry(&mut self, entry: &Value) {
        if !self.persist {
            return;
        }
        let Some(ref path) = self.session_file else {
            return;
        };
        if !self.has_assistant() {
            // Not ready to flush yet
            self.flushed = false;
            return;
        }
        let path = path.clone();
        if !self.flushed {
            let mut f = match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(f) => f,
                Err(_) => return,
            };
            for e in &self.file_entries {
                let _ = writeln!(f, "{}", serde_json::to_string(e).unwrap_or_default());
            }
            self.flushed = true;
        } else {
            let mut f = match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(f) => f,
                Err(_) => return,
            };
            let _ = writeln!(f, "{}", serde_json::to_string(entry).unwrap_or_default());
        }
    }

    fn append_entry(&mut self, entry: Value) {
        let id = entry_id(&entry).to_string();
        self.file_entries.push(entry.clone());
        self.by_id.insert(id.clone(), entry.clone());
        self.leaf_id = Some(id);
        self.persist_entry(&entry);
    }

    fn make_base(&self) -> (String, Option<String>, String) {
        let id = generate_id(|id| self.by_id.contains_key(id));
        let parent_id = self.leaf_id.clone();
        let timestamp = now_iso();
        (id, parent_id, timestamp)
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    pub fn is_persisted(&self) -> bool {
        self.persist
    }

    pub fn get_cwd(&self) -> &str {
        &self.cwd
    }

    pub fn get_session_dir(&self) -> &Path {
        &self.session_dir
    }

    pub fn get_session_id(&self) -> &str {
        &self.session_id
    }

    pub fn get_session_file(&self) -> Option<&Path> {
        self.session_file.as_deref()
    }

    pub fn get_leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn get_leaf_entry(&self) -> Option<&Value> {
        self.leaf_id.as_ref().and_then(|id| self.by_id.get(id))
    }

    pub fn get_entry(&self, id: &str) -> Option<&Value> {
        self.by_id.get(id)
    }

    pub fn get_children(&self, parent_id: &str) -> Vec<&Value> {
        self.by_id
            .values()
            .filter(|e| entry_parent_id(e) == Some(parent_id))
            .collect()
    }

    pub fn get_label(&self, id: &str) -> Option<&str> {
        self.labels_by_id.get(id).map(|s| s.as_str())
    }

    pub fn get_header(&self) -> Option<SessionHeader> {
        self.file_entries
            .iter()
            .find(|e| entry_type(e) == "session")
            .and_then(|e| serde_json::from_value(e.clone()).ok())
    }

    /// Returns all session entries (excludes header).
    pub fn get_entries(&self) -> Vec<Value> {
        self.file_entries
            .iter()
            .filter(|e| is_session_entry(e))
            .cloned()
            .collect()
    }

    pub fn get_session_name(&self) -> Option<String> {
        for entry in self.get_entries().iter().rev() {
            if entry_type(entry) == "session_info" {
                let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
                return if name.trim().is_empty() {
                    None
                } else {
                    Some(name.trim().to_string())
                };
            }
        }
        None
    }

    // -------------------------------------------------------------------------
    // Append operations
    // -------------------------------------------------------------------------

    pub fn append_message(&mut self, message: Value) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let entry = serde_json::json!({
            "type": "message",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "message": message
        });
        let ret = id.clone();
        self.append_entry(entry);
        ret
    }

    pub fn append_thinking_level_change(&mut self, thinking_level: &str) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let entry = serde_json::json!({
            "type": "thinking_level_change",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "thinkingLevel": thinking_level
        });
        let ret = id.clone();
        self.append_entry(entry);
        ret
    }

    pub fn append_model_change(&mut self, provider: &str, model_id: &str) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let entry = serde_json::json!({
            "type": "model_change",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "provider": provider,
            "modelId": model_id
        });
        let ret = id.clone();
        self.append_entry(entry);
        ret
    }

    pub fn append_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
        details: Option<Value>,
        from_hook: Option<bool>,
    ) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let mut entry = serde_json::json!({
            "type": "compaction",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before
        });
        if let Some(d) = details {
            entry["details"] = d;
        }
        if let Some(fh) = from_hook {
            entry["fromHook"] = Value::Bool(fh);
        }
        let ret = entry_id(&entry).to_string();
        self.append_entry(entry);
        ret
    }

    pub fn append_custom_entry(&mut self, custom_type: &str, data: Option<Value>) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let mut entry = serde_json::json!({
            "type": "custom",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "customType": custom_type
        });
        if let Some(d) = data {
            entry["data"] = d;
        }
        let ret = entry_id(&entry).to_string();
        self.append_entry(entry);
        ret
    }

    pub fn append_session_info(&mut self, name: &str) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let entry = serde_json::json!({
            "type": "session_info",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "name": name.trim()
        });
        let ret = entry_id(&entry).to_string();
        self.append_entry(entry);
        ret
    }

    pub fn append_custom_message_entry(
        &mut self,
        custom_type: &str,
        content: Value,
        display: bool,
        details: Option<Value>,
    ) -> String {
        let (id, parent_id, timestamp) = self.make_base();
        let mut entry = serde_json::json!({
            "type": "custom_message",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "customType": custom_type,
            "content": content,
            "display": display
        });
        if let Some(d) = details {
            entry["details"] = d;
        }
        let ret = entry_id(&entry).to_string();
        self.append_entry(entry);
        ret
    }

    /// Set or clear a label on an entry. Returns the label entry id.
    /// Errors if `target_id` is not found.
    pub fn append_label_change(&mut self, target_id: &str, label: Option<&str>) -> Result<String> {
        if !self.by_id.contains_key(target_id) {
            return Err(anyhow!("Entry {} not found", target_id));
        }
        let (id, parent_id, timestamp) = self.make_base();
        let entry = serde_json::json!({
            "type": "label",
            "id": id,
            "parentId": parent_id,
            "timestamp": timestamp,
            "targetId": target_id,
            "label": label
        });
        // Update label index
        let ts = timestamp.clone();
        match label {
            Some(lbl) if !lbl.is_empty() => {
                self.labels_by_id.insert(target_id.to_string(), lbl.to_string());
                self.label_timestamps_by_id.insert(target_id.to_string(), ts);
            }
            _ => {
                self.labels_by_id.remove(target_id);
                self.label_timestamps_by_id.remove(target_id);
            }
        }
        let ret = entry_id(&entry).to_string();
        self.append_entry(entry);
        Ok(ret)
    }

    // -------------------------------------------------------------------------
    // Tree traversal
    // -------------------------------------------------------------------------

    /// Walk from the given entry (or current leaf) to root, returning path in root-first order.
    pub fn get_branch(&self, from_id: Option<&str>) -> Vec<Value> {
        let start_id = from_id.or(self.leaf_id.as_deref());
        let mut path: Vec<Value> = vec![];
        let mut current_id = start_id.map(|s| s.to_string());
        while let Some(id) = &current_id {
            match self.by_id.get(id.as_str()) {
                Some(entry) => {
                    path.insert(0, entry.clone());
                    current_id = entry_parent_id(entry).map(|s| s.to_string());
                }
                None => break,
            }
        }
        path
    }

    pub fn build_session_context(&self) -> SessionContext {
        match &self.leaf_id {
            None => SessionContext {
                messages: vec![],
                thinking_level: "off".to_string(),
                model: None,
            },
            Some(id) => {
                let entries = self.get_entries();
                build_session_context_with_index(&entries, Some(id.as_str()), Some(&self.by_id))
            }
        }
    }

    /// Get the session as a tree of nodes, with label information resolved.
    pub fn get_tree(&self) -> Vec<SessionTreeNode> {
        let entries = self.get_entries();
        let mut node_map: HashMap<String, SessionTreeNode> = HashMap::new();

        for entry in &entries {
            let id = entry_id(entry).to_string();
            let label = self.labels_by_id.get(&id).cloned();
            let label_timestamp = self.label_timestamps_by_id.get(&id).cloned();
            node_map.insert(
                id,
                SessionTreeNode {
                    entry: entry.clone(),
                    children: vec![],
                    label,
                    label_timestamp,
                },
            );
        }

        let mut roots: Vec<String> = vec![];
        let parent_ids: Vec<(String, Option<String>)> = entries
            .iter()
            .map(|e| {
                let id = entry_id(e).to_string();
                let pid = entry_parent_id(e).map(|s| s.to_string());
                (id, pid)
            })
            .collect();

        for (id, pid) in &parent_ids {
            match pid {
                None => roots.push(id.clone()),
                Some(p) if p == id => roots.push(id.clone()),
                Some(p) => {
                    if node_map.contains_key(p.as_str()) {
                        // We'll attach after collecting
                    } else {
                        roots.push(id.clone());
                    }
                }
            }
        }

        // Attach children to parents (need to be careful with borrow)
        // Collect (parent_id, child_id) pairs first
        let attachments: Vec<(String, String)> = parent_ids
            .iter()
            .filter_map(|(id, pid)| {
                let p = pid.as_deref()?;
                if p == id {
                    return None;
                }
                if node_map.contains_key(p) {
                    Some((p.to_string(), id.clone()))
                } else {
                    None
                }
            })
            .collect();

        // We need to build the tree iteratively; use an index-based approach
        // Convert node_map to Vec<SessionTreeNode> with stable ordering
        let order: Vec<String> = entries.iter().map(|e| entry_id(e).to_string()).collect();

        // Use a temporary flat representation then assemble
        let mut children_map: HashMap<String, Vec<String>> = HashMap::new();
        for (parent, child) in &attachments {
            children_map
                .entry(parent.clone())
                .or_default()
                .push(child.clone());
        }

        // Build tree recursively (session trees are shallow in practice)
        fn build_node(
            id: &str,
            node_map: &mut HashMap<String, SessionTreeNode>,
            children_map: &HashMap<String, Vec<String>>,
        ) -> SessionTreeNode {
            let mut node = node_map.remove(id).unwrap();
            if let Some(child_ids) = children_map.get(id) {
                let mut kids: Vec<SessionTreeNode> = child_ids
                    .iter()
                    .map(|cid| build_node(cid, node_map, children_map))
                    .collect();
                // Sort children by timestamp (oldest first)
                kids.sort_by(|a, b| {
                    let ta = entry_timestamp(&a.entry);
                    let tb = entry_timestamp(&b.entry);
                    ta.cmp(tb)
                });
                node.children = kids;
            }
            node
        }

        // Determine root ids (entries with parentId = null or broken parent)
        let root_ids: Vec<String> = order
            .iter()
            .filter(|id| {
                let entry = &self.by_id[id.as_str()];
                let pid = entry_parent_id(entry);
                match pid {
                    None => true,
                    Some(p) => p == id.as_str() || !node_map.contains_key(p),
                }
            })
            .cloned()
            .collect();

        root_ids
            .iter()
            .filter_map(|id| {
                if node_map.contains_key(id.as_str()) {
                    Some(build_node(id, &mut node_map, &children_map))
                } else {
                    None
                }
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // Branching
    // -------------------------------------------------------------------------

    /// Move the leaf pointer to the specified entry (start a new branch).
    pub fn branch(&mut self, branch_from_id: &str) -> Result<()> {
        if !self.by_id.contains_key(branch_from_id) {
            return Err(anyhow!("Entry {} not found", branch_from_id));
        }
        self.leaf_id = Some(branch_from_id.to_string());
        Ok(())
    }

    /// Reset the leaf pointer to null (before any entries).
    pub fn reset_leaf(&mut self) {
        self.leaf_id = None;
    }

    /// Branch with a summary of the abandoned path.
    pub fn branch_with_summary(
        &mut self,
        branch_from_id: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_hook: Option<bool>,
    ) -> Result<String> {
        if let Some(id) = branch_from_id {
            if !self.by_id.contains_key(id) {
                return Err(anyhow!("Entry {} not found", id));
            }
        }
        self.leaf_id = branch_from_id.map(|s| s.to_string());

        let id = generate_id(|id| self.by_id.contains_key(id));
        let timestamp = now_iso();
        let mut entry = serde_json::json!({
            "type": "branch_summary",
            "id": id,
            "parentId": branch_from_id,
            "timestamp": timestamp,
            "fromId": branch_from_id.unwrap_or("root"),
            "summary": summary,
        });
        if let Some(d) = details {
            entry["details"] = d;
        }
        if let Some(fh) = from_hook {
            entry["fromHook"] = Value::Bool(fh);
        }
        let ret = entry_id(&entry).to_string();
        self.append_entry(entry);
        Ok(ret)
    }

    /// Create a new session containing only the path from root to the specified leaf.
    /// In persist mode, returns the new session file path. In memory mode, replaces
    /// the current in-memory state and returns None.
    pub fn create_branched_session(&mut self, leaf_id: &str) -> Result<Option<PathBuf>> {
        let path = self.get_branch(Some(leaf_id));
        if path.is_empty() {
            return Err(anyhow!("Entry {} not found", leaf_id));
        }

        let path_without_labels: Vec<Value> = path
            .iter()
            .filter(|e| entry_type(e) != "label")
            .cloned()
            .collect();

        let new_session_id = create_session_id();
        let timestamp = now_iso();
        let file_ts = timestamp.replace([':', '.'], "-");

        let new_header = {
            let previous_file = self.session_file.clone();
            let parent_session = if self.persist {
                previous_file
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
            } else {
                None
            };
            let mut h = serde_json::json!({
                "type": "session",
                "version": CURRENT_SESSION_VERSION,
                "id": new_session_id,
                "timestamp": timestamp,
                "cwd": self.cwd,
            });
            if let Some(ps) = parent_session {
                h["parentSession"] = Value::String(ps);
            }
            h
        };

        // Collect labels for entries in the path
        let path_ids: std::collections::HashSet<String> =
            path_without_labels.iter().map(|e| entry_id(e).to_string()).collect();

        let labels_to_write: Vec<(String, String, String)> = self
            .labels_by_id
            .iter()
            .filter(|(target_id, _)| path_ids.contains(*target_id))
            .map(|(target_id, label)| {
                let ts = self
                    .label_timestamps_by_id
                    .get(target_id)
                    .cloned()
                    .unwrap_or_else(now_iso);
                (target_id.clone(), label.clone(), ts)
            })
            .collect();

        if self.persist {
            let new_file = self.session_dir.join(format!("{}_{}.jsonl", file_ts, new_session_id));

            // Build label entries
            let last_entry_id = path_without_labels
                .last()
                .map(|e| entry_id(e).to_string());
            let mut parent_id_for_labels = last_entry_id;
            let mut label_entries: Vec<Value> = vec![];
            let mut all_ids = path_ids.clone();

            for (target_id, label, lts) in &labels_to_write {
                let lid = generate_id(|id| all_ids.contains(id));
                all_ids.insert(lid.clone());
                let le = serde_json::json!({
                    "type": "label",
                    "id": lid,
                    "parentId": parent_id_for_labels,
                    "timestamp": lts,
                    "targetId": target_id,
                    "label": label,
                });
                parent_id_for_labels = Some(entry_id(&le).to_string());
                label_entries.push(le);
            }

            let new_entries: Vec<Value> = std::iter::once(new_header.clone())
                .chain(path_without_labels.iter().cloned())
                .chain(label_entries.iter().cloned())
                .collect();

            self.file_entries = new_entries;
            self.session_id = new_session_id;
            self.session_file = Some(new_file.clone());
            self.build_index();

            let has_assistant = self.has_assistant();
            if has_assistant {
                self.rewrite_file();
                self.flushed = true;
            } else {
                self.flushed = false;
            }

            return Ok(Some(new_file));
        }

        // In-memory mode
        let mut all_ids = path_ids.clone();
        let mut label_entries: Vec<Value> = vec![];
        let mut parent_id_for_labels = path_without_labels
            .last()
            .map(|e| entry_id(e).to_string());

        for (target_id, label, lts) in &labels_to_write {
            let lid = generate_id(|id| {
                all_ids.contains(id) || label_entries.iter().any(|e| entry_id(e) == id)
            });
            all_ids.insert(lid.clone());
            let le = serde_json::json!({
                "type": "label",
                "id": lid,
                "parentId": parent_id_for_labels,
                "timestamp": lts,
                "targetId": target_id,
                "label": label,
            });
            parent_id_for_labels = Some(entry_id(&le).to_string());
            label_entries.push(le);
        }

        self.file_entries = std::iter::once(new_header)
            .chain(path_without_labels.iter().cloned())
            .chain(label_entries.iter().cloned())
            .collect();
        self.session_id = new_session_id;
        self.build_index();

        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Session listing (async)
// ---------------------------------------------------------------------------

async fn build_session_info(file_path: impl AsRef<Path>) -> Option<SessionInfo> {
    let path = file_path.as_ref();
    let content = tokio::fs::read_to_string(path).await.ok()?;
    let entries: Vec<Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if entries.is_empty() {
        return None;
    }
    let header = entries.first()?;
    if entry_type(header) != "session" {
        return None;
    }

    let meta = tokio::fs::metadata(path).await.ok()?;
    let stats_mtime: chrono::DateTime<chrono::Utc> = meta
        .modified()
        .ok()
        .map(|t| t.into())
        .unwrap_or_else(Utc::now);

    let mut message_count = 0usize;
    let mut first_message = String::new();
    let mut all_messages: Vec<String> = vec![];
    let mut name: Option<String> = None;
    let mut last_activity: Option<u64> = None;

    for entry in &entries {
        if entry_type(entry) == "session_info" {
            let n = entry.get("name").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            name = if n.is_empty() { None } else { Some(n) };
        }

        if entry_type(entry) != "message" {
            continue;
        }
        message_count += 1;
        let msg = match entry.get("message") {
            Some(m) => m,
            None => continue,
        };
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role != "user" && role != "assistant" {
            continue;
        }

        // Track last activity time
        if let Some(ts) = msg.get("timestamp").and_then(|v| v.as_u64()) {
            last_activity = Some(last_activity.unwrap_or(0).max(ts));
        }

        let text = extract_text_content(msg);
        if text.is_empty() {
            continue;
        }
        all_messages.push(text.clone());
        if first_message.is_empty() && role == "user" {
            first_message = text;
        }
    }

    let cwd = header.get("cwd").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let parent_session_path = header
        .get("parentSession")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let created = header
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or(stats_mtime);

    let modified = if let Some(ts) = last_activity {
        chrono::DateTime::from_timestamp_millis(ts as i64)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or(stats_mtime)
    } else {
        stats_mtime
    };

    Some(SessionInfo {
        path: path.to_string_lossy().to_string(),
        id: header.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        cwd,
        name,
        parent_session_path,
        created,
        modified,
        message_count,
        first_message: if first_message.is_empty() {
            "(no messages)".to_string()
        } else {
            first_message
        },
        all_messages_text: all_messages.join(" "),
    })
}

fn extract_text_content(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

pub async fn list_sessions(cwd: &str, session_dir: Option<&Path>) -> Result<Vec<SessionInfo>> {
    let dir = session_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| get_default_session_dir(cwd));
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut sessions = vec![];
    let mut rd = tokio::fs::read_dir(&dir).await?;
    let mut files = vec![];
    while let Some(entry) = rd.next_entry().await? {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            files.push(p);
        }
    }
    let results = futures::future::join_all(files.iter().map(|f| build_session_info(f))).await;
    for info in results.into_iter().flatten() {
        sessions.push(info);
    }
    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(sessions)
}
