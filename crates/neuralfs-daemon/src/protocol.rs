use serde::{Deserialize, Serialize};

/// CLI -> Daemon IPC requests, sent as a single line of JSON over the named pipe.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "lowercase")]
pub enum Request {
    Find { query: String },
    Open { query: String },
    Status,
    Reindex,
    ConfigGet { key: String },
    ConfigSet { key: String, value: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPath {
    pub path: String,
    pub score: f64,
}

/// Daemon -> CLI IPC response, sent as a single line of JSON over the named pipe.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Response {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<ScoredPath>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_files: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_retrain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opened_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    pub fn error(msg: impl Into<String>) -> Self {
        Response {
            error: Some(msg.into()),
            ..Default::default()
        }
    }

    pub fn ok(msg: impl Into<String>) -> Self {
        Response {
            ok: Some(true),
            message: Some(msg.into()),
            ..Default::default()
        }
    }
}
