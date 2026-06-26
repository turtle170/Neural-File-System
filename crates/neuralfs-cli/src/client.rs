use std::time::Duration;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Mirrors neuralfs-daemon's wire protocol (crates/neuralfs-daemon/src/protocol.rs).
/// Kept as a separate, intentionally minimal copy since the CLI only needs to
/// speak the JSON-over-pipe protocol, not share daemon internals.
#[derive(Debug, Serialize)]
#[serde(tag = "cmd", rename_all = "lowercase")]
pub enum Request {
    Find { query: String },
    Open { query: String },
    Status,
    Reindex,
    ConfigGet { key: String },
    ConfigSet { key: String, value: String },
}

#[derive(Debug, Deserialize, Default)]
pub struct ScoredPath {
    pub path: String,
    pub score: f64,
}

#[derive(Debug, Deserialize, Default)]
pub struct Response {
    pub results: Option<Vec<ScoredPath>>,
    pub status: Option<String>,
    pub indexed_files: Option<usize>,
    pub last_retrain: Option<String>,
    #[allow(dead_code)]
    pub ok: Option<bool>,
    pub message: Option<String>,
    pub opened_path: Option<String>,
    pub error: Option<String>,
}

#[cfg(windows)]
const PIPE_NAME: &str = r"\\.\pipe\neuralfs";
#[cfg(unix)]
const SOCKET_PATH: &str = "/tmp/neuralfs.sock";

pub async fn send(req: Request) -> Result<Response> {
    #[cfg(windows)]
    let stream = connect_windows().await?;
    #[cfg(unix)]
    let stream = connect_unix().await?;

    let (reader, mut writer) = tokio::io::split(stream);
    let mut line = serde_json::to_string(&req)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;

    let mut lines = BufReader::new(reader).lines();
    match tokio::time::timeout(Duration::from_secs(10), lines.next_line()).await {
        Ok(Ok(Some(resp_line))) => Ok(serde_json::from_str(&resp_line)?),
        Ok(Ok(None)) => bail!("daemon closed the connection without responding"),
        Ok(Err(e)) => bail!("failed reading daemon response: {e}"),
        Err(_) => bail!("timed out waiting for daemon response"),
    }
}

#[cfg(windows)]
async fn connect_windows() -> Result<tokio::net::windows::named_pipe::NamedPipeClient> {
    use tokio::net::windows::named_pipe::ClientOptions;

    let mut last_err = None;
    for attempt in 0..5 {
        match ClientOptions::new().open(PIPE_NAME) {
            Ok(client) => return Ok(client),
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(50 * (attempt + 1))).await;
            }
        }
    }
    bail!(
        "could not connect to neuralfs daemon on {PIPE_NAME} (is it running?): {}",
        last_err.unwrap()
    )
}

#[cfg(unix)]
async fn connect_unix() -> Result<tokio::net::UnixStream> {
    tokio::net::UnixStream::connect(SOCKET_PATH)
        .await
        .map_err(|e| anyhow::anyhow!("could not connect to neuralfs daemon on {SOCKET_PATH} (is it running?): {e}"))
}
