use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

use crate::indexer::Indexer;
use crate::protocol::{Request, Response};
use crate::search;
use crate::state::DaemonState;

#[cfg(windows)]
pub const PIPE_NAME: &str = r"\\.\pipe\neuralfs";

#[cfg(unix)]
pub const SOCKET_PATH: &str = "/tmp/neuralfs.sock";

/// Runs the daemon's IPC server forever, accepting one client connection at a
/// time and spawning an async task per connection so a slow client can never
/// block another, or block the classifier retrain loop.
#[cfg(windows)]
pub async fn run_server(state: Arc<DaemonState>, retrain_tx: UnboundedSender<()>) -> Result<()> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let mut server = ServerOptions::new().create(PIPE_NAME)?;
    log::info!("ipc server listening on {PIPE_NAME}");

    loop {
        server.connect().await?;
        let connected = server;
        server = ServerOptions::new().create(PIPE_NAME)?;

        let state = state.clone();
        let retrain_tx = retrain_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(connected, state, retrain_tx).await {
                log::error!("ipc client error: {e}");
            }
        });
    }
}

#[cfg(unix)]
pub async fn run_server(state: Arc<DaemonState>, retrain_tx: UnboundedSender<()>) -> Result<()> {
    use tokio::net::UnixListener;

    let _ = std::fs::remove_file(SOCKET_PATH);
    let listener = UnixListener::bind(SOCKET_PATH)?;
    log::info!("ipc server listening on {SOCKET_PATH}");

    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        let retrain_tx = retrain_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state, retrain_tx).await {
                log::error!("ipc client error: {e}");
            }
        });
    }
}

async fn handle_connection<S>(
    stream: S,
    state: Arc<DaemonState>,
    retrain_tx: UnboundedSender<()>,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => handle_request(req, &state, &retrain_tx).await,
            Err(e) => Response::error(format!("bad request: {e}")),
        };
        let mut out = serde_json::to_string(&response)?;
        out.push('\n');
        writer.write_all(out.as_bytes()).await?;
    }
    Ok(())
}

async fn handle_request(
    req: Request,
    state: &Arc<DaemonState>,
    retrain_tx: &UnboundedSender<()>,
) -> Response {
    match req {
        Request::Find { query } => {
            let cfg = state.config.read().await;
            let clf = state.classifier.read().await;
            let results = search::find(state.store.as_ref(), &clf, &query, cfg.lambda, cfg.max_results);
            Response {
                results: Some(results),
                ..Default::default()
            }
        }
        Request::Open { query } => {
            let (lambda,) = {
                let cfg = state.config.read().await;
                (cfg.lambda,)
            };
            let clf = state.classifier.read().await;
            let results = search::find(state.store.as_ref(), &clf, &query, lambda, 1);
            match results.into_iter().next() {
                Some(top) => {
                    let now = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64;
                    let _ = state.store.touch_open(&top.path, now);
                    Response {
                        opened_path: Some(top.path),
                        ..Default::default()
                    }
                }
                None => Response::error("no match found"),
            }
        }
        Request::Status => {
            let last_retrain = state.last_retrain.read().await.clone();
            Response {
                status: Some("running".to_string()),
                indexed_files: Some(state.store.len()),
                last_retrain,
                ..Default::default()
            }
        }
        Request::Reindex => {
            let roots = state.config.read().await.root_dirs.clone();
            let store = state.store.clone();
            let count = tokio::task::spawn_blocking(move || {
                let indexer = Indexer::new(&store);
                indexer.reindex_all(&roots).unwrap_or(0)
            })
            .await
            .unwrap_or(0);
            let _ = retrain_tx.send(());
            Response::ok(format!("reindexed {count} files"))
        }
        Request::ConfigGet { key } => {
            let cfg = state.config.read().await;
            match cfg.get(&key) {
                Some(v) => Response::ok(v),
                None => Response::error(format!("unknown key: {key}")),
            }
        }
        Request::ConfigSet { key, value } => {
            let mut cfg = state.config.write().await;
            match cfg.set(&key, &value) {
                Ok(()) => {
                    let _ = cfg.save(&state.config_path);
                    Response::ok(format!("{key} = {value}"))
                }
                Err(e) => Response::error(e.to_string()),
            }
        }
    }
}
