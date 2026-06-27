use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use anyhow::Result;
use base64::Engine;
use notify::{RecursiveMode, Watcher};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

use crate::indexer::Indexer;
use crate::protocol::{Request, Response};
use crate::search;
use crate::state::DaemonState;

/// Learning rate for online (per-access) classifier updates.
const ONLINE_LR: f64 = 0.3;
/// Cap on binary payloads moved over the JSON-line IPC (fs read/write).
const MAX_IPC_BYTES: usize = 16 * 1024 * 1024;

fn b64() -> base64::engine::general_purpose::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

fn lines(v: Vec<String>) -> Response {
    Response {
        lines: Some(v),
        ..Default::default()
    }
}

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
            let lambda = state.config.read().await.lambda;
            let top = {
                let clf = state.classifier.read().await;
                search::find(state.store.as_ref(), &clf, &query, lambda, 1)
                    .into_iter()
                    .next()
            };
            match top {
                Some(top) => {
                    let now = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64;
                    let _ = state.store.touch_open(&top.path, now);

                    // Continuous learning: nudge the model toward predicting
                    // this file's directory for this query, right now.
                    if let Some(parent) = Path::new(&top.path).parent() {
                        let parent = parent.to_string_lossy().to_string();
                        let mut clf = state.classifier.write().await;
                        clf.online_update(&query, &parent, ONLINE_LR);
                    }

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

        // ---- virtual filesystem -----------------------------------------
        Request::FsWrite { path, data_b64 } => {
            let bytes = match b64().decode(data_b64.as_bytes()) {
                Ok(b) if b.len() <= MAX_IPC_BYTES => b,
                Ok(_) => return Response::error("payload exceeds 16 MiB IPC limit"),
                Err(e) => return Response::error(format!("bad base64: {e}")),
            };
            let vfs = state.vfs.clone();
            let n = bytes.len();
            let res = tokio::task::spawn_blocking(move || vfs.write_file(&path, &bytes)).await;
            match res {
                Ok(Ok(())) => Response::ok(format!("wrote {n} bytes")),
                Ok(Err(e)) => Response::error(e.to_string()),
                Err(e) => Response::error(format!("task error: {e}")),
            }
        }
        Request::FsRead { path } => {
            let vfs = state.vfs.clone();
            let res = tokio::task::spawn_blocking(move || vfs.read_file(&path)).await;
            match res {
                Ok(Ok(bytes)) => {
                    if bytes.len() > MAX_IPC_BYTES {
                        return Response::error("file exceeds 16 MiB IPC limit; too large to read over IPC");
                    }
                    Response {
                        data_b64: Some(b64().encode(&bytes)),
                        ..Default::default()
                    }
                }
                Ok(Err(e)) => Response::error(e.to_string()),
                Err(e) => Response::error(format!("task error: {e}")),
            }
        }
        Request::FsLs { path } => match state.vfs.readdir(&path) {
            Ok(entries) => {
                if entries.is_empty() {
                    return lines(vec!["(empty)".into()]);
                }
                lines(
                    entries
                        .into_iter()
                        .map(|(name, kind)| format!("{kind:<4}  {name}"))
                        .collect(),
                )
            }
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsMkdir { path } => match state.vfs.mkdir(&path) {
            Ok(()) => Response::ok(format!("created directory {path}")),
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsRm { path } => match state.vfs.remove(&path) {
            Ok(()) => Response::ok(format!("removed {path}")),
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsStat { path } => match state.vfs.stat(&path) {
            Ok(s) => lines(vec![
                format!("kind:    {}", s.kind),
                format!("size:    {} bytes", s.size),
                format!("blocks:  {}", s.blocks),
                format!("entries: {}", s.entries),
                format!("mtime:   {}", s.mtime),
            ]),
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsInfo => {
            let vfs = state.vfs.clone();
            match tokio::task::spawn_blocking(move || vfs.info()).await {
                Ok(Ok(i)) => lines(vec![
                    format!("transaction (txg):   {}", i.txg),
                    format!("files:               {}", i.files),
                    format!("directories:         {}", i.dirs),
                    format!("unique blocks:       {}", i.unique_blocks),
                    format!("logical bytes:       {}", i.logical_bytes),
                    format!("physical (live):     {}", i.physical_referenced),
                    format!("physical (total):    {}", i.physical_total),
                    format!("reclaimable bytes:   {}", i.reclaimable_bytes),
                    format!("dedup ratio:         {:.2}x", i.dedup_ratio),
                ]),
                Ok(Err(e)) => Response::error(e.to_string()),
                Err(e) => Response::error(format!("task error: {e}")),
            }
        }
        Request::FsSnapshot { name } => match state.vfs.snapshot(&name) {
            Ok(()) => Response::ok(format!("snapshot '{name}' created")),
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsSnapshots => match state.vfs.list_snapshots() {
            Ok(snaps) if snaps.is_empty() => lines(vec!["(no snapshots)".into()]),
            Ok(snaps) => lines(
                snaps
                    .into_iter()
                    .map(|(name, txg)| format!("{name}  (txg {txg})"))
                    .collect(),
            ),
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsRollback { name } => match state.vfs.rollback(&name) {
            Ok(()) => Response::ok(format!("rolled back to snapshot '{name}'")),
            Err(e) => Response::error(e.to_string()),
        },
        Request::FsScrub => {
            let vfs = state.vfs.clone();
            match tokio::task::spawn_blocking(move || vfs.scrub()).await {
                Ok(Ok((checked, bad))) => {
                    let mut out = vec![format!("scrubbed {checked} blocks")];
                    if bad.is_empty() {
                        out.push("no errors — all checksums valid".into());
                    } else {
                        out.push(format!("{} CORRUPT block(s):", bad.len()));
                        out.extend(bad);
                    }
                    lines(out)
                }
                Ok(Err(e)) => Response::error(e.to_string()),
                Err(e) => Response::error(format!("task error: {e}")),
            }
        }

        // ---- hook onto the real filesystem ------------------------------
        Request::Hook { dir } => handle_hook(state, retrain_tx, dir).await,
        Request::HookStatus => {
            let cfg = state.config.read().await;
            let mut out = vec![format!("indexed files: {}", state.store.len())];
            out.push("hooked directories:".into());
            for d in &cfg.root_dirs {
                out.push(format!("  {d}"));
            }
            lines(out)
        }

        // ---- AI model status --------------------------------------------
        Request::Ai => {
            let clf = state.classifier.read().await;
            let last_retrain = state.last_retrain.read().await.clone();
            lines(vec![
                format!("trained:        {}", clf.is_trained()),
                format!("model version:  {}", clf.version()),
                format!("online updates: {}", clf.online_updates()),
                format!("saved version:  {}", state.saved_version.load(Ordering::SeqCst)),
                format!("classes (dirs): {}", clf.num_classes()),
                format!("vocabulary:     {}", clf.vocab_size()),
                format!("last full train: {}", last_retrain.unwrap_or_else(|| "never".into())),
            ])
        }

        // ---- benchmark ---------------------------------------------------
        Request::Bench { mb } => {
            let vfs = state.vfs.clone();
            match tokio::task::spawn_blocking(move || run_bench(&vfs, mb)).await {
                Ok(Ok(out)) => lines(out),
                Ok(Err(e)) => Response::error(e.to_string()),
                Err(e) => Response::error(format!("task error: {e}")),
            }
        }
    }
}

async fn handle_hook(
    state: &Arc<DaemonState>,
    retrain_tx: &UnboundedSender<()>,
    dir: String,
) -> Response {
    let path = Path::new(&dir);
    if !path.is_dir() {
        return Response::error(format!("not a directory: {dir}"));
    }
    let canonical = {
        let c = path
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| dir.clone());
        // Strip the Windows extended-length (verbatim) prefix for readability.
        c.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(c)
    };

    // Record the new root in config (deduped) and persist.
    {
        let mut cfg = state.config.write().await;
        if !cfg.root_dirs.iter().any(|d| d == &canonical) {
            cfg.root_dirs.push(canonical.clone());
            let _ = cfg.save(&state.config_path);
        }
    }

    // Start watching it live, if we have a watcher.
    {
        let mut guard = state.watcher.lock();
        if let Some(w) = guard.as_mut() {
            if let Err(e) = w.watch(Path::new(&canonical), RecursiveMode::Recursive) {
                log::warn!("failed to watch hooked dir {canonical}: {e}");
            }
        }
    }

    // Index it now, then trigger a retrain so the AI immediately knows it.
    let store = state.store.clone();
    let to_index = canonical.clone();
    let count = tokio::task::spawn_blocking(move || {
        let indexer = Indexer::new(&store);
        indexer.index_root(Path::new(&to_index))
    })
    .await
    .unwrap_or(0);
    let _ = state.store.flush();
    let _ = retrain_tx.send(());

    Response::ok(format!(
        "hooked onto {canonical} — indexed {count} files, learning enabled"
    ))
}

/// In-process throughput benchmark over the virtual filesystem. Uses unique
/// per-block content so dedup doesn't mask real write cost.
fn run_bench(vfs: &neuralfs_fs::Filesystem, mb: usize) -> Result<Vec<String>> {
    let mb = mb.clamp(1, 1024);
    let total = mb * 1024 * 1024;
    let mut payload = vec![0u8; total];
    // Fill with a non-repeating pattern (Knuth multiplicative) so every 64 KiB
    // block is distinct and the block store cannot dedup it away.
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (((i as u64).wrapping_mul(2654435761)) >> 13) as u8;
    }

    let t0 = Instant::now();
    vfs.write_file("/_bench/data.bin", &payload)?;
    vfs.flush()?;
    let write_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let read_back = vfs.read_file("/_bench/data.bin")?;
    let read_s = t1.elapsed().as_secs_f64();

    let ok = read_back.len() == payload.len();
    let _ = vfs.remove("/_bench");

    let mbf = mb as f64;
    Ok(vec![
        format!("payload:        {mb} MiB ({total} bytes, unique blocks)"),
        format!("write:          {:.3} s  ->  {:.1} MB/s", write_s, mbf / write_s),
        format!("read+verify:    {:.3} s  ->  {:.1} MB/s", read_s, mbf / read_s),
        format!("integrity:      {}", if ok { "verified" } else { "MISMATCH" }),
    ])
}
