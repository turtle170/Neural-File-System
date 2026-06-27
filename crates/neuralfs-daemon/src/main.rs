mod classifier;
mod config;
mod indexer;
mod install;
mod ipc;
mod logging;
#[cfg(all(target_os = "linux", feature = "fuse"))]
mod mountfs;
mod protocol;
mod scorer;
mod search;
mod state;
mod store;
mod watcher;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use neuralfs_fs::Filesystem;
use tokio::sync::mpsc::UnboundedReceiver;

use classifier::Classifier;
use config::Config;
use state::DaemonState;
use store::Store;

fn data_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("neuralfs")
}

/// Extract the value following `--flag` in the argument list.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

/// Run the FUSE user-mode filesystem hook (Linux/WSL, `--features fuse`).
fn run_mount_mode(args: &[String], mountpoint: &str) {
    #[cfg(all(target_os = "linux", feature = "fuse"))]
    {
        let backing = flag_value(args, "--backing").unwrap_or_else(|| {
            data_dir().join("backing").to_string_lossy().to_string()
        });
        let cache_bytes = flag_value(args, "--cache-mb")
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(neuralfs_fs::DEFAULT_CACHE_BYTES);

        // Minimal stderr logging for the foreground mount process.
        let _ = logging::init(
            &data_dir().join("neuralfs-mount.log"),
            log::LevelFilter::Info,
        );
        eprintln!(
            "NeuralFS hook: mounting at {mountpoint} (backing {backing}, cache {} MiB)",
            cache_bytes / (1024 * 1024)
        );
        if let Err(e) = mountfs::run_mount(
            std::path::Path::new(mountpoint),
            std::path::Path::new(&backing),
            cache_bytes,
        ) {
            eprintln!("mount failed: {e:#}");
            std::process::exit(1);
        }
    }
    #[cfg(not(all(target_os = "linux", feature = "fuse")))]
    {
        let _ = (args, mountpoint);
        eprintln!(
            "--mount requires a Linux build with the `fuse` feature: \
             cargo build --release --features fuse"
        );
        std::process::exit(1);
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--install") {
        if let Err(e) = install::install() {
            eprintln!("install failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--uninstall") {
        if let Err(e) = install::uninstall() {
            eprintln!("uninstall failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    // User-mode filesystem hook: `neuralfs --mount <mountpoint> [--backing <dir>]`.
    if let Some(mountpoint) = flag_value(&args, "--mount") {
        run_mount_mode(&args, &mountpoint);
        return;
    }

    if let Err(e) = run().await {
        log::error!("fatal startup error: {e:#}");
        eprintln!("neuralfs failed to start: {e:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let dir = data_dir();
    std::fs::create_dir_all(&dir)?;

    let config_path = dir.join("config.toml");
    let config = Config::load_or_create(&config_path)?;

    logging::init(&dir.join("neuralfs.log"), logging::level_from_str(&config.log_level))?;
    log::info!("neuralfs daemon starting, data dir = {}", dir.display());

    let store = Arc::new(Store::open(&dir.join("index.db"))?);
    let vfs = Arc::new(Filesystem::open(&dir.join("volume"))?);
    log::info!("opened virtual filesystem volume at {}", dir.join("volume").display());
    let state = Arc::new(DaemonState::new(
        store.clone(),
        vfs,
        config.clone(),
        config_path,
    ));

    // Initial full re-index on startup.
    {
        let store = store.clone();
        let roots = config.root_dirs.clone();
        let count = tokio::task::spawn_blocking(move || {
            let idx = indexer::Indexer::new(&store);
            idx.reindex_all(&roots).unwrap_or(0)
        })
        .await
        .unwrap_or(0);
        log::info!("initial index complete: {count} files");
    }

    // Load a persisted model if present, otherwise train from scratch.
    {
        let loaded = store
            .load_model()
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<Classifier>(&bytes).ok())
            .filter(|c| c.is_trained());

        let clf = match loaded {
            Some(c) => {
                log::info!("loaded persisted classifier model");
                c
            }
            None => {
                let store_for_train = store.clone();
                let trained = tokio::task::spawn_blocking(move || {
                    let entries = store_for_train.all_entries().unwrap_or_default();
                    Classifier::train(&entries)
                })
                .await
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or_default();
                if let Ok(bytes) = bincode::serialize(&trained) {
                    let _ = store.save_model(&bytes);
                }
                log::info!("trained initial classifier model");
                trained
            }
        };
        state.saved_version.store(clf.version(), Ordering::SeqCst);
        *state.classifier.write().await = clf;
        *state.last_retrain.write().await = Some(chrono::Local::now().to_rfc2822());
    }

    let (retrain_tx, retrain_rx) = tokio::sync::mpsc::unbounded_channel::<()>();

    let watcher = watcher::spawn_watcher(
        store.clone(),
        config.root_dirs.clone(),
        config.retrain_threshold,
        retrain_tx.clone(),
    )?;
    *state.watcher.lock() = Some(watcher);

    tokio::spawn(retrain_loop(state.clone(), retrain_rx));
    tokio::spawn(periodic_flush(store.clone(), state.vfs.clone()));
    tokio::spawn(ai_checkpoint_loop(state.clone()));

    ipc::run_server(state.clone(), retrain_tx).await?;
    Ok(())
}

async fn retrain_loop(state: Arc<DaemonState>, mut rx: UnboundedReceiver<()>) {
    while rx.recv().await.is_some() {
        let store = state.store.clone();
        let result = tokio::task::spawn_blocking(move || {
            let entries = store.all_entries()?;
            Classifier::train(&entries)
        })
        .await;

        match result {
            Ok(Ok(clf)) => {
                if let Ok(bytes) = bincode::serialize(&clf) {
                    if let Err(e) = state.store.save_model(&bytes) {
                        log::error!("failed to persist classifier model: {e}");
                    }
                }
                state.saved_version.store(clf.version(), Ordering::SeqCst);
                *state.classifier.write().await = clf;
                *state.last_retrain.write().await = Some(chrono::Local::now().to_rfc2822());
                log::info!("classifier retrained");
            }
            Ok(Err(e)) => log::error!("classifier retrain failed: {e}"),
            Err(e) => log::error!("classifier retrain task panicked: {e}"),
        }
    }
}

async fn periodic_flush(store: Arc<Store>, vfs: Arc<Filesystem>) {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        if let Err(e) = store.flush() {
            log::error!("periodic flush failed: {e}");
        }
        if let Err(e) = vfs.flush() {
            log::error!("vfs flush failed: {e}");
        }
    }
}

/// Persists the classifier whenever online learning has advanced its version
/// past what's on disk — so the AI is "stored and updated as long as it is
/// active," not just on full retrains.
async fn ai_checkpoint_loop(state: Arc<DaemonState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let current = state.classifier.read().await.version();
        if current == state.saved_version.load(Ordering::SeqCst) {
            continue;
        }
        let bytes = {
            let clf = state.classifier.read().await;
            bincode::serialize(&*clf)
        };
        match bytes {
            Ok(bytes) => {
                if let Err(e) = state.store.save_model(&bytes) {
                    log::error!("ai checkpoint failed: {e}");
                } else {
                    state.saved_version.store(current, Ordering::SeqCst);
                    log::info!("ai checkpoint saved (version {current})");
                }
            }
            Err(e) => log::error!("ai checkpoint serialize failed: {e}"),
        }
    }
}
