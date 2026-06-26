mod classifier;
mod config;
mod indexer;
mod install;
mod ipc;
mod logging;
mod protocol;
mod scorer;
mod search;
mod state;
mod store;
mod watcher;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
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
    let state = Arc::new(DaemonState::new(store.clone(), config.clone(), config_path));

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
        *state.classifier.write().await = clf;
        *state.last_retrain.write().await = Some(chrono::Local::now().to_rfc2822());
    }

    let (retrain_tx, retrain_rx) = tokio::sync::mpsc::unbounded_channel::<()>();

    let _watcher_guard = watcher::spawn_watcher(
        store.clone(),
        config.root_dirs.clone(),
        config.retrain_threshold,
        retrain_tx.clone(),
    )?;

    tokio::spawn(retrain_loop(state.clone(), retrain_rx));
    tokio::spawn(periodic_flush(store.clone()));

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
                *state.classifier.write().await = clf;
                *state.last_retrain.write().await = Some(chrono::Local::now().to_rfc2822());
                log::info!("classifier retrained");
            }
            Ok(Err(e)) => log::error!("classifier retrain failed: {e}"),
            Err(e) => log::error!("classifier retrain task panicked: {e}"),
        }
    }
}

async fn periodic_flush(store: Arc<Store>) {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        if let Err(e) = store.flush() {
            log::error!("periodic flush failed: {e}");
        }
    }
}
