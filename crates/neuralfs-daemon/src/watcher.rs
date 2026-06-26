use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use notify::{Event, EventKind, RecursiveMode, Watcher};
use tokio::sync::mpsc::UnboundedSender;

use crate::indexer::Indexer;
use crate::store::Store;

/// Spawns a background thread that watches `roots` for filesystem changes and
/// keeps the index up to date. Returns the underlying watcher; it must be kept
/// alive (not dropped) for the duration of the daemon's lifetime.
///
/// Note: Windows' `ReadDirectoryChangesW` (which `notify` wraps) has no "file
/// opened" event, so open-frequency tracking is driven by the CLI's `open`
/// command via IPC rather than passive FS watching.
pub fn spawn_watcher(
    store: Arc<Store>,
    roots: Vec<String>,
    retrain_threshold: usize,
    retrain_tx: UnboundedSender<()>,
) -> Result<notify::RecommendedWatcher> {
    let (tx, rx) = std::sync::mpsc::channel::<notify::Result<Event>>();

    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;

    for root in &roots {
        if let Err(e) = watcher.watch(Path::new(root), RecursiveMode::Recursive) {
            log::error!("failed to watch root {root}: {e}");
        } else {
            log::info!("watching {root}");
        }
    }

    let event_count = Arc::new(AtomicUsize::new(0));
    std::thread::spawn(move || {
        let indexer = Indexer::new(&store);
        for res in rx {
            match res {
                Ok(event) => {
                    handle_event(&indexer, &event);
                    let prev = event_count.fetch_add(1, Ordering::SeqCst);
                    if prev + 1 >= retrain_threshold {
                        event_count.store(0, Ordering::SeqCst);
                        if retrain_tx.send(()).is_err() {
                            log::warn!("retrain channel closed; daemon shutting down?");
                        }
                    }
                }
                Err(e) => log::warn!("watch error: {e}"),
            }
        }
    });

    Ok(watcher)
}

fn handle_event(indexer: &Indexer, event: &Event) {
    match event.kind {
        EventKind::Create(_) | EventKind::Modify(_) => {
            for path in &event.paths {
                indexer.upsert_path(path);
            }
        }
        EventKind::Remove(_) => {
            for path in &event.paths {
                indexer.remove_path(path);
            }
        }
        _ => {}
    }
}
