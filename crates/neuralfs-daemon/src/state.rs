use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use neuralfs_fs::Filesystem;
use notify::RecommendedWatcher;
use parking_lot::Mutex;
use tokio::sync::RwLock;

use crate::classifier::Classifier;
use crate::config::Config;
use crate::store::Store;

pub struct DaemonState {
    pub store: Arc<Store>,
    /// The NeuralFS copy-on-write virtual filesystem volume.
    pub vfs: Arc<Filesystem>,
    pub classifier: RwLock<Classifier>,
    pub config: RwLock<Config>,
    pub config_path: PathBuf,
    pub last_retrain: RwLock<Option<String>>,
    /// Live FS watcher, so `hook` can start watching new roots at runtime.
    pub watcher: Mutex<Option<RecommendedWatcher>>,
    /// Model version most recently persisted to disk by the checkpoint loop.
    pub saved_version: AtomicU64,
}

impl DaemonState {
    pub fn new(
        store: Arc<Store>,
        vfs: Arc<Filesystem>,
        config: Config,
        config_path: PathBuf,
    ) -> Self {
        Self {
            store,
            vfs,
            classifier: RwLock::new(Classifier::default()),
            config: RwLock::new(config),
            config_path,
            last_retrain: RwLock::new(None),
            watcher: Mutex::new(None),
            saved_version: AtomicU64::new(0),
        }
    }
}
