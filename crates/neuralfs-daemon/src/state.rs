use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::classifier::Classifier;
use crate::config::Config;
use crate::store::Store;

pub struct DaemonState {
    pub store: Arc<Store>,
    pub classifier: RwLock<Classifier>,
    pub config: RwLock<Config>,
    pub config_path: PathBuf,
    pub last_retrain: RwLock<Option<String>>,
}

impl DaemonState {
    pub fn new(store: Arc<Store>, config: Config, config_path: PathBuf) -> Self {
        Self {
            store,
            classifier: RwLock::new(Classifier::default()),
            config: RwLock::new(config),
            config_path,
            last_retrain: RwLock::new(None),
        }
    }
}
