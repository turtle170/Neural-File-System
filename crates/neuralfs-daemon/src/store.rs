use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A single indexed file's metadata, frequency, and recency state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub file_name: String,
    pub extension: String,
    pub parent: String,
    pub size: u64,
    pub modified: i64,
    pub depth: usize,
    pub freq: u64,
    pub last_open: i64,
}

impl FileEntry {
    pub fn key(&self) -> u64 {
        hash_path(&self.path)
    }
}

pub fn hash_path(path: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

fn key_bytes(hash: u64) -> [u8; 8] {
    hash.to_be_bytes()
}

/// Persists the file index and trained classifier model in an embedded sled DB.
pub struct Store {
    db: sled::Db,
    index: sled::Tree,
    by_parent: sled::Tree,
    model: sled::Tree,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let db = sled::Config::new()
            .path(path)
            .cache_capacity(16 * 1024 * 1024)
            .open()
            .with_context(|| format!("opening sled db at {}", path.display()))?;
        let index = db.open_tree("index")?;
        let by_parent = db.open_tree("by_parent")?;
        let model = db.open_tree("model")?;
        Ok(Self {
            db,
            index,
            by_parent,
            model,
        })
    }

    pub fn upsert(&self, entry: &FileEntry) -> Result<()> {
        let hash = entry.key();
        let kb = key_bytes(hash);

        if let Some(old) = self.index.get(kb)? {
            let old: FileEntry = bincode::deserialize(&old)?;
            if old.parent != entry.parent {
                self.remove_parent_link(&old.parent, hash)?;
            }
        }

        let bytes = bincode::serialize(entry)?;
        self.index.insert(kb, bytes)?;
        self.add_parent_link(&entry.parent, hash)?;
        Ok(())
    }

    /// Upsert but preserve existing freq/last_open if the entry already exists.
    pub fn upsert_preserving_stats(&self, mut entry: FileEntry) -> Result<()> {
        let kb = key_bytes(entry.key());
        if let Some(old) = self.index.get(kb)? {
            let old: FileEntry = bincode::deserialize(&old)?;
            entry.freq = old.freq;
            entry.last_open = old.last_open;
        }
        self.upsert(&entry)
    }

    pub fn remove(&self, path: &str) -> Result<()> {
        let hash = hash_path(path);
        let kb = key_bytes(hash);
        if let Some(old) = self.index.remove(kb)? {
            let old: FileEntry = bincode::deserialize(&old)?;
            self.remove_parent_link(&old.parent, hash)?;
        }
        Ok(())
    }

    pub fn touch_open(&self, path: &str, now: i64) -> Result<Option<FileEntry>> {
        let kb = key_bytes(hash_path(path));
        if let Some(bytes) = self.index.get(kb)? {
            let mut entry: FileEntry = bincode::deserialize(&bytes)?;
            entry.freq += 1;
            entry.last_open = now;
            self.index.insert(kb, bincode::serialize(&entry)?)?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    pub fn files_in(&self, parent: &str) -> Result<Vec<FileEntry>> {
        let prefix = parent_prefix(parent);
        let mut out = Vec::new();
        for kv in self.by_parent.scan_prefix(&prefix) {
            let (_, hash_bytes) = kv?;
            if let Some(bytes) = self.index.get(&hash_bytes)? {
                out.push(bincode::deserialize(&bytes)?);
            }
        }
        Ok(out)
    }

    pub fn all_entries(&self) -> Result<Vec<FileEntry>> {
        let mut out = Vec::with_capacity(self.index.len());
        for kv in self.index.iter() {
            let (_, bytes) = kv?;
            out.push(bincode::deserialize(&bytes)?);
        }
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn save_model(&self, bytes: &[u8]) -> Result<()> {
        self.model.insert("classifier", bytes)?;
        Ok(())
    }

    pub fn load_model(&self) -> Result<Option<Vec<u8>>> {
        Ok(self.model.get("classifier")?.map(|v| v.to_vec()))
    }

    pub fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }

    fn add_parent_link(&self, parent: &str, hash: u64) -> Result<()> {
        let mut key = parent_prefix(parent);
        key.extend_from_slice(&key_bytes(hash));
        self.by_parent.insert(key, key_bytes(hash).to_vec())?;
        Ok(())
    }

    fn remove_parent_link(&self, parent: &str, hash: u64) -> Result<()> {
        let mut key = parent_prefix(parent);
        key.extend_from_slice(&key_bytes(hash));
        self.by_parent.remove(key)?;
        Ok(())
    }
}

fn parent_prefix(parent: &str) -> Vec<u8> {
    let mut v = parent.to_lowercase().into_bytes();
    v.push(0u8);
    v
}
