use std::path::Path;
use std::time::SystemTime;

use anyhow::Result;
use walkdir::WalkDir;

use crate::store::{FileEntry, Store};

/// Walks one or more root directories and (re)populates the index.
pub struct Indexer<'a> {
    store: &'a Store,
}

impl<'a> Indexer<'a> {
    pub fn new(store: &'a Store) -> Self {
        Self { store }
    }

    /// Full re-index of all configured roots. Returns number of files indexed.
    pub fn reindex_all(&self, roots: &[String]) -> Result<usize> {
        let mut count = 0;
        for root in roots {
            count += self.index_root(Path::new(root));
        }
        let _ = self.store.flush();
        Ok(count)
    }

    /// Walk a single root directory, indexing every file found.
    /// Per-entry errors (permission denied, broken symlinks, etc.) are logged and skipped.
    pub fn index_root(&self, root: &Path) -> usize {
        let mut count = 0;
        for dent in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| match e {
                Ok(e) => Some(e),
                Err(err) => {
                    log::warn!("walk error: {err}");
                    None
                }
            })
        {
            if !dent.file_type().is_file() {
                continue;
            }
            match build_entry(dent.path(), dent.depth()) {
                Ok(entry) => {
                    if let Err(e) = self.store.upsert_preserving_stats(entry) {
                        log::error!("failed to upsert {}: {e}", dent.path().display());
                    } else {
                        count += 1;
                    }
                }
                Err(e) => log::warn!("failed to stat {}: {e}", dent.path().display()),
            }
        }
        count
    }

    pub fn upsert_path(&self, path: &Path) {
        if !path.is_file() {
            return;
        }
        let depth = path.components().count();
        match build_entry(path, depth) {
            Ok(entry) => {
                if let Err(e) = self.store.upsert_preserving_stats(entry) {
                    log::error!("failed to upsert {}: {e}", path.display());
                }
            }
            Err(e) => log::warn!("failed to stat {}: {e}", path.display()),
        }
    }

    pub fn remove_path(&self, path: &Path) {
        if let Err(e) = self.store.remove(&path.to_string_lossy()) {
            log::error!("failed to remove {}: {e}", path.display());
        }
    }
}

fn build_entry(path: &Path, depth: usize) -> Result<FileEntry> {
    let meta = path.metadata()?;
    let modified = meta
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let extension = path
        .extension()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let parent = path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    Ok(FileEntry {
        path: path.to_string_lossy().to_string(),
        file_name,
        extension,
        parent,
        size: meta.len(),
        modified,
        depth,
        freq: 0,
        last_open: 0,
    })
}
