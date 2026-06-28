use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::blockstore::Hash;

/// A filesystem object. Inodes are immutable once written: every modification
/// allocates a new inode id (copy-on-write), which is what makes snapshots O(1)
/// and crash-consistent — old roots keep pointing at old, untouched inodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Inode {
    File {
        size: u64,
        /// Content-addressed blocks (empty when the file is stored inline).
        blocks: Vec<Hash>,
        /// Small-file fast path: bytes for files at or below the inline
        /// threshold live directly in the inode, skipping the block store
        /// (no blake3 hashing, no blocks tree, no data.log append) entirely.
        /// Mutually exclusive with `blocks` being non-empty.
        #[serde(default)]
        inline: Vec<u8>,
        mtime: i64,
        mode: u32,
    },
    Dir {
        entries: BTreeMap<String, u64>,
        mtime: i64,
    },
}

impl Inode {
    pub fn is_dir(&self) -> bool {
        matches!(self, Inode::Dir { .. })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Inode::File { .. } => "file",
            Inode::Dir { .. } => "dir",
        }
    }
}

/// The filesystem's root pointer. Advancing this single value atomically
/// publishes a new consistent view of the entire tree (ZFS-style uberblock).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SuperBlock {
    pub root: u64,
    pub next_inode: u64,
    pub txg: u64,
}

/// Metadata returned by `stat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stat {
    pub kind: String,
    pub size: u64,
    pub mtime: i64,
    pub blocks: usize,
    pub entries: usize,
}
