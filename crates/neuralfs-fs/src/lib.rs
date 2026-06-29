//! NeuralFS filesystem engine.
//!
//! A lightweight copy-on-write, content-addressed, checksummed object
//! filesystem — ZFS-like integrity and snapshots without the weight.

mod blockstore;
mod cache;
mod fs;
mod inode;

pub use blockstore::{hash_hex, Hash, BLOCK_SIZE};
pub use cache::{CacheStats, RamCache};
pub use fs::{Filesystem, FsInfo, GcReport, DEFAULT_CACHE_BYTES};
pub use inode::{Inode, Stat, SuperBlock};
