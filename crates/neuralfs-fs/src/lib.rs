//! NeuralFS filesystem engine.
//!
//! A lightweight copy-on-write, content-addressed, checksummed object
//! filesystem — ZFS-like integrity and snapshots without the weight.

mod blockstore;
mod fs;
mod inode;

pub use blockstore::{hash_hex, Hash, BLOCK_SIZE};
pub use fs::{Filesystem, FsInfo};
pub use inode::{Inode, Stat, SuperBlock};
