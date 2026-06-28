use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{bail, Context, Result};
use hashlink::LruCache;
use parking_lot::Mutex;

use crate::blockstore::{BlockStore, BLOCK_SIZE};
use crate::cache::CacheStats;
use crate::inode::{Inode, Stat, SuperBlock};

/// Default RAM budget for the block cache: a strict 1 GiB, ZFS-ARC style.
pub const DEFAULT_CACHE_BYTES: u64 = 1024 * 1024 * 1024;
/// Number of (immutable) inodes kept hot in RAM to avoid sled reads.
const INODE_CACHE_ENTRIES: usize = 65_536;
/// Files at or below this size are stored inline in the inode (small-file fast
/// path) instead of as content-addressed blocks. 4 KiB matches a typical page.
pub const INLINE_MAX: usize = 4096;

/// A lightweight copy-on-write, content-addressed filesystem.
///
/// Layout under the given directory:
///   data.log  — append-only content-addressed block data
///   meta      — sled db with `blocks`, `inodes`, `super`, `snapshots` trees
///
/// Design goals (ZFS-like, but lighter): end-to-end blake3 checksums,
/// copy-on-write with an atomic root swap per transaction, O(1) snapshots,
/// transparent block-level dedup, a strict-capped frequency-aware RAM block
/// cache, and an immutable-inode cache so hot metadata never touches disk.
pub struct Filesystem {
    blocks: BlockStore,
    db: sled::Db,
    inodes: sled::Tree,
    super_tree: sled::Tree,
    snapshots: sled::Tree,
    state: Mutex<SuperBlock>,
    /// Inodes are immutable once written (CoW allocates a new id per change),
    /// so caching them by id never goes stale.
    inode_cache: Mutex<LruCache<u64, Arc<Inode>>>,
}

impl Filesystem {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_cache(dir, DEFAULT_CACHE_BYTES)
    }

    pub fn open_with_cache(dir: &Path, cache_bytes: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating fs dir {}", dir.display()))?;
        let db = sled::Config::new()
            .path(dir.join("meta"))
            .cache_capacity(32 * 1024 * 1024)
            .open()?;
        let block_index = db.open_tree("blocks")?;
        let inodes = db.open_tree("inodes")?;
        let super_tree = db.open_tree("super")?;
        let snapshots = db.open_tree("snapshots")?;
        let blocks = BlockStore::open(&dir.join("data.log"), block_index, cache_bytes)?;

        let state = match super_tree.get("root")? {
            Some(v) => bincode::deserialize(&v)?,
            None => {
                // Fresh filesystem: inode 0 is the empty root directory.
                let root = Inode::Dir {
                    entries: BTreeMap::new(),
                    mtime: now(),
                };
                inodes.insert(id_key(0), bincode::serialize(&root)?)?;
                let st = SuperBlock {
                    root: 0,
                    next_inode: 1,
                    txg: 0,
                };
                super_tree.insert("root", bincode::serialize(&st)?)?;
                db.flush()?;
                st
            }
        };

        Ok(Self {
            blocks,
            db,
            inodes,
            super_tree,
            snapshots,
            state: Mutex::new(state),
            inode_cache: Mutex::new(LruCache::new(INODE_CACHE_ENTRIES)),
        })
    }

    /// Block-cache statistics (resident bytes, hit rate, promotions, ...).
    pub fn cache_stats(&self) -> CacheStats {
        self.blocks.cache_stats()
    }

    // ---- public filesystem API ------------------------------------------

    /// Write (creating or overwriting) a file, auto-creating parent dirs.
    pub fn write_file(&self, path: &str, data: &[u8]) -> Result<()> {
        let comps = split(path);
        if comps.is_empty() {
            bail!("cannot write to root");
        }

        // Small-file fast path: store the bytes directly in the inode and skip
        // the block store (no blake3, no blocks tree, no data.log append).
        let (blocks, inline) = if data.len() <= INLINE_MAX {
            (Vec::new(), data.to_vec())
        } else {
            let mut block_hashes = Vec::new();
            for chunk in data.chunks(BLOCK_SIZE).filter(|c| !c.is_empty()) {
                block_hashes.push(self.blocks.put(chunk)?);
            }
            (block_hashes, Vec::new())
        };

        let mut state = self.state.lock();
        let file_id = alloc(&mut state);
        self.put_inode(
            file_id,
            &Inode::File {
                size: data.len() as u64,
                blocks,
                inline,
                mtime: now(),
                mode: 0o644,
            },
        )?;
        let new_root = self.rebuild(state.root, &comps, Some(file_id), &mut state)?;
        self.commit(&mut state, new_root)
    }

    /// Read a file's full contents, verifying every block's checksum.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let id = self.resolve(path)?;
        match self.read_inode(id)? {
            Inode::File {
                size,
                blocks,
                inline,
                ..
            } => {
                if blocks.is_empty() {
                    // Inline (or empty) file: bytes live in the inode.
                    return Ok(inline);
                }
                let mut out = Vec::with_capacity(size as usize);
                for h in &blocks {
                    out.extend_from_slice(&self.blocks.get(h)?);
                }
                Ok(out)
            }
            Inode::Dir { .. } => bail!("{path} is a directory"),
        }
    }

    pub fn mkdir(&self, path: &str) -> Result<()> {
        let comps = split(path);
        if comps.is_empty() {
            return Ok(());
        }
        let mut state = self.state.lock();
        let dir_id = alloc(&mut state);
        self.put_inode(
            dir_id,
            &Inode::Dir {
                entries: BTreeMap::new(),
                mtime: now(),
            },
        )?;
        let new_root = self.rebuild(state.root, &comps, Some(dir_id), &mut state)?;
        self.commit(&mut state, new_root)
    }

    pub fn readdir(&self, path: &str) -> Result<Vec<(String, String)>> {
        let id = self.resolve(path)?;
        match self.read_inode(id)? {
            Inode::Dir { entries, .. } => {
                let mut out = Vec::with_capacity(entries.len());
                for (name, child) in entries {
                    let kind = self.read_inode(child)?.kind().to_string();
                    out.push((name, kind));
                }
                Ok(out)
            }
            Inode::File { .. } => bail!("{path} is a file"),
        }
    }

    /// Remove a file or (empty or non-empty) directory entry.
    pub fn remove(&self, path: &str) -> Result<()> {
        let comps = split(path);
        if comps.is_empty() {
            bail!("cannot remove root");
        }
        let mut state = self.state.lock();
        let new_root = self.rebuild(state.root, &comps, None, &mut state)?;
        self.commit(&mut state, new_root)
    }

    pub fn rename(&self, from: &str, to: &str) -> Result<()> {
        let from_comps = split(from);
        let to_comps = split(to);
        if from_comps.is_empty() || to_comps.is_empty() {
            bail!("cannot rename root");
        }
        let id = self.resolve(from)?;
        let mut state = self.state.lock();
        let r1 = self.rebuild(state.root, &to_comps, Some(id), &mut state)?;
        let r2 = self.rebuild(r1, &from_comps, None, &mut state)?;
        self.commit(&mut state, r2)
    }

    pub fn stat(&self, path: &str) -> Result<Stat> {
        let id = self.resolve(path)?;
        Ok(match self.read_inode(id)? {
            Inode::File {
                size,
                blocks,
                mtime,
                ..
            } => Stat {
                kind: "file".into(),
                size,
                mtime,
                blocks: blocks.len(),
                entries: 0,
            },
            Inode::Dir { entries, mtime } => Stat {
                kind: "dir".into(),
                size: 0,
                mtime,
                blocks: 0,
                entries: entries.len(),
            },
        })
    }

    pub fn exists(&self, path: &str) -> bool {
        self.resolve(path).is_ok()
    }

    // ---- snapshots -------------------------------------------------------

    /// Create a named snapshot of the current state. O(1): records the root
    /// pointer; CoW guarantees the referenced inodes are never mutated.
    pub fn snapshot(&self, name: &str) -> Result<()> {
        let state = *self.state.lock();
        self.snapshots.insert(name, bincode::serialize(&state)?)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn list_snapshots(&self) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        for kv in self.snapshots.iter() {
            let (k, v) = kv?;
            let st: SuperBlock = bincode::deserialize(&v)?;
            out.push((String::from_utf8_lossy(&k).to_string(), st.txg));
        }
        Ok(out)
    }

    /// Roll the live filesystem back to a snapshot's root. Monotonic inode
    /// allocation is preserved so ids are never reused.
    pub fn rollback(&self, name: &str) -> Result<()> {
        let snap: SuperBlock = match self.snapshots.get(name)? {
            Some(v) => bincode::deserialize(&v)?,
            None => bail!("no such snapshot: {name}"),
        };
        let mut state = self.state.lock();
        state.root = snap.root;
        state.next_inode = state.next_inode.max(snap.next_inode);
        let new_root = state.root;
        self.commit(&mut state, new_root)
    }

    // ---- integrity & stats ----------------------------------------------

    pub fn scrub(&self) -> Result<(usize, Vec<String>)> {
        self.blocks.scrub()
    }

    pub fn info(&self) -> Result<FsInfo> {
        let state = *self.state.lock();
        let mut logical = 0u64;
        let mut referenced = 0u64; // live block refs, counting duplicates
        let mut live: std::collections::HashSet<crate::blockstore::Hash> =
            std::collections::HashSet::new();
        let mut files = 0usize;
        let mut dirs = 0usize;
        self.walk(state.root, &mut |inode| {
            match inode {
                Inode::File { size, blocks, .. } => {
                    files += 1;
                    logical += *size;
                    for h in blocks {
                        referenced += self.blocks.len_for(h).unwrap_or(0) as u64;
                        live.insert(*h);
                    }
                }
                Inode::Dir { .. } => dirs += 1,
            }
            Ok(())
        })?;
        // Unique bytes actually referenced by live files.
        let mut physical_referenced = 0u64;
        for h in &live {
            physical_referenced += self.blocks.len_for(h).unwrap_or(0) as u64;
        }
        let physical_total = self.blocks.physical_bytes()?;
        // Dedup ratio is measured over live data only (ZFS-style): how much
        // logical data the unique referenced blocks stand in for.
        let dedup_ratio = if physical_referenced == 0 {
            1.0
        } else {
            referenced as f64 / physical_referenced as f64
        };
        Ok(FsInfo {
            txg: state.txg,
            files,
            dirs,
            unique_blocks: self.blocks.block_count(),
            logical_bytes: logical,
            physical_referenced,
            physical_total,
            reclaimable_bytes: physical_total.saturating_sub(physical_referenced),
            dedup_ratio,
        })
    }

    pub fn flush(&self) -> Result<()> {
        self.blocks.flush()?;
        self.db.flush()?;
        Ok(())
    }

    // ---- internals -------------------------------------------------------

    fn resolve(&self, path: &str) -> Result<u64> {
        let comps = split(path);
        let mut id = self.state.lock().root;
        for c in comps {
            match self.read_inode(id)? {
                Inode::Dir { entries, .. } => {
                    id = *entries
                        .get(c)
                        .with_context(|| format!("no such path component: {c}"))?;
                }
                Inode::File { .. } => bail!("{c} is not a directory"),
            }
        }
        Ok(id)
    }

    /// Copy-on-write rebuild of the directory chain along `comps`, returning the
    /// id of the new (sub)tree root. `leaf = Some(id)` sets the final component
    /// to `id` (creating intermediate dirs as needed); `None` removes it.
    fn rebuild(
        &self,
        dir_id: u64,
        comps: &[&str],
        leaf: Option<u64>,
        state: &mut SuperBlock,
    ) -> Result<u64> {
        let mut entries = match self.read_inode(dir_id)? {
            Inode::Dir { entries, .. } => entries,
            Inode::File { .. } => bail!("path traverses a file"),
        };
        let name = comps[0].to_string();
        if comps.len() == 1 {
            match leaf {
                Some(id) => {
                    entries.insert(name, id);
                }
                None => {
                    if entries.remove(&name).is_none() {
                        bail!("no such entry: {name}");
                    }
                }
            }
        } else {
            let child_id = match entries.get(&name) {
                Some(&id) => id,
                None => {
                    let id = alloc(state);
                    self.put_inode(
                        id,
                        &Inode::Dir {
                            entries: BTreeMap::new(),
                            mtime: now(),
                        },
                    )?;
                    id
                }
            };
            let new_child = self.rebuild(child_id, &comps[1..], leaf, state)?;
            entries.insert(name, new_child);
        }
        let new_id = alloc(state);
        self.put_inode(
            new_id,
            &Inode::Dir {
                entries,
                mtime: now(),
            },
        )?;
        Ok(new_id)
    }

    fn walk<F: FnMut(&Inode) -> Result<()>>(&self, id: u64, f: &mut F) -> Result<()> {
        let inode = self.read_inode(id)?;
        f(&inode)?;
        if let Inode::Dir { entries, .. } = inode {
            for child in entries.values() {
                self.walk(*child, f)?;
            }
        }
        Ok(())
    }

    fn read_inode(&self, id: u64) -> Result<Inode> {
        if let Some(a) = self.inode_cache.lock().get(&id) {
            return Ok((**a).clone());
        }
        let v = self
            .inodes
            .get(id_key(id))?
            .with_context(|| format!("dangling inode {id}"))?;
        let inode: Inode = bincode::deserialize(&v)?;
        self.inode_cache.lock().insert(id, Arc::new(inode.clone()));
        Ok(inode)
    }

    fn put_inode(&self, id: u64, inode: &Inode) -> Result<()> {
        self.inodes.insert(id_key(id), bincode::serialize(inode)?)?;
        self.inode_cache.lock().insert(id, Arc::new(inode.clone()));
        Ok(())
    }

    fn commit(&self, state: &mut SuperBlock, new_root: u64) -> Result<()> {
        state.root = new_root;
        state.txg += 1;
        self.super_tree
            .insert("root", bincode::serialize(&*state)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FsInfo {
    pub txg: u64,
    pub files: usize,
    pub dirs: usize,
    pub unique_blocks: usize,
    pub logical_bytes: u64,
    /// Unique bytes referenced by live files.
    pub physical_referenced: u64,
    /// All stored block bytes, including orphans not yet reclaimed.
    pub physical_total: u64,
    /// Orphaned bytes from deleted/overwritten data (no GC yet).
    pub reclaimable_bytes: u64,
    pub dedup_ratio: f64,
}

fn alloc(state: &mut SuperBlock) -> u64 {
    let id = state.next_inode;
    state.next_inode += 1;
    id
}

fn id_key(id: u64) -> [u8; 8] {
    id.to_be_bytes()
}

fn split(path: &str) -> Vec<&str> {
    path.split(|c| c == '/' || c == '\\')
        .filter(|s| !s.is_empty())
        .collect()
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let n: u64 = std::process::id() as u64 * 1_000_003
            + SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as u64;
        p.push(format!("nfs-fs-test-{n}"));
        p
    }

    #[test]
    fn write_read_roundtrip() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        fs.mkdir("/docs").unwrap();
        fs.write_file("/docs/a.txt", b"hello world").unwrap();
        assert_eq!(fs.read_file("/docs/a.txt").unwrap(), b"hello world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn overwrite_updates_content() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        fs.write_file("/f", b"one").unwrap();
        fs.write_file("/f", b"two-longer").unwrap();
        assert_eq!(fs.read_file("/f").unwrap(), b"two-longer");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn small_files_are_inlined_not_blocked() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        // A small file (<= INLINE_MAX) must read back correctly...
        let small = vec![0xABu8; INLINE_MAX];
        fs.write_file("/small.bin", &small).unwrap();
        assert_eq!(fs.read_file("/small.bin").unwrap(), small);
        // ...and must NOT have touched the block store at all.
        assert_eq!(
            fs.info().unwrap().unique_blocks,
            0,
            "small file should be inline, not stored as blocks"
        );

        // A file just over the threshold uses blocks as before.
        let big = vec![0xCDu8; INLINE_MAX + 1];
        fs.write_file("/big.bin", &big).unwrap();
        assert_eq!(fs.read_file("/big.bin").unwrap(), big);
        assert!(fs.info().unwrap().unique_blocks > 0);

        // Growing past the threshold then shrinking back round-trips both ways.
        fs.write_file("/g", &vec![1u8; INLINE_MAX + 100]).unwrap();
        assert_eq!(fs.read_file("/g").unwrap().len(), INLINE_MAX + 100);
        fs.write_file("/g", b"tiny").unwrap();
        assert_eq!(fs.read_file("/g").unwrap(), b"tiny");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dedup_stores_identical_blocks_once() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        let payload = vec![7u8; 100_000];
        fs.write_file("/a", &payload).unwrap();
        fs.write_file("/b", &payload).unwrap();
        let info = fs.info().unwrap();
        // two files referencing the same blocks -> dedup ratio ~2x
        assert!(info.dedup_ratio > 1.5, "ratio was {}", info.dedup_ratio);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn snapshot_and_rollback() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        fs.write_file("/x", b"v1").unwrap();
        fs.snapshot("s1").unwrap();
        fs.write_file("/x", b"v2").unwrap();
        assert_eq!(fs.read_file("/x").unwrap(), b"v2");
        fs.rollback("s1").unwrap();
        assert_eq!(fs.read_file("/x").unwrap(), b"v1");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scrub_clean_after_writes() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        fs.write_file("/a", b"abc").unwrap();
        fs.write_file("/b", &vec![1u8; 200_000]).unwrap();
        let (checked, bad) = fs.scrub().unwrap();
        assert!(checked > 0);
        assert!(bad.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn persists_across_reopen() {
        let dir = tmp();
        {
            let fs = Filesystem::open(&dir).unwrap();
            fs.write_file("/keep.txt", b"durable").unwrap();
            fs.flush().unwrap();
        }
        {
            let fs = Filesystem::open(&dir).unwrap();
            assert_eq!(fs.read_file("/keep.txt").unwrap(), b"durable");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn remove_and_readdir() {
        let dir = tmp();
        let fs = Filesystem::open(&dir).unwrap();
        fs.write_file("/d/a", b"1").unwrap();
        fs.write_file("/d/b", b"2").unwrap();
        assert_eq!(fs.readdir("/d").unwrap().len(), 2);
        fs.remove("/d/a").unwrap();
        assert_eq!(fs.readdir("/d").unwrap().len(), 1);
        assert!(!fs.exists("/d/a"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
