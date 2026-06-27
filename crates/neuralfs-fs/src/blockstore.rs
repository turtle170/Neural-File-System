use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use parking_lot::Mutex;

use crate::cache::{CacheStats, RamCache};

/// Logical block size. Files are split into chunks of at most this many bytes.
pub const BLOCK_SIZE: usize = 64 * 1024;

/// A blake3 digest. Doubles as the content address (key) and the integrity
/// checksum for a block — verified on every read, ZFS-style.
pub type Hash = [u8; 32];

pub fn hash_bytes(bytes: &[u8]) -> Hash {
    *blake3::hash(bytes).as_bytes()
}

pub fn hash_hex(h: &Hash) -> String {
    let mut s = String::with_capacity(64);
    for b in h {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

struct DataLog {
    file: File,
    len: u64,
}

/// A content-addressed, append-only, deduplicating block store.
///
/// Blocks are written once and never mutated (copy-on-write happens at the
/// inode layer). Identical content is stored a single time. Each read
/// re-hashes the bytes and compares against the requested address, so silent
/// corruption is detected rather than returned.
pub struct BlockStore {
    log: Mutex<DataLog>,
    index: sled::Tree,
    cache: Mutex<RamCache<Hash>>,
}

impl BlockStore {
    /// `cache_bytes` is the strict RAM budget for cached block data.
    pub fn open(data_path: &Path, index: sled::Tree, cache_bytes: u64) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(data_path)
            .with_context(|| format!("opening data log {}", data_path.display()))?;
        let len = file.metadata()?.len();
        Ok(Self {
            log: Mutex::new(DataLog { file, len }),
            index,
            cache: Mutex::new(RamCache::new(cache_bytes)),
        })
    }

    /// Store a block, returning its content address. A no-op (beyond hashing)
    /// if an identical block already exists.
    pub fn put(&self, bytes: &[u8]) -> Result<Hash> {
        let hash = hash_bytes(bytes);
        if self.index.contains_key(hash)? {
            return Ok(hash);
        }
        let mut log = self.log.lock();
        let write_at = log.len;
        let data_offset = write_at + 4;
        let mut record = Vec::with_capacity(4 + bytes.len());
        record.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        record.extend_from_slice(bytes);
        log.file.seek(SeekFrom::Start(write_at))?;
        log.file.write_all(&record)?;
        log.len += record.len() as u64;
        drop(log);

        self.index
            .insert(hash, enc(data_offset, bytes.len() as u32).to_vec())?;
        self.cache.lock().insert(hash, Arc::new(bytes.to_vec()));
        Ok(hash)
    }

    /// Fetch a block by content address, verifying its integrity. Served from
    /// the RAM cache on a hit (no disk I/O, no re-hash).
    pub fn get(&self, hash: &Hash) -> Result<Arc<Vec<u8>>> {
        if let Some(v) = self.cache.lock().get(hash) {
            return Ok(v);
        }
        let (offset, len) = match self.index.get(hash)? {
            Some(v) => dec(&v),
            None => bail!("block not found: {}", hash_hex(hash)),
        };
        let mut buf = vec![0u8; len as usize];
        {
            let mut log = self.log.lock();
            log.file.seek(SeekFrom::Start(offset))?;
            log.file.read_exact(&mut buf)?;
        }
        let actual = hash_bytes(&buf);
        if &actual != hash {
            bail!(
                "integrity check failed: block {} hashed to {}",
                hash_hex(hash),
                hash_hex(&actual)
            );
        }
        let arc = Arc::new(buf);
        self.cache.lock().insert(*hash, arc.clone());
        Ok(arc)
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.lock().stats()
    }

    /// Number of unique stored blocks.
    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    /// Total physical bytes occupied by unique block data (excludes record headers).
    pub fn physical_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for kv in self.index.iter() {
            let (_, v) = kv?;
            let (_, len) = dec(&v);
            total += len as u64;
        }
        Ok(total)
    }

    /// Verify every stored block against its checksum. Returns
    /// (blocks_checked, list_of_corrupt_block_hashes).
    pub fn scrub(&self) -> Result<(usize, Vec<String>)> {
        let mut checked = 0;
        let mut bad = Vec::new();
        for kv in self.index.iter() {
            let (k, v) = kv?;
            let mut hash = [0u8; 32];
            if k.len() == 32 {
                hash.copy_from_slice(&k);
            } else {
                continue;
            }
            let (offset, len) = dec(&v);
            let mut buf = vec![0u8; len as usize];
            {
                let mut log = self.log.lock();
                log.file.seek(SeekFrom::Start(offset))?;
                if log.file.read_exact(&mut buf).is_err() {
                    bad.push(hash_hex(&hash));
                    checked += 1;
                    continue;
                }
            }
            if hash_bytes(&buf) != hash {
                bad.push(hash_hex(&hash));
            }
            checked += 1;
        }
        Ok((checked, bad))
    }

    pub fn flush(&self) -> Result<()> {
        let log = self.log.lock();
        log.file.sync_data()?;
        Ok(())
    }

    pub fn len_for(&self, hash: &Hash) -> Result<u32> {
        match self.index.get(hash)? {
            Some(v) => Ok(dec(&v).1),
            None => Ok(0),
        }
    }
}

fn enc(offset: u64, len: u32) -> [u8; 12] {
    let mut b = [0u8; 12];
    b[..8].copy_from_slice(&offset.to_le_bytes());
    b[8..].copy_from_slice(&len.to_le_bytes());
    b
}

fn dec(v: &[u8]) -> (u64, u32) {
    let mut o = [0u8; 8];
    let mut l = [0u8; 4];
    o.copy_from_slice(&v[..8]);
    l.copy_from_slice(&v[8..12]);
    (u64::from_le_bytes(o), u32::from_le_bytes(l))
}
