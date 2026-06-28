//! User-mode filesystem hook (FUSE passthrough + hot-file RAM cache).
//!
//! NeuralFS mounts *in front of* a real backing directory. Metadata operations
//! (lookup/getattr/readdir/create/...) pass straight through to the backing
//! filesystem at native speed, while file *reads* go through a strict,
//! frequency-aware 1 GiB RAM cache: once a file has been read often enough it
//! is pulled wholesale into memory (ZFS-ARC style), so subsequent reads are
//! served without touching disk.
//!
//! Linux/WSL only, behind the `fuse` cargo feature.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyCreate, ReplyData,
    ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request,
};
use neuralfs_fs::RamCache;

const TTL: Duration = Duration::from_secs(1);
const ROOT_INO: u64 = 1;
/// Files at or below this size are eligible to be cached whole in RAM.
const MAX_CACHE_FILE: u64 = 256 * 1024 * 1024;
/// Read count at which a file is promoted into the RAM cache.
const HOT_THRESHOLD: u32 = 2;

pub struct PassthroughFs {
    ino_to_path: HashMap<u64, PathBuf>,
    path_to_ino: HashMap<PathBuf, u64>,
    next_ino: u64,
    handles: HashMap<u64, File>,
    next_fh: u64,
    cache: RamCache<u64>,
    freq: HashMap<u64, u32>,
}

impl PassthroughFs {
    pub fn new(backing: PathBuf, cache_bytes: u64) -> Self {
        let mut ino_to_path = HashMap::new();
        let mut path_to_ino = HashMap::new();
        ino_to_path.insert(ROOT_INO, backing.clone());
        path_to_ino.insert(backing, ROOT_INO);
        Self {
            ino_to_path,
            path_to_ino,
            next_ino: 2,
            handles: HashMap::new(),
            next_fh: 1,
            cache: RamCache::new(cache_bytes),
            freq: HashMap::new(),
        }
    }

    fn ino_for(&mut self, path: &Path) -> u64 {
        if let Some(&i) = self.path_to_ino.get(path) {
            return i;
        }
        let i = self.next_ino;
        self.next_ino += 1;
        self.ino_to_path.insert(i, path.to_path_buf());
        self.path_to_ino.insert(path.to_path_buf(), i);
        i
    }

    fn path_of(&self, ino: u64) -> Option<PathBuf> {
        self.ino_to_path.get(&ino).cloned()
    }

    fn forget_path(&mut self, path: &Path) {
        if let Some(ino) = self.path_to_ino.remove(path) {
            self.ino_to_path.remove(&ino);
            self.cache.invalidate(&ino);
            self.freq.remove(&ino);
        }
    }
}

fn attr_from_meta(ino: u64, m: &std::fs::Metadata) -> FileAttr {
    let kind = if m.is_dir() {
        FileType::Directory
    } else if m.file_type().is_symlink() {
        FileType::Symlink
    } else {
        FileType::RegularFile
    };
    FileAttr {
        ino,
        size: m.len(),
        blocks: (m.len() + 511) / 512,
        atime: m.accessed().unwrap_or(UNIX_EPOCH),
        mtime: m.modified().unwrap_or(UNIX_EPOCH),
        ctime: UNIX_EPOCH + Duration::from_secs(m.ctime().max(0) as u64),
        crtime: UNIX_EPOCH,
        kind,
        perm: (m.mode() & 0o7777) as u16,
        nlink: m.nlink() as u32,
        uid: m.uid(),
        gid: m.gid(),
        rdev: m.rdev() as u32,
        flags: 0,
        blksize: 512,
    }
}

impl Filesystem for PassthroughFs {
    fn init(
        &mut self,
        _req: &Request,
        config: &mut fuser::KernelConfig,
    ) -> Result<(), libc::c_int> {
        // Ask the kernel to send larger write/read chunks so big sequential I/O
        // crosses the FUSE boundary in 1 MiB units instead of 128 KiB ones —
        // fewer userspace round trips per megabyte.
        let _ = config.set_max_write(1 << 20);
        let _ = config.set_max_readahead(1 << 20);
        // Writeback caching: the kernel buffers and coalesces writes in its page
        // cache and flushes them back in big batches, instead of forwarding every
        // small write() straight to userspace. This is the single biggest win for
        // small-file / small-write workloads — it removes most of the per-write
        // kernel<->userspace round trips that make a plain FUSE passthrough slow.
        let _ = config.add_capabilities(fuser::consts::FUSE_WRITEBACK_CACHE);
        Ok(())
    }

    fn lookup(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let Some(pp) = self.path_of(parent) else {
            return reply.error(libc::ENOENT);
        };
        let child = pp.join(name);
        match std::fs::symlink_metadata(&child) {
            Ok(m) => {
                let ino = self.ino_for(&child);
                reply.entry(&TTL, &attr_from_meta(ino, &m), 0);
            }
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    fn getattr(&mut self, _req: &Request, ino: u64, reply: ReplyAttr) {
        let Some(p) = self.path_of(ino) else {
            return reply.error(libc::ENOENT);
        };
        match std::fs::symlink_metadata(&p) {
            Ok(m) => reply.attr(&TTL, &attr_from_meta(ino, &m)),
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn setattr(
        &mut self,
        _req: &Request,
        ino: u64,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<fuser::TimeOrNow>,
        _mtime: Option<fuser::TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        let Some(p) = self.path_of(ino) else {
            return reply.error(libc::ENOENT);
        };
        if let Some(sz) = size {
            if let Ok(f) = OpenOptions::new().write(true).open(&p) {
                let _ = f.set_len(sz);
            }
            self.cache.invalidate(&ino);
        }
        match std::fs::symlink_metadata(&p) {
            Ok(m) => reply.attr(&TTL, &attr_from_meta(ino, &m)),
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let Some(p) = self.path_of(ino) else {
            return reply.error(libc::ENOENT);
        };
        let mut entries: Vec<(u64, FileType, String)> = vec![
            (ino, FileType::Directory, ".".to_string()),
            (ino, FileType::Directory, "..".to_string()),
        ];
        match std::fs::read_dir(&p) {
            Ok(rd) => {
                for ent in rd.flatten() {
                    let cp = ent.path();
                    let kind = match ent.file_type() {
                        Ok(t) if t.is_dir() => FileType::Directory,
                        Ok(t) if t.is_symlink() => FileType::Symlink,
                        _ => FileType::RegularFile,
                    };
                    let cino = self.ino_for(&cp);
                    entries.push((cino, kind, ent.file_name().to_string_lossy().to_string()));
                }
            }
            Err(_) => return reply.error(libc::ENOTDIR),
        }
        for (i, (cino, kind, name)) in entries.into_iter().enumerate().skip(offset as usize) {
            if reply.add(cino, (i + 1) as i64, kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn open(&mut self, _req: &Request, ino: u64, flags: i32, reply: ReplyOpen) {
        let Some(p) = self.path_of(ino) else {
            return reply.error(libc::ENOENT);
        };
        let acc = flags & libc::O_ACCMODE;
        let mut opts = OpenOptions::new();
        match acc {
            libc::O_WRONLY => {
                opts.write(true);
            }
            libc::O_RDWR => {
                opts.read(true).write(true);
            }
            _ => {
                opts.read(true);
            }
        }
        opts.custom_flags(flags & !libc::O_ACCMODE);
        match opts.open(&p) {
            Ok(f) => {
                let fh = self.next_fh;
                self.next_fh += 1;
                self.handles.insert(fh, f);

                // Frequency accounting is per-open: a file opened for reading
                // often enough is promoted wholesale into the RAM cache (this
                // is the "read frequency is high enough → load into RAM" rule).
                if flags & libc::O_ACCMODE != libc::O_WRONLY {
                    let c = self.freq.entry(ino).or_insert(0);
                    *c += 1;
                    if *c >= HOT_THRESHOLD && !self.cache.contains(&ino) {
                        if let Ok(m) = std::fs::metadata(&p) {
                            if m.len() <= MAX_CACHE_FILE {
                                if let Ok(whole) = std::fs::read(&p) {
                                    self.cache.insert(ino, Arc::new(whole));
                                }
                            }
                        }
                    }
                }
                // Keep the kernel page cache across opens: repeated reads of an
                // unchanged file are served from the kernel's cache without ever
                // crossing into userspace (on top of our own RAM cache below).
                reply.opened(fh, fuser::consts::FOPEN_KEEP_CACHE);
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn create(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let Some(pp) = self.path_of(parent) else {
            return reply.error(libc::ENOENT);
        };
        let child = pp.join(name);
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true);
        if flags & libc::O_TRUNC != 0 {
            opts.truncate(true);
        }
        match opts.open(&child) {
            Ok(f) => {
                let ino = self.ino_for(&child);
                let fh = self.next_fh;
                self.next_fh += 1;
                self.handles.insert(fh, f);
                self.cache.invalidate(&ino);
                match std::fs::symlink_metadata(&child) {
                    Ok(m) => reply.created(
                        &TTL,
                        &attr_from_meta(ino, &m),
                        0,
                        fh,
                        fuser::consts::FOPEN_KEEP_CACHE,
                    ),
                    Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
                }
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn read(
        &mut self,
        _req: &Request,
        ino: u64,
        fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock: Option<u64>,
        reply: ReplyData,
    ) {
        let off = offset.max(0) as usize;
        let want = size as usize;

        // Fast path: whole file already resident in the RAM cache.
        if let Some(buf) = self.cache.get(&ino) {
            let end = (off + want).min(buf.len());
            let slice = if off < buf.len() { &buf[off..end] } else { &[] };
            return reply.data(slice);
        }

        let Some(p) = self.path_of(ino) else {
            return reply.error(libc::ENOENT);
        };

        // Serve the requested range from the backing file (cold path).
        let mut buf = vec![0u8; want];
        let n = match self.handles.get(&fh) {
            Some(f) => f.read_at(&mut buf, off as u64),
            None => match File::open(&p) {
                Ok(f) => f.read_at(&mut buf, off as u64),
                Err(e) => return reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
            },
        };
        match n {
            Ok(n) => reply.data(&buf[..n]),
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &mut self,
        _req: &Request,
        ino: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock: Option<u64>,
        reply: ReplyWrite,
    ) {
        let res = match self.handles.get(&fh) {
            Some(f) => f.write_at(data, offset.max(0) as u64),
            None => {
                let Some(p) = self.path_of(ino) else {
                    return reply.error(libc::ENOENT);
                };
                match OpenOptions::new().write(true).open(&p) {
                    Ok(f) => f.write_at(data, offset.max(0) as u64),
                    Err(e) => return reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
                }
            }
        };
        // Write-through: the cached copy is now stale.
        self.cache.invalidate(&ino);
        self.freq.remove(&ino);
        match res {
            Ok(n) => reply.written(n as u32),
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn mkdir(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let Some(pp) = self.path_of(parent) else {
            return reply.error(libc::ENOENT);
        };
        let child = pp.join(name);
        match std::fs::create_dir(&child) {
            Ok(()) => {
                let ino = self.ino_for(&child);
                match std::fs::symlink_metadata(&child) {
                    Ok(m) => reply.entry(&TTL, &attr_from_meta(ino, &m), 0),
                    Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
                }
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn unlink(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(pp) = self.path_of(parent) else {
            return reply.error(libc::ENOENT);
        };
        let child = pp.join(name);
        match std::fs::remove_file(&child) {
            Ok(()) => {
                self.forget_path(&child);
                reply.ok();
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn rmdir(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(pp) = self.path_of(parent) else {
            return reply.error(libc::ENOENT);
        };
        let child = pp.join(name);
        match std::fs::remove_dir(&child) {
            Ok(()) => {
                self.forget_path(&child);
                reply.ok();
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn rename(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let (Some(pp), Some(np)) = (self.path_of(parent), self.path_of(newparent)) else {
            return reply.error(libc::ENOENT);
        };
        let from = pp.join(name);
        let to = np.join(newname);
        match std::fs::rename(&from, &to) {
            Ok(()) => {
                self.forget_path(&from);
                self.forget_path(&to);
                reply.ok();
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn flush(&mut self, _req: &Request, _ino: u64, fh: u64, _lock: u64, reply: ReplyEmpty) {
        if let Some(f) = self.handles.get(&fh) {
            let _ = f.sync_data();
        }
        reply.ok();
    }

    fn release(
        &mut self,
        _req: &Request,
        _ino: u64,
        fh: u64,
        _flags: i32,
        _lock: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        self.handles.remove(&fh);
        reply.ok();
    }

    fn fsync(&mut self, _req: &Request, _ino: u64, fh: u64, _datasync: bool, reply: ReplyEmpty) {
        if let Some(f) = self.handles.get(&fh) {
            let _ = f.sync_all();
        }
        reply.ok();
    }

    fn statfs(&mut self, _req: &Request, _ino: u64, reply: fuser::ReplyStatfs) {
        // blocks, bfree, bavail, files, ffree, bsize, namelen, frsize
        reply.statfs(1 << 30, 1 << 29, 1 << 29, 1 << 20, 1 << 19, 4096, 255, 4096);
    }
}

/// Mount the passthrough caching filesystem at `mountpoint`, backed by
/// `backing`. Blocks until the filesystem is unmounted.
pub fn run_mount(mountpoint: &Path, backing: &Path, cache_bytes: u64) -> Result<()> {
    std::fs::create_dir_all(backing)?;
    std::fs::create_dir_all(mountpoint)?;
    let fs = PassthroughFs::new(backing.to_path_buf(), cache_bytes);
    let opts = vec![
        MountOption::FSName("neuralfs".to_string()),
        MountOption::DefaultPermissions,
        MountOption::AutoUnmount,
    ];
    log::info!(
        "mounting NeuralFS hook at {} (backing {}, {} MiB cache)",
        mountpoint.display(),
        backing.display(),
        cache_bytes / (1024 * 1024)
    );
    fuser::mount2(fs, mountpoint, &opts)?;
    Ok(())
}
