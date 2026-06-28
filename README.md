# NeuralFS

NeuralFS is three things in one daemon:

1. **A learned file finder** — it indexes your real files, learns which
   directories a query resolves to (TF-IDF + softmax/logistic regression),
   ranks results by an exponential-decay open-frequency score, and **keeps
   learning online from every access while it runs**.
2. **A real filesystem engine** — a lightweight copy-on-write,
   content-addressed, checksummed object store (`neuralfs-fs`): ZFS-like
   integrity, O(1) snapshots, transparent block dedup, and a strict 1 GiB
   frequency-aware RAM cache — but far lighter than ZFS.
3. **Real filesystem mounts** — a FUSE passthrough caching mount on Linux/WSL
   (serves metadata at native speed, pulls hot files into the 1 GiB RAM cache),
   and a **WinFsp drive-letter mount on Windows** that exposes NeuralFS as a real
   drive `N:` through the WinFsp kernel-mode driver (verified working).

A small CLI (`nfs`) drives it all over a Windows named pipe.

## Workspace layout

```
neuralfs/
├── Cargo.toml                  # workspace
├── crates/
│   ├── neuralfs-fs/            # the copy-on-write filesystem engine (library)
│   │   └── src/
│   │       ├── blockstore.rs   # blake3 content-addressed append-only blocks + LRU cache
│   │       ├── inode.rs        # CoW inodes + superblock
│   │       └── fs.rs           # path ops, snapshots, scrub, dedup stats (+ tests)
│   ├── neuralfs-daemon/        # daemon binary -> neuralfs.exe
│   │   └── src/
│   │       ├── main.rs         # startup, retrain loop, AI checkpoint loop, flush
│   │       ├── indexer.rs      # walkdir-based filesystem indexer
│   │       ├── classifier.rs   # TF-IDF + softmax regression + online SGD updates
│   │       ├── scorer.rs       # frequency + exponential decay scoring
│   │       ├── search.rs       # predict -> backtrack -> fallback lookup
│   │       ├── watcher.rs      # notify-based FS event watcher
│   │       ├── ipc.rs          # named-pipe server: find/open/fs/hook/ai/cache/bench
│   │       ├── pathcache.rs    # 500 MiB, 5-min sliding-TTL cache of found paths
│   │       ├── mountfs.rs      # FUSE passthrough caching mount (Linux/WSL, feature=fuse)
│   │       ├── winfsphost.rs   # WinFsp drive-letter mount (Windows, feature=winfsp)
│   │       ├── store.rs        # sled-backed index persistence
│   │       └── ...             # config, logging, install, state, protocol
│   └── neuralfs-cli/           # CLI binary -> nfs.exe
└── README.md
```

(`neuralfs-fs` also contains `cache.rs` — the strict, frequency-aware RAM cache.)

## Building

```sh
cargo build --release --workspace
```

Binaries: `target/release/neuralfs.exe` (daemon) and `target/release/nfs.exe` (CLI).

## The filesystem engine (`neuralfs-fs`)

A genuine copy-on-write object filesystem, usable through `nfs fs ...`:

- **End-to-end integrity** — every block is addressed and checksummed with
  blake3; reads re-hash and verify, so silent corruption is caught, not served.
- **Copy-on-write** — writes never mutate existing blocks or inodes; a single
  atomic root-pointer swap publishes each transaction (ZFS-style uberblock).
- **O(1) snapshots & rollback** — a snapshot just records the current root;
  CoW guarantees its inodes are never overwritten.
- **Transparent dedup** — identical blocks (across files *and* within a file)
  are stored once; `nfs fs info` reports the live dedup ratio.
- **Scrub** — `nfs fs scrub` walks and re-verifies every block's checksum.
- **Speed** — append-only block log + the RAM cache below. Honest `nfs bench 64`
  on this machine: **~258 MB/s write to disk** (CoW + blake3 + sled metadata,
  competitive with raw ext4's ~226 MB/s while doing strictly more work), and
  cache-warm reads at **~1.8 GB/s**.

## The 1 GiB frequency-aware RAM cache (ZFS-ARC style)

`neuralfs-fs/cache.rs` is a strict, byte-bounded, frequency-aware cache:

- **Strict cap** — a hard byte ceiling (default **1 GiB**), enforced after every
  insert; resident bytes never exceed it (unit-tested).
- **Frequency promotion (SLRU)** — new data lands in a *probation* segment; data
  read **again** is promoted to a *protected* segment, so genuinely hot data
  survives eviction pressure that churns through one-shot reads. This is exactly
  "if a file's read frequency is high enough, keep it in RAM."
- **Used everywhere** — backs the CoW volume's block cache and the FUSE hot-file
  cache. `nfs cache` shows resident bytes, hit rate, promotions, and evictions.

The filesystem also keeps a separate **immutable-inode cache**: because CoW gives
every modified inode a brand-new id, cached inodes can never go stale, so hot
metadata is served from RAM with no sled lookups.

## The 500 MiB, 5-minute path cache (sliding TTL)

A second, independent cache (`neuralfs-daemon/pathcache.rs`) accelerates *search*:
when `nfs find` returns results — the paths the AI guessed or that were otherwise
found — they're cached keyed by the query.

- **Separate 500 MiB budget**, distinct from the 1 GiB block cache, strictly
  enforced (soonest-to-expire entries evicted first if it would overflow).
- **5-minute sliding TTL** — each entry expires 5 minutes after its *last* use.
  Re-running the same query within the window serves it from RAM **and refreshes
  the timer**, so hot queries stay resident while one-off queries age out.
- A background sweeper drops expired entries every 30 s; `nfs cache` shows both
  caches (resident bytes, hits/misses, expirations, evictions).

This is unit-tested for sliding-refresh, expiry, sweep reclamation, and strict
cap enforcement.

## Hooking onto your real filesystem

Two hooks, depending on what you want:

### 1. Learned search/access layer (cross-platform)

```sh
nfs hook "C:/Users/me/Documents"   # index it, watch it live, learn from it
nfs hook                           # show currently hooked directories
```

This indexes the tree, starts a live `notify` watcher, and turns on
access-driven learning — NeuralFS becomes the smart search layer over your files.

### 2. User-mode filesystem mount (FUSE — Linux/WSL)

```sh
cargo build --release --features fuse
neuralfs --mount /mnt/nfs --backing /data/store --cache-mb 1024
```

This is a **real user-mode filesystem**: the OS routes file I/O for everything
under `/mnt/nfs` through the NeuralFS daemon. Metadata operations pass through to
the backing directory at native speed; file reads go through the 1 GiB
frequency-aware cache, so a file opened often enough is served from RAM.

**Measured in a clean WSL2 VM** (passthrough over ext4):

| workload | raw ext4 | NeuralFS FUSE hook | vs. old `nfs fs` CLI path |
|---|---|---|---|
| small-file read (500 × 2 KiB) | 820 files/s | **408 files/s** | was 185 files/s — **2.2× faster** |
| hot 200 MiB file, repeat read | — | **1465 MB/s from RAM** (vs 538 MB/s cold) | — |

The honest takeaway: FUSE pays an inherent userspace round-trip tax, so raw ext4
still wins metadata-heavy small-file workloads (~2×). But the mount more than
doubled small-file throughput over the previous CLI/IPC approach, and the cache
makes repeated reads of hot data RAM-fast — both of the user's "make it faster"
goals, within the limits of a user-mode (non-kernel) hook.

### 3. Windows-native drive-letter mount (WinFsp — working)

WinFsp *is* the kernel driver — it ships a pre-signed kernel-mode filesystem
driver, and NeuralFS provides the *user-mode host* against it
([`winfsphost.rs`](crates/neuralfs-daemon/src/winfsphost.rs), the `winfsp`
feature). This mounts NeuralFS as a **real Windows drive letter**, with the OS
routing all file I/O through the WinFsp kernel driver into the daemon.

```powershell
# Build (needs WinFsp + its Developer package, and LLVM/libclang for bindgen):
$env:LIBCLANG_PATH = "C:\path\to\LLVM\bin"
cargo build --release -p neuralfs-daemon --features winfsp

# Mount as drive N: (Ctrl-C to unmount):
.\target\release\neuralfs.exe --mount-winfsp N:
```

**Verified working** against installed WinFsp 2.1: `N:\` appears as a real drive;
create / read / write files, `mkdir`, and directory listing all succeed through
the kernel driver, and the volume unmounts cleanly on stop. The current host is a
self-contained in-memory filesystem proving the kernel-driver integration
end-to-end; backing it with the CoW `neuralfs-fs` engine + the 1 GiB cache (so
the drive letter gets checksums, dedup, snapshots, and hot-file RAM caching) is
the next step, reusing the exact same engine the FUSE mount already uses.

Notes:
- `winfsp` is a GPL-3.0 crate, so a `--features winfsp` build is GPL-licensed;
  the default build pulls none of it.
- `winfsp-sys` finds the SDK via the registry (the WinFsp **Developer** feature
  must be installed so `inc\` and `lib\` are present), and uses `bindgen`, which
  needs `libclang` — point `LIBCLANG_PATH` at an LLVM `bin` directory.

## The continuously-learning AI

The classifier is trained from the index, then **keeps updating online** while
the daemon is alive:

- Every `nfs open` applies a single online SGD step nudging the model toward the
  directory you actually opened — no full retrain needed.
- A background checkpoint loop persists the evolving model to disk whenever its
  version advances, so learning survives restarts.
- Full retrains still fire on startup, reindex, and every `retrain_threshold`
  FS events.
- `nfs ai` shows model version, online-update count, last-saved version,
  classes, and vocabulary size.

## CLI reference

```sh
# learned finder over your real files
nfs find <query>             # ranked matches (AI + frequency)
nfs open <query>             # open top match, record an access (online-learns)
nfs status                   # daemon status, index size, last retrain
nfs reindex                  # full re-index of hooked dirs
nfs hook <dir> | nfs hook    # attach a real dir / list hooked dirs
nfs ai                       # continuously-updated model status
nfs cache                    # both caches: 1 GiB block cache + 500 MiB path TTL cache
nfs config get|set <k> [v]   # lambda, root_dirs, retrain_threshold, ...

# the copy-on-write virtual filesystem
nfs fs write <vpath> <src|-> # write a local file (or stdin) into the volume
nfs fs read  <vpath> [dest]  # read a vpath to stdout or a local file
nfs fs ls    <vpath>
nfs fs mkdir <vpath>
nfs fs rm    <vpath>
nfs fs stat  <vpath>
nfs fs info                  # sizes, unique blocks, dedup ratio, reclaimable
nfs fs snapshot <name>       # O(1) snapshot
nfs fs snapshots
nfs fs rollback <name>
nfs fs scrub                 # verify every block checksum

nfs bench [MiB]              # virtual-fs write/read throughput (default 64)
```

## Daemon lifecycle (Windows)

```sh
neuralfs.exe                 # run in foreground (index + search + AI + CoW volume)
neuralfs.exe --install       # register a logon startup task (Task Scheduler)
neuralfs.exe --uninstall
```

Platform-specific mount modes (feature-gated, so the default build is untouched):

```sh
# Linux/WSL, built --features fuse: passthrough caching mount
neuralfs --mount <mountpoint> --backing <dir> [--cache-mb 1024]

# Windows, built --features winfsp: real drive-letter mount via the WinFsp driver
neuralfs --mount-winfsp N:
neuralfs --winfsp-probe          # confirm the WinFsp driver/library is reachable
```

State lives under `%APPDATA%/neuralfs/`:

```
config.toml      lambda, root_dirs (hooked dirs), retrain_threshold, ...
index.db         sled index + persisted classifier model
volume/          the neuralfs-fs CoW filesystem (data.log + meta)
neuralfs.log
```

## Design notes / deviations

- **Classifier** is a from-scratch TF-IDF + gradient-descent softmax regression
  on `ndarray` (not `linfa`) — small dependency tree, trains in milliseconds at
  the capped vocab/sample sizes, and supports cheap online SGD updates.
- **Open-frequency & learning** are driven by `nfs open` over IPC, since
  Windows' `ReadDirectoryChangesW` (wrapped by `notify`) has no "file opened"
  event. The watcher still handles create/modify/remove to keep the index live.
- **No block GC yet.** The block log is append-only with dedup; deleting or
  overwriting data leaves orphaned blocks (reported as `reclaimable bytes` by
  `nfs fs info`). Compaction is a planned addition.
- **IPC transport** is Windows named pipes (`\\.\pipe\neuralfs`) with a Unix
  socket fallback compiled under `cfg(unix)`. Binary `fs read/write` over IPC is
  capped at 16 MiB; the in-process `bench` bypasses IPC.
- **FUSE hook is Linux/WSL only**, behind the off-by-default `fuse` feature
  (optional `fuser`/`libc` deps), so the default Windows build is untouched. It
  is a *passthrough caching* layer over a real directory, not the CoW volume;
  the CoW volume is reached via `nfs fs`. Windows uses WinFsp (see above).
- **Benchmark honesty.** An earlier `nfs bench` fill repeated every 32 blocks, so
  dedup silently collapsed it and inflated throughput. The fill is now a
  long-period xorshift64 stream (genuinely unique blocks), and the numbers above
  reflect that correction. The FUSE figures come from a clean, disposable WSL2 VM.
```
