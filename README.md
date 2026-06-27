# NeuralFS

NeuralFS is two things in one daemon:

1. **A learned file finder** — it indexes your real files, learns which
   directories a query resolves to (TF-IDF + softmax/logistic regression),
   ranks results by an exponential-decay open-frequency score, and **keeps
   learning online from every access while it runs**.
2. **A real filesystem engine** — a lightweight copy-on-write,
   content-addressed, checksummed object store (`neuralfs-fs`): ZFS-like
   integrity, O(1) snapshots, and transparent block dedup, but far lighter.

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
│   │       ├── ipc.rs          # named-pipe server: find/open/fs/hook/ai/bench
│   │       ├── store.rs        # sled-backed index persistence
│   │       └── ...             # config, logging, install, state, protocol
│   └── neuralfs-cli/           # CLI binary -> nfs.exe
└── README.md
```

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
- **Speed** — append-only block log + LRU block cache. Measured on this machine
  (`nfs bench 256`): **~1.8 GB/s write, ~2.0 GB/s read *including* blake3
  verification.**

## Hooking onto your real filesystem

```sh
nfs hook "C:/Users/me/Documents"   # index it, watch it live, learn from it
nfs hook                           # show currently hooked directories
```

`hook` attaches NeuralFS to a real directory at runtime: it indexes the tree,
starts a live `notify` watcher on it, and turns on access-driven learning. This
is a **userspace hook** — NeuralFS becomes the smart access/search layer over
your real files.

> **Kernel mount (future).** Exposing NeuralFS as an actual mounted drive
> letter (so the OS routes *all* file I/O through it) requires a kernel
> filesystem driver. The supported path is [WinFsp](https://winfsp.dev/) (the
> Windows FUSE equivalent): a `mount` mode would back a WinFsp volume with the
> `neuralfs-fs` engine. It needs the WinFsp driver installed and signed, so it
> is documented here as the extension point rather than shipped untested.

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
neuralfs.exe                 # run in foreground
neuralfs.exe --install       # register a logon startup task (Task Scheduler)
neuralfs.exe --uninstall
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
```
