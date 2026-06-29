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

## Build variants (4 binaries)

NeuralFS ships as three named variants from one shared codebase (the brand is
passed into a shared `entry()` at runtime, so the variants never duplicate
logic — they differ only in which mount compiles in):

| Variant | Binary | Targets | Native mount |
|---|---|---|---|
| **NeuralFS Cross** | `neuralfs-cross` | ELF **and** PE | FUSE on Linux; none on Windows (portable, GPL-free) |
| **NeuralFS Windows** | `neuralfs-windows` | PE | WinFsp kernel drive-letter mount + Windows tuning |
| **NeuralFS Linux** | `neuralfs-linux` | ELF | FUSE (`fuse.ko`) VFS mount + Linux tuning |

```sh
cargo build --release -p neuralfs-cross               # portable (ELF or PE per host)
cargo build --release -p neuralfs-linux               # Linux/ELF, needs fuse3
cargo build --release -p neuralfs-windows             # Windows/PE, needs WinFsp SDK + libclang
./target/release/neuralfs-cross --version             # -> "NeuralFS Cross 0.1.0"
```

> **On "in-kernel" Linux.** NeuralFS Linux uses FUSE, and that *is* the
> in-kernel-driver path on Linux: `fuse.ko` is a real kernel module in mainline
> Linux. FUSE keeps the **filesystem logic** in userspace, which is exactly what
> lets the Linux variant reuse the CoW engine, the AI, and the caches, and stay
> crash-safe (a bug can't panic the kernel). A "pure" in-kernel FS would be a
> from-scratch C kernel module that could reuse none of that and could panic the
> kernel — so FUSE is the deliberate choice, not a fallback.

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
- **Garbage collection** — `nfs fs gc` reclaims storage orphaned by overwritten
  or deleted files. Without it the append-only `data.log` only ever grows; GC
  builds the set of blocks still reachable from the live root *and every snapshot
  root* (snapshots pin data, so they count), then compacts the log down to just
  those (restic's `forget`+`prune` model). It's stop-the-world — a daemon
  `fs_gate` lock serializes it against writes — and integrity-checked throughout.
  Measured in a release container: 20 overwrites of a 10 MiB file left 4,114
  blocks of which only 160 were live; GC reclaimed all **257 MiB of orphans in
  0.196 s (~1.3 GB/s compaction)**, bringing reclaimable bytes to zero.
- **Small-file fast path (inline data)** — files at or below 4 KiB are stored
  **directly in the inode**, skipping the block store entirely: no blake3 hash,
  no blocks-tree lookup/insert, no `data.log` append. (This is what ext4/Btrfs/
  NTFS do for tiny "resident" files.) Verified: a ≤4 KiB file uses **zero**
  blocks. This speeds up the engine / `nfs fs` for small files; it does *not*
  apply to the FUSE passthrough, which writes to backing files, not the engine.
- **Speed** — append-only block log + the RAM cache below. Honest `nfs bench 64`
  on this machine: **~258 MB/s write to disk** (CoW + blake3 + sled metadata,
  competitive with raw ext4's ~226 MB/s while doing strictly more work), and
  cache-warm reads at **~1.8 GB/s**. A release `nfs bench 256` in a Linux
  container measured **294 MB/s write / 1063 MB/s cache-warm read**.

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
cargo build --release -p neuralfs-linux        # or: --features fuse
neuralfs-linux --mount /mnt/nfs --backing /data/store --cache-mb 1024
```

This is a **real filesystem** routed through the in-kernel `fuse.ko` driver: the
OS sends file I/O for everything under `/mnt/nfs` to the NeuralFS daemon.
Metadata operations pass through to the backing directory at native speed; file
reads go through the 1 GiB frequency-aware cache, so a file opened often enough
is served from RAM.

**Tuned for throughput.** The mount enables FUSE **writeback caching**
(`FUSE_WRITEBACK_CACHE`), keeps the kernel page cache across opens
(`FOPEN_KEEP_CACHE`), and raises `max_write`/`max_readahead` to 1 MiB. Net effect
in a privileged Linux container: sequential write went from **66 MB/s → 191 MB/s
(~2.9×)**. Honest caveat: writeback caching helps *throughput* (big writes), not
metadata-heavy *small-file* create/open/close, which stay bound by per-file
kernel↔userspace round trips (and are noisy run-to-run). The only thing that
removes that round-trip tax is an in-kernel module — a deliberate non-goal here,
since it would abandon the engine, AI, and crash-safety (see the variants note
above).

**Tuned for metadata, too.** On top of throughput, the mount hands the kernel
longer **entry/attr cache timeouts** (5 s, up from 1 s) so repeated `lookup`/
`getattr` of the same paths are answered from the kernel's own dcache instead of
upcalling the daemon, and it raises the async request ceiling
(`max_background` 12 → 64, with a matching congestion threshold) so readahead and
writeback pipeline more deeply. Measured before/after in a privileged container
(20 000 files, median of 3 runs): a repeated full-tree stat traversal went from
**4.21 s → 4.04 s (~4–8% faster)**. Honest scope: this only helps workloads that
*re-touch* the same metadata (editor re-stat, `git status`, incremental builds);
a single-pass create-many run was unchanged (~17 s either way), because it is
round-trip-bound, exactly as the FUSE literature predicts. Caching is also safe
for the AI — the classifier learns from `find`/`open` and the inotify watcher,
never from these FUSE upcalls, so caching them harder does not starve the model.

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

**Re-measured in two isolated, disposable Linux Docker containers** (privileged,
`--device /dev/fuse`; A mounts ext4 over a loopback file, B mounts NeuralFS via
FUSE over a backing dir — a FUSE crash here is userspace, no BSOD risk):

| workload | Container A: raw ext4 | Container B: NeuralFS (FUSE) |
|---|---|---|
| sequential write (256 MiB) | **223 MB/s** | 66 MB/s |
| sequential read | 743 cold / 7159 warm MB/s | 2035 MB/s |
| small-file create (500 × 2 KiB) | **942 files/s** | 286 files/s |
| small-file read | **1240 files/s** | 640 files/s |
| small-file delete | **1163 files/s** | 911 files/s |
| hot 200 MiB file, repeat read | — | 628 → **1301 MB/s** (cold → RAM-cached) |

Same conclusion as the WSL2 run, reproduced in a cleaner container A/B setup: ext4
wins raw throughput (native syscalls vs. FUSE's kernel↔userspace round trip per
op), and NeuralFS's edge is the RAM cache turning repeat reads of hot data ~2×
faster than even ext4's cold read. Container A/B note: container B's sequential
*write* (66 MB/s) is the FUSE-mount path specifically — the underlying CoW engine
itself does ~258 MB/s in-process (`nfs bench`); the gap is FUSE syscall overhead,
not the storage engine.

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

**Metadata caching.** The volume sets WinFsp's metadata cache timeouts —
`FileInfoTimeout`, `DirInfoTimeout`, `VolumeInfoTimeout` (10 s) and
`SecurityTimeout` (60 s, since the descriptor is a process-lifetime constant),
up from the original 1 s. Within each window the kernel answers a query without
a round trip down to the user-mode host, and on Windows that round trip is *two
process context switches* — WinFsp's own documented bottleneck for repeated
opens and stats. This is safe to cache aggressively here because the volume is
authoritative and only ever mutated *through* WinFsp, so there is no out-of-band
writer to go stale against. (Compile-verified against winfsp 2.1; a drive-letter
throughput number awaits the engine-backed host.)

Notes:
- `winfsp` is a GPL-3.0 crate, so a `--features winfsp` build is a GPL-3.0
  combined work; the default build and `--features fuse` pull none of it and
  stay Apache-2.0. See [Licensing](#licensing) and [NOTICE](NOTICE).
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
- **Usage-weighted retrains (gets better over long-term use).** When the index
  is larger than the training budget (`MAX_SAMPLES`), a retrain doesn't take a
  positional stride (an arbitrary slice of one store snapshot) — it draws a
  **weighted reservoir sample** (Efraimidis–Spirakis A-Res) keyed by
  `(freq+1)·e^(−λ·age)`, the same recency/frequency shape as the runtime scorer.
  So the model trains on a representative sample of how you *actually* use your
  files across all history — recent habits are favoured (tracking drift) while
  the long tail still gets sampled. This is what lets it keep *improving*, not
  just stay bounded, the longer it runs.
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
nfs fs gc                    # reclaim storage orphaned by overwrites/deletes

nfs bench [MiB]              # virtual-fs write/read throughput (default 64)
```

## Daemon lifecycle

Any variant binary (`neuralfs-cross` / `neuralfs-windows` / `neuralfs-linux`)
takes the same flags; substitute the one you built:

```sh
neuralfs-cross                  # run in foreground (index + search + AI + CoW volume)
neuralfs-cross --version        # print the variant brand + version
neuralfs-cross --install        # Windows: register a logon startup task (Task Scheduler)
neuralfs-cross --uninstall

# NeuralFS Linux (ELF, fuse): in-kernel-fuse.ko mount, FS logic in userspace
neuralfs-linux --mount <mountpoint> --backing <dir> [--cache-mb 1024]

# NeuralFS Windows (PE, winfsp): real drive-letter mount via the WinFsp driver
neuralfs-windows --mount-winfsp N:
neuralfs-windows --winfsp-probe   # confirm the WinFsp driver/library is reachable
```

**Responsive startup.** The initial full re-index of `root_dirs` runs in the
background, so the daemon answers commands immediately instead of blocking on it.
Measured in a release container with `root_dirs = /usr` (~25,500 files): the IPC
socket was up and `nfs status` answered in **0.41 s** while that index ran behind
it. A persisted model loads synchronously, so the AI is ready at once when one
exists; `find`/`open` work during the warm-up via the exact-name fast path and
the recency scorer.

State lives under `%APPDATA%/neuralfs/`:

```
config.toml      lambda, root_dirs (hooked dirs), retrain_threshold, ...
index.db         sled index + persisted classifier model
volume/          the neuralfs-fs CoW filesystem (data.log + meta)
neuralfs.log
```

## Design notes / deviations

- **Variant structure.** One shared codebase: `neuralfs-daemon` is a *library*
  exposing `entry(brand)`, and three tiny binary crates (`neuralfs-cross/
  -windows/-linux`) each call it with their brand and pull in the right mount
  feature. The brand is passed at runtime (not a compile-time feature) on
  purpose — Cargo unifies a shared lib's features across crates built together,
  which would otherwise collapse a feature-derived brand to one value. Platform
  variants are excluded from `default-members` so a plain `cargo build` never
  tries to compile WinFsp (needs the SDK) or `fuser` (Linux) on the wrong host.
- **Search exact-match fast path.** The index keeps a `by_name` tree
  (lowercased filename → paths). A `find` whose query is exactly an indexed
  filename returns straight from it — no classifier, no scan. This is the
  "drop the AI when a trivial exact match wins" shortcut.
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

## Licensing

NeuralFS is **Apache-2.0** (see [LICENSE](LICENSE)) — the default build and any
`--features fuse` build are Apache-2.0 in their entirety; every dependency they
pull in is permissively licensed (MIT, Apache-2.0, BSD, ISC, CC0-1.0,
Unicode-3.0, plus one unmodified transitive MPL-2.0 dependency that doesn't
affect this project's own licensing).

**One carve-out:** the optional `winfsp` feature (the Windows drive-letter
mount, [`winfsphost.rs`](crates/neuralfs-daemon/src/winfsphost.rs)) links the
`winfsp`/`winfsp-sys` crates, which are **GPL-3.0** — licensed by their authors,
not by this project, and not something this project can relicense. A binary
built with `--features winfsp` is therefore a GPL-3.0 combined work and subject
to GPL-3.0's terms as a whole. Builds without that feature (the default, and
`--features fuse`) are unaffected. Full details in [NOTICE](NOTICE).
