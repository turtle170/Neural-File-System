# NeuralFS

A learned, frequency-aware file system helper for Windows: a background
daemon that indexes your files, learns which directories a query is likely
to resolve to (TF-IDF + softmax/logistic regression), and ranks results by
an exponential-decay open-frequency score. A small CLI (`nfs`) talks to the
daemon over a named pipe.

## Workspace layout

```
neuralfs/
├── Cargo.toml                  # workspace
├── crates/
│   ├── neuralfs-daemon/        # daemon binary -> neuralfs.exe
│   │   └── src/
│   │       ├── main.rs         # startup, retrain loop, periodic flush
│   │       ├── indexer.rs      # walkdir-based filesystem indexer
│   │       ├── classifier.rs   # TF-IDF vectorizer + softmax regression
│   │       ├── scorer.rs       # frequency + exponential decay scoring
│   │       ├── search.rs       # predict -> backtrack -> fallback lookup
│   │       ├── watcher.rs      # notify-based FS event watcher
│   │       ├── ipc.rs          # named pipe / Unix socket IPC server
│   │       ├── store.rs        # sled-backed persistence
│   │       ├── config.rs       # config.toml load/save
│   │       ├── logging.rs      # file logger
│   │       ├── install.rs      # Task Scheduler register/unregister
│   │       └── state.rs        # shared daemon state
│   └── neuralfs-cli/            # CLI binary -> nfs.exe
│       └── src/
│           ├── main.rs
│           └── client.rs       # IPC client
```

## Building

```sh
cargo build --release --workspace
```

Binaries land at `target/release/neuralfs.exe` and `target/release/nfs.exe`.

## Running

```sh
# start the daemon in the foreground (Ctrl+C to stop)
./target/release/neuralfs.exe

# register it to launch at logon via Windows Task Scheduler
./target/release/neuralfs.exe --install
./target/release/neuralfs.exe --uninstall

# query it
./target/release/nfs.exe find quarterly report
./target/release/nfs.exe open quarterly report
./target/release/nfs.exe status
./target/release/nfs.exe reindex
./target/release/nfs.exe config get lambda
./target/release/nfs.exe config set lambda 0.2
```

Config lives at `%APPDATA%/neuralfs/config.toml`, the index/model at
`%APPDATA%/neuralfs/index.db` (sled), logs at `%APPDATA%/neuralfs/neuralfs.log`.

## Design notes / deviations from the original spec

- **Classifier**: implemented as a from-scratch TF-IDF vectorizer + softmax
  (multinomial logistic) regression trained by gradient descent on `ndarray`,
  rather than pulling in `linfa`/`linfa-logistic`. This keeps the dependency
  tree small, trains in milliseconds for the capped vocab/sample sizes used
  here, and gives full control over the top-3 confidence output the lookup
  algorithm needs.
- **Open-frequency tracking**: Windows' `ReadDirectoryChangesW` (which the
  `notify` crate wraps) has no "file opened" event, so frequency/recency is
  bumped when the CLI's `nfs open` command resolves and opens a file via IPC,
  not via passive FS watching. The watcher still handles create/modify/remove
  to keep the index and parent-directory lookups current, and triggers a
  classifier retrain every `retrain_threshold` events.
- **IPC transport**: Windows named pipes (`\\.\pipe\neuralfs`) are the primary
  transport; a Unix domain socket fallback is compiled in under `cfg(unix)`
  for portability, though Windows is the only tested/primary target.
