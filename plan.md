\# NeuralFS — Build Plan for Claude Code



\## Project Overview

Build \*\*NeuralFS\*\*: a Rust daemon + CLI tool that acts as a learned, frequency-aware file system plugin. It indexes file paths and metadata, uses a TF-IDF + logistic regression classifier to predict file locations on query, backtracks to parent directories on miss, and ranks results using an exponential decay frequency score.



\---



\## Target Platform

\- Windows (primary) — ships as `neuralfs.exe` (daemon) + `nfs.exe` (CLI)

\- Cross-platform Rust code where possible, but Windows service integration is priority

\- Daemon launches at startup via Windows Task Scheduler or Service



\---



\## Workspace Structure

```

neuralfs/

├── Cargo.toml                  # workspace

├── crates/

│   ├── neuralfs-daemon/        # daemon binary (neuralfs.exe)

│   │   ├── Cargo.toml

│   │   └── src/

│   │       ├── main.rs

│   │       ├── indexer.rs      # walks FS, builds index

│   │       ├── classifier.rs   # TF-IDF vectorizer + logistic regression

│   │       ├── scorer.rs       # frequency + exponential decay scoring

│   │       ├── watcher.rs      # file system event watcher (notify crate)

│   │       ├── ipc.rs          # named pipe / Unix socket IPC server

│   │       └── store.rs        # persists index + model to disk (sled or bincode)

│   └── neuralfs-cli/           # CLI binary (nfs.exe)

│       ├── Cargo.toml

│       └── src/

│           ├── main.rs

│           └── client.rs       # sends queries to daemon over IPC

└── README.md

```



\---



\## Core Dependencies

```toml

\[dependencies]

\# Daemon

notify = "6"             # cross-platform FS watcher

sled = "0.34"            # embedded key-value store for index persistence

bincode = "1"            # fast serialization

serde = { version = "1", features = \["derive"] }

tokio = { version = "1", features = \["full"] }

walkdir = "2"            # recursive directory walker

linfa = "0.7"            # ML framework (logistic regression)

linfa-logistic = "0.7"   # logistic regression implementation

ndarray = "0.15"         # required by linfa

tf-idf = "0.1"           # or implement manually (see below)

chrono = "0.4"           # timestamps for decay

log = "0.4"

env\_logger = "0.10"



\# CLI

clap = { version = "4", features = \["derive"] }

```

> Note: If `tf-idf` crate is insufficient, implement TF-IDF manually — it's straightforward (term frequency × inverse document frequency over path tokens).



\---



\## Module Specifications



\### 1. `indexer.rs`

\- Walk a root directory (configurable, default: user home) using `walkdir`

\- For each file collect:

&#x20; - Full path (string)

&#x20; - Filename + extension

&#x20; - Parent directory path

&#x20; - File size, modified timestamp

&#x20; - Depth from root

\- Store entries in `sled` DB keyed by path hash

\- Re-index on startup and on FS change events from watcher



\### 2. `classifier.rs` — TF-IDF + Logistic Regression

\*\*Vectorization:\*\*

\- Tokenize paths by splitting on `/`, `\\`, `\_`, `-`, `.`, spaces

\- Build TF-IDF matrix over path tokens across all indexed files

\- Each file = one document; tokens = path components + filename parts



\*\*Training:\*\*

\- Label = directory bucket (top-N most common parent dirs become classes)

\- Train `linfa` logistic regression on the TF-IDF matrix

\- Retrain triggered by: startup, re-index, or every 500 new file events



\*\*Prediction:\*\*

\- Query string → tokenize → TF-IDF vector → predict class (directory)

\- Return top-3 predicted directories with confidence scores



\### 3. `scorer.rs` — Frequency + Exponential Decay

```

score(file) = base\_freq × e^(-λ × hours\_since\_last\_open) + classifier\_confidence

```

\- `λ = 0.1` (default, configurable) — controls cooldown speed

\- `base\_freq` = total open count for this file

\- On file open event: increment `base\_freq`, reset `last\_open` timestamp

\- Score is computed at query time, not stored (derived from stored freq + timestamp)

\- Files/dirs sorted descending by score when presenting results



\### 4. `watcher.rs`

\- Use `notify` crate to watch configured root directories

\- On `Create` / `Modify` / `Remove` events:

&#x20; - Update index entry

&#x20; - Update frequency score

&#x20; - Queue classifier retrain if event count threshold exceeded

\- On `Access` event (file opened):

&#x20; - Increment frequency counter for that path

&#x20; - Update last-opened timestamp



\### 5. `store.rs`

\- `sled` tree structure:

&#x20; - `index` tree: `path\_hash → FileEntry { path, metadata, freq, last\_open }`

&#x20; - `model` tree: serialized trained classifier weights (bincode)

&#x20; - `config` tree: λ, root dirs, retrain threshold

\- Load on daemon startup, flush periodically (every 60s) and on shutdown



\### 6. `ipc.rs` — Daemon ↔ CLI Communication

\- Use a \*\*named pipe\*\* on Windows (`\\\\.\\pipe\\neuralfs`) / Unix socket on Linux

\- JSON protocol over the pipe:

```json

// CLI → Daemon

{ "cmd": "find", "query": "quarterly report" }

{ "cmd": "status" }

{ "cmd": "reindex" }



// Daemon → CLI

{ "results": \[ { "path": "C:/...", "score": 0.94 }, ... ] }

{ "status": "running", "indexed\_files": 42301 }

```

\- Daemon runs async IPC server (tokio)

\- CLI connects, sends command, prints response, exits



\---



\## Lookup Algorithm (implement in `main.rs` or `search.rs`)

```

fn find(query: \&str) -> Vec<ScoredPath>:

&#x20; 1. predicted\_dirs = classifier.predict\_top3(query)

&#x20; 2. for each predicted\_dir (sorted by confidence):

&#x20;      candidates = index.files\_in(predicted\_dir)

&#x20;      matches = fuzzy\_match(query, candidates)   // simple contains / token overlap

&#x20;      if matches not empty:

&#x20;          return sort\_by\_score(matches)

&#x20;      else:

&#x20;          parent = predicted\_dir.parent()

&#x20;          repeat with parent (max 4 levels up)

&#x20; 3. if still no match:

&#x20;      fallback = index.search\_all(query)         // linear scan sorted by score

&#x20;      return sort\_by\_score(fallback)

```



\---



\## CLI (`nfs.exe`) Commands

```

nfs find <query>          # find a file, returns ranked list of paths

nfs open <query>          # find + open the top result in OS default app

nfs status                # show daemon status, index size, last retrain time

nfs reindex               # trigger full re-index

nfs config set <key> <v>  # set config values (lambda, root dirs, etc.)

nfs config get <key>

```



\---



\## Daemon Startup (Windows)

\- Register as a Windows startup task via Task Scheduler XML or a simple registry entry

\- On first run: `neuralfs.exe --install` creates the startup task

\- On `neuralfs.exe --uninstall` removes it

\- Log to `%APPDATA%/neuralfs/neuralfs.log`

\- Store DB at `%APPDATA%/neuralfs/index.db`



\---



\## Configuration File

`%APPDATA%/neuralfs/config.toml`:

```toml

root\_dirs = \["C:/Users/username", "D:/Projects"]

lambda = 0.1

retrain\_threshold = 500   # retrain after N file events

max\_results = 10

log\_level = "info"

```



\---



\## Build Order for Claude Code

Implement in this exact order:

1\. Workspace `Cargo.toml` + both crate skeletons

2\. `store.rs` — get persistence working first

3\. `indexer.rs` — walk + store file entries

4\. `scorer.rs` — frequency + decay math

5\. `classifier.rs` — TF-IDF vectorizer, then logistic regression training + prediction

6\. `watcher.rs` — hook FS events into indexer + scorer

7\. `ipc.rs` — named pipe server (daemon side)

8\. `neuralfs-daemon/main.rs` — wire everything together, startup logic

9\. `neuralfs-cli/` — IPC client + clap CLI

10\. Windows startup registration (`--install` / `--uninstall`)



\---



\## Key Constraints

\- Daemon must use <50MB RAM at idle after index is built

\- Query response must return in <100ms for indexes up to 500k files

\- Classifier retrain must be async — never block the IPC server

\- All errors must be logged, never panic in daemon

\- Index must survive daemon restart (persisted in sled)

