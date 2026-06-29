mod client;

use std::io::{Read, Write};

use anyhow::{Context, Result};
use base64::Engine;
use clap::{Parser, Subcommand};

use client::{Request, Response};

#[derive(Parser)]
#[command(name = "nfs", about = "NeuralFS CLI - query the neuralfs daemon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Find a file in the indexed real filesystem, ranked by the AI + frequency
    Find { query: Vec<String> },
    /// Find + open the top result in the OS default app (records an access)
    Open { query: Vec<String> },
    /// Show daemon status, index size, last retrain time
    Status,
    /// Trigger a full re-index of all hooked directories
    Reindex,
    /// Get or set config values (lambda, root dirs, etc.)
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Operate on the NeuralFS copy-on-write virtual filesystem volume
    Fs {
        #[command(subcommand)]
        action: FsAction,
    },
    /// Hook NeuralFS onto a real directory of your filesystem (index + watch + learn)
    Hook {
        /// Directory to attach to; omit to show currently hooked directories
        dir: Option<String>,
    },
    /// Show the continuously-updated AI model status
    Ai,
    /// Show RAM cache (ZFS-ARC-style, 1 GiB cap) statistics
    Cache,
    /// Benchmark virtual filesystem throughput (write/read MB/s)
    Bench {
        /// Payload size in MiB
        #[arg(default_value_t = 64)]
        mb: usize,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    Set { key: String, value: String },
    Get { key: String },
}

#[derive(Subcommand)]
enum FsAction {
    /// Write a local file (or stdin with '-') to a virtual path
    Write {
        vpath: String,
        /// Local source file, or '-' for stdin
        source: String,
    },
    /// Read a virtual path to stdout (or to a local file)
    Read {
        vpath: String,
        /// Optional local destination file; default is stdout
        dest: Option<String>,
    },
    /// List a virtual directory
    Ls { vpath: String },
    /// Create a virtual directory (parents created as needed)
    Mkdir { vpath: String },
    /// Remove a virtual file or directory
    Rm { vpath: String },
    /// Show metadata for a virtual path
    Stat { vpath: String },
    /// Show volume statistics (sizes, blocks, dedup ratio)
    Info,
    /// Create a named snapshot
    Snapshot { name: String },
    /// List snapshots
    Snapshots,
    /// Roll back the volume to a named snapshot
    Rollback { name: String },
    /// Verify every block's checksum (ZFS-style scrub)
    Scrub,
    /// Reclaim storage orphaned by overwritten/deleted files (compact the log)
    Gc,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli).await {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Find { query } => {
            let resp = client::send(Request::Find {
                query: query.join(" "),
            })
            .await?;
            print_results(&resp);
        }
        Command::Open { query } => {
            let resp = client::send(Request::Open {
                query: query.join(" "),
            })
            .await?;
            match resp.opened_path {
                Some(path) => {
                    println!("opening {path}");
                    open_in_os(&path)?;
                }
                None => println!("{}", resp.error.as_deref().unwrap_or("no match found")),
            }
        }
        Command::Status => {
            let resp = client::send(Request::Status).await?;
            println!("status: {}", resp.status.unwrap_or_else(|| "unknown".into()));
            println!("indexed_files: {}", resp.indexed_files.unwrap_or(0));
            println!(
                "last_retrain: {}",
                resp.last_retrain.unwrap_or_else(|| "never".into())
            );
        }
        Command::Reindex => {
            let resp = client::send(Request::Reindex).await?;
            println!(
                "{}",
                resp.message.unwrap_or_else(|| "reindex triggered".into())
            );
        }
        Command::Config { action } => match action {
            ConfigAction::Get { key } => {
                print_message_or_error(&client::send(Request::ConfigGet { key }).await?);
            }
            ConfigAction::Set { key, value } => {
                print_message_or_error(&client::send(Request::ConfigSet { key, value }).await?);
            }
        },
        Command::Fs { action } => run_fs(action).await?,
        Command::Hook { dir } => match dir {
            Some(dir) => print_message_or_error(&client::send(Request::Hook { dir }).await?),
            None => print_lines(&client::send(Request::HookStatus).await?),
        },
        Command::Ai => print_lines(&client::send(Request::Ai).await?),
        Command::Cache => print_lines(&client::send(Request::Cache).await?),
        Command::Bench { mb } => print_lines(&client::send(Request::Bench { mb }).await?),
    }
    Ok(())
}

async fn run_fs(action: FsAction) -> Result<()> {
    match action {
        FsAction::Write { vpath, source } => {
            let data = if source == "-" {
                let mut buf = Vec::new();
                std::io::stdin().read_to_end(&mut buf)?;
                buf
            } else {
                std::fs::read(&source).with_context(|| format!("reading {source}"))?
            };
            let data_b64 = base64::engine::general_purpose::STANDARD.encode(&data);
            print_message_or_error(
                &client::send(Request::FsWrite {
                    path: vpath,
                    data_b64,
                })
                .await?,
            );
        }
        FsAction::Read { vpath, dest } => {
            let resp = client::send(Request::FsRead { path: vpath }).await?;
            if let Some(err) = resp.error {
                println!("error: {err}");
                return Ok(());
            }
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(resp.data_b64.unwrap_or_default().as_bytes())?;
            match dest {
                Some(path) => {
                    std::fs::write(&path, &bytes)?;
                    println!("wrote {} bytes to {path}", bytes.len());
                }
                None => {
                    std::io::stdout().write_all(&bytes)?;
                }
            }
        }
        FsAction::Ls { vpath } => print_lines(&client::send(Request::FsLs { path: vpath }).await?),
        FsAction::Mkdir { vpath } => {
            print_message_or_error(&client::send(Request::FsMkdir { path: vpath }).await?)
        }
        FsAction::Rm { vpath } => {
            print_message_or_error(&client::send(Request::FsRm { path: vpath }).await?)
        }
        FsAction::Stat { vpath } => {
            print_lines(&client::send(Request::FsStat { path: vpath }).await?)
        }
        FsAction::Info => print_lines(&client::send(Request::FsInfo).await?),
        FsAction::Snapshot { name } => {
            print_message_or_error(&client::send(Request::FsSnapshot { name }).await?)
        }
        FsAction::Snapshots => print_lines(&client::send(Request::FsSnapshots).await?),
        FsAction::Rollback { name } => {
            print_message_or_error(&client::send(Request::FsRollback { name }).await?)
        }
        FsAction::Scrub => print_lines(&client::send(Request::FsScrub).await?),
        FsAction::Gc => print_lines(&client::send(Request::FsGc).await?),
    }
    Ok(())
}

fn print_results(resp: &Response) {
    if let Some(err) = &resp.error {
        println!("error: {err}");
        return;
    }
    match &resp.results {
        Some(results) if !results.is_empty() => {
            for r in results {
                println!("{:>8.3}  {}", r.score, r.path);
            }
        }
        _ => println!("no matches found"),
    }
}

fn print_lines(resp: &Response) {
    if let Some(err) = &resp.error {
        println!("error: {err}");
        return;
    }
    match &resp.lines {
        Some(lines) => {
            for l in lines {
                println!("{l}");
            }
        }
        None => print_message_or_error(resp),
    }
}

fn print_message_or_error(resp: &Response) {
    match &resp.message {
        Some(m) => println!("{m}"),
        None => println!("{}", resp.error.as_deref().unwrap_or("unknown error")),
    }
}

#[cfg(windows)]
fn open_in_os(path: &str) -> Result<()> {
    std::process::Command::new("cmd")
        .args(["/C", "start", "", path])
        .spawn()?;
    Ok(())
}

#[cfg(unix)]
fn open_in_os(path: &str) -> Result<()> {
    std::process::Command::new("xdg-open").arg(path).spawn()?;
    Ok(())
}
