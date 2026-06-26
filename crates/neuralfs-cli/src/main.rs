mod client;

use anyhow::Result;
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
    /// Find a file, returns ranked list of paths
    Find { query: Vec<String> },
    /// Find + open the top result in the OS default app
    Open { query: Vec<String> },
    /// Show daemon status, index size, last retrain time
    Status,
    /// Trigger a full re-index
    Reindex,
    /// Get or set config values (lambda, root dirs, etc.)
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    Set { key: String, value: String },
    Get { key: String },
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
                None => match resp.error {
                    Some(err) => println!("{err}"),
                    None => println!("no match found"),
                },
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
                let resp = client::send(Request::ConfigGet { key }).await?;
                print_message_or_error(&resp);
            }
            ConfigAction::Set { key, value } => {
                let resp = client::send(Request::ConfigSet { key, value }).await?;
                print_message_or_error(&resp);
            }
        },
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
