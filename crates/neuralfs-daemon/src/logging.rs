use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use log::{LevelFilter, Log, Metadata, Record};

struct FileLogger {
    file: Mutex<std::fs::File>,
    level: LevelFilter,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let line = format!("[{now}] [{}] {}\n", record.level(), record.args());
        if let Ok(mut f) = self.file.lock() {
            let _ = f.write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}

/// Initializes a process-wide file logger writing to `log_path`. Never panics;
/// returns an error if the log file can't be opened so the caller can decide
/// how to proceed.
pub fn init(log_path: &Path, level: LevelFilter) -> anyhow::Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(log_path)?;
    let logger = FileLogger {
        file: Mutex::new(file),
        level,
    };
    log::set_boxed_logger(Box::new(logger)).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    log::set_max_level(level);
    Ok(())
}

pub fn level_from_str(s: &str) -> LevelFilter {
    match s.to_lowercase().as_str() {
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "warn" => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _ => LevelFilter::Info,
    }
}
