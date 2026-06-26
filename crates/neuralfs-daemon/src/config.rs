use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub root_dirs: Vec<String>,
    pub lambda: f64,
    pub retrain_threshold: usize,
    pub max_results: usize,
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        let home = dirs::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "C:/".to_string());
        Config {
            root_dirs: vec![home],
            lambda: 0.1,
            retrain_threshold: 500,
            max_results: 10,
            log_level: "info".to_string(),
        }
    }
}

impl Config {
    pub fn load_or_create(path: &Path) -> Result<Config> {
        if path.exists() {
            let text = fs::read_to_string(path)?;
            Ok(toml::from_str(&text)?)
        } else {
            let cfg = Config::default();
            cfg.save(path)?;
            Ok(cfg)
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        fs::write(path, text)?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "lambda" => Some(self.lambda.to_string()),
            "retrain_threshold" => Some(self.retrain_threshold.to_string()),
            "max_results" => Some(self.max_results.to_string()),
            "log_level" => Some(self.log_level.clone()),
            "root_dirs" => Some(self.root_dirs.join(";")),
            _ => None,
        }
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "lambda" => self.lambda = value.parse()?,
            "retrain_threshold" => self.retrain_threshold = value.parse()?,
            "max_results" => self.max_results = value.parse()?,
            "log_level" => self.log_level = value.to_string(),
            "root_dirs" => self.root_dirs = value.split(';').map(|s| s.to_string()).collect(),
            _ => anyhow::bail!("unknown config key: {key}"),
        }
        Ok(())
    }
}
