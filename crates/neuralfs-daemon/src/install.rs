use anyhow::{bail, Result};

const TASK_NAME: &str = "NeuralFS";

/// Registers the daemon to start automatically at user logon via Windows
/// Task Scheduler. No-op (with an explanatory error) on non-Windows targets.
pub fn install() -> Result<()> {
    #[cfg(windows)]
    {
        let exe = std::env::current_exe()?;
        let exe_str = exe.to_string_lossy();
        let status = std::process::Command::new("schtasks")
            .args([
                "/Create",
                "/TN",
                TASK_NAME,
                "/TR",
                &format!("\"{exe_str}\""),
                "/SC",
                "ONLOGON",
                "/RL",
                "LIMITED",
                "/F",
            ])
            .status()?;
        if !status.success() {
            bail!("schtasks /Create failed with status {status}");
        }
        println!("Installed NeuralFS startup task ({TASK_NAME}). It will launch at next logon.");
        Ok(())
    }
    #[cfg(not(windows))]
    {
        bail!("--install is only supported on Windows (Task Scheduler)");
    }
}

pub fn uninstall() -> Result<()> {
    #[cfg(windows)]
    {
        let status = std::process::Command::new("schtasks")
            .args(["/Delete", "/TN", TASK_NAME, "/F"])
            .status()?;
        if !status.success() {
            bail!("schtasks /Delete failed with status {status}");
        }
        println!("Removed NeuralFS startup task ({TASK_NAME}).");
        Ok(())
    }
    #[cfg(not(windows))]
    {
        bail!("--uninstall is only supported on Windows (Task Scheduler)");
    }
}
