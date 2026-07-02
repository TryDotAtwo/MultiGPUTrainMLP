use crate::config::TrainerConfig;
use anyhow::Result;
use std::path::Path;

pub fn run_training(cfg: &TrainerConfig, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let snapshot = toml::to_string_pretty(cfg)?;
    std::fs::write(output_dir.join("config.snapshot.toml"), snapshot)?;
    println!("native_training_entry_ready");
    Ok(())
}