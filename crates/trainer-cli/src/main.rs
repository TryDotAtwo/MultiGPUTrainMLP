mod config;
mod native;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::TrainerConfig;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mgt")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    PrintDefaultConfig,
    ValidateConfig {
        path: PathBuf,
    },
    Train {
        config: PathBuf,
        output_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::PrintDefaultConfig => {
            let text = toml::to_string_pretty(&TrainerConfig::p888_default())?;
            println!("{text}");
        }
        Command::ValidateConfig { path } => {
            let text = std::fs::read_to_string(path)?;
            let cfg: TrainerConfig = toml::from_str(&text)?;
            cfg.validate()?;
            println!("config_ok");
        }
        Command::Train { config, output_dir } => {
            let text = std::fs::read_to_string(config)?;
            let cfg: TrainerConfig = toml::from_str(&text)?;
            cfg.validate()?;
            native::run_training(&cfg, &output_dir)?;
        }
    }
    Ok(())
}
