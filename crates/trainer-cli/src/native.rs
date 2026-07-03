use crate::config::TrainerConfig;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run_training(cfg: &TrainerConfig, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    std::fs::write(output_dir.join("config.snapshot.toml"), config_snapshot(cfg))?;

    let world_size = read_env_u32("MGT_WORLD_SIZE", 1)?;
    let global_rank = read_env_u32("MGT_GLOBAL_RANK", 0)?;
    let local_rank = read_env_u32("MGT_LOCAL_RANK", global_rank)?;
    let device_id = read_env_u32("MGT_DEVICE_ID", local_rank)?;
    let steps = read_env_u64("MGT_TRAIN_STEPS", cfg.epochs)?;
    let batch_size = read_env_u64("MGT_TRAIN_BATCH_SIZE", cfg.batch_states_per_rank())?;
    let k_min = read_env_u32("MGT_TRAIN_K_MIN", cfg.k_min)?;
    let k_max = read_env_u32("MGT_TRAIN_K_MAX", cfg.k_max)?;
    let hd1 = read_env_u32("MGT_TRAIN_HD1", cfg.hd1)?;
    let hd2 = read_env_u32("MGT_TRAIN_HD2", cfg.hd2)?;
    let nrd = read_env_u32("MGT_TRAIN_NRD", cfg.residual_blocks)?;
    let nccl_id_file = std::env::var_os("MGT_NCCL_ID_FILE").map(PathBuf::from);
    let resume_checkpoint = std::env::var_os("MGT_RESUME_CHECKPOINT").map(PathBuf::from);
    let bin = resolve_native_binary()?;

    if steps == 0 || batch_size == 0 {
        bail!("MGT_TRAIN_STEPS and MGT_TRAIN_BATCH_SIZE must be positive");
    }
    if k_min == 0 || k_min > k_max {
        bail!("invalid MGT_TRAIN_K_MIN/MGT_TRAIN_K_MAX range");
    }

    let mut command = Command::new(&bin);
    command
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--steps")
        .arg(steps.to_string())
        .arg("--device-id")
        .arg(device_id.to_string())
        .arg("--world-size")
        .arg(world_size.to_string())
        .arg("--global-rank")
        .arg(global_rank.to_string())
        .arg("--local-rank")
        .arg(local_rank.to_string())
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--k-min")
        .arg(k_min.to_string())
        .arg("--k-max")
        .arg(k_max.to_string())
        .arg("--hd1")
        .arg(hd1.to_string())
        .arg("--hd2")
        .arg(hd2.to_string())
        .arg("--nrd")
        .arg(nrd.to_string());
    if let Some(path) = nccl_id_file {
        command.arg("--nccl-id-file").arg(path);
    }
    if let Some(path) = resume_checkpoint {
        command.arg("--resume-checkpoint").arg(path);
    }

    let output = command
        .output()
        .with_context(|| format!("failed to launch native trainer {}", bin.display()))?;
    std::fs::write(output_dir.join("native.stdout"), &output.stdout)?;
    std::fs::write(output_dir.join("native.stderr"), &output.stderr)?;
    if !output.status.success() {
        bail!("native trainer failed with status {}", output.status);
    }
    require_file(output_dir.join("metadata.env"))?;
    require_file(output_dir.join("layers.json"))?;
    require_file(output_dir.join("train.log"))?;
    require_file(output_dir.join("profile.jsonl"))?;
    require_file(output_dir.join("weights").join("manifest.json"))?;
    require_file(output_dir.join("weights").join("weights.f32.bin"))?;
    require_file(output_dir.join("checkpoint").join("manifest.json"))?;
    require_file(output_dir.join("checkpoint").join("state.f32.bin"))?;
    println!("native_training_ok");
    Ok(())
}

fn resolve_native_binary() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MGT_NATIVE_TRAIN_BIN") {
        return Ok(PathBuf::from(path));
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let candidates = [
        format!("build-gpu-smoke/mgt_native_train{exe}"),
        format!("build-kaggle-2xt4/mgt_native_train{exe}"),
        format!("build-gpu-smoke/mgt_native_train_smoke{exe}"),
        format!("build-kaggle-2xt4/mgt_native_train_smoke{exe}"),
    ];
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    bail!("native trainer binary not found; set MGT_NATIVE_TRAIN_BIN or build mgt_native_train")
}

fn require_file(path: PathBuf) -> Result<()> {
    let metadata = std::fs::metadata(&path).with_context(|| format!("missing artifact {}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        bail!("artifact {} is empty or not a file", path.display());
    }
    Ok(())
}

fn read_env_u32(name: &str, default_value: u32) -> Result<u32> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u32>()
            .with_context(|| format!("{name} must be an unsigned integer")),
        Err(std::env::VarError::NotPresent) => Ok(default_value),
        Err(err) => Err(err).with_context(|| format!("failed to read {name}")),
    }
}

fn read_env_u64(name: &str, default_value: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .with_context(|| format!("{name} must be an unsigned integer")),
        Err(std::env::VarError::NotPresent) => Ok(default_value),
        Err(err) => Err(err).with_context(|| format!("failed to read {name}")),
    }
}

fn config_snapshot(cfg: &TrainerConfig) -> String {
    format!(
        "group_id = {}\ntarget_id = {}\nstate_len = {}\nstate_alignment = {}\nstate_value_pad = {}\nmove_count = {}\noutput_dim = {}\nhd1 = {}\nhd2 = {}\nresidual_blocks = {}\nwalkers_per_depth = {}\nk_min = {}\nk_max = {}\nepochs = {}\nlearning_rate = {}\nweight_decay = {}\nadam_beta1 = {}\nadam_beta2 = {}\nadam_eps = {}\nbase_seed = \"0x{:016x}\"\ncheckpoint_period_steps = {}\nweight_export_period_steps = {}\n",
        cfg.group_id,
        cfg.target_id,
        cfg.state_len,
        cfg.state_alignment,
        cfg.state_value_pad,
        cfg.move_count,
        cfg.output_dim,
        cfg.hd1,
        cfg.hd2,
        cfg.residual_blocks,
        cfg.walkers_per_depth,
        cfg.k_min,
        cfg.k_max,
        cfg.epochs,
        cfg.learning_rate,
        cfg.weight_decay,
        cfg.adam_beta1,
        cfg.adam_beta2,
        cfg.adam_eps,
        cfg.base_seed,
        cfg.checkpoint_period_steps,
        cfg.weight_export_period_steps,
    )
}