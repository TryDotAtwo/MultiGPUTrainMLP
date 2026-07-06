use crate::config::TrainerConfig;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run_training(cfg: &TrainerConfig, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    std::fs::write(
        output_dir.join("config.snapshot.toml"),
        config_snapshot(cfg),
    )?;

    let world_size = read_env_u32("MGT_WORLD_SIZE", 1)?;
    let global_rank = read_env_u32("MGT_GLOBAL_RANK", 0)?;
    let local_rank = read_env_u32("MGT_LOCAL_RANK", global_rank)?;
    let device_id = read_env_u32("MGT_DEVICE_ID", local_rank)?;
    let steps = read_env_u64("MGT_TRAIN_STEPS", cfg.epochs)?;
    let batch_size = read_env_u64("MGT_TRAIN_BATCH_SIZE", cfg.batch_states_per_rank())?;
    let k_min = read_env_u32("MGT_TRAIN_K_MIN", cfg.k_min)?;
    let k_max = read_env_u32("MGT_TRAIN_K_MAX", cfg.k_max)?;
    let group_id = read_env_u32("MGT_GROUP_ID", cfg.group_id)?;
    let target_id = read_env_u32("MGT_TARGET_ID", cfg.target_id)?;
    let state_len = read_env_u32("MGT_STATE_LEN", cfg.state_len)?;
    let state_value_count = read_env_u32("MGT_STATE_VALUE_COUNT", cfg.state_value_pad)?;
    let move_count = read_env_u32("MGT_MOVE_COUNT", cfg.move_count)?;
    let state_alignment = read_env_u32("MGT_STATE_ALIGNMENT", cfg.state_alignment)?;
    let hd1 = read_env_u32("MGT_TRAIN_HD1", cfg.hd1)?;
    let hd2 = read_env_u32("MGT_TRAIN_HD2", cfg.hd2)?;
    let nrd = read_env_u32("MGT_TRAIN_NRD", cfg.residual_blocks)?;
    let output_dim = read_env_u32("MGT_OUTPUT_DIM", cfg.output_dim)?;
    let hidden_alignment = read_env_u32("MGT_HIDDEN_ALIGNMENT", cfg.hidden_alignment)?;
    let walkers = read_env_u32("MGT_WALKERS", cfg.walkers)?;
    let gradient_carousel_slots =
        read_env_u32("MGT_GRADIENT_CAROUSEL_SLOTS", cfg.gradient_carousel_slots)?;
    let input_grad_partial_chunks = read_env_u32(
        "MGT_INPUT_GRAD_PARTIAL_CHUNKS",
        cfg.input_grad_partial_chunks,
    )?;
    let input_grad_positions_per_block = read_env_u32(
        "MGT_INPUT_GRAD_POSITIONS_PER_BLOCK",
        cfg.input_grad_positions_per_block,
    )?;
    let input_grad_sparse = read_env_bool("MGT_INPUT_GRAD_SPARSE", cfg.input_grad_sparse)?;
    let input_grad_fp16 = read_env_bool("MGT_INPUT_GRAD_FP16", cfg.input_grad_fp16)?;
    let linear_fp16 = read_env_bool("MGT_LINEAR_FP16", cfg.linear_fp16)?;
    let overlap_allreduce = read_env_bool("MGT_OVERLAP_ALLREDUCE", true)?;
    let allreduce_bucket_bytes =
        read_env_u64("MGT_ALLREDUCE_BUCKET_BYTES", cfg.allreduce_bucket_bytes)?;
    let seed = read_env_u64("MGT_SEED", cfg.base_seed)?;
    let learning_rate = read_env_f32("MGT_LR", cfg.learning_rate)?;
    let weight_decay = read_env_f32("MGT_WEIGHT_DECAY", cfg.weight_decay)?;
    let write_artifacts = read_env_bool("MGT_WRITE_ARTIFACTS", true)?;
    let nccl_id_file = std::env::var_os("MGT_NCCL_ID_FILE").map(PathBuf::from);
    let resume_checkpoint = std::env::var_os("MGT_RESUME_CHECKPOINT").map(PathBuf::from);
    let bin = resolve_native_binary()?;

    if steps == 0 || steps > u32::MAX as u64 || batch_size == 0 || batch_size > u32::MAX as u64 {
        bail!("MGT_TRAIN_STEPS and MGT_TRAIN_BATCH_SIZE must be positive u32 values");
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
        .arg("--group-id")
        .arg(group_id.to_string())
        .arg("--target-id")
        .arg(target_id.to_string())
        .arg("--state-len")
        .arg(state_len.to_string())
        .arg("--state-value-count")
        .arg(state_value_count.to_string())
        .arg("--move-count")
        .arg(move_count.to_string())
        .arg("--state-alignment")
        .arg(state_alignment.to_string())
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--k-min")
        .arg(k_min.to_string())
        .arg("--k-max")
        .arg(k_max.to_string())
        .arg("--walkers")
        .arg(walkers.to_string())
        .arg("--hd1")
        .arg(hd1.to_string())
        .arg("--hd2")
        .arg(hd2.to_string())
        .arg("--nrd")
        .arg(nrd.to_string())
        .arg("--output-dim")
        .arg(output_dim.to_string())
        .arg("--hidden-alignment")
        .arg(hidden_alignment.to_string())
        .arg("--gradient-carousel-slots")
        .arg(gradient_carousel_slots.to_string())
        .arg("--input-grad-partial-chunks")
        .arg(input_grad_partial_chunks.to_string())
        .arg("--input-grad-positions-per-block")
        .arg(input_grad_positions_per_block.to_string())
        .arg("--input-grad-sparse")
        .arg(if input_grad_sparse { "1" } else { "0" })
        .arg("--input-grad-fp16")
        .arg(if input_grad_fp16 { "1" } else { "0" })
        .arg("--linear-fp16")
        .arg(if linear_fp16 { "1" } else { "0" })
        .arg("--overlap-allreduce")
        .arg(if overlap_allreduce { "1" } else { "0" })
        .arg("--allreduce-bucket-bytes")
        .arg(allreduce_bucket_bytes.to_string())
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--lr")
        .arg(learning_rate.to_string())
        .arg("--weight-decay")
        .arg(weight_decay.to_string())
        .arg("--write-artifacts")
        .arg(if write_artifacts { "1" } else { "0" });
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
    if write_artifacts {
        require_file(output_dir.join("weights").join("manifest.json"))?;
        require_file(output_dir.join("weights").join("weights.f32.bin"))?;
        require_file(output_dir.join("checkpoint").join("manifest.json"))?;
        require_file(output_dir.join("checkpoint").join("state.f32.bin"))?;
    }
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
        format!("build-cuda-plan-docker/mgt_native_train{exe}"),
        format!("build-kaggle-2xt4/mgt_native_train{exe}"),
        format!("build-gpu-smoke/mgt_native_train_smoke{exe}"),
        format!("build-cuda-plan-docker/mgt_native_train_smoke{exe}"),
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
    let metadata =
        std::fs::metadata(&path).with_context(|| format!("missing artifact {}", path.display()))?;
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

fn read_env_f32(name: &str, default_value: f32) -> Result<f32> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<f32>()
            .with_context(|| format!("{name} must be a float")),
        Err(std::env::VarError::NotPresent) => Ok(default_value),
        Err(err) => Err(err).with_context(|| format!("failed to read {name}")),
    }
}

fn read_env_bool(name: &str, default_value: bool) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => bail!("{name} must be one of 1,0,true,false,yes,no,on,off"),
        },
        Err(std::env::VarError::NotPresent) => Ok(default_value),
        Err(err) => Err(err).with_context(|| format!("failed to read {name}")),
    }
}

fn config_snapshot(cfg: &TrainerConfig) -> String {
    format!(
        "group_id = {}\ntarget_id = {}\nstate_len = {}\nstate_alignment = {}\nstate_storage_len = {}\nnum_classes = {}\nmove_count = {}\noutput_dim = {}\nhd1 = {}\nhd2 = {}\nresidual_blocks = {}\nhidden_alignment = {}\nbatch_size = {}\nwalkers = {}\nk_min = {}\nk_max = {}\nepochs = {}\nlearning_rate = {}\nweight_decay = {}\nadam_beta1 = {}\nadam_beta2 = {}\nadam_eps = {}\nbase_seed = \"0x{:016x}\"\ncheckpoint_period_steps = {}\nweight_export_period_steps = {}\ngradient_carousel_slots = {}\ninput_grad_partial_chunks = {}\ninput_grad_positions_per_block = {}\ninput_grad_sparse = {}\ninput_grad_fp16 = {}\nlinear_fp16 = {}\noverlap_allreduce = true\nallreduce_bucket_bytes = {}\n",
        cfg.group_id,
        cfg.target_id,
        cfg.state_len,
        cfg.state_alignment,
        cfg.state_storage_len(),
        cfg.state_value_pad,
        cfg.move_count,
        cfg.output_dim,
        cfg.hd1,
        cfg.hd2,
        cfg.residual_blocks,
        cfg.hidden_alignment,
        cfg.batch_size,
        cfg.walkers,
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
        cfg.gradient_carousel_slots,
        cfg.input_grad_partial_chunks,
        cfg.input_grad_positions_per_block,
        cfg.input_grad_sparse,
        cfg.input_grad_fp16,
        cfg.linear_fp16,
        cfg.allreduce_bucket_bytes,
    )
}
