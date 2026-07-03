use crate::config::TrainerConfig;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct LayerManifest {
    model_mode: &'static str,
    group_id: u32,
    target_id: u32,
    state_len: u32,
    state_storage_len: u32,
    state_value_pad: u32,
    output_dim: u32,
    hd1: u32,
    hd2: u32,
    residual_blocks: u32,
    num_parameters: u64,
}

pub fn run_training(cfg: &TrainerConfig, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    std::fs::write(output_dir.join("config.snapshot.toml"), config_snapshot(cfg))?;

    let world_size = read_env_u32("MGT_WORLD_SIZE", 1)?;
    let global_rank = read_env_u32("MGT_GLOBAL_RANK", 0)?;
    let local_rank = read_env_u32("MGT_LOCAL_RANK", global_rank)?;
    let device_id = read_env_u32("MGT_DEVICE_ID", local_rank)?;

    let manifest = LayerManifest {
        model_mode: "MLP2RB",
        group_id: cfg.group_id,
        target_id: cfg.target_id,
        state_len: cfg.state_len,
        state_storage_len: cfg.state_storage_len(),
        state_value_pad: cfg.state_value_pad,
        output_dim: cfg.output_dim,
        hd1: cfg.hd1,
        hd2: cfg.hd2,
        residual_blocks: cfg.residual_blocks,
        num_parameters: estimate_parameters(cfg),
    };
    let layers = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(output_dir.join("layers.json"), layers)?;

    let metadata = format!(
        "MODEL_MODE=MLP2RB\nGROUP_ID={}\nTARGET_ID={}\nOUTPUT_DIM={}\nHD1={}\nHD2={}\nRESIDUAL_BLOCKS={}\nNUM_PARAMETERS={}\nK_MIN={}\nK_MAX={}\nBATCH_STATES_PER_RANK={}\nLR={}\nWEIGHT_DECAY={}\nWORLD_SIZE={}\nGLOBAL_RANK={}\nLOCAL_RANK={}\nDEVICE_ID={}\n",
        cfg.group_id,
        cfg.target_id,
        cfg.output_dim,
        cfg.hd1,
        cfg.hd2,
        cfg.residual_blocks,
        manifest.num_parameters,
        cfg.k_min,
        cfg.k_max,
        cfg.batch_states_per_rank(),
        cfg.learning_rate,
        cfg.weight_decay,
        world_size,
        global_rank,
        local_rank,
        device_id,
    );
    std::fs::write(output_dir.join("metadata.env"), metadata)?;

    let log = format!(
        "rank={} local_rank={} device={} world_size={} phase=entry_ready batch_states={} output_dim={} weight_decay={}\n",
        global_rank,
        local_rank,
        device_id,
        world_size,
        cfg.batch_states_per_rank(),
        cfg.output_dim,
        cfg.weight_decay,
    );
    std::fs::write(output_dir.join("train.log"), log)?;
    println!("native_training_entry_ready");
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

fn estimate_parameters(cfg: &TrainerConfig) -> u64 {
    let input = cfg.state_len as u64 * cfg.state_value_pad as u64;
    let hd1 = cfg.hd1 as u64;
    let hd2 = cfg.hd2 as u64;
    let output = cfg.output_dim as u64;
    let input_block = input * hd1 + hd1 + hd1 * hd2 + hd2;
    let residual = cfg.residual_blocks as u64 * (hd2 * hd2 + hd2 + hd2 * hd2 + hd2);
    let head = hd2 * output + output;
    input_block + residual + head
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