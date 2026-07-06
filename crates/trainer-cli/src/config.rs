use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainerConfig {
    pub group_id: u32,
    pub target_id: u32,
    pub state_len: u32,
    pub state_alignment: u32,
    pub state_value_pad: u32,
    pub move_count: u32,
    pub output_dim: u32,
    pub hd1: u32,
    pub hd2: u32,
    pub residual_blocks: u32,
    pub hidden_alignment: u32,
    pub batch_size: u64,
    pub walkers: u32,
    pub k_min: u32,
    pub k_max: u32,
    pub epochs: u64,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub base_seed: u64,
    pub checkpoint_period_steps: u64,
    pub weight_export_period_steps: u64,
    pub gradient_carousel_slots: u32,
    pub input_grad_partial_chunks: u32,
    pub input_grad_positions_per_block: u32,
    pub input_grad_sparse: bool,
    pub input_grad_fp16: bool,
    pub linear_fp16: bool,
    pub allreduce_bucket_bytes: u64,
}

impl TrainerConfig {
    pub fn p888_default() -> Self {
        Self {
            group_id: 888,
            target_id: 0,
            state_len: 72,
            state_alignment: 16,
            state_value_pad: 72,
            move_count: 18,
            output_dim: 1,
            hd1: 2556,
            hd2: 218,
            residual_blocks: 16,
            hidden_alignment: 8,
            batch_size: 100_000,
            walkers: 34_482,
            k_min: 1,
            k_max: 29,
            epochs: 32692,
            learning_rate: 0.0001,
            weight_decay: 0.0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1.0e-8,
            base_seed: 0x0888_0000_0000_0001,
            checkpoint_period_steps: 4096,
            weight_export_period_steps: 4096,
            gradient_carousel_slots: 3,
            input_grad_partial_chunks: 1,
            input_grad_positions_per_block: 1,
            input_grad_sparse: false,
            input_grad_fp16: false,
            linear_fp16: false,
            allreduce_bucket_bytes: 4 * 1024 * 1024,
        }
    }

    pub fn state_storage_len(&self) -> u32 {
        round_up(self.state_len, self.state_alignment)
    }

    pub fn batch_states_per_rank(&self) -> u64 {
        self.batch_size
    }

    pub fn validate(&self) -> Result<()> {
        if self.state_len == 0 {
            bail!("state_len must be positive");
        }
        if self.state_alignment == 0 || !self.state_alignment.is_power_of_two() {
            bail!("state_alignment must be a non-zero power of two");
        }
        if self.state_value_pad == 0 {
            bail!("state_value_pad must be positive");
        }
        if self.move_count == 0 {
            bail!("move_count must be positive");
        }
        if self.output_dim != 1 {
            bail!("output_dim must be 1 for the current native trainer");
        }
        if self.hd1 == 0 || self.hd2 == 0 {
            bail!("hidden dimensions must be positive");
        }
        if self.hidden_alignment == 0 || !self.hidden_alignment.is_power_of_two() {
            bail!("hidden_alignment must be a non-zero power of two");
        }
        if self.batch_size == 0 || self.batch_size > u32::MAX as u64 {
            bail!("batch_size must fit into u32 and be positive");
        }
        if self.walkers == 0 {
            bail!("walkers must be positive");
        }
        if self.k_min == 0 || self.k_min > self.k_max {
            bail!("invalid walk depth range");
        }
        if self.epochs == 0 || self.epochs > u32::MAX as u64 {
            bail!("epochs must fit into u32 and be positive");
        }
        if self.learning_rate <= 0.0 {
            bail!("learning_rate must be positive");
        }
        if self.weight_decay < 0.0 {
            bail!("weight_decay must be non-negative");
        }
        if !(0.0..1.0).contains(&self.adam_beta1) {
            bail!("adam_beta1 must be in [0, 1)");
        }
        if !(0.0..1.0).contains(&self.adam_beta2) {
            bail!("adam_beta2 must be in [0, 1)");
        }
        if self.adam_eps <= 0.0 {
            bail!("adam_eps must be positive");
        }
        if self.gradient_carousel_slots == 0 {
            bail!("gradient_carousel_slots must be positive");
        }
        if self.input_grad_partial_chunks == 0 {
            bail!("input_grad_partial_chunks must be positive");
        }
        if self.input_grad_positions_per_block == 0 {
            bail!("input_grad_positions_per_block must be positive");
        }
        if self.allreduce_bucket_bytes == 0 {
            bail!("allreduce_bucket_bytes must be positive");
        }
        Ok(())
    }
}

fn round_up(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    ((value + alignment - 1) / alignment) * alignment
}
