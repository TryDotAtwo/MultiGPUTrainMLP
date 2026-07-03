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
    pub walkers_per_depth: u32,
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
}

impl TrainerConfig {
    pub fn p888_default() -> Self {
        Self {
            group_id: 888,
            target_id: 0,
            state_len: 72,
            state_alignment: 16,
            state_value_pad: 128,
            move_count: 18,
            output_dim: 1,
            hd1: 2556,
            hd2: 218,
            residual_blocks: 16,
            walkers_per_depth: 3449,
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
        }
    }

    pub fn state_storage_len(&self) -> u32 {
        let min_len = self.state_len.max(self.state_len + 4);
        round_up(min_len, self.state_alignment)
    }

    pub fn batch_states_per_rank(&self) -> u64 {
        self.walkers_per_depth as u64 * (self.k_max - self.k_min + 1) as u64
    }

    pub fn validate(&self) -> Result<()> {
        if self.group_id != 888 {
            bail!("group_id must be 888 for version one");
        }
        if self.target_id != 0 {
            bail!("target_id must be 0 for version one");
        }
        if self.state_len != 72 {
            bail!("state_len must be 72 for version one");
        }
        if self.state_storage_len() != 80 {
            bail!("state_storage_len must be 80 for p888");
        }
        if self.state_alignment != 16 {
            bail!("state_alignment must be 16");
        }
        if self.state_value_pad != 128 {
            bail!("state_value_pad must be 128");
        }
        if self.move_count != 18 {
            bail!("move_count must be 18");
        }
        if self.output_dim != 1 {
            bail!("output_dim must be 1");
        }
        if self.k_min == 0 || self.k_min > self.k_max {
            bail!("invalid walk depth range");
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
        if self.batch_states_per_rank() == 0 {
            bail!("batch_states_per_rank must be positive");
        }
        Ok(())
    }
}

fn round_up(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}