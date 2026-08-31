use anyhow::{bail, Result};
use std::ffi::{c_char, c_void, CString};
use std::fmt;
use std::ptr;

const STATUS_OK: i32 = 0;

#[repr(C)]
struct RawConfig {
    struct_size: u32,
    abi_version: u32,
    device_id: u32,
    capacity_rows: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    reserved_u32: [u32; 3],
    group_json_utf8: *const c_char,
    target_bin_utf8: *const c_char,
    base_seed: u64,
    k_min: u32,
    k_max: u32,
}

#[repr(C)]
struct RawExecutionOptions {
    struct_size: u32,
    abi_version: u32,
    execution_mode: u32,
    reserved_u32: u32,
}

#[repr(C)]
struct RawStep {
    struct_size: u32,
    active_rows: u32,
    optimizer_step: u64,
    semantic_epoch: u64,
    epoch_sample_offset: u64,
}

#[repr(C)]
struct RawMetrics {
    struct_size: u32,
    reserved_u32: u32,
    completed_sequence: u64,
    optimizer_step: u64,
    loss: f32,
    reserved_f32: f32,
}

unsafe extern "C" {
    fn mgt_single_gpu_v1_create(config: *const RawConfig, out: *mut *mut c_void) -> i32;
    fn mgt_single_gpu_v1_create_with_options(
        config: *const RawConfig,
        options: *const RawExecutionOptions,
        out: *mut *mut c_void,
    ) -> i32;
    fn mgt_single_gpu_v1_prepare(handle: *mut c_void) -> i32;
    fn mgt_single_gpu_v1_train_step(
        handle: *mut c_void,
        step: *const RawStep,
        metrics: *mut RawMetrics,
    ) -> i32;
    fn mgt_single_gpu_v1_read_metrics(handle: *mut c_void, metrics: *mut RawMetrics) -> i32;
    fn mgt_single_gpu_v1_checkpoint(handle: *mut c_void, path: *const c_char) -> i32;
    fn mgt_single_gpu_v1_destroy(handle: *mut *mut c_void) -> i32;
    fn mgt_single_gpu_v1_last_error(
        handle: *mut c_void,
        destination: *mut c_char,
        capacity: usize,
    ) -> usize;
}

#[derive(Debug, Clone)]
pub struct SingleGpuConfig {
    pub device_id: u32,
    pub capacity_rows: u32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub group_json: String,
    pub target_bin: String,
    pub base_seed: u64,
    pub k_min: u32,
    pub k_max: u32,
}

impl SingleGpuConfig {
    pub const ABI_VERSION: u32 = 1;
    pub const RAW_SIZE: usize = std::mem::size_of::<RawConfig>();
}

impl Default for SingleGpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            capacity_rows: 4,
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            group_json: "native/tests/fixtures/p888.json".into(),
            target_bin: "native/tests/fixtures/p888-target.bin".into(),
            base_seed: 0x8881,
            k_min: 1,
            k_max: 29,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SingleGpuMetrics {
    pub completed_sequence: u64,
    pub optimizer_step: u64,
    pub loss: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SingleGpuExecutionMode {
    Eager = 0,
    /// Capture the configured capacity once; smaller valid batches use eager.
    /// Requires graph support in the linked native library (CUDA >=12.8).
    FixedBatchGraph = 1,
}

impl SingleGpuExecutionMode {
    pub const RAW_OPTIONS_SIZE: usize = std::mem::size_of::<RawExecutionOptions>();
}

pub struct SingleGpuTrainer {
    handle: *mut c_void,
}

unsafe impl Send for SingleGpuTrainer {}

impl fmt::Debug for SingleGpuTrainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleGpuTrainer")
            .field("owned", &(!self.handle.is_null()))
            .finish()
    }
}

fn error_text(handle: *mut c_void, context: &str, status: i32) -> anyhow::Error {
    let mut bytes = vec![0u8; 4096];
    unsafe {
        mgt_single_gpu_v1_last_error(handle, bytes.as_mut_ptr().cast::<c_char>(), bytes.len());
    }
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    let native = String::from_utf8_lossy(&bytes[..end]);
    anyhow::anyhow!("{context} failed ({status}): {native}")
}

impl SingleGpuTrainer {
    pub fn create(config: SingleGpuConfig) -> Result<Self> {
        Self::create_with_mode(config, SingleGpuExecutionMode::Eager)
    }

    pub fn create_with_mode(config: SingleGpuConfig, mode: SingleGpuExecutionMode) -> Result<Self> {
        let group_json = CString::new(config.group_json)?;
        let target_bin = CString::new(config.target_bin)?;
        let raw = RawConfig {
            struct_size: std::mem::size_of::<RawConfig>() as u32,
            abi_version: SingleGpuConfig::ABI_VERSION,
            device_id: config.device_id,
            capacity_rows: config.capacity_rows,
            learning_rate: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            weight_decay: config.weight_decay,
            reserved_u32: [0; 3],
            group_json_utf8: group_json.as_ptr(),
            target_bin_utf8: target_bin.as_ptr(),
            base_seed: config.base_seed,
            k_min: config.k_min,
            k_max: config.k_max,
        };
        let mut handle = ptr::null_mut();
        let options = RawExecutionOptions {
            struct_size: std::mem::size_of::<RawExecutionOptions>() as u32,
            abi_version: SingleGpuConfig::ABI_VERSION,
            execution_mode: mode as u32,
            reserved_u32: 0,
        };
        let status = unsafe {
            match mode {
                SingleGpuExecutionMode::Eager => mgt_single_gpu_v1_create(&raw, &mut handle),
                SingleGpuExecutionMode::FixedBatchGraph => {
                    mgt_single_gpu_v1_create_with_options(&raw, &options, &mut handle)
                }
            }
        };
        if status != STATUS_OK || handle.is_null() {
            return Err(error_text(ptr::null_mut(), "trainer creation", status));
        }
        Ok(Self { handle })
    }

    pub fn prepare(&mut self) -> Result<()> {
        let status = unsafe { mgt_single_gpu_v1_prepare(self.handle) };
        if status != STATUS_OK {
            return Err(error_text(self.handle, "trainer prepare", status));
        }
        Ok(())
    }

    pub fn train_step(
        &mut self,
        active_rows: u32,
        optimizer_step: u64,
        semantic_epoch: u64,
        epoch_sample_offset: u64,
    ) -> Result<SingleGpuMetrics> {
        let step = RawStep {
            struct_size: std::mem::size_of::<RawStep>() as u32,
            active_rows,
            optimizer_step,
            semantic_epoch,
            epoch_sample_offset,
        };
        let mut metrics = RawMetrics {
            struct_size: std::mem::size_of::<RawMetrics>() as u32,
            reserved_u32: 0,
            completed_sequence: 0,
            optimizer_step: 0,
            loss: 0.0,
            reserved_f32: 0.0,
        };
        let status = unsafe { mgt_single_gpu_v1_train_step(self.handle, &step, &mut metrics) };
        if status != STATUS_OK {
            return Err(error_text(self.handle, "train step", status));
        }
        Ok(SingleGpuMetrics {
            completed_sequence: metrics.completed_sequence,
            optimizer_step: metrics.optimizer_step,
            loss: metrics.loss,
        })
    }

    /// Enqueue without a metrics copy or per-step synchronization. Use
    /// read_metrics to observe asynchronous execution errors; Drop waits for
    /// outstanding work but cannot report destruction errors.
    pub fn enqueue_step(
        &mut self,
        active_rows: u32,
        optimizer_step: u64,
        semantic_epoch: u64,
        epoch_sample_offset: u64,
    ) -> Result<()> {
        let step = RawStep {
            struct_size: std::mem::size_of::<RawStep>() as u32,
            active_rows,
            optimizer_step,
            semantic_epoch,
            epoch_sample_offset,
        };
        let status = unsafe { mgt_single_gpu_v1_train_step(self.handle, &step, ptr::null_mut()) };
        if status != STATUS_OK {
            return Err(error_text(self.handle, "step enqueue", status));
        }
        Ok(())
    }

    /// Synchronize the latest submitted step and copy its loss/sequence.
    pub fn read_metrics(&mut self) -> Result<SingleGpuMetrics> {
        let mut raw = RawMetrics {
            struct_size: std::mem::size_of::<RawMetrics>() as u32,
            reserved_u32: 0,
            completed_sequence: 0,
            optimizer_step: 0,
            loss: 0.0,
            reserved_f32: 0.0,
        };
        let status = unsafe { mgt_single_gpu_v1_read_metrics(self.handle, &mut raw) };
        if status != STATUS_OK {
            return Err(error_text(self.handle, "metrics read", status));
        }
        Ok(SingleGpuMetrics {
            completed_sequence: raw.completed_sequence,
            optimizer_step: raw.optimizer_step,
            loss: raw.loss,
        })
    }

    pub fn checkpoint(&mut self, directory: &str) -> Result<()> {
        let path = CString::new(directory)?;
        let status = unsafe { mgt_single_gpu_v1_checkpoint(self.handle, path.as_ptr()) };
        if status != STATUS_OK {
            bail!(error_text(self.handle, "checkpoint", status));
        }
        Ok(())
    }
}

impl Drop for SingleGpuTrainer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                mgt_single_gpu_v1_destroy(&mut self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}
