use trainer_cli::single_gpu_ffi::{SingleGpuConfig, SingleGpuTrainer};

#[test]
fn ffi_layout_and_raii_owner() {
    assert_eq!(SingleGpuConfig::ABI_VERSION, 1);
    assert_eq!(SingleGpuConfig::RAW_SIZE, 48);
    let mut config = SingleGpuConfig::default();
    config.capacity_rows = 0;
    let error = SingleGpuTrainer::create(config).unwrap_err();
    assert!(error.to_string().contains("creation"));

    let mut trainer = SingleGpuTrainer::create(SingleGpuConfig::default()).unwrap();
    trainer.prepare().unwrap();
    let metrics = trainer.train_step(4, 1).unwrap();
    assert_eq!(metrics.completed_sequence, 1);
    assert_eq!(metrics.optimizer_step, 1);
    assert!(metrics.loss.is_finite());
}
