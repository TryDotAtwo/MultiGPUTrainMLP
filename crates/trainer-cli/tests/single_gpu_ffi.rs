use trainer_cli::single_gpu_ffi::{SingleGpuConfig, SingleGpuTrainer};

#[test]
fn ffi_layout_and_raii_owner() {
    assert_eq!(SingleGpuConfig::ABI_VERSION, 1);
    assert_eq!(SingleGpuConfig::RAW_SIZE, 80);
    let mut config = SingleGpuConfig::default();
    config.capacity_rows = 0;
    let error = SingleGpuTrainer::create(config).unwrap_err();
    assert!(error.to_string().contains("creation"));

    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../native/tests/fixtures");
    let mut valid = SingleGpuConfig::default();
    valid.group_json = fixtures.join("p888.json").to_string_lossy().into_owned();
    valid.target_bin = fixtures.join("p888-target.bin").to_string_lossy().into_owned();
    let mut trainer = SingleGpuTrainer::create(valid).unwrap();
    trainer.prepare().unwrap();
    let metrics = trainer.train_step(4, 1).unwrap();
    assert_eq!(metrics.completed_sequence, 1);
    assert_eq!(metrics.optimizer_step, 1);
    assert!(metrics.loss.is_finite());
}
