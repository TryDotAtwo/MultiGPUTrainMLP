use trainer_cli::single_gpu_ffi::{SingleGpuConfig, SingleGpuExecutionMode, SingleGpuTrainer};

#[test]
fn ffi_layout_and_raii_owner() {
    assert_eq!(SingleGpuConfig::ABI_VERSION, 1);
    assert_eq!(SingleGpuConfig::RAW_SIZE, 80);
    assert_eq!(SingleGpuExecutionMode::RAW_OPTIONS_SIZE, 16);
    let mut config = SingleGpuConfig::default();
    config.capacity_rows = 0;
    let error = SingleGpuTrainer::create(config).unwrap_err();
    assert!(error.to_string().contains("creation"));

    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../native/tests/fixtures");
    let mut valid = SingleGpuConfig::default();
    valid.group_json = fixtures.join("p888.json").to_string_lossy().into_owned();
    valid.target_bin = fixtures
        .join("p888-target.bin")
        .to_string_lossy()
        .into_owned();
    let mut trainer = SingleGpuTrainer::create(valid.clone()).unwrap();
    trainer.prepare().unwrap();
    let metrics = trainer.train_step(4, 1, 0, 0).unwrap();
    assert_eq!(metrics.completed_sequence, 1);
    assert_eq!(metrics.optimizer_step, 1);
    assert!(metrics.loss.is_finite());
    drop(trainer);

    valid.group_json = fixtures
        .join("../../production_inputs/p888.json")
        .to_string_lossy()
        .into_owned();
    let mut graph =
        SingleGpuTrainer::create_with_mode(valid, SingleGpuExecutionMode::FixedBatchGraph).unwrap();
    assert!(graph.read_metrics().is_err());
    graph.prepare().unwrap();
    assert_eq!(graph.read_metrics().unwrap().optimizer_step, 0);
    graph.enqueue_step(4, 1, 0, 0).unwrap();
    graph.enqueue_step(3, 2, 0, 4).unwrap();
    graph.enqueue_step(4, 3, 1, 0).unwrap();
    let last = graph.read_metrics().unwrap();
    assert_eq!(last.completed_sequence, 3);
    assert_eq!(last.optimizer_step, 3);
    assert!(last.loss.is_finite());
    assert!(graph.enqueue_step(4, 5, 1, 4).is_err());
    assert_eq!(graph.read_metrics().unwrap().optimizer_step, 3);
    graph.enqueue_step(4, 4, 1, 4).unwrap();
    drop(graph); // Drains the outstanding graph before releasing its arena.
}
