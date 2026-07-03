#[path = "../src/config.rs"]
mod config;
#[path = "../src/native.rs"]
mod native;

use config::TrainerConfig;
use std::fs;

#[test]
fn training_entry_writes_rank_artifacts() {
    let output = std::env::temp_dir().join(format!("mgt-native-artifacts-{}", std::process::id()));
    if output.exists() {
        fs::remove_dir_all(&output).unwrap();
    }

    let cfg = TrainerConfig::p888_default();
    native::run_training(&cfg, &output).unwrap();

    let metadata = fs::read_to_string(output.join("metadata.env")).unwrap();
    assert!(metadata.contains("MODEL_MODE=MLP2RB"));
    assert!(metadata.contains("OUTPUT_DIM=1"));
    assert!(metadata.contains("WEIGHT_DECAY=0"));

    let layers = fs::read_to_string(output.join("layers.json")).unwrap();
    assert!(layers.contains("\"hd1\": 2556"));
    assert!(layers.contains("\"output_dim\": 1"));

    let log = fs::read_to_string(output.join("train.log")).unwrap();
    assert!(log.contains("rank=0"));
    assert!(log.contains("phase=entry_ready"));

    fs::remove_dir_all(output).unwrap();
}