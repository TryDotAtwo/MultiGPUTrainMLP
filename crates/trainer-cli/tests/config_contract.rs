#[path = "../src/config.rs"]
mod config;

use config::TrainerConfig;

#[test]
fn p888_default_matches_static_contract() {
    let cfg = TrainerConfig::p888_default();
    assert_eq!(cfg.group_id, 888);
    assert_eq!(cfg.target_id, 0);
    assert_eq!(cfg.state_len, 72);
    assert_eq!(cfg.state_storage_len(), 80);
    assert_eq!(cfg.state_alignment, 16);
    assert_eq!(cfg.state_value_pad, 72);
    assert_eq!(cfg.move_count, 18);
    assert_eq!(cfg.output_dim, 1);
    assert_eq!(cfg.hd1, 2556);
    assert_eq!(cfg.hd2, 218);
    assert_eq!(cfg.hidden_alignment, 8);
    assert_eq!(cfg.batch_states_per_rank(), 100_000);
    assert_eq!(cfg.walkers, 34_482);
    assert_eq!(cfg.gradient_carousel_slots, 3);
    assert_eq!(cfg.input_grad_partial_chunks, 1);
    assert_eq!(cfg.input_grad_positions_per_block, 1);
    assert!(!cfg.input_grad_sparse);
    assert!(!cfg.input_grad_fp16);
    assert!(!cfg.linear_fp16);
    assert_eq!(cfg.allreduce_bucket_bytes, 4 * 1024 * 1024);
    assert!(cfg.validate().is_ok());
}

#[test]
fn state_padding_is_nearest_alignment() {
    let mut cfg = TrainerConfig::p888_default();
    cfg.group_id = 123;
    cfg.target_id = 7;
    cfg.state_len = 31;
    cfg.state_alignment = 16;
    cfg.state_value_pad = 11;
    cfg.move_count = 5;
    cfg.hd1 = 33;
    cfg.hd2 = 37;
    assert_eq!(cfg.state_storage_len(), 32);
    assert!(cfg.validate().is_ok());
}

#[test]
fn rejects_wrong_output_dim() {
    let mut cfg = TrainerConfig::p888_default();
    cfg.output_dim = 2;
    let err = cfg.validate().unwrap_err().to_string();
    assert!(err.contains("output_dim"));
}

#[test]
fn rejects_bad_alignment() {
    let mut cfg = TrainerConfig::p888_default();
    cfg.state_alignment = 10;
    let err = cfg.validate().unwrap_err().to_string();
    assert!(err.contains("state_alignment"));
}

#[test]
fn default_config_toml_roundtrips() {
    let cfg = TrainerConfig::p888_default();
    let text = toml::to_string_pretty(&cfg).unwrap();
    let parsed: TrainerConfig = toml::from_str(&text).unwrap();
    assert_eq!(parsed, cfg);
    assert!(parsed.validate().is_ok());
}
