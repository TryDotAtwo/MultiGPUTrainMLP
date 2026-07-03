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
    assert_eq!(cfg.batch_states_per_rank(), 100_021);
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
fn default_config_toml_roundtrips() {
    let cfg = TrainerConfig::p888_default();
    let text = toml::to_string_pretty(&cfg).unwrap();
    let parsed: TrainerConfig = toml::from_str(&text).unwrap();
    assert_eq!(parsed, cfg);
    assert!(parsed.validate().is_ok());
}
