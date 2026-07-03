#[path = "../src/config.rs"]
mod config;
#[path = "../src/native.rs"]
mod native;

use config::TrainerConfig;
use std::fs;
use std::io::Write;

#[test]
fn training_entry_invokes_native_rank_binary() {
    let root = std::env::temp_dir().join(format!("mgt-native-artifacts-{}", std::process::id()));
    let output = root.join("run");
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    fs::create_dir_all(&root).unwrap();

    let bin = root.join("fake_native_train.sh");
    let mut file = fs::File::create(&bin).unwrap();
    file.write_all(
        br#"#!/usr/bin/env bash
set -euo pipefail
out=""
hd1=""
hd2=""
steps=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-dir) out="$2"; shift 2 ;;
    --hd1) hd1="$2"; shift 2 ;;
    --hd2) hd2="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    *) shift 2 ;;
  esac
done
mkdir -p "$out/weights" "$out/checkpoint"
printf 'MODEL_MODE=MLP2RB
OUTPUT_DIM=1
WEIGHT_DECAY=0
HD1=%s
HD2=%s
' "$hd1" "$hd2" > "$out/metadata.env"
printf '{"hd1": %s, "hd2": %s, "output_dim": 1}
' "$hd1" "$hd2" > "$out/layers.json"
printf 'rank=0 phase=train steps=%s
' "$steps" > "$out/train.log"
printf '{"milliseconds":1.0,"memory_bytes":1024,"status":"ok"}
' > "$out/profile.jsonl"
printf '{"format":"stream1_weights"}
' > "$out/weights/manifest.json"
printf 'weights' > "$out/weights/weights.f32.bin"
printf '{"format":"mgt_train_checkpoint"}
' > "$out/checkpoint/manifest.json"
printf 'checkpoint' > "$out/checkpoint/state.f32.bin"
printf 'fake native ok
'
"#,
    )
    .unwrap();
    drop(file);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).unwrap();
    }

    let cfg = TrainerConfig::p888_default();
    std::env::set_var("MGT_NATIVE_TRAIN_BIN", &bin);
    std::env::set_var("MGT_TRAIN_STEPS", "2");
    std::env::set_var("MGT_TRAIN_BATCH_SIZE", "17");
    std::env::set_var("MGT_TRAIN_HD1", "7");
    std::env::set_var("MGT_TRAIN_HD2", "4");
    native::run_training(&cfg, &output).unwrap();
    std::env::remove_var("MGT_NATIVE_TRAIN_BIN");
    std::env::remove_var("MGT_TRAIN_STEPS");
    std::env::remove_var("MGT_TRAIN_BATCH_SIZE");
    std::env::remove_var("MGT_TRAIN_HD1");
    std::env::remove_var("MGT_TRAIN_HD2");

    let metadata = fs::read_to_string(output.join("metadata.env")).unwrap();
    assert!(metadata.contains("MODEL_MODE=MLP2RB"));
    assert!(metadata.contains("OUTPUT_DIM=1"));
    assert!(metadata.contains("HD1=7"));
    assert!(metadata.contains("HD2=4"));

    let stdout = fs::read_to_string(output.join("native.stdout")).unwrap();
    assert!(stdout.contains("fake native ok"));
    assert!(output.join("weights").join("weights.f32.bin").exists());

    fs::remove_dir_all(root).unwrap();
}
