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
batch_size=""
state_len=""
state_value_count=""
move_count=""
state_alignment=""
hidden_alignment=""
gradient_slots=""
input_grad_partial_chunks=""
input_grad_fp16=""
linear_fp16=""
overlap_allreduce=""
allreduce_bucket_bytes=""
lr=""
weight_decay=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-dir) out="$2"; shift 2 ;;
    --hd1) hd1="$2"; shift 2 ;;
    --hd2) hd2="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --state-len) state_len="$2"; shift 2 ;;
    --state-value-count) state_value_count="$2"; shift 2 ;;
    --move-count) move_count="$2"; shift 2 ;;
    --state-alignment) state_alignment="$2"; shift 2 ;;
    --hidden-alignment) hidden_alignment="$2"; shift 2 ;;
    --gradient-carousel-slots) gradient_slots="$2"; shift 2 ;;
    --input-grad-partial-chunks) input_grad_partial_chunks="$2"; shift 2 ;;
    --input-grad-fp16) input_grad_fp16="$2"; shift 2 ;;
    --linear-fp16) linear_fp16="$2"; shift 2 ;;
    --overlap-allreduce) overlap_allreduce="$2"; shift 2 ;;
    --allreduce-bucket-bytes) allreduce_bucket_bytes="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --weight-decay) weight_decay="$2"; shift 2 ;;
    *) shift 2 ;;
  esac
done
mkdir -p "$out/weights" "$out/checkpoint"
printf 'MODEL_MODE=MLP2RB
OUTPUT_DIM=1
WEIGHT_DECAY=%s
HD1=%s
HD2=%s
STATE_LEN=%s
STATE_VALUE_COUNT=%s
MOVE_COUNT=%s
STATE_ALIGNMENT=%s
HIDDEN_ALIGNMENT=%s
BATCH_SIZE=%s
GRADIENT_CAROUSEL_SLOTS=%s
INPUT_GRAD_PARTIAL_CHUNKS=%s
INPUT_GRAD_FP16=%s
LINEAR_FP16=%s
OVERLAP_ALLREDUCE=%s
ALLREDUCE_BUCKET_BYTES=%s
LR=%s
' "$weight_decay" "$hd1" "$hd2" "$state_len" "$state_value_count" "$move_count" "$state_alignment" "$hidden_alignment" "$batch_size" "$gradient_slots" "$input_grad_partial_chunks" "$input_grad_fp16" "$linear_fp16" "$overlap_allreduce" "$allreduce_bucket_bytes" "$lr" > "$out/metadata.env"
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

    let mut cfg = TrainerConfig::p888_default();
    cfg.state_len = 31;
    cfg.state_alignment = 16;
    cfg.state_value_pad = 11;
    cfg.move_count = 5;
    cfg.hidden_alignment = 8;
    cfg.gradient_carousel_slots = 4;
    cfg.input_grad_partial_chunks = 1;
    cfg.input_grad_fp16 = true;
    cfg.linear_fp16 = true;
    cfg.allreduce_bucket_bytes = 1_048_576;
    cfg.learning_rate = 0.0003;
    cfg.weight_decay = 0.0;
    cfg.validate().unwrap();

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
    assert!(metadata.contains("STATE_LEN=31"));
    assert!(metadata.contains("STATE_VALUE_COUNT=11"));
    assert!(metadata.contains("MOVE_COUNT=5"));
    assert!(metadata.contains("STATE_ALIGNMENT=16"));
    assert!(metadata.contains("HIDDEN_ALIGNMENT=8"));
    assert!(metadata.contains("BATCH_SIZE=17"));
    assert!(metadata.contains("INPUT_GRAD_PARTIAL_CHUNKS=1"));
    assert!(metadata.contains("INPUT_GRAD_FP16=1"));
    assert!(metadata.contains("LINEAR_FP16=1"));
    assert!(metadata.contains("OVERLAP_ALLREDUCE=1"));
    assert!(metadata.contains("GRADIENT_CAROUSEL_SLOTS=4"));
    assert!(metadata.contains("ALLREDUCE_BUCKET_BYTES=1048576"));
    assert!(metadata.contains("LR=0.0003"));
    assert!(metadata.contains("WEIGHT_DECAY=0"));

    let stdout = fs::read_to_string(output.join("native.stdout")).unwrap();
    assert!(stdout.contains("fake native ok"));
    assert!(output.join("weights").join("weights.f32.bin").exists());

    fs::remove_dir_all(root).unwrap();
}
