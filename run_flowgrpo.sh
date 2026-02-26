#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/workspace/verl/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flowgrpo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

IMG=/hf_cache/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
REWARD_IMG=/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3


CUDA_VISIBLE_DEVICES=0,1 bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh \
  trainer.n_gpus_per_node=2 \
  trainer.logger='["console"]' \
  actor_rollout_ref.model.path=$IMG \
  actor_rollout_ref.model.tokenizer_path=$IMG/tokenizer \
  reward.reward_model.model_path=$REWARD_IMG \
  data.train_files=/root/data/ocr/train.parquet \
  data.val_files=/root/data/ocr/test.parquet \
