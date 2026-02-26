# FlowGRPO Run Preparation (for `examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh`)

This note summarizes what the script does at runtime and what you can prepare in advance to reduce waiting time.

## 1) What happens when you run the script

The script launches:

```bash
python3 -m verl.trainer.main_ppo --config-name='ppo_diffusion_trainer.yaml' ...
```

Major stages:

1. Hydra merges config + CLI overrides.
2. Ray initializes local runtime (`ray.init(...)`) and starts worker actors.
3. Trainer builds resource pools from `trainer.nnodes` and `trainer.n_gpus_per_node`.
4. Model/tokenizer paths are copied to local cache (`copy_to_local`) and loaded:
   - Actor diffusion model: `~/models/Qwen/Qwen-Image`
   - Actor tokenizer: `~/models/Qwen/Qwen-Image/tokenizer`
   - Reward model: `~/models/Qwen/Qwen2.5-VL-3B-Instruct`
5. OCR parquet datasets are loaded:
   - `~/data/ocr/train.parquet`
   - `~/data/ocr/test.parquet`
6. Workers initialize FSDP + vLLM-Omni rollout + reward rollout.
7. First rollout/validation batches run (usually the longest startup delay before step logs look normal).

## 2) Important note for your 2-GPU node

The script hard-codes:

- `trainer.n_gpus_per_node=4`

On a 2-GPU node, run with override:

```bash
bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh trainer.n_gpus_per_node=2
```

You may also need to reduce batch sizes to avoid OOM/timeouts during warmup.

## 3) High-impact prep you can do in advance

## A. Pre-download models locally (largest time saver)

Prepare both model directories before training starts:

- `~/models/Qwen/Qwen-Image`
- `~/models/Qwen/Qwen2.5-VL-3B-Instruct`

If using Hugging Face CLI:

```bash
huggingface-cli download Qwen/Qwen-Image --local-dir ~/models/Qwen/Qwen-Image
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ~/models/Qwen/Qwen2.5-VL-3B-Instruct
```

## B. Pre-build OCR parquet dataset

The script expects parquet files already present. Build once:

```bash
python3 examples/data_preprocess/qwenimage_ocr.py \
  --local_dataset_path ~/dataset/ocr \
  --local_save_dir ~/data/ocr
```

Expected outputs:

- `~/data/ocr/train.parquet`
- `~/data/ocr/test.parquet`

## C. Warm package/runtime caches

Run once in the same environment to fill caches:

```bash
python3 -c "import torch, ray, transformers, diffusers"
python3 -m verl.trainer.main_ppo --help >/dev/null
```

This does not fully initialize training, but avoids some first-import penalties.

## D. Start Ray before training (optional)

```bash
ray stop --force || true
ray start --head
```

The script can start Ray automatically; prestarting is optional but can reduce startup jitter.

## E. Use local fast storage for caches

If available, place/copy models and HF caches on local NVMe SSD (not NFS).

Common cache env vars to pin to fast storage:

```bash
export HF_HOME=~/hf_cache
export TRANSFORMERS_CACHE=~/hf_cache/transformers
export HUGGINGFACE_HUB_CACHE=~/hf_cache/hub
```

## 4) Suggested 2-GPU quick-start profile

For first successful run on 2 GPUs, consider overriding to smaller settings:

```bash
bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh \
  trainer.n_gpus_per_node=2 \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=4 \
  trainer.total_epochs=1
```

After stability is confirmed, scale parameters back up.

## 5) Preflight checklist

Before each run:

- `nvidia-smi` shows 2 healthy GPUs.
- `~/models/Qwen/Qwen-Image` exists.
- `~/models/Qwen/Qwen2.5-VL-3B-Instruct` exists.
- `~/data/ocr/train.parquet` and `~/data/ocr/test.parquet` exist.
- Enough disk in cache/model directories.
- `ray status` is healthy (or no stale Ray processes before run).

