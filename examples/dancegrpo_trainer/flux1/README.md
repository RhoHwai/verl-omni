# FLUX.1-dev DanceGRPO

This directory contains one Ascend NPU, full-parameter FLUX.1-dev DanceGRPO
launcher with selectable HPSv2 or HPSv3 reward. Its training settings are
based on the validated `v2_2.sh` run:
720x720 images, 16 denoising steps, rollout `guidance_scale=3.5`, actor
`guidance_scale=1.0`, 0.3 DanceSDE noise, and a random 60% subset after dropping
the final transition.

FLUX.1-dev uses a distilled guidance embedding. This adapter does not run a
negative-prompt CFG branch and deliberately has no `true_cfg_scale` option.
The rollout and actor guidance values are intentionally different; they keep
their existing `guidance_scale` config names because their locations already
distinguish sampling from actor log-probability recomputation.

## Prepare prompts

Prompt files are not included. Supply two UTF-8 text files with one prompt per
line so the train/test boundary is explicit and reviewable. Public sources you
can evaluate include the [HPSv2 project and HPD benchmark](https://github.com/tgxs002/HPSv2)
and the [DanceGRPO reference implementation](https://github.com/XueZeyue/DanceGRPO).

Convert the files to the same chat-style parquet schema used by the other
DanceGRPO examples:

```bash
python3 examples/dancegrpo_trainer/flux1/dataprocess/prepare_prompts.py \
  --train-path /path/to/train.txt \
  --test-path /path/to/test.txt \
  --output-dir /path/to/flux1_prompts
```

This writes `train.parquet` and `test.parquet`. The processor never downloads,
embeds, randomly splits, or copies prompt text into this repository.

## Prepare FLUX.1-dev and HPSv2

The HPSv2 entry point requires `open_clip_torch`:

```bash
pip install open_clip_torch
```

Prepare a local diffusers-format
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) checkpoint and
download these HPSv2.1 files:

- [`HPS_v2.1_compressed.pt`](https://huggingface.co/xswu/HPSv2/tree/main)
- [`open_clip_pytorch_model.bin`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)

Only use checkpoints you trust: the HPSv2 loader uses PyTorch checkpoint
deserialization for compatibility with the published weights.

## Run with HPSv2

From the repository root:

```bash
export MODEL_NAME=/path/to/FLUX.1-dev
export TRAIN_FILES_PATH=/path/to/flux1_prompts/train.parquet
export VAL_FILES_PATH=/path/to/flux1_prompts/test.parquet
export HPSV2_PRETRAINED_PATH=/path/to/open_clip_pytorch_model.bin
export CUSTOM_REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt

REWARD_NAME=hpsv2 \
  bash examples/dancegrpo_trainer/flux1/run_flux1_dev_dancegrpo_fullparam.sh
```

HPSv2 defaults to two CPU-resident reward workers (`REWARD_NUM_WORKERS=2` and
`REWARD_DEVICE=cpu`). The OpenCLIP model and both checkpoints are loaded on
CPU, so reward scoring does not consume rollout NPU memory. Synchronous scorer
calls run through the reward manager's executor, and HPSv2 uses autocast during
inference to retain the throughput of the validated recipe. To opt into NPU
scoring, set `REWARD_DEVICE=npu:0` explicitly. This is not the default because
verl RewardLoopWorkers do not reserve Ray accelerator resources; an explicit
NPU scorer can therefore contend with actor and rollout allocations.

The launcher defaults to eight NPUs and full-parameter training
(`model.lora_rank=0`). Hardware and batch settings can be overridden with
`ASCEND_RT_VISIBLE_DEVICES`, `NUM_GPUS`, `ROLLOUT_TP`, `TRAIN_BATCH_SIZE`,
`ROLLOUT_N`, and `REWARD_NUM_WORKERS`.

## Run with HPSv3

The same launcher can select the repository's existing `hpsv3_reward.py`. This
example does not modify or reimplement HPSv3. It does not require
`HPSV2_PRETRAINED_PATH`.

```bash
export MODEL_NAME=/path/to/FLUX.1-dev
export TRAIN_FILES_PATH=/path/to/flux1_prompts/train.parquet
export VAL_FILES_PATH=/path/to/flux1_prompts/test.parquet
export CUSTOM_REWARD_MODEL_PATH=/path/to/HPSv3.safetensors

REWARD_NAME=hpsv3 \
  bash examples/dancegrpo_trainer/flux1/run_flux1_dev_dancegrpo_fullparam.sh
```

HPSv3 uses conservative defaults because each RewardLoopWorker loads one
Qwen2-VL-7B reward model: one reward worker, training batch size 8, and rollout
memory utilization 0.3. Rollout TP 2 and FP32 actor model storage are aligned
with the validated FLUX full-parameter recipe. Parameter and optimizer offload
remain enabled. The single reward worker and reduced batch/memory utilization
lower memory pressure, but they do not add Ray NPU isolation to the existing
HPSv3 implementation. Keep offload enabled when validating this mode on the
target server.
