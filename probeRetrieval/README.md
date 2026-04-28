# Contact-Aware Retrieval for LIBERO Object

This directory implements the prototype described in `codex.md`: baseline OpenVLA evaluation, contact-triggered probing, memory-bank construction, image retrieval, probe-based re-ranking, action fusion, and success-rate reporting for `libero_object`.

## Environment

Create the project environment under the lab path requested for this machine:

```bash
cd /network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval
bash scripts/create_env.sh
conda activate /network/rit/lab/wang_lab_cs/yhan/envs/probeRetrieval
```

For real LIBERO/OpenVLA runs, install the official OpenVLA-OFT, LIBERO, robosuite, MuJoCo, torch, and transformers dependencies into that same environment. Keep Hugging Face downloads under `/network/rit/lab/wang_lab_cs/yhan/hf_cache`.

With the local repos used in this workspace, install the real dependencies with:

```bash
bash scripts/install_real_deps.sh
```

## Teammate Setup

`requirements.txt` is only the minimal dependency set for the lightweight utilities in this repo. To run the real OpenVLA/LIBERO experiments, your teammate also needs:

- this `probeRetrieval` repo
- the `openvla-oft` repo
- the `LIBERO` repo

A portable setup flow looks like:

```bash
git clone <your-probeRetrieval-repo>
git clone <openvla-oft-repo>
git clone <LIBERO-repo>

cd probeRetrieval
ENV_DIR=/path/to/envs/probeRetrieval bash scripts/create_env.sh
conda activate /path/to/envs/probeRetrieval

ENV_DIR=/path/to/envs/probeRetrieval \
OPENVLA_OFT_REPO=/path/to/openvla-oft \
LIBERO_REPO=/path/to/LIBERO \
bash scripts/install_real_deps.sh
```

If your teammate only wants to run the real experiments, sending `requirements.txt` by itself is not enough; the editable installs from `openvla-oft` and `LIBERO` are also required.

## Baseline OpenVLA

The baseline script wraps the official evaluation command from the spec:

```bash
bash scripts/run_baseline.sh
```

It executes:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /network/rit/lab/wang_lab_cs/yhan/repos/openvla-7b-oft \
  --task_suite_name libero_object \
  --center_crop True
```

## Goal Baseline OpenVLA

To evaluate the weaker LIBERO-goal checkpoint on the goal suite, use:

```bash
bash scripts/run_goal_baseline.sh
```

Default settings:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --num_trials_per_task 50 \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_l1_regression False \
  --center_crop True
```

## Smoke Test Without MuJoCo

The mock mode validates the collection, memory, retrieval, re-ranking, fusion, and evaluation plumbing:

```bash
python -m rollout.collect_data --config default.yaml --mock --num_trials_per_task 1
python -m inference.run_inference --config default.yaml --mock --num_trials_per_task 1
python -m eval.evaluate --input_dir artifacts
```

## Real Collection and Inference

The real runner uses the local OpenVLA-OFT and LIBERO repos configured in `configs/default.yaml`:

- `/network/rit/lab/wang_lab_cs/yhan/repos/openvla-oft`
- `/network/rit/lab/wang_lab_cs/yhan/repos/openvla-7b-oft`
- `/network/rit/lab/wang_lab_cs/yhan/repos/LIBERO`

Start small:

```bash
bash scripts/run_real_collect.sh --num_trials_per_task 1
bash scripts/run_real_inference.sh --num_trials_per_task 1
python -m eval.evaluate --input_dir artifacts/real_eval
```

To run one inference variant at a time:

```bash
bash scripts/run_real_inference.sh --num_trials_per_task 1 --variant image_retrieval_only
bash scripts/run_real_inference.sh --num_trials_per_task 1 --variant full_probe_rerank
```

## Outputs

Collection writes `collection_episodes.jsonl`, `collection_summary.json`, debug contact frames, and a memory bank with `metadata.jsonl` plus `arrays.npz`.

Inference writes one JSONL file per variant:

- `baseline_vla_episodes.jsonl`
- `image_retrieval_only_episodes.jsonl`
- `full_probe_rerank_episodes.jsonl`

`eval.evaluate` writes per-task and average success rates to `evaluation_summary.json`.
