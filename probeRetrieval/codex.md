# Codex Implementation Spec

## Contact-Aware Retrieval on LIBERO-Object with an OpenVLA LIBERO Checkpoint

### Project status

This document is the execution spec for Codex. It is intentionally narrow. Do not expand scope beyond what is written here unless a blocking issue makes the current plan impossible.

### Verified assumptions

- Use **LIBERO** and restrict experiments to **`libero_object`**.
- Use a **pretrained LIBERO checkpoint** instead of re-training the VLA.
- Start from **`moojink/openvla-7b-oft-finetuned-libero-object`** by default, because the OpenVLA-OFT LIBERO instructions explicitly provide this checkpoint and the matching `run_libero_eval.py` flow for `libero_object`.
- `libero_object` contains ten object-centric pick-and-place tasks with the same high-level skill template: pick up the target object and place it in the basket.
- `run_libero_eval.py` is the correct baseline entry point.
- robosuite supports image observations, proprioceptive observations, and MuJoCo-native force-torque sensor access. In practice, the first implementation should rely primarily on contact and motion features, with force-torque as optional if it is stable.

### Objective

Implement a minimal but real prototype that augments an OpenVLA policy with:

1. image-based first-stage retrieval,
2. a short contact-triggered probe,
3. second-stage re-ranking using probe response features,
4. action fusion between retrieved action and VLA action.

The goal is to compare:

- baseline VLA,
- image retrieval only,
- full two-stage retrieval with probe re-ranking.

---

## 1. Scope lock

### Included

- LIBERO suite: `libero_object`
- checkpoint: `moojink/openvla-7b-oft-finetuned-libero-object`
- one unified probe policy for all tasks in `libero_object`
- memory bank built from collected rollouts
- top-k image retrieval followed by probe-based re-ranking
- success-rate evaluation

### Excluded

- no VLA fine-tuning
- no RL
- no multi-suite training
- no drawer / cabinet / push tasks
- no task-specific probe variants in the first version
- no new simulator asset editing in the first version

### Important framing

Do **not** claim strict IID after inserting the probe. The correct framing is:
the same contact-triggered probe protocol is applied during both memory construction and test-time inference, so retrieval keys are computed under a matched intervention protocol.

---

## 2. Environment and baseline setup

### 2.1 Python environment

Use Python 3.10 if possible.

### 2.2 Required repositories

Codex should assume these repos are needed:

- OpenVLA-OFT repo that contains the LIBERO evaluation script
- LIBERO repo
- robosuite / MuJoCo dependencies required by LIBERO

### 2.3 Baseline command

Codex should wire the baseline around the official evaluation path:

```bash
python experiments/robot/libero/run_libero_eval.py   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object   --task_suite_name libero_object   --center_crop True
```

If the local codebase uses a wrapper entry point, keep behavior equivalent to the command above.

### 2.4 Reproducibility notes

Keep hooks for:

- `--seed`
- `--num_trials_per_task`
- output logging directory
- cache directories for Hugging Face checkpoints

Do not spend time reproducing paper numbers exactly. The goal is a working research prototype.

---

## 3. High-level method

### 3.1 Data collection

For each episode:

1. reset environment
2. let pretrained VLA act normally
3. monitor for first meaningful contact with the target object
4. at contact time `t0`, save:
   - current image observation,
   - current policy action `action_v`,
   - robot / object state needed for probe features
5. pause normal rollout and execute a short fixed probe sequence
6. compute probe response features from the `t0:t1` window
7. resume VLA from the post-probe observation
8. run to episode termination
9. save final outcome: success or failure

### 3.2 Inference

For each evaluation episode:

1. let VLA approach normally
2. before contact or near contact, run stage-1 image retrieval and get top-k candidates
3. when contact occurs, execute the same probe
4. compute query probe response features
5. re-rank the top-k candidates with stage-2 probe similarity
6. aggregate retrieved successful candidates into `action_r`
7. combine `action_r` with VLA action `action_v`

### 3.3 Fusion rule

Start simple:

```python
action = alpha * action_r + (1 - alpha) * action_v
```

Default:

```python
alpha = 0.5
```

Alpha must be a config value.

---

## 4. Contact event detection

### 4.1 Required behavior

Implement a contact trigger that fires only once per episode for the first version.

### 4.2 Minimal implementation

The first implementation may use simulator-level contact count:

```python
env.sim.data.ncon > 0
```

### 4.3 Better implementation

Prefer filtering for contact involving the gripper and the target object if target object body / geom ids are available.

### 4.4 Safety

Avoid firing on irrelevant scene contacts if possible. Add logging so we can inspect what triggered the probe.

---

## 5. Probe design

### 5.1 First-version probe

Use the same short probe for all `libero_object` tasks.

Recommended initial sequence:

1. slightly close gripper
2. apply a very small upward motion
3. hold briefly

Keep the probe extremely short: roughly 3 to 6 control steps.

### 5.2 Why this probe

`libero_object` is a pick-and-place suite, so the probe should test graspability / followability rather than pushing behavior.

### 5.3 Hard constraints

- probe must be identical during collection and inference
- probe should not be long enough to become a second policy
- keep magnitudes conservative to reduce destructive distribution shift

---

## 6. Probe response features

### 6.1 Required first-version features

Implement these first:

- `contact_ratio`
- `total_ee_motion`
- `total_obj_motion`
- `motion_ratio = total_obj_motion / (total_ee_motion + eps)`
- `gripper_delta`

### 6.2 Optional features

Only add if straightforward and stable:

- `mean_force`
- `max_force`
- `force_std`
- object vertical displacement
- whether object follows the gripper during lift
- object-to-EEF relative displacement

### 6.3 Definitions

```python
total_ee_motion = sum(||eef_pos[t+1] - eef_pos[t]||)
total_obj_motion = sum(||obj_pos[t+1] - obj_pos[t]||)
contact_ratio = num_contact_steps / probe_num_steps
motion_ratio = total_obj_motion / (total_ee_motion + 1e-6)
gripper_delta = gripper_state[t1] - gripper_state[t0]
```

### 6.4 Recommendation

Do not make force features mandatory for milestone 1. Motion and contact features are enough for the first working experiment.

---

## 7. Memory bank design

Each stored item should contain at least:

```python
{
    "episode_id": str,
    "task_name": str,
    "image_embedding": np.ndarray,
    "raw_image_path": str | None,
    "action_v_t0": np.ndarray,
    "probe_features": dict,
    "post_probe_action_chunk": np.ndarray | None,
    "success": bool,
    "metadata": {...}
}
```

### 7.1 Recommended storage

Use a simple serialized format first:

- `.jsonl` for metadata
- `.npy` or `.npz` for arrays
- optionally one directory per task

### 7.2 Success filtering

For `action_r`, aggregate from successful candidates only in the first version.

---

## 8. Retrieval design

### 8.1 Stage 1: image retrieval

Use image similarity to retrieve top-k candidates.

The exact embedding backend may be:

- CLIP,
- a visual embedding exported from the VLA vision encoder,
- another simple frozen image encoder.

Do not block the project on perfect image embeddings. Start with the easiest stable option.

### 8.2 Stage 2: probe-based re-ranking

Within the stage-1 top-k set only, compute similarity between query probe features and candidate probe features.

### 8.3 Simple similarity

Start with normalized L2 or cosine similarity over a small feature vector derived from the probe features.

### 8.4 Candidate aggregation

Compute `action_r` by weighted average over top successful re-ranked candidates.

If action chunks are unavailable, use the single best candidate action first.

---

## 9. Evaluation plan

### 9.1 Required baselines

Implement exactly these first:

1. `baseline_vla`: plain OpenVLA rollout
2. `image_retrieval_only`
3. `full_probe_rerank`

### 9.2 Main metric

Success rate.

Report:

- per-task success rate
- average across all `libero_object` tasks

### 9.3 Recommended ablations if time allows

- different `k`
- different `alpha`
- re-rank on / off
- success-only vs all-candidate aggregation
- motion-only features vs motion + force

---

## 10. Project file layout

Use this structure unless the current repo already enforces a close equivalent:

```text
project/
├── configs/
│   ├── default.yaml
│   ├── collect_libero_object.yaml
│   └── infer_libero_object.yaml
├── env/
│   └── libero_wrapper.py
├── vla/
│   └── openvla_policy.py
├── probe/
│   ├── contact_detector.py
│   ├── probe_runner.py
│   └── feature_extractor.py
├── retrieval/
│   ├── image_embedder.py
│   ├── image_retrieval.py
│   ├── probe_rerank.py
│   └── memory_bank.py
├── rollout/
│   ├── collect_data.py
│   └── rollout_utils.py
├── inference/
│   └── run_inference.py
├── eval/
│   └── evaluate.py
├── scripts/
│   ├── run_baseline.sh
│   ├── run_collect.sh
│   └── run_full_eval.sh
└── README.md
```

---

## 11. Required deliverables for Codex

Codex should produce:

### 11.1 Working code

- baseline runner
- data collection runner
- memory bank builder
- inference runner with two-stage retrieval
- evaluation script

### 11.2 Logging

Every episode should log:

- task name
- seed
- whether contact triggered
- probe start and end step
- probe feature vector
- retrieved candidate ids
- fused action statistics
- final success / failure

### 11.3 Artifacts

Save:

- rollout logs
- evaluation summaries
- per-task JSON results
- optional debug videos or frame dumps for a few episodes

---

## 12. Acceptance criteria

Milestone 1 is complete when all of the following are true:

1. the official `libero_object` baseline checkpoint runs end-to-end,
2. contact-triggered probe executes without crashing,
3. memory bank can be built from collected episodes,
4. stage-1 retrieval and stage-2 re-ranking both run,
5. the three required baselines produce comparable success-rate reports,
6. code is configurable and documented enough for a labmate to run.

Milestone 2 is complete when:

- there is at least one clean comparison table for baseline vs retrieval variants,
- and one or two short qualitative rollout examples are saved.

---

## 13. Implementation priorities

### Priority A

Get the baseline running.

### Priority B

Get contact trigger + probe + feature extraction working.

### Priority C

Collect a small dataset and build the memory bank.

### Priority D

Implement retrieval and re-ranking.

### Priority E

Run experiments and package results.

If blocked, never jump to more ambitious ideas before earlier priorities work.

---

## 14. Common failure modes

### 14.1 Contact trigger is noisy

Fix by filtering to target-object contact if possible.

### 14.2 Probe destabilizes the rollout

Reduce probe magnitude and number of steps.

### 14.3 Force features are unstable or unavailable

Disable them and continue with motion/contact features only.

### 14.4 Retrieval helps little

Check:

- image embedding quality,
- whether contact was triggered at the right time,
- whether successful candidates dominate the action aggregation,
- whether `alpha` is too high.

### 14.5 Version mismatch

If the official checkpoint or eval script is fragile, prioritize matching the package versions recommended by the OpenVLA-OFT LIBERO instructions.

---

## 15. Suggested execution order for two weeks

### Week 1

- Day 1: baseline setup
- Day 2: confirm `libero_object` eval works
- Day 3: implement contact detector
- Day 4: implement probe runner
- Day 5: implement feature extraction
- Day 6: collect first rollout set
- Day 7: build memory bank and test stage-1 retrieval

### Week 2

- Day 8: implement stage-2 re-ranking
- Day 9: implement action fusion
- Day 10: run baseline comparisons
- Day 11: debug failure cases
- Day 12: run final sweeps over `alpha` and `k`
- Day 13: save tables, logs, and examples
- Day 14: clean repo and write concise experiment notes

---

## 16. References that informed this spec

Use these only as implementation anchors, not as extra scope:

- OpenVLA-OFT LIBERO instructions and checkpoint names
- LIBERO task suite definition for `libero_object`
- robosuite sensor / observable documentation

End of spec.