Goal probe retrieval test eval -- 1-GPU serial run (NYU HPC)
run_dir: /home/cy3064/workdir/CausalRetrieval/probeRetrieval/artifacts/goal_probe_retrieval_test_eval_1gpu_nyu/random_retrieve/20260502_163803
variant: random_retrieve
episodes_per_setting: 10
memory_dir: artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/memory_bank
response_jsonl: artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/episodes.jsonl
test_frictions: 0.05, 0.5
test_mass_scales: 0.05, 5.0, 10.0
mujoco_gl: osmesa

Tasks (serial):
  - push_the_plate_to_the_front_of_the_stove
  - put_the_cream_cheese_in_the_bowl
  - put_the_bowl_on_the_stove
  - put_the_bowl_on_top_of_the_cabinet
  - put_the_wine_bottle_on_top_of_the_cabinet

Each task writes:
  <task>/probe_retrieval_test_episodes.jsonl
  <task>/probe_retrieval_test_summary.json
