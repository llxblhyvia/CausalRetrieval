Goal probe retrieval test eval 4-GPU run
run_dir: /network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/goal_probe_retrieval_test_eval_4gpu/20260429_161402
episodes_per_setting: 10
memory_dir: artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/memory_bank
response_jsonl: artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822/merged/episodes.jsonl
test_frictions: 0.05, 0.5
test_mass_scales: 0.05, 5.0, 10.0

Task shards:
  gpu0: push_the_plate_to_the_front_of_the_stove, put_the_cream_cheese_in_the_bowl
  gpu1: put_the_bowl_on_the_stove
  gpu2: put_the_bowl_on_top_of_the_cabinet
  gpu3: put_the_wine_bottle_on_top_of_the_cabinet

Each shard writes:
  probe_retrieval_test_episodes.jsonl
  probe_retrieval_test_summary.json
