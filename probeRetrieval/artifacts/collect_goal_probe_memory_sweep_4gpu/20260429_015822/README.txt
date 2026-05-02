Goal probe memory collection 4-GPU run
run_dir: /network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/collect_goal_probe_memory_sweep_4gpu/20260429_015822
episodes_per_setting: 10
frictions: 0.2, 0.7
mass_scales: 0.5, 3.0, 7.0

Task shards:
  gpu0: push_the_plate_to_the_front_of_the_stove, put_the_cream_cheese_in_the_bowl
  gpu1: put_the_bowl_on_the_stove
  gpu2: put_the_bowl_on_top_of_the_cabinet
  gpu3: put_the_wine_bottle_on_top_of_the_cabinet

Each shard writes:
  episodes.jsonl
  summary.json
  collection_summary.json
  memory_bank/metadata.jsonl
  memory_bank/arrays.npz
