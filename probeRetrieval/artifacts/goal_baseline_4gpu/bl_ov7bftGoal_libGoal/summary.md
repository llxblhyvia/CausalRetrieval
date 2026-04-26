# Goal Baseline Summary

Run directory: `/network/rit/lab/wang_lab_cs/yhan/CausalRetrieval/probeRetrieval/artifacts/goal_baseline_4gpu/bl_ov7bftGoal_libGoal`

Model: `openvla/openvla-7b-finetuned-libero-goal`
Task suite: `libero_goal`
Episodes per task: `50`
Total tasks: `10`
Total episodes: `500`
Total successes: `201`
Overall success rate: `40.2%`

## Per-task results

| Task | Successes / 50 | Success Rate |
| --- | ---: | ---: |
| turn on the stove | 45 | 90.0% |
| put the wine bottle on top of the cabinet | 34 | 68.0% |
| put the bowl on top of the cabinet | 33 | 66.0% |
| put the bowl on the plate | 23 | 46.0% |
| put the bowl on the stove | 22 | 44.0% |
| open the middle drawer of the cabinet | 20 | 40.0% |
| put the cream cheese in the bowl | 18 | 36.0% |
| push the plate to the front of the stove | 4 | 8.0% |
| put the wine bottle on the rack | 2 | 4.0% |
| open the top drawer and put the bowl inside | 0 | 0.0% |

## Shard results

| GPU shard | Tasks | Episodes | Successes | Success Rate |
| --- | --- | ---: | ---: | ---: |
| gpu0 | 0,1,2 | 150 | 76 | 50.7% |
| gpu1 | 3,4,5 | 150 | 37 | 24.7% |
| gpu2 | 6,7 | 100 | 63 | 63.0% |
| gpu3 | 8,9 | 100 | 25 | 25.0% |

## Source logs

- `gpu0/logs/EVAL-libero_goal-openvla-2026_04_25-02_00_26.txt`
- `gpu1/logs/EVAL-libero_goal-openvla-2026_04_25-02_00_27.txt`
- `gpu2/logs/EVAL-libero_goal-openvla-2026_04_25-02_00_26.txt`
- `gpu3/logs/EVAL-libero_goal-openvla-2026_04_25-02_00_27.txt`
