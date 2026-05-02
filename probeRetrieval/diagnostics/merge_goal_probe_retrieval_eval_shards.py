from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rollout.rollout_utils import ensure_dir, write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_task": {}}
    if not rows:
        return summary
    summary["overall"] = {
        "num_episodes": len(rows),
        "success_rate": float(sum(bool(r.get("success", False)) for r in rows) / len(rows)),
    }
    task_names = sorted({str(r.get("task_name", "")) for r in rows})
    for task_name in task_names:
        task_rows = [row for row in rows if row.get("task_name") == task_name]
        per_setting: dict[str, Any] = {}
        for setting_name in sorted({str(r.get("setting_name", "")) for r in task_rows}):
            setting_rows = [row for row in task_rows if row.get("setting_name") == setting_name]
            per_setting[setting_name] = {
                "n": len(setting_rows),
                "success_rate": float(sum(bool(r.get("success", False)) for r in setting_rows) / len(setting_rows)),
                "mean_num_ranked": float(sum(float(r.get("num_ranked_candidates", 0.0)) for r in setting_rows) / len(setting_rows)),
                "mean_num_successful_top5": float(sum(float(r.get("num_successful_top5", 0.0)) for r in setting_rows) / len(setting_rows)),
            }
        summary["by_task"][task_name] = per_setting
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge 4-GPU probe retrieval eval shards.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = ensure_dir(args.output_dir or (run_dir / "merged"))
    all_rows: list[dict[str, Any]] = []
    shard_info = []

    for shard_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("gpu")):
        episodes_path = shard_dir / "probe_retrieval_test_episodes.jsonl"
        summary_path = shard_dir / "probe_retrieval_test_summary.json"
        rows = load_jsonl(episodes_path)
        all_rows.extend(rows)
        shard_info.append(
            {
                "shard": shard_dir.name,
                "num_rows": len(rows),
                "episodes_path": str(episodes_path) if episodes_path.exists() else None,
                "summary_path": str(summary_path) if summary_path.exists() else None,
            }
        )

    merged_episodes_path = output_dir / "probe_retrieval_test_episodes.jsonl"
    with merged_episodes_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize_rows(all_rows)
    summary["shards"] = shard_info
    write_json(output_dir / "probe_retrieval_test_summary.json", summary)


if __name__ == "__main__":
    main()
