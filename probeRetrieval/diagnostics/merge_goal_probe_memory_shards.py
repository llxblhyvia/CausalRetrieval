from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retrieval.memory_bank import MemoryBank
from rollout.rollout_utils import ensure_dir, write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge 4-GPU goal probe memory collection shards.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = ensure_dir(args.output_dir or (run_dir / "merged"))
    merged_memory = MemoryBank()
    merged_rows: list[dict[str, Any]] = []
    shard_summaries = []

    for shard_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("gpu")):
        memory_dir = shard_dir / "memory_bank"
        episodes_path = shard_dir / "episodes.jsonl"
        summary_path = shard_dir / "collection_summary.json"
        shard_bank = MemoryBank.load(memory_dir)
        merged_memory.items.extend(shard_bank.items)
        shard_rows = load_jsonl(episodes_path)
        merged_rows.extend(shard_rows)
        shard_summaries.append(
            {
                "shard": shard_dir.name,
                "num_items": len(shard_bank),
                "num_rows": len(shard_rows),
                "summary_path": str(summary_path) if summary_path.exists() else None,
            }
        )

    merged_memory.save(output_dir / "memory_bank")
    episodes_out = output_dir / "episodes.jsonl"
    with episodes_out.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    success_rate = 0.0
    if merged_rows:
        success_rate = sum(bool(row.get("final_success", False)) for row in merged_rows) / len(merged_rows)

    write_json(
        output_dir / "merge_summary.json",
        {
            "run_dir": str(run_dir),
            "num_shards": len(shard_summaries),
            "num_items": len(merged_memory),
            "num_rows": len(merged_rows),
            "success_rate": float(success_rate),
            "memory_dir": str(output_dir / "memory_bank"),
            "episodes_path": str(episodes_out),
            "shards": shard_summaries,
        },
    )


if __name__ == "__main__":
    main()
