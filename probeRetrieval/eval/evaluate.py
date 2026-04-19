from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from rollout.rollout_utils import write_json


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(paths: list[Path]) -> Dict[str, Any]:
    rows = []
    for path in paths:
        fallback_variant = path.stem.replace("_episodes", "")
        for row in load_jsonl(path):
            row = dict(row)
            row["variant"] = str(row.get("variant") or fallback_variant)
            rows.append(row)
    out: Dict[str, Any] = {"variants": {}}
    variants = sorted({str(r["variant"]) for r in rows})
    for variant in variants:
        vrows = [r for r in rows if str(r["variant"]) == variant]
        if not vrows:
            continue
        per_task = {}
        for task in sorted({str(r["task_name"]) for r in vrows}):
            trows = [r for r in vrows if r["task_name"] == task]
            per_task[task] = {"success_rate": float(np.mean([bool(r["success"]) for r in trows])), "n": len(trows)}
        out["variants"][variant] = {
            "average_success_rate": float(np.mean([bool(r["success"]) for r in vrows])),
            "n": len(vrows),
            "per_task": per_task,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-task LIBERO object success rates.")
    parser.add_argument("--input_dir", default="artifacts")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(path for path in input_dir.glob("*_episodes.jsonl") if path.name != "collection_episodes.jsonl")
    summary = summarize(paths)
    output = Path(args.output) if args.output else input_dir / "evaluation_summary.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
