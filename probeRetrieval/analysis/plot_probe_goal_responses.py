from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("success_rate", "Success Rate", True),
    ("mean_force", "Mean Force", False),
    ("contact_ratio", "Contact Ratio", False),
    ("probe_obj_displacement", "Probe Object Displacement", False),
    ("post_probe_object_to_target_distance", "Post-Probe Object-to-Target Distance", False),
    ("motion_ratio", "Motion Ratio", False),
    ("max_object_step_displacement", "Max Object Step Displacement", False),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dedup_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str, int]]]:
    seen = {}
    duplicates = []
    for row in rows:
        key = (str(row["task_name"]), str(row["setting_name"]), int(row["episode_idx"]))
        if key in seen:
            duplicates.append(key)
            continue
        seen[key] = row
    return list(seen.values()), duplicates


def setting_sort_key(setting: str) -> tuple[int, float]:
    if setting.startswith("friction_mu_"):
        return (0, float(setting.split("_")[-1]))
    if setting.startswith("mass_scale_"):
        return (1, float(setting.split("_")[-1]))
    return (2, 0.0)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_name"]), str(row["setting_name"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (task_name, setting_name), rs in sorted(grouped.items(), key=lambda item: (item[0][0], setting_sort_key(item[0][1]))):
        agg = {
            "task_name": task_name,
            "setting_name": setting_name,
            "n": len(rs),
            "probe_trigger_rate": float(np.mean([bool(r["probe_triggered"]) for r in rs])),
            "success_rate": float(np.mean([bool(r["final_success"]) for r in rs])),
        }
        for key, _, _ in METRICS:
            if key == "success_rate":
                continue
            values = [r[key] for r in rs if r.get(key) is not None]
            agg[key] = float(np.mean(values)) if values else None
        summary_rows.append(agg)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_by_task(summary_rows: list[dict[str, Any]], metric_key: str, metric_label: str, output_path: Path) -> None:
    task_names = sorted({row["task_name"] for row in summary_rows})
    setting_names = sorted({row["setting_name"] for row in summary_rows}, key=setting_sort_key)
    x = np.arange(len(task_names))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, setting_name in enumerate(setting_names):
        y = []
        for task_name in task_names:
            match = next((row for row in summary_rows if row["task_name"] == task_name and row["setting_name"] == setting_name), None)
            value = np.nan if match is None or match.get(metric_key) is None else float(match[metric_key])
            y.append(value)
        ax.bar(x + (idx - 2) * width, y, width=width, label=setting_name.replace("_", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=20, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} Across Tasks and Physics Settings")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_setting(summary_rows: list[dict[str, Any]], metric_key: str, metric_label: str, output_path: Path) -> None:
    setting_names = sorted({row["setting_name"] for row in summary_rows}, key=setting_sort_key)
    means = []
    for setting_name in setting_names:
        vals = [row[metric_key] for row in summary_rows if row["setting_name"] == setting_name and row.get(metric_key) is not None]
        means.append(float(np.mean(vals)) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(len(setting_names)), means, marker="o", linewidth=2)
    ax.set_xticks(range(len(setting_names)))
    ax.set_xticklabels([s.replace("_", " ") for s in setting_names], rotation=20, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Average {metric_label} by Physics Setting")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scatter(rows: list[dict[str, Any]], x_key: str, y_key: str, output_path: Path) -> None:
    colors = {"friction": "#1f77b4", "mass": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for sweep_type in ("friction", "mass"):
        rs = [r for r in rows if r.get("sweep_type") == sweep_type and r.get(x_key) is not None and r.get(y_key) is not None]
        xs = [float(r[x_key]) for r in rs]
        ys = [float(r[y_key]) for r in rs]
        ax.scatter(xs, ys, alpha=0.7, label=sweep_type, color=colors[sweep_type])
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.set_title(f"{y_key.replace('_', ' ').title()} vs {x_key.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]], duplicates: list[tuple[str, str, int]], raw_count: int, dedup_count: int) -> None:
    lines = [
        "# Probe Goal Response Summary",
        "",
        f"- Raw episode rows: `{raw_count}`",
        f"- Deduplicated episode rows: `{dedup_count}`",
        f"- Duplicate keys removed: `{len(duplicates)}`",
        "",
        "## Best Success Setting Per Task",
        "",
        "| Task | Best Setting | Success Rate |",
        "| --- | --- | ---: |",
    ]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_task[str(row["task_name"])].append(row)
    for task_name in sorted(by_task):
        best = max(by_task[task_name], key=lambda row: float(row["success_rate"]))
        lines.append(f"| {task_name} | {best['setting_name']} | {best['success_rate']:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot response comparisons for probe-goal physics sweep results.")
    parser.add_argument("--episodes", required=True, help="Path to episodes.jsonl")
    parser.add_argument("--output_dir", default=None, help="Directory for plots/csv/notes")
    args = parser.parse_args()

    episodes_path = Path(args.episodes)
    output_dir = Path(args.output_dir) if args.output_dir else episodes_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(episodes_path)
    rows, duplicates = dedup_rows(raw_rows)
    summary_rows = aggregate_rows(rows)

    write_csv(output_dir / "response_summary.csv", summary_rows)
    write_csv(output_dir / "episodes_dedup.csv", rows)
    write_markdown_summary(output_dir / "notes.md", summary_rows, duplicates, len(raw_rows), len(rows))

    for metric_key, metric_label, _ in METRICS:
        plot_metric_by_task(summary_rows, metric_key, metric_label, output_dir / f"{metric_key}_by_task.png")
        plot_metric_by_setting(summary_rows, metric_key, metric_label, output_dir / f"{metric_key}_by_setting.png")

    plot_scatter(rows, "mean_force", "final_object_to_target_distance", output_dir / "force_vs_final_distance.png")
    plot_scatter(rows, "probe_obj_displacement", "final_object_to_target_distance", output_dir / "probe_disp_vs_final_distance.png")

    with (output_dir / "duplicates.json").open("w", encoding="utf-8") as f:
        json.dump([list(item) for item in duplicates], f, indent=2)


if __name__ == "__main__":
    main()
