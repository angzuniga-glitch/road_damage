"""
model_comparison.py

Comparison across all models and training configurations.
Uses models trained from scratch as baselinnes for each model.

Reads eval JSON files already produced by eval.py / eval_det.py / eval_yolo.py
and produces:
  - Baseline comparison table where the models trained from scratch are the baselines.
  - Cross-model comparison table of finetuned models.
  - Per-class breakdown across all models
  - Summary markdown report

Usage:
    python -m src.model_comparison --results_dir outputs/ --out_dir outputs/model_comparisons

The script discovers results automatically by scanning for eval JSON files.
You can also pass explicit paths:

    python -m src.model_comparison \
        --fasterrcnn_finetune outputs/fasterrcnn_finetune/logs/eval_det_val.json \
        --fasterrcnn_frozen   outputs/fasterrcnn_frozen/logs/eval_det_val.json \
        --fasterrcnn_scratch  outputs/fasterrcnn_scratch/logs/eval_det_val.json \
        --resnet18_finetune   outputs/resnet18_finetune/logs/eval_val.json \
        --resnet18_frozen     outputs/resnet18_frozen/logs/eval_val.json \
        --resnet18_scratch    outputs/resnet18_scratch/logs/eval_val.json \
        --vit_finetune        outputs/vit_finetune/logs/eval_val.json \
        --vit_frozen          outputs/vit_frozen/logs/eval_val.json \
        --vit_scratch         outputs/vit_scratch/logs/eval_val.json \
        --yolo_finetune       outputs/yolov8n_finetune/logs/eval_yolo_val.json \
        --yolo_frozen         outputs/yolov8n_frozen/logs/eval_yolo_val.json \
        --yolo_scratch        outputs/yolov8n_scratch/logs/eval_yolo_val.json \
        --customcnn           outputs/customcnn/logs/eval_val.json \
        --out_dir outputs/model_comparisons
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import ensure_dir, save_json, setup_logging

SEP = "-" * 100
logger = logging.getLogger(__name__)

# RDD2022 class descriptions
CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal cracking  — cracks parallel to road direction",
    "D10": "Transverse cracking    — cracks perpendicular to road direction",
    "D20": "Alligator cracking     — interconnected mesh-pattern cracks",
    "D40": "Pothole                — bowl-shaped holes in road surface",
}


def load_result(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load an eval JSON file. Returns None if path is None or file missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("Result file not found: %s", path)
        return None
    with open(p, "r") as f:
        return json.load(f)


def extract_metrics(result: Dict[str, Any], model_family: str) -> Dict[str, float]:
    """
    Extract a normalised metrics dict from any eval JSON format.
    Detection models (fasterrcnn, yolo) use different key names than
    classification models (resnet, vit, customcnn).
    """
    if result is None:
        return {}

    # for fasterrcnn and yolo
    if "precision@50" in result:
        return {
            "precision": result.get("precision@50", 0.0),
            "recall": result.get("recall@50", 0.0),
            "f1": result.get("f1@50", 0.0),
            "map50": result.get("map50_approx", result.get("map50", 0.0)),
        }

    # Classification models — eval.py format
    if "accuracy" in result:

        # macro-F1 is primary metric, map accuracy to map50 plot
        return {
            "precision": result.get("macro_f1", 0.0),
            "recall": result.get("accuracy", 0.0),
            "f1": result.get("macro_f1", 0.0),
            "map50": result.get("macro_f1", 0.0),
            "accuracy": result.get("accuracy", 0.0),
        }

    return {}


def extract_per_class(result: Dict[str, Any]) -> Dict[str, Dict]:
    """Extract per-class metrics from eval JSON."""
    if result is None:
        return {}
    return result.get("per_class", {})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified model comparison using scratch variants as baselines."
    )

    # Auto-discovery
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Root outputs/ directory. Script will auto-discover eval JSONs.",
    )

    # Explicit paths
    for name in [
        "fasterrcnn_finetune",
        "fasterrcnn_frozen",
        "fasterrcnn_scratch",
        "resnet18_finetune",
        "resnet18_frozen",
        "resnet18_scratch",
        "vit_finetune",
        "vit_frozen",
        "vit_scratch",
        "yolo_finetune",
        "yolo_frozen",
        "yolo_scratch",
        "customcnn",
    ]:
        p.add_argument(
            f"--{name}", type=str, default=None, help=f"Path to eval JSON for {name}."
        )

    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/model_comparisons",
        help="Directory for comparison outputs.",
    )
    return p.parse_args()


# Auto-discovery
DISCOVERY_PATHS = {
    "fasterrcnn_finetune": "fasterrcnn_finetune/logs/eval_det_val.json",
    "fasterrcnn_frozen": "fasterrcnn_frozen/logs/eval_det_val.json",
    "fasterrcnn_scratch": "fasterrcnn_scratch/logs/eval_det_val.json",
    "resnet18_finetune": "resnet18_finetune/logs/eval_val.json",
    "resnet18_frozen": "resnet18_frozen/logs/eval_val.json",
    "resnet18_scratch": "resnet18_scratch/logs/eval_val.json",
    "vit_finetune": "vit_finetune/logs/eval_val.json",
    "vit_frozen": "vit_frozen/logs/eval_val.json",
    "vit_scratch": "vit_scratch/logs/eval_val.json",
    "yolo_finetune": "yolov8n_finetune/logs/eval_yolo_val.json",
    "yolo_frozen": "yolov8n_frozen/logs/eval_yolo_val.json",
    "yolo_scratch": "yolov8n_scratch/logs/eval_yolo_val.json",
    "customcnn": "customcnn/logs/eval_val.json",
}


def discover_results(results_dir: str, args: argparse.Namespace) -> Dict[str, str]:
    """Build dict combining auto-discovery with explicit --model_key overrides."""
    root = Path(results_dir)
    paths = {}

    for key, rel in DISCOVERY_PATHS.items():
        candidate = root / rel
        if candidate.exists():
            paths[key] = str(candidate)

    # CLI overrides take priority
    for key in DISCOVERY_PATHS:
        val = getattr(args, key, None)
        if val is not None:
            paths[key] = val

    return paths


# Formatting helpers
def fmt(v: Any, decimals: int = 4) -> str:
    """Format float to fixed decimals or return N/A for missing vals."""
    if v is None or v == {}:
        return "  N/A  "
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "  N/A  "


def delta(new: float, base: float) -> str:
    """Format signed delta with +/-."""
    if new is None or base is None:
        return "     —"
    d = new - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


# Table printers
def print_training_mode_ablation(
    family: str,
    scratch: Dict,
    frozen: Dict,
    finetune: Dict,
    metric: str = "map50",
) -> str:
    """show scratch to frozen to finetune progression."""
    s = scratch.get(metric)
    fr = frozen.get(metric)
    ft = finetune.get(metric)

    lines = [
        f"\n  {family.upper()} — Training mode ablation  (metric: {metric})",
        f"  {'Mode':<20} {'Value':>8}  {'delta vs scratch':>14}",
        f"  {'-'*45}",
        f"  {'Scratch (baseline)':<20} {fmt(s):>8}  {'—':>14}",
        f"  {'Frozen backbone':<20} {fmt(fr):>8}  {delta(fr, s) if fr and s else '—':>14}",
        f"  {'Full finetune':<20} {fmt(ft):>8}  {delta(ft, s) if ft and s else '—':>14}",
    ]
    return "\n".join(lines)


def build_summary_table(all_results: Dict[str, Dict]) -> str:
    """Builds main comparison table with all models and configs with scratch as [BASE]."""
    col_w = 28
    header = (
        f"{'Model':<{col_w}} {'Precision':>12} {'Recall':>10} "
        f"{'F1':>10} {'mAP50*':>10}  Notes"
    )
    rows = [SEP, header, SEP]

    families = [
        (
            "FASTER R-CNN",
            ["fasterrcnn_scratch", "fasterrcnn_frozen", "fasterrcnn_finetune"],
        ),
        ("RESNET-18", ["resnet18_scratch", "resnet18_frozen", "resnet18_finetune"]),
        ("VIT", ["vit_scratch", "vit_frozen", "vit_finetune"]),
        ("YOLOV8N", ["yolo_scratch", "yolo_frozen", "yolo_finetune"]),
        ("CUSTOM CNN", ["customcnn"]),
    ]

    mode_labels = {
        "scratch": "[BASE] Scratch",
        "frozen": "       Frozen ",
        "finetune": "       Finetune",
        "customcnn": "       CustomCNN",
    }

    for family_name, keys in families:
        rows.append(f"\n  {family_name}")
        for key in keys:
            m = all_results.get(key, {})
            if not m:
                continue

            # Determine mode label
            if "scratch" in key or key == "customcnn":
                label = mode_labels.get(
                    "scratch" if "scratch" in key else "customcnn", key
                )
                note = "baseline (no pretraining)"
            elif "frozen" in key:
                label = mode_labels["frozen"]
                note = ""
            else:
                label = mode_labels["finetune"]
                note = "best variant"

            rows.append(
                f"  {label:<{col_w}} {fmt(m.get('precision')):>10} "
                f"{fmt(m.get('recall')):>10} {fmt(m.get('f1')):>10} "
                f"{fmt(m.get('map50')):>10}  {note}"
            )

        rows.append("")

    rows.append(SEP)

    return "\n".join(rows)


def build_per_class_table(all_results: Dict[str, Dict], raw_results: Dict) -> str:
    """Per-class AP50/F1 across all finetuned models."""
    classes = list(CLASS_DESCRIPTIONS.keys())
    col_w = 26

    header_parts = [f"{'Model':<{col_w}}"]
    for cls in classes:
        header_parts.append(f"{cls:>10}")
    header = " ".join(header_parts)

    rows = [
        SEP,
        "  Per-class AP50 / F1 — finetune models only",
        "  "
        + "\n  ".join(f"{cls}: {desc}" for cls, desc in CLASS_DESCRIPTIONS.items()),
        SEP,
        "  " + header,
        SEP,
    ]

    finetune_keys = [
        "fasterrcnn_finetune",
        "resnet18_finetune",
        "vit_finetune",
        "yolo_finetune",
        "customcnn",
    ]

    for key in finetune_keys:
        result = raw_results.get(key)
        if result is None:
            continue
        per_class = extract_per_class(result)
        if not per_class:
            continue

        row_parts = [f"  {key:<{col_w}}"]
        for cls in classes:
            cm = per_class.get(cls, {})
            val = cm.get("ap50", cm.get("f1", None))
            row_parts.append(f"{fmt(val):>10}")
        rows.append(" ".join(row_parts))

    rows.append(SEP)

    return "\n".join(rows)


def build_markdown_report(
    summary_table: str,
    per_class_table: str,
    ablation_blocks: str,
    all_metrics: Dict,
) -> str:
    return f"""# Road Damage Detection — Model Comparison Report

## Baseline approach

Scratch variants (no pretrained weights) serve as the baseline for each model
family. This isolates the contribution of ImageNet pretraining and end-to-end
fine-tuning directly from the RDD2022 dataset, without reference to external
literature benchmarks.

## Class definitions

All macro metrics are averaged equally across these 4 classes:

| Class | Description |
|-------|-------------|
```
{CLASS_DESCRIPTIONS}
```

## All models comparison

```
{summary_table}
```

## Ablation (scratch to frozen to finetune)

```
{ablation_blocks}
```

## Per-class breakdown across finetuned models

```
{per_class_table}
```
"""


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    setup_logging(out_dir / "model_comparisons.log")

    # Discover / load results
    if args.results_dir:
        paths = discover_results(args.results_dir, args)
    else:
        paths = {}
        for key in DISCOVERY_PATHS:
            val = getattr(args, key, None)
            if val is not None:
                paths[key] = val

    if not paths:
        logger.warning(
            "No result files found. Run eval scripts first, then provide "
            "--results_dir or explicit --model_key paths."
        )

    # Load raw JSON results
    raw: Dict[str, Optional[Dict]] = {key: load_result(p) for key, p in paths.items()}

    # Extract normalized metrics
    all_metrics: Dict[str, Dict] = {}
    for key, result in raw.items():
        family = key.rsplit("_", 1)[0] if "_" in key else key
        m = extract_metrics(result, family)
        if m:
            all_metrics[key] = m

    logger.info("Loaded results for: %s", ", ".join(all_metrics.keys()) or "none")

    summary_table = build_summary_table(all_metrics)
    logger.info("\n%s\nMODEL COMPARISON SUMMARY\n%s", SEP, summary_table)

    per_class_table = build_per_class_table(all_metrics, raw)
    logger.info("\n%s\nPER-CLASS BREAKDOWN\n%s", SEP, per_class_table)

    ablation_blocks = []
    for family, scratch_key, frozen_key, finetune_key in [
        (
            "Faster R-CNN",
            "fasterrcnn_scratch",
            "fasterrcnn_frozen",
            "fasterrcnn_finetune",
        ),
        ("ResNet-18", "resnet18_scratch", "resnet18_frozen", "resnet18_finetune"),
        ("ViT", "vit_scratch", "vit_frozen", "vit_finetune"),
        ("YOLOv8n", "yolo_scratch", "yolo_frozen", "yolo_finetune"),
    ]:
        block = print_training_mode_ablation(
            family=family,
            scratch=all_metrics.get(scratch_key, {}),
            frozen=all_metrics.get(frozen_key, {}),
            finetune=all_metrics.get(finetune_key, {}),
        )
        ablation_blocks.append(block)
        logger.info(block)

    save_json(
        {
            "all_metrics": all_metrics,
            "class_descriptions": CLASS_DESCRIPTIONS,
        },
        out_dir / "model_comparisons.json",
    )

    md = build_markdown_report(
        summary_table=summary_table,
        per_class_table=per_class_table,
        ablation_blocks="\n".join(ablation_blocks),
        all_metrics=all_metrics,
    )
    (out_dir / "model_comparisons_report.md").write_text(md)

    logger.info(SEP)
    logger.info(
        "Comparison complete\n"
        "Outputs saved to: %s\n"
        "  model_comparisons.json\n"
        "  model_comparisons_report.md\n"
        "  model_comparisons.log",
        out_dir,
    )
    logger.info(SEP)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Comparison failed with error: {e}", flush=True)
        raise
