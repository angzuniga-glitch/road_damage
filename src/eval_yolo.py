from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.utils import ensure_dir, save_json, set_seed, setup_logging
from src.data.xml_utils import load_detection_records, parse_voc_xml

SEP = "-" * 100
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 model on RDD2022."
    )
    p.add_argument(
        "--config", type=str, required=True, help="Path to YOLO YAML config."
    )
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained weights."
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--max_viz", type=int, default=8)
    return p.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_boxes(
    image_path: str | Path,
    gt_boxes: List[List[float]],
    gt_labels: List[int],
    pred_boxes: List[List[float]],
    pred_labels: List[int],
    pred_scores: List[float],
    out_path: str | Path,
    id_to_label: Dict[int, str],
    score_thresh: float = 0.5,
) -> None:
    """Save a single image with GT (green) and predicted (red dashed) boxes."""

    img = PILImage.open(image_path).convert("RGB")
    img_arr = np.array(img)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_arr)
    ax.axis("off")

    # Ground truth — green
    for box, label in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            max(ymin - 4, 0),
            f"GT:{id_to_label.get(label, label)}",
            color="lime",
            fontsize=8,
        )

    # Predictions — red dashed
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            min(ymax + 10, img_arr.shape[0] - 1),
            f"PR:{id_to_label.get(label, label)} {score:.2f}",
            color="red",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    except ImportError:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics")

    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(Path(cfg["outputs"]["root_dir"]) / "logs" / "eval_yolo.log")
    set_seed(cfg.get("seed", 1337))

    SETTINGS.update({"runs_dir": str(Path.cwd())})
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    out_cfg = cfg["outputs"]
    data_cfg = cfg["data"]
    allowed_labels = data_cfg["allowed_labels"]
    label_map = {lab: i for i, lab in enumerate(allowed_labels)}
    id_to_label = {i: lab for lab, i in label_map.items()}

    logs_dir = Path(out_cfg["root_dir"]) / "logs"
    fig_dir = Path(out_cfg["root_dir"]) / "figures" / f"predictions_{args.split}"
    ensure_dir(logs_dir)
    ensure_dir(fig_dir)

    model = YOLO(args.checkpoint)
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    yolo_data_dir = Path(cfg.get("yolo_data_dir", "yolo_dataset"))
    dataset_yaml = yolo_data_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}. "
            "Run train_yolo.py first to convert the dataset."
        )

    val_results = model.val(
        data=str(dataset_yaml),
        split=args.split,
        conf=args.score_thresh,
        iou=args.iou_thresh,
        device=0,
        verbose=False,
        save_json=False,
        project=out_cfg["root_dir"],
        name=f"eval_{args.split}",
        exist_ok=True,
    )

    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)

    ap50_per_class = val_results.box.ap50
    p_per_class = val_results.box.p
    r_per_class = val_results.box.r

    per_class: Dict[str, Any] = {}
    for i, lab in enumerate(allowed_labels):
        ap = float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0
        p_c = float(p_per_class[i]) if i < len(p_per_class) else 0.0
        r_c = float(r_per_class[i]) if i < len(r_per_class) else 0.0
        f1 = 0.0 if (p_c + r_c) == 0 else 2 * p_c * r_c / (p_c + r_c)
        per_class[lab] = {
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "f1": round(f1, 4),
            "ap50": round(ap, 4),
        }

    header = "%-12s %10s %10s %10s %10s" % (
        "Class",
        "Precision",
        "Recall",
        "F1",
        "AP50",
    )
    class_rows = "\n".join(
        "%-12s %10.4f %10.4f %10.4f %10.4f"
        % (
            lab,
            m["precision"],
            m["recall"],
            m["f1"],
            m["ap50"],
        )
        for lab, m in per_class.items()
    )

    logger.info(
        "\n%s\n"
        "Config:          %s\n"
        "Checkpoint:      %s\n"
        "Split:           %s\n"
        "Score threshold: %s\n"
        "IoU threshold:   %s\n"
        "%s\n"
        "Precision@50:    %.4f\n"
        "Recall@50:       %.4f\n"
        "mAP50:           %.4f\n"
        "mAP50-95:        %.4f\n"
        "%s\n"
        "%s\n"
        "%s\n"
        "%s\n"
        "%s",
        SEP,
        args.config,
        args.checkpoint,
        args.split,
        args.score_thresh,
        args.iou_thresh,
        SEP,
        precision,
        recall,
        map50,
        map50_95,
        SEP,
        header,
        SEP,
        class_rows,
        SEP,
    )  # pylint: disable=logging-too-many-args

    metrics_out = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "precision@50": round(precision, 4),
        "recall@50": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class": per_class,
    }
    log_path = logs_dir / f"eval_yolo_{args.split}.json"
    save_json(metrics_out, log_path)

    # Visualization
    records = load_detection_records(
        rdd_root=data_cfg["rdd_root"],
        allowed_labels=allowed_labels,
        countries=data_cfg.get("countries"),
        split_mode=data_cfg.get("split_mode", "random"),
        split=args.split,
        xml_glob=data_cfg.get("xml_glob", "**/annotations/xmls/*.xml"),
        image_dir_hint=data_cfg.get("image_dir_hint", "images"),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 1337),
    )

    n_viz = min(args.max_viz, len(records))
    logger.info("Saving %s prediction figures to %s", n_viz, fig_dir)

    for i, rec in enumerate(records[:n_viz]):
        img_path = rec["image_path"]
        ann_path = rec["ann_path"]

        # Ground truth
        _img_w, _img_h, objects = parse_voc_xml(ann_path)
        gt_boxes = [[float(x) for x in box] for _, box in objects if _ in label_map]
        gt_labels = [label_map[lab] for lab, _ in objects if lab in label_map]

        # YOLO prediction
        results = model.predict(
            source=img_path,
            conf=args.score_thresh,
            iou=args.iou_thresh,
            device=0,
            verbose=False,
        )

        r = results[0]
        pred_boxes = r.boxes.xyxy.cpu().tolist()
        pred_labels = r.boxes.cls.cpu().int().tolist()
        pred_scores = r.boxes.conf.cpu().tolist()

        draw_boxes(
            image_path=img_path,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            out_path=fig_dir / f"sample_{i:03d}.png",
            id_to_label=id_to_label,
            score_thresh=args.score_thresh,
        )

    logger.info(
        "Saved metrics to:      %s\n" "Saved prediction figs: %s",
        log_path,
        fig_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Evaluation failed with error: {e}", flush=True)
        raise
