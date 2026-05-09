from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.data.dataset_det import RDDDetectionDataset, DetectionTransform, detection_collate_fn
from src.models.detection_factory import create_detection_model
from src.utils import (
    compute_detection_metrics, 
    ensure_dir, 
    get_device, 
    load_checkpoint, 
    save_json, 
    set_seed,
    setup_logging,
)

SEP = "=" * 100
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Faster R-CNN on RDD2022.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--score_thresh", type=float, default=0.5, help="Confidence threshold for predicted boxes.")
    p.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for a correct detection.")
    p.add_argument("--max_viz", type=int, default=8, help="Number of qualitative prediction images to save.")
    return p.parse_args()

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def run_predictions(model, loader, device, max_viz: int):

    # val loss
    model.train()
    total_loss = 0.0
    total_batches = 0
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images_dev, targets_dev)
        total_loss += float(sum(loss_dict.values()).item())
        total_batches += 1
    val_loss = total_loss / max(total_batches, 1)

    # predictions
    model.eval()
    all_targets: List[Dict] = []
    all_preds: List[Dict] = []
    viz_images: List = []
    viz_targets: List = []
    viz_preds: List = []

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        batch_preds = model(images_dev)

        for img, tgt, pred in zip(images, targets, batch_preds):
            pred_cpu = {k: v.detach().cpu() for k, v in pred.items()}
            tgt_cpu  = {k: v.detach().cpu() for k, v in tgt.items()}

            all_targets.append({"boxes": tgt_cpu["boxes"], "labels": tgt_cpu["labels"]})
            all_preds.append(pred_cpu)

            if len(viz_images) < max_viz:
                viz_images.append(img.cpu())
                viz_targets.append(tgt_cpu)
                viz_preds.append(pred_cpu)

    return val_loss, all_targets, all_preds, viz_images, viz_targets, viz_preds

def draw_boxes(
    image_tensor: torch.Tensor,
    target: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    out_path: str | Path,
    id_to_label: Dict[int, str],
    score_thresh: float = 0.5,
) -> None:
    image = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")

    # Ground truth in green
    for box, label in zip(target["boxes"], target["labels"]):
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(xmin, max(ymin - 4, 0), f"GT:{id_to_label[int(label)]}", color="lime", fontsize=8)

    # Predictions in red
    keep = pred["scores"] >= score_thresh
    boxes = pred["boxes"][keep]
    labels = pred["labels"][keep]
    scores = pred["scores"][keep]

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box.tolist()
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
            min(ymax + 10, image.shape[0] - 1),
            f"PR:{id_to_label[int(label)]} {float(score):.2f}",
            color="red",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(Path(cfg["outputs"]["logs_dir"]) / "eval_det.log")
    set_seed(cfg.get("seed", 1337))
    device = get_device()

    out_cfg = cfg["outputs"]
    ensure_dir(out_cfg["root_dir"])
    ensure_dir(out_cfg["logs_dir"])
    ensure_dir(Path(out_cfg["root_dir"]) / "figures")

    data_cfg = cfg["data"]
    allowed_labels = data_cfg["allowed_labels"]

    cache_path = data_cfg.get(
        "ann_cache_path",
        str(Path("outputs") / f".ann_cache_{args.split}.pkl"),
    )

    ds = RDDDetectionDataset(
        rdd_root=data_cfg["rdd_root"],
        split=args.split,
        allowed_labels=allowed_labels,
        transform=DetectionTransform(train=False),
        countries=data_cfg.get("countries"),
        split_mode=data_cfg.get("split_mode", "random"),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 1337),
        xml_glob=data_cfg.get("xml_glob", "**/annotations/xmls/*.xml"),
        image_dir_hint=data_cfg.get("image_dir_hint", "images"),
        cache_path=cache_path,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )

    num_classes = len(ds.label_map) + 1
    model_cfg = cfg["model"]
    model = create_detection_model(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    ckpt_meta = load_checkpoint(args.checkpoint, model=model, optimizer=None, map_location=device)

    val_loss, all_targets, all_preds, viz_images, viz_targets, viz_preds = run_predictions(
        model, loader, device, max_viz=args.max_viz,
    )

    metrics = compute_detection_metrics(
        targets=all_targets,
        preds=all_preds,
        label_map=ds.label_map,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    id_to_label = {v: k for k, v in ds.label_map.items()}

    logger.info(
        "\n%s\n"
        "Config:              %s\n"
        "Checkpoint:          %s\n"
        "Split:               %s\n"
        "Loaded epoch:        %s\n"
        "Samples:             %s\n"
        "%s\n"
        "Val loss:            %.4f\n"
        "Precision@50:        %.4f\n"
        "Recall@50:           %.4f\n"
        "F1@50:               %.4f\n"
        "mAP50 approx:        %.4f\n"
        "%s",
        SEP,
        args.config,
        args.checkpoint,
        args.split,
        ckpt_meta.get("epoch", "N/A"),
        len(ds),
        SEP,
        val_loss,
        metrics["precision@50"],
        metrics["recall@50"],
        metrics["f1@50"],
        metrics["map50_approx"],
        SEP,
    )  # pylint: disable=logging-too-many-args

    metrics_out = {
        "config": cfg,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "val_loss": val_loss,
        "precision@50": metrics["precision@50"],
        "recall@50": metrics["recall@50"],
        "f1@50": metrics["f1@50"],
        "map50_approx": metrics["map50_approx"],
        "per_class": metrics["per_class"],
        "checkpoint_meta": ckpt_meta,
    }

    log_path = Path(out_cfg["logs_dir"]) / f"eval_det_{args.split}.json"
    save_json(metrics_out, log_path)

    fig_dir = Path(out_cfg["root_dir"]) / "figures" / f"predictions_{args.split}"
    ensure_dir(fig_dir)

    for i, (img, tgt, pred) in enumerate(zip(viz_images, viz_targets, viz_preds)):
        draw_boxes(
            image_tensor=img,
            target=tgt,
            pred=pred,
            out_path=fig_dir / f"sample_{i:03d}.png",
            id_to_label=id_to_label,
            score_thresh=args.score_thresh,
        )

    logger.info(
        "Saved metrics to:      %s\n"
        "Saved prediction figs: %s",
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
