from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_det import RDDDetectionDataset, DetectionTransform, detection_collate_fn
from src.models.detection_factory import create_detection_model
from src.utils import ensure_dir, get_device, load_checkpoint, save_json


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


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: [N,4], boxes2: [M,4] in xyxy format
    returns IoU matrix [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N,M,2]
    wh = (rb - lt).clamp(min=0)                          # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]                   # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def compute_val_loss(model, loader, device) -> float:
    """
    Detection models in torchvision return losses in train mode when targets are passed.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_images = []
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        preds = model(images_dev)

        for img, tgt, pred in zip(images, targets, preds):
            # move prediction tensors to cpu
            pred_cpu = {k: v.detach().cpu() for k, v in pred.items()}
            tgt_cpu = {k: v.detach().cpu() for k, v in tgt.items()}
            all_images.append(img.cpu())
            all_targets.append(tgt_cpu)
            all_preds.append(pred_cpu)

    return all_images, all_targets, all_preds


def match_detections_for_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> Tuple[List[int], int]:
    """
    Greedy matching for a single class.
    Returns:
      - list of TP flags aligned with sorted predictions
      - number of GT boxes
    """
    if pred_boxes.numel() == 0:
        return [], int(gt_boxes.shape[0])

    if gt_boxes.numel() == 0:
        return [0] * pred_boxes.shape[0], 0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    tp_flags: List[int] = []

    for i in range(pred_boxes.shape[0]):
        best_iou, best_j = torch.max(ious[i], dim=0)
        j = int(best_j.item())
        if float(best_iou.item()) >= iou_thresh and j not in matched_gt:
            tp_flags.append(1)
            matched_gt.add(j)
        else:
            tp_flags.append(0)

    return tp_flags, int(gt_boxes.shape[0])


def compute_detection_metrics(
    targets: List[Dict[str, torch.Tensor]],
    preds: List[Dict[str, torch.Tensor]],
    label_map: Dict[str, int],
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes:
      - precision@IoU
      - recall@IoU
      - F1@IoU
      - per-class versions
      - a simple AP50 approximation from precision-recall ranking
    """
    class_ids = sorted(label_map.values())
    id_to_label = {v: k for k, v in label_map.items()}

    per_class = {}
    macro_precision = []
    macro_recall = []
    macro_f1 = []
    macro_ap50 = []

    for cid in class_ids:
        scores_all: List[float] = []
        tp_flags_all: List[int] = []
        total_gt = 0

        for tgt, pred in zip(targets, preds):
            gt_mask = tgt["labels"] == cid
            gt_boxes = tgt["boxes"][gt_mask]

            pred_mask = (pred["labels"] == cid) & (pred["scores"] >= score_thresh)
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            tp_flags, gt_count = match_detections_for_class(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                gt_boxes=gt_boxes,
                iou_thresh=iou_thresh,
            )

            if pred_scores.numel() > 0:
                scores_sorted, _ = torch.sort(pred_scores, descending=True)
                scores_all.extend([float(x) for x in scores_sorted.tolist()])
                tp_flags_all.extend(tp_flags)

            total_gt += gt_count

        # Sort again globally by score
        if len(scores_all) > 0:
            order = sorted(range(len(scores_all)), key=lambda i: scores_all[i], reverse=True)
            tp_flags_all = [tp_flags_all[i] for i in order]

        fp_flags_all = [1 - x for x in tp_flags_all]

        tp_cum = []
        fp_cum = []
        running_tp = 0
        running_fp = 0
        for tp, fp in zip(tp_flags_all, fp_flags_all):
            running_tp += tp
            running_fp += fp
            tp_cum.append(running_tp)
            fp_cum.append(running_fp)

        if len(tp_cum) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            ap50 = 0.0
        else:
            precisions = [tp / max(tp + fp, 1) for tp, fp in zip(tp_cum, fp_cum)]
            recalls = [tp / max(total_gt, 1) for tp in tp_cum]

            precision = precisions[-1]
            recall = recalls[-1]
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

            # Simple AP approximation using trapezoidal area under PR curve
            prev_recall = 0.0
            ap50 = 0.0
            for p, r in zip(precisions, recalls):
                ap50 += p * max(r - prev_recall, 0.0)
                prev_recall = r

        per_class[id_to_label[cid]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap50,
            "support_gt": total_gt,
            "num_predictions": len(tp_flags_all),
        }

        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)
        macro_ap50.append(ap50)

    metrics = {
        "precision@50": sum(macro_precision) / max(len(macro_precision), 1),
        "recall@50": sum(macro_recall) / max(len(macro_recall), 1),
        "f1@50": sum(macro_f1) / max(len(macro_f1), 1),
        "map50_approx": sum(macro_ap50) / max(len(macro_ap50), 1),
        "per_class": per_class,
    }
    return metrics


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
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
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
    device = get_device()

    out_cfg = cfg["outputs"]
    ensure_dir(out_cfg["root_dir"])
    ensure_dir(out_cfg["logs_dir"])
    ensure_dir(Path(out_cfg["root_dir"]) / "figures")

    data_cfg = cfg["data"]
    allowed_labels = data_cfg["allowed_labels"]

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
    )

    num_workers = cfg["train"].get("num_workers", 2)
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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

    val_loss = compute_val_loss(model, loader, device)
    images, targets, preds = collect_predictions(model, loader, device)
    metrics = compute_detection_metrics(
        targets=targets,
        preds=preds,
        label_map=ds.label_map,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    id_to_label = {v: k for k, v in ds.label_map.items()}

    print("=" * 100)
    print(f"Config:              {args.config}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Split:               {args.split}")
    print(f"Loaded epoch:        {ckpt_meta.get('epoch', 'N/A')}")
    print(f"Samples:             {len(ds)}")
    print("-" * 100)
    print(f"Val loss:            {val_loss:.4f}")
    print(f"Precision@50:        {metrics['precision@50']:.4f}")
    print(f"Recall@50:           {metrics['recall@50']:.4f}")
    print(f"F1@50:               {metrics['f1@50']:.4f}")
    print(f"mAP50 approx:        {metrics['map50_approx']:.4f}")
    print("=" * 100)

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

    n_viz = min(args.max_viz, len(images))
    for i in range(n_viz):
        draw_boxes(
            image_tensor=images[i],
            target=targets[i],
            pred=preds[i],
            out_path=fig_dir / f"sample_{i:03d}.png",
            id_to_label=id_to_label,
            score_thresh=args.score_thresh,
        )

    print(f"Saved metrics to:      {log_path}")
    print(f"Saved prediction figs: {fig_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
