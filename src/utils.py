from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List,Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    average: str = "macro",
) -> Dict[str, Any]:

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1_per_class.tolist(),
        "per_class_support": support.tolist(),
    }


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: float,
    config: Optional[Dict[str, Any]] = None,
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
    }
    torch.save(ckpt, path)

def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["model_state_dict"]

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {}
        for k, v in state_dict.items():
            new_k = k
            for prefix in ("_orig_mod.", "_orig_mod_"):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            stripped[new_k] = v
        model.load_state_dict(stripped)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "best_metric": ckpt.get("best_metric", 0.0),
        "config": ckpt.get("config", None),
    }

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU matrix [N, M] for xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)

def _match_detections_for_class(pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor, gt_boxes: torch.Tensor,
    iou_thresh: float,) -> Tuple[List[int], int]:

    if pred_boxes.numel() == 0:
        return [], int(gt_boxes.shape[0])
    if gt_boxes.numel() == 0:
        return [0] * pred_boxes.shape[0], 0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt: set = set()
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

def compute_detection_metrics(targets: List[Dict[str, torch.Tensor]],
    preds: List[Dict[str, torch.Tensor]], label_map: Dict[str, int],
    score_thresh: float = 0.5, iou_thresh: float = 0.5,) -> Dict[str, Any]:
    
    class_ids = sorted(label_map.values())
    id_to_label = {v: k for k, v in label_map.items()}

    per_class: Dict[str, Any] = {}
    macro_precision, macro_recall, macro_f1, macro_ap50 = [], [], [], []

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

            tp_flags, gt_count = _match_detections_for_class(
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

        if scores_all:
            order = sorted(range(len(scores_all)), key=lambda i: scores_all[i], reverse=True)
            tp_flags_all = [tp_flags_all[i] for i in order]

        fp_flags_all = [1 - x for x in tp_flags_all]
        tp_cum, fp_cum, running_tp, running_fp = [], [], 0, 0
        for tp, fp in zip(tp_flags_all, fp_flags_all):
            running_tp += tp
            running_fp += fp
            tp_cum.append(running_tp)
            fp_cum.append(running_fp)

        if not tp_cum:
            precision = recall = f1 = ap50 = 0.0
        else:
            precisions = [tp / max(tp + fp, 1) for tp, fp in zip(tp_cum, fp_cum)]
            recalls = [tp / max(total_gt, 1) for tp in tp_cum]
            precision = precisions[-1]
            recall = recalls[-1]
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            prev_recall, ap50 = 0.0, 0.0
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

    n = max(len(macro_precision), 1)
    return {
        "precision@50": sum(macro_precision) / n,
        "recall@50": sum(macro_recall) / n,
        "f1@50": sum(macro_f1) / n,
        "map50_approx": sum(macro_ap50) / n,
        "per_class": per_class,
    }