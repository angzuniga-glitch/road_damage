"""
analysis.py

Produces:

  1. GradCAM/Attention Rollout visualizations for sampled images
  2. Best-3/Worst-3 qualitative analysis with side/by/side image + heatmap panels
  3. Model metrics summary
  4. Metric breakdown per-class with class descriptions
  5. Summary report saved as JSON and markdown

Usage:

ViT
============================================================================================================================================
python -m src.analysis --config configs/vit_finetune.yaml --checkpoint outputs/vit_finetune/checkpoints/best.pt --model_type vit --split val
python -m src.analysis --config configs/vit_frozen.yaml --checkpoint outputs/vit_frozen/checkpoints/best.pt --model_type vit --split val
python -m src.analysis --config configs/vit_scratch.yaml --checkpoint outputs/vit_scratch/checkpoints/best.pt --model_type vit --split val

ResNet
======================================================================================================================================================
python -m src.analysis --config configs/resnet18_finetune.yaml --checkpoint outputs/resnet18_finetune/checkpoints/best.pt --model_type resnet18 --split val
python -m src.analysis --config configs/resnet18_frozen.yaml --checkpoint outputs/resnet18_frozen/checkpoints/best.pt --model_type resnet18 --split val
python -m src.analysis --config configs/resnet18_scratch.yaml --checkpoint outputs/resnet18_scratch/checkpoints/best.pt --model_type resnet18 --split val

YOLO
===================================================================================================================================================
python -m src.analysis --config configs/yolo_finetune.yaml --checkpoint outputs/yolov8n_finetune/train/weights/best.pt --model_type yolo --split val
python -m src.analysis --config configs/yolo_frozen.yaml --checkpoint outputs/yolov8n_frozen/train/weights/best.pt --model_type yolo --split val
python -m src.analysis --config configs/yolo_scratch.yaml --checkpoint outputs/yolov8n_scratch/train/weights/best.pt --model_type yolo --split val

Faster R-CNN
====================================================================================================================================================
python -m src.analysis --config configs/fasterrcnn_finetune.yaml --checkpoint outputs/fasterrcnn_finetune/checkpoints/best.pt --model_type fasterrcnn
python -m src.analysis --config configs/fasterrcnn_frozen.yaml --checkpoint outputs/fasterrcnn_frozen/checkpoints/best.pt --model_type fasterrcnn
python -m src.analysis --config configs/fasterrcnn_scratch.yaml --checkpoint outputs/fasterrcnn_scratch/checkpoints/best.pt --model_type fasterrcnn

Custom CNN
=============================================================================================================================================
python -m src.analysis --config configs/customcnn.yaml --checkpoint outputs/customcnn/checkpoints/best.pt --model_type custom_cnn --split val
-------------------------------------------------------------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchvision.transforms import functional as F_tv
from torchvision import transforms as T

from src.data.xml_utils import load_detection_records, parse_voc_xml
from src.data.dataset_det import RDDDetectionDataset, DetectionTransform
from src.data.dataset import RDDBboxCropDataset, load_label_map
from src.models.factory import create_model
from src.models.detection_factory import create_detection_model
from src.utils import compute_detection_metrics, box_iou


from src.utils import (
    ensure_dir,
    get_device,
    load_checkpoint,
    save_json,
    set_seed,
    setup_logging,
)

SEP = "-" * 100
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# RDD2022 class descriptions
CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal cracking  — cracks running parallel to the road direction",
    "D10": "Transverse cracking    — cracks running perpendicular to the road direction",
    "D20": "Alligator cracking     — interconnected cracks forming a mesh pattern",
    "D40": "Pothole                — bowl-shaped holes in the road surface",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive model analysis for road damage detection."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet18", "resnet34", "custom_cnn", "vit", "fasterrcnn", "yolo"],
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--n_samples", type=int, default=500, help="Number of images to run GradCAM on."
    )
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/<model_type>/analysis/",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5, help="GradCAM heatmap overlay transparency."
    )
    return p.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def __call__(self, tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        logits = self.model(tensor)
        pred = int(logits.argmax(dim=1).item())
        cls = target_class if target_class is not None else pred
        self.model.zero_grad()
        logits[0, cls].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        cam = np.clip(cam, np.percentile(cam, 5), np.percentile(cam, 95))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, pred, float(logits.softmax(dim=1)[0, pred].item())

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


class AttentionRollout:
    """Attention Rollout for ViT"""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attention_maps: List[torch.Tensor] = []
        self._hooks = []
        for block in model.blocks:
            hook = block.attn.register_forward_hook(self._save_attention)
            self._hooks.append(hook)

    def _save_attention(self, module, input, output) -> None:
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = (C // module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        self.attention_maps.append(attn.detach().cpu())

    def __call__(self, input_tensor: torch.Tensor):
        self.attention_maps = []
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
        pred_class = int(logits.argmax(dim=1).item())
        result = torch.eye(self.attention_maps[0].shape[-1])
        for attn in self.attention_maps:
            attn_avg = attn.mean(dim=1)[0]
            attn_aug = attn_avg + torch.eye(attn_avg.shape[0])
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
            result = attn_aug @ result
        num_patches = result.shape[0] - 1
        patch_side = int(num_patches**0.5)
        mask = result[0, 1:].reshape(patch_side, patch_side).numpy()
        if mask.max() > 0:
            mask = mask / mask.max()

        mask[0, :] = 0  # top row
        mask[-1, :] = 0  # bottom row
        mask[:, 0] = 0  # left col
        mask[:, -1] = 0  # right col

        if mask.max() > 0:
            mask = mask / mask.max()

        return mask, pred_class

    def remove(self):
        for h in self._hooks:
            h.remove()


class ViTGradCAM:
    """Grad-CAM adapted for ViT by hooking the last transformer block output."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = model.blocks[-1].register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        self._bwd = model.blocks[-1].register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def __call__(self, tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        logits = self.model(tensor)
        pred = int(logits.argmax(dim=1).item())
        cls = target_class if target_class is not None else pred
        self.model.zero_grad()
        logits[0, cls].backward()
        act = self.activations[0, 1:, :]
        grad = self.gradients[0, 1:, :]
        weights = grad.mean(dim=0)
        cam = (act * weights).sum(dim=1)
        cam = F.relu(cam)
        num_patches = cam.shape[0]
        patch_side = int(num_patches**0.5)
        cam = cam.reshape(patch_side, patch_side).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, pred, float(logits.softmax(dim=1)[0, pred].item())

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


class DetectionGradCAM:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        layer = self._find_layer4(model)
        self._fwd = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        self._bwd = layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def _find_layer4(self, model):
        backbone = model.backbone
        body = getattr(backbone, "_orig_mod", backbone)
        body = getattr(body, "body", body)
        if hasattr(body, "layer4"):
            return body.layer4
        raise RuntimeError("Could not find backbone.body.layer4.")

    def __call__(self, tensor: torch.Tensor, score_thresh: float = 0.5):
        self.model.eval()
        for p in self.model.backbone.parameters():
            p.requires_grad_(True)
        tensor = tensor.requires_grad_(True)
        preds = self.model([tensor.squeeze(0)])
        pred = preds[0]
        keep = pred["scores"] >= score_thresh
        if keep.sum() == 0:
            return np.zeros((7, 7)), []
        scores = pred["scores"][keep]
        boxes = pred["boxes"][keep]
        labels = pred["labels"][keep]
        self.model.zero_grad()
        scores.max().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        detections = [
            {
                "box": b.detach().cpu().tolist(),
                "label": int(l.detach().cpu().item()),
                "score": float(s.detach().cpu().item()),
            }
            for b, l, s in zip(boxes, labels, scores)
        ]
        return cam, detections

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


# Visualization
def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = image.shape[:2]
    cam_r = cv2.resize(cam, (w, h))

    cam_r = np.power(cam_r, 0.5)

    hmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)

    return np.clip((1 - alpha) * image + alpha * hmap, 0, 255).astype(np.uint8)


def save_panel(
    original: np.ndarray,
    cam_overlay: np.ndarray,
    out_path: Path,
    title: str,
    detections: Optional[List[Dict]] = None,
    id_to_label: Optional[Dict] = None,
    score_thresh: float = 0.5,
    quality_label: str = "",
    gt_boxes:    Optional[List] = None,   
    gt_labels:   Optional[List] = None,    
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colour_map = {"BEST": "lime", "WORST": "red", "": "white"}
    border_col = colour_map.get(quality_label, "white")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(cam_overlay)
    axes[1].set_title("Grad-CAM", fontsize=11)
    axes[1].axis("off")

    if gt_boxes:
        for box, lab_idx in zip(gt_boxes, gt_labels or []):
            xmin, ymin, xmax, ymax = box
            lab = id_to_label.get(lab_idx, str(lab_idx)) if id_to_label else str(lab_idx)
            for ax in [axes[0], axes[1]]:
                rect = mpatches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor="yellow", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    xmin, max(ymin - 4, 0),
                    f"GT:{lab}",
                    color="yellow", fontsize=8, fontweight="bold",
                )

    if detections:
        for det in detections:
            if det["score"] < score_thresh:
                continue
            xmin, ymin, xmax, ymax = det["box"]
            lab = (
                id_to_label.get(det["label"], str(det["label"]))
                if id_to_label
                else str(det["label"])
            )
            for ax in [axes[0], axes[1]]:
                rect = mpatches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    xmin, max(ymin - 4, 0),
                    f"{lab} {det['score']:.2f}",
                    color="lime", fontsize=8, fontweight="bold",
                )

    if quality_label:
        fig.patch.set_facecolor("#1a1a1a" if quality_label == "WORST" else "#0a1a0a")
        fig.text(
            0.01,
            0.97,
            quality_label,
            color=border_col,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        title, fontsize=10, color="white" if quality_label else "black", y=1.01
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# Baseline comparison
def print_model_metrics(
    model_name: str,
    model_metrics: Dict[str, float],
    out_dir: Path,
) -> None:
    """Prints model metrics and save to JSON."""
    logger.info(
        "\n%s\n"
        "MODEL METRICS — %s"
        "Precision: %.4f"
        "Recall:    %.4f"
        "F1:        %.4f"
        "mAP50:     %.4f"
        "%s\n",
        SEP,
        model_name,
        model_metrics.get("precision", 0.0),
        model_metrics.get("recall", 0.0),
        model_metrics.get("f1", 0.0),
        model_metrics.get("map50", 0.0),
        SEP,
    )
    save_json(model_metrics, out_dir / "model_metrics.json")


def print_per_class_metrics(
    per_class: Dict[str, Dict],
    macro_metrics: Dict[str, float],
    out_dir: Path,
) -> None:

    logger.info(
        "\n%s\n"
        "PER-CLASS METRIC BREAKDOWN\n"
        "Macro metrics (Precision, Recall, F1, mAP50) are computed by averaging\n"
        "equally across all %s damage classes listed below. Each class contributes\n"
        "equally regardless of instance count, which means rare classes like \n"
        "D40: potholes have the same weight as common classes like D00 longitudinal\n"
        "cracking, which can mask poor performance on minority classes.\n"
        "%s\n",
        SEP,
        len(per_class),
        SEP,
    )  # pylint: disable=logging-too-many-args

    class_lines = "\n".join(
        f"  {cls}  —  {desc}"
        for cls, desc in CLASS_DESCRIPTIONS.items()
        if cls in per_class
    )
    logger.info(
        "\n%s\n" "Classes included in macro metric computation:\n" "%s\n" "%s\n",
        SEP,
        class_lines,
        SEP,
    )  # pylint: disable=logging-too-many-args

    header = "%-6s %10s %10s %10s %10s %12s  Description" % (
        "Class",
        "Precision",
        "Recall",
        "F1",
        "AP50",
        "Support GT",
    )
    rows = [SEP, header, SEP]
    for cls, m in per_class.items():
        desc = CLASS_DESCRIPTIONS.get(cls, "")
        support = m.get("support_gt", m.get("support", "N/A"))
        rows.append(
            "%-6s %10.4f %10.4f %10.4f %10.4f %12s  %s"
            % (
                cls,
                m.get("precision", 0.0),
                m.get("recall", 0.0),
                m.get("f1", 0.0),
                m.get("ap50", 0.0),
                support,
                desc,
            )
        )

    rows.append(SEP)
    logger.info("\n" + "\n".join(rows))

    logger.info(
        "\n%s\n"
        "MACRO averages (equal class weighting):\n"
        "  Precision: %.4f\n"
        "  Recall:    %.4f\n"
        "  F1:        %.4f\n"
        "  mAP50:     %.4f\n"
        "%s\n",
        SEP,
        macro_metrics.get("precision", 0.0),
        macro_metrics.get("recall", 0.0),
        macro_metrics.get("f1", 0.0),
        macro_metrics.get("map50", 0.0),
        SEP,
    )  # pylint: disable=logging-too-many-args

    save_json(
        {
            "per_class": per_class,
            "macro": macro_metrics,
            "class_descriptions": CLASS_DESCRIPTIONS,
        },
        out_dir / "per_class_metrics.json",
    )


# Best/Worst
def select_best_worst(
    scored_samples: List[Dict],
    n: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    scored_samples = list of dicts
    Returns (best_n, worst_n) sorted by score descending/ascending.
    """
    sorted_s = sorted(scored_samples, key=lambda x: x["score"], reverse=True)
    best = sorted_s[:n]
    worst = sorted_s[-n:][::-1]
    return best, worst


def save_vit_panel(
    original: np.ndarray,
    cam_rollout: np.ndarray,
    cam_gradcam: np.ndarray,
    out_path: Path,
    title: str,
    alpha: float = 0.5,
) -> None:
    """3-panel output for ViT: Original, Attention Rollout, Grad-CAM."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    overlay_rollout = overlay_cam(original, cam_rollout, alpha=alpha)
    overlay_gradcam = overlay_cam(original, cam_gradcam, alpha=alpha)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay_rollout)
    axes[1].set_title("Attention Rollout\n(global attention flow)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay_gradcam)
    axes[2].set_title("ViT Grad-CAM\n(class-discriminative)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def save_best_worst_panel(
    samples: List[Dict],
    quality: str,
    out_dir: Path,
    id_to_label: Dict,
    alpha: float,
    score_thresh: float,
) -> None:
    """Save 3-column panel showing best/worst detections."""
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    colour = "lime" if quality == "BEST" else "red"
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"{'BEST 3: Highest confidence correct' if quality == 'BEST' else 'WORST 3: Highest confidence incorrect'}",
        fontsize=16,
        color=colour,
        fontweight="bold",
        y=1.01,
    )

    for col, sample in enumerate(samples):
        original = sample["original"]
        cam_overlay = sample["cam_overlay"]
        detections = sample.get("detections", [])
        gt_boxes = sample.get("gt_boxes", [])
        gt_labels = sample.get("gt_labels", [])
        img_name = sample.get("image_name", f"sample_{col}")
        top_score = sample.get("score", 0.0)
        pred_label = sample.get("pred_label", "")
        true_label = sample.get("true_label", "")

        confidence_display = sample.get("confidence", top_score)

        if pred_label and true_label:
            title_str = f"{img_name}\npred={pred_label} true={true_label}\nconf={confidence_display:.3f}"
        else:
            title_str = f"{img_name}\nconf={confidence_display:.3f}"

        for row_ax in [axes[0, col], axes[1, col]]:
            # GT boxes in yellow
            for box, lab_idx in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = box
                lab = id_to_label.get(lab_idx, str(lab_idx)) if id_to_label else str(lab_idx)
                rect = mpatches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor="yellow", facecolor="none",
                )
                row_ax.add_patch(rect)
                row_ax.text(
                    xmin, max(ymin - 4, 0),
                    f"GT:{lab}", color="yellow", fontsize=7, fontweight="bold",
                )
            # Predicted boxes
            for det in detections:
                if det["score"] < score_thresh:
                    continue
                xmin, ymin, xmax, ymax = det["box"]
                lab = id_to_label.get(det["label"], str(det["label"])) if id_to_label else str(det["label"])
                rect = mpatches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor=colour, facecolor="none", linestyle="--",
                )
                row_ax.add_patch(rect)
                row_ax.text(
                    xmin, min(ymax + 10, original.shape[0] - 1),
                    f"PR:{lab} {det['score']:.2f}",
                    color=colour, fontsize=7, fontweight="bold",
                )

        pred_label = sample.get("pred_label", "")
        true_label = sample.get("true_label", "")

        if pred_label and true_label:
            title_str = f"{img_name}\npred={pred_label} true={true_label}\nconf={confidence_display:.3f}"
        else:
            title_str = f"{img_name}\nconf={confidence_display:.3f}"

        axes[0, col].imshow(original)
        axes[0, col].set_title(title_str, fontsize=8, color="white")
        axes[0, col].axis("off")

        axes[1, col].imshow(cam_overlay)
        axes[1, col].set_title("Grad-CAM", fontsize=9, color="white")
        axes[1, col].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"{quality.lower()}_3_panel.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved %s panel: %s", quality, out_path)


def save_vit_best_worst_panel(
    samples: List[Dict],
    quality: str,
    out_dir: Path,
    alpha: float,
) -> None:
    n = len(samples)
    fig, axes = plt.subplots(3, n, figsize=(7 * n, 14))
    if n == 1:
        axes = axes.reshape(3, 1)

    colour = "lime" if quality == "BEST" else "red"
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"{'BEST 3: Highest confidence correct' if quality == 'BEST' else 'WORST 3: Highest confidence incorrect'}",
        fontsize=16,
        color=colour,
        fontweight="bold",
        y=1.01,
    )

    for col, sample in enumerate(samples):
        original = sample["original"]
        cam_gradcam = sample["cam_overlay"]
        cam_rollout = sample.get("cam_rollout")
        img_name = sample.get("image_name", f"sample_{col}")
        top_score = sample.get("score", 0.0)
        pred_label = sample.get("pred_label", "")
        true_label = sample.get("true_label", "")

        confidence_display = sample.get("confidence", top_score)

        axes[0, col].imshow(original)
        axes[0, col].set_title(
            f"{img_name}\npred={pred_label} true={true_label}\nconf={confidence_display:.3f}",
            fontsize=8,
            color="white",
        )
        axes[0, col].axis("off")

        if cam_rollout is not None:
            axes[1, col].imshow(cam_rollout)
        else:
            axes[1, col].imshow(np.zeros_like(original))
        axes[1, col].set_title("Attention Rollout", fontsize=9, color="white")
        axes[1, col].axis("off")

        axes[2, col].imshow(cam_gradcam)
        axes[2, col].set_title("ViT Grad-CAM", fontsize=9, color="white")
        axes[2, col].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"{quality.lower()}_3_panel.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved %s panel: %s", quality, out_path)


def _find_wrong_detections(
    scored_samples: List[Dict],
    all_targets: List[Dict],
    all_preds: List[Dict],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
) -> List[Dict]:
    wrong = []
    for sample, targets, preds in zip(scored_samples, all_targets, all_preds):
        gt_boxes = targets["boxes"]
        pred_boxes = preds["boxes"]
        pred_scores = preds["scores"]
        pred_labels = preds["labels"]

        # Only consider predictions above score threshold
        keep = pred_scores >= score_thresh
        if keep.sum() == 0:
            continue

        high_boxes = pred_boxes[keep]
        high_scores = pred_scores[keep]
        high_labels = pred_labels[keep]

        if len(gt_boxes) == 0:
            # No ground truth all detections are false positives
            wrong.append({**sample, "score": float(high_scores.max().item())})
            continue

        # Check each high-confidence prediction against ground truth
        ious = box_iou(high_boxes, gt_boxes)  # [num_pred, num_gt]
        max_iou_per_pred = ious.max(dim=1).values

        # A prediction is wrong if its best IoU with any GT box is below threshold
        is_wrong = max_iou_per_pred < iou_thresh
        if is_wrong.any():
            # Score this sample by the highest-confidence wrong detection
            wrong_scores = high_scores[is_wrong]
            wrong.append(
                {
                    **sample,
                    "score": float(wrong_scores.max().item()),
                }
            )

    return sorted(wrong, key=lambda x: x["score"], reverse=True)


def run_detection_analysis(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:

    device = get_device()
    data_cfg = cfg["data"]
    allowed = data_cfg["allowed_labels"]
    id_to_label = {i + 1: lab for i, lab in enumerate(allowed)}

    cache_path = str(Path("outputs") / f".ann_cache_{args.split}.pkl")

    ds = RDDDetectionDataset(
        rdd_root=data_cfg["rdd_root"],
        split=args.split,
        allowed_labels=allowed,
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

    num_classes = len(ds.label_map) + 1
    model_cfg = cfg["model"]
    model = create_detection_model(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    gradcam = DetectionGradCAM(model)

    indices = list(range(len(ds)))
    random.shuffle(indices)
    sample_indices = indices[: args.n_samples]

    scored_samples = []
    all_targets = []
    all_preds = []

    gradcam_dir = out_dir / "gradcam"
    ensure_dir(gradcam_dir)

    DISPLAY_W, DISPLAY_H = 640, 480

    logger.info("Running GradCAM on %s samples.", args.n_samples)

    for i, idx in enumerate(sample_indices):
        image_tensor, target = ds[idx]
        img_path = ds.records[idx]["image_path"]
        img_name = Path(img_path).stem

        original_pil = Image.open(img_path).convert("RGB")

        tensor = image_tensor.unsqueeze(0).to(device)

        try:
            cam, detections = gradcam(tensor, score_thresh=args.score_thresh)
        except Exception as e:
            logger.warning("GradCAM failed for %s: %s", img_name, e)
            continue

        if not detections:
            logger.info("No detections for %s — skipping.", img_name)
            continue

        # Accumulate for metrics
        all_targets.append(
            {
                "boxes": target["boxes"],
                "labels": target["labels"],
            }
        )

        model.eval()
        with torch.no_grad():
            preds = model([image_tensor.to(device)])
        pred_cpu = {k: v.cpu() for k, v in preds[0].items()}
        all_preds.append(pred_cpu)

        disp_img = np.array(original_pil.resize((DISPLAY_W, DISPLAY_H)))
        cam_overlay = overlay_cam(disp_img, cam, alpha=args.alpha)

        orig_w, orig_h = original_pil.size
        sx = DISPLAY_W / orig_w
        sy = DISPLAY_H / orig_h
        scaled_detections = []
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            scaled_detections.append(
                {
                    "box": [xmin * sx, ymin * sy, xmax * sx, ymax * sy],
                    "label": det["label"],
                    "score": det["score"],
                }
            )

        top_score = max([d["score"] for d in scaled_detections], default=0.0)

        gt_scaled = []
        for box in target["boxes"].tolist():
            xmin, ymin, xmax, ymax = box
            gt_scaled.append([xmin * sx, ymin * sy, xmax * sx, ymax * sy])

        scored_samples.append({
            "image_name":  img_name,
            "image_path":  img_path,
            "original":    disp_img,
            "cam_overlay": cam_overlay,
            "detections":  scaled_detections,
            "gt_boxes":    gt_scaled,
            "gt_labels":   target["labels"].tolist(),
            "score":       top_score,
            "idx":         idx,
        })

        save_panel(
            original=disp_img,
            cam_overlay=cam_overlay,
            out_path=gradcam_dir / f"{i:03d}_{img_name}_gradcam.png",
            title=f"{img_name} | top_score={top_score:.3f}",
            detections=scaled_detections,
            id_to_label=id_to_label,
            score_thresh=args.score_thresh,
        )

    gradcam.remove()

    metrics = compute_detection_metrics(
        targets=all_targets,
        preds=all_preds,
        label_map=ds.label_map,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    macro = {
        "precision": metrics["precision@50"],
        "recall": metrics["recall@50"],
        "f1": metrics["f1@50"],
        "map50": metrics["map50_approx"],
    }

    print_per_class_metrics(metrics["per_class"], macro, out_dir)

    # Best/Worst panels
    if len(scored_samples) >= 3:
        best, _ = select_best_worst(scored_samples, n=3)
        best_names = {s["image_name"] for s in best}

        wrong_samples = _find_wrong_detections(
            scored_samples,
            all_targets,
            all_preds,
            iou_thresh=args.iou_thresh,
            score_thresh=args.score_thresh,
        )
        wrong_samples = [s for s in wrong_samples if s["image_name"] not in best_names]

        if len(wrong_samples) >= 3:
            worst = wrong_samples[:3]
        elif len(wrong_samples) > 0:
            remaining = [
                s
                for s in scored_samples
                if s["score"] >= args.score_thresh
                and s["image_name"] not in best_names
                and s["image_name"] not in {w["image_name"] for w in wrong_samples}
            ]
            worst = (
                wrong_samples
                + sorted(remaining, key=lambda x: x["score"], reverse=True)
            )[:3]
        else:
            has_detections = [
                s
                for s in scored_samples
                if s["score"] > 0 and s["image_name"] not in best_names
            ]
            worst = sorted(has_detections, key=lambda x: x["score"], reverse=True)[:3]

        save_best_worst_panel(
            best,
            "BEST",
            out_dir / "qualitative",
            id_to_label,
            args.alpha,
            args.score_thresh,
        )
        save_best_worst_panel(
            worst,
            "WORST",
            out_dir / "qualitative",
            id_to_label,
            args.alpha,
            args.score_thresh,
        )

    return macro, metrics["per_class"]


def run_classification_analysis(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    out_dir: Path,
    model_type: str,
) -> None:

    device = get_device()
    data_cfg = cfg["data"]
    allowed = data_cfg.get("allowed_labels", [])

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    image_size = data_cfg.get("image_size", 224)

    eval_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    label_map_path = cfg["outputs"].get("label_map_path")
    label_map = (
        load_label_map(label_map_path)
        if label_map_path and Path(label_map_path).exists()
        else None
    )

    split_dir = data_cfg.get("split_dir")
    if split_dir:
        split_dir_path = Path(split_dir)
        npy = split_dir_path / args.split / f"{args.split}_annotations.npy"
        pkl = split_dir_path / args.split / f"{args.split}_annotations.pkl"
        ds = RDDBboxCropDataset(
            npy_path=str(npy) if npy.exists() else None,
            pkl_path=str(pkl) if pkl.exists() else None,
            split=None,
            transform=eval_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=allowed,
            label_map=label_map,
        )
    else:
        ds = RDDBboxCropDataset(
            csv_path=data_cfg["csv_path"],
            split=args.split,
            transform=eval_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=allowed,
            label_map=label_map,
        )

    if label_map is None:
        label_map = ds.label_map
    id_to_label = {v: k for k, v in label_map.items()}

    model_cfg = cfg["model"]
    num_classes = len(label_map)
    model = create_model(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    if model_type in ("resnet18", "resnet34"):
        gradcam_obj = GradCAM(model, model.layer4[-1])
        vitgradcam_obj = None
        rollout_obj = None
    elif model_type == "custom_cnn":
        gradcam_obj = GradCAM(model, model.features[-2].block[-2])
        vitgradcam_obj = None
        rollout_obj = None
    elif model_type == "vit":
        gradcam_obj = None
        vitgradcam_obj = ViTGradCAM(model)
        rollout_obj = AttentionRollout(model)
    else:
        gradcam_obj = None
        vitgradcam_obj = None
        rollout_obj = None

    indices = list(range(len(ds)))
    random.shuffle(indices)
    sample_indices = indices[: args.n_samples]

    scored_samples = []
    y_true, y_pred = [], []

    gradcam_dir = out_dir / "gradcam"
    ensure_dir(gradcam_dir)

    logger.info("Running GradCAM on %s classification samples.", args.n_samples)

    for i, idx in enumerate(sample_indices):
        tensor, label_id = ds[idx]
        img_tensor = tensor.unsqueeze(0).to(device)

        if model_type == "vit":
            cam_rollout, pred_id = rollout_obj(img_tensor)
            cam_gradcam, _, _ = vitgradcam_obj(img_tensor)
            with torch.no_grad():
                logits = model(img_tensor)
                confidence = float(logits.softmax(dim=1)[0, pred_id].item())
            cam = cam_gradcam
        elif gradcam_obj is not None:
            cam_rollout = None
            cam_gradcam, pred_id, confidence = gradcam_obj(img_tensor)
            cam = cam_gradcam
        else:
            cam_rollout = None
            cam_gradcam, pred_id, confidence = np.zeros((7, 7)), 0, 0.0
            cam = cam_gradcam

        y_true.append(label_id)
        y_pred.append(pred_id)

        # Reconstruct image
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        disp = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        disp = (disp * 255).astype(np.uint8)

        cam_overlay = overlay_cam(disp, cam, alpha=args.alpha)
        pred_label = id_to_label.get(pred_id, str(pred_id))
        true_label = id_to_label.get(label_id, str(label_id))
        correct = pred_id == label_id

        display_alpha = min(args.alpha, 0.35) if model_type == "vit" else args.alpha

        scored_samples.append(
            {
                "image_name": f"sample_{idx}",
                "original": disp,
                "cam_overlay": overlay_cam(disp, cam_gradcam, alpha=display_alpha),
                "cam_rollout": (
                    overlay_cam(disp, cam_rollout, alpha=display_alpha)
                    if cam_rollout is not None
                    else None
                ),
                "score": confidence if correct else (1.0 - confidence),
                "correct": correct,
                "pred_label": pred_label,
                "true_label": true_label,
                "confidence": confidence,
            }
        )

        if model_type == "vit" and cam_rollout is not None:
            vit_alpha = min(args.alpha, 0.35)
            save_vit_panel(
                original=disp,
                cam_rollout=overlay_cam(disp, cam_rollout, alpha=args.alpha),
                cam_gradcam=overlay_cam(disp, cam_gradcam, alpha=args.alpha),
                out_path=gradcam_dir
                / f"{i:03d}_pred{pred_label}_true{true_label}_vit.png",
                title=f"pred={pred_label}  true={true_label}  conf={confidence:.3f}  {'✓' if correct else '✗'}",
                alpha=args.alpha,
            )
        else:
            cam_overlay = overlay_cam(disp, cam, alpha=args.alpha)
            save_panel(
                original=disp,
                cam_overlay=cam_overlay,
                out_path=gradcam_dir / f"{i:03d}_pred{pred_label}_true{true_label}.png",
                title=f"pred={pred_label}  true={true_label}  conf={confidence:.3f}  {'✓' if correct else '✗'}",
            )

    if gradcam_obj:
        gradcam_obj.remove()
    if vitgradcam_obj:
        vitgradcam_obj.remove()
    if rollout_obj:
        rollout_obj.remove()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes))
    )

    per_class = {}
    for i, lab in enumerate(allowed):
        per_class[lab] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "ap50": float(f1[i]),  # proxy for classification
            "support_gt": int(support[i]),
        }

    macro = {
        "precision": float(prec.mean()),
        "recall": float(rec.mean()),
        "f1": float(f1.mean()),
        "map50": float(f1.mean()),
        "accuracy": float(acc),
    }

    print_per_class_metrics(per_class, macro, out_dir)

    correct_samples = [s for s in scored_samples if s["correct"]]
    wrong_samples = [s for s in scored_samples if not s["correct"]]

    if len(correct_samples) >= 3:
        best, _ = select_best_worst(correct_samples, n=3)
        best_names = {s["image_name"] for s in best}
    else:
        best = correct_samples
        best_names = {s["image_name"] for s in best}

    high_conf_wrong = sorted(
        [
            s
            for s in wrong_samples
            if s["confidence"] >= 0.5 and s["image_name"] not in best_names
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    for s in wrong_samples:
        logger.info(
            "wrong sample: %s  pred=%s  true=%s  confidence=%.3f  in_high_conf=%s",
            s["image_name"],
            s["pred_label"],
            s["true_label"],
            s["confidence"],
            s["confidence"] >= 0.5,
        )

    if len(high_conf_wrong) >= 3:
        worst = high_conf_wrong[:3]
    elif len(high_conf_wrong) > 0:
        remaining = sorted(
            [
                s
                for s in wrong_samples
                if s not in high_conf_wrong and s["image_name"] not in best_names
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )
        worst = (high_conf_wrong + remaining)[:3]
    else:
        worst = sorted(
            [s for s in wrong_samples if s["image_name"] not in best_names],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]

    if model_type == "vit":
        if len(best) >= 3:
            save_vit_best_worst_panel(best, "BEST", out_dir / "qualitative", args.alpha)
        if len(worst) >= 3:
            save_vit_best_worst_panel(
                worst, "WORST", out_dir / "qualitative", args.alpha
            )
    else:
        if len(best) >= 3:
            save_best_worst_panel(
                best, "BEST", out_dir / "qualitative", {}, args.alpha, 0.0
            )
        if len(worst) >= 3:
            save_best_worst_panel(
                worst, "WORST", out_dir / "qualitative", {}, args.alpha, 0.0
            )

    return macro, per_class


def run_yolo_analysis(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> None:
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    except ImportError:
        raise SystemExit("ultralytics is not installed.")

    SETTINGS.update({"runs_dir": str(Path.cwd())})

    data_cfg = cfg["data"]
    allowed = data_cfg["allowed_labels"]
    label_map = {lab: i for i, lab in enumerate(allowed)}
    id_to_label = {i: lab for i, lab in enumerate(allowed)}

    model = YOLO(args.checkpoint)

    yolo_data_dir = Path(cfg.get("yolo_data_dir", "yolo_dataset"))
    dataset_yaml = yolo_data_dir / "dataset.yaml"

    val_results = model.val(
        data=str(dataset_yaml),
        split=args.split,
        conf=args.score_thresh,
        iou=args.iou_thresh,
        device=0,
        verbose=False,
        project=str(Path(out_dir).resolve()),
        name="val_metrics",
        exist_ok=True,
    )

    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    prec = float(val_results.box.mp)
    rec = float(val_results.box.mr)

    ap50_list = val_results.box.ap50
    p_list = val_results.box.p
    r_list = val_results.box.r

    per_class = {}
    for i, lab in enumerate(allowed):
        p_c = float(p_list[i]) if i < len(p_list) else 0.0
        r_c = float(r_list[i]) if i < len(r_list) else 0.0
        ap = float(ap50_list[i]) if i < len(ap50_list) else 0.0
        f1 = 0.0 if (p_c + r_c) == 0 else 2 * p_c * r_c / (p_c + r_c)
        per_class[lab] = {
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "f1": round(f1, 4),
            "ap50": round(ap, 4),
        }

    macro = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round((2 * prec * rec / max(prec + rec, 1e-6)), 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
    }

    print_per_class_metrics(per_class, macro, out_dir)

    records = load_detection_records(
        rdd_root=data_cfg["rdd_root"],
        allowed_labels=allowed,
        countries=data_cfg.get("countries"),
        split_mode=data_cfg.get("split_mode", "random"),
        split=args.split,
        xml_glob=data_cfg.get("xml_glob", "**/annotations/xmls/*.xml"),
        image_dir_hint=data_cfg.get("image_dir_hint", "images"),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 1337),
    )

    random.shuffle(records)
    sample_records = records[: args.n_samples]
    scored_samples = []
    gradcam_dir = out_dir / "gradcam"
    ensure_dir(gradcam_dir)

    imgsz = cfg.get("train", {}).get("imgsz", 640)

    for i, rec in enumerate(sample_records):
        img_path = rec["image_path"]
        ann_path = rec["ann_path"]
        img_name = Path(img_path).stem

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        original = np.array(img.resize((imgsz, imgsz)))

        tensor = F_tv.to_tensor(img).unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=(imgsz, imgsz), mode="bilinear", align_corners=False
        )
        tensor = tensor.to(next(model.model.parameters()).device)

        results = model.predict(
            source=img_path,
            conf=args.score_thresh,
            verbose=False,
        )
        r = results[0]
        sx, sy = imgsz / orig_w, imgsz / orig_h
        detections = []
        if len(r.boxes) > 0:
            for box, cls, score in zip(
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.cls.cpu().int().tolist(),
                r.boxes.conf.cpu().tolist(),
            ):
                detections.append(
                    {
                        "box": [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
                        "label": cls,
                        "score": score,
                    }
                )

        top_score = max([d["score"] for d in detections], default=0.0)

        _img_w, _img_h, objects = parse_voc_xml(ann_path)
        gt_boxes_raw = [
            [float(xmin) * sx, float(ymin) * sy, float(xmax) * sx, float(ymax) * sy]
            for lab, (xmin, ymin, xmax, ymax) in objects
            if lab in label_map
        ]

        is_wrong = False
        if detections and gt_boxes_raw:
            pred_t = torch.tensor(
                [
                    [d["box"][0], d["box"][1], d["box"][2], d["box"][3]]
                    for d in detections
                ]
            )
            gt_t = torch.tensor(gt_boxes_raw)
            ious = box_iou(pred_t, gt_t)
            max_iou_per_pred = ious.max(dim=1).values
            is_wrong = bool((max_iou_per_pred < args.iou_thresh).any())
        elif detections and not gt_boxes_raw:
            is_wrong = True

        yolo_nn = model.model
        act_holder = {}

        def _hook(m, inp, out):
            act_holder["act"] = out.detach()

        h = yolo_nn.model[6].register_forward_hook(_hook)
        with torch.no_grad():
            yolo_nn(tensor)
        h.remove()
        act = act_holder.get("act")
        if act is not None:
            cam = act.mean(dim=1).squeeze().cpu().numpy()
            cam = np.clip(cam, np.percentile(cam, 5), np.percentile(cam, 95))
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam_overlay = overlay_cam(original, cam, alpha=args.alpha)

        scored_samples.append({
            "image_name":  img_name,
            "original":    original,
            "cam_overlay": cam_overlay,
            "detections":  detections,
            "gt_boxes":    gt_boxes_raw,
            "gt_labels":   [label_map[lab] for lab, _ in objects if lab in label_map],
            "score":       top_score,
            "is_wrong":    is_wrong,
        })

        save_panel(
            original     = original,
            cam_overlay  = cam_overlay,
            out_path     = gradcam_dir / f"{i:03d}_{img_name}_gradcam.png",
            title        = f"{img_name} | top_score={top_score:.3f}",
            detections   = detections,
            id_to_label  = id_to_label,
            score_thresh = args.score_thresh,
            gt_boxes     = gt_boxes_raw,
            gt_labels    = [label_map[lab] for lab, _ in objects if lab in label_map],
        )

    if len(scored_samples) >= 3:
        best, _ = select_best_worst(scored_samples, n=3)
        best_names = {s["image_name"] for s in best}

        wrong_samples = sorted(
            [
                s
                for s in scored_samples
                if s.get("is_wrong", False)
                and s["score"] >= args.score_thresh
                and s["image_name"] not in best_names
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        if len(wrong_samples) >= 3:
            worst = wrong_samples[:3]
        elif len(wrong_samples) > 0:
            remaining = [
                s
                for s in scored_samples
                if s not in wrong_samples
                and s["score"] >= args.score_thresh
                and s["image_name"] not in best_names
            ]
            remaining_sorted = sorted(remaining, key=lambda x: x["score"], reverse=True)
            worst = (wrong_samples + remaining_sorted)[:3]
        else:
            has_detections = [
                s
                for s in scored_samples
                if s["score"] > 0 and s["image_name"] not in best_names
            ]
            worst = sorted(has_detections, key=lambda x: x["score"], reverse=True)[:3]

        if len(best) >= 3:
            save_best_worst_panel(
                best,
                "BEST",
                out_dir / "qualitative",
                id_to_label,
                args.alpha,
                args.score_thresh,
            )
        else:
            logger.warning(
                "Not enough best samples to generate panel (%s found).", len(best)
            )

        if len(worst) >= 3:
            save_best_worst_panel(
                worst,
                "WORST",
                out_dir / "qualitative",
                id_to_label,
                args.alpha,
                args.score_thresh,
            )
        else:
            logger.warning(
                "Not enough worst samples to generate panel (%s found).", len(worst)
            )

    return macro, per_class


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(cfg["outputs"]["root_dir"]) / "analysis"
    )
    ensure_dir(out_dir)

    setup_logging(out_dir / "analysis.log")
    set_seed(cfg.get("seed", 1337))

    logger.info(
        "\n%s\n"
        "ROAD DAMAGE DETECTION ANALYSIS\n"
        "Model type:  %s\n"
        "Checkpoint:  %s\n"
        "Split:       %s\n"
        "Output dir:  %s\n"
        "%s\n",
        SEP,
        args.model_type,
        args.checkpoint,
        args.split,
        out_dir,
        SEP,
    )

    if args.model_type == "fasterrcnn":
        macro, per_class = run_detection_analysis(args, cfg, out_dir)
    elif args.model_type == "yolo":
        macro, per_class = run_yolo_analysis(args, cfg, out_dir)
    else:
        macro, per_class = run_classification_analysis(
            args, cfg, out_dir, args.model_type
        )

    summary = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "macro": macro,
        "per_class": per_class,
        "class_descriptions": CLASS_DESCRIPTIONS,
    }
    save_json(summary, out_dir / "summary.json")

    logger.info(
        "\n%s\n"
        "Analysis complete\n"
        "Outputs saved to: %s\n"
        "  gradcam/     — per-image GradCAM overlays\n"
        "  qualitative/ — best_3 and worst_3 panels\n"
        "  model_metrics.json\n"
        "  per_class_metrics.json\n"
        "  summary.json\n"
        "  analysis.log\n"
        "%s\n",
        SEP,
        out_dir,
        SEP,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Analysis failed with error: {e}", flush=True)
        raise
