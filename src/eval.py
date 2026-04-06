from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.dataset import RDDBboxCropDataset, load_label_map
from src.data.transforms import get_eval_transforms
from src.models.factory import create_model
from src.utils import (
    compute_classification_metrics,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_json,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eval.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate road-damage crop classifier.")
    p.add_argument("--config", type=str, required=True, 
                   help="Path to YAML config.")
    p.add_argument("--checkpoint", type=str, required=True, 
                   help="Path to model checkpoint.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                   help="Dataset split to evaluate.")
    return p.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_count = 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: str | Path,
    normalize: str | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize=normalize)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix" if normalize is None else "Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 1337))
    device = get_device()

    logger.info("\nStarting evaluation with config: %s\n", args.config)

    out_cfg = cfg["outputs"]
    ensure_dir(out_cfg["root_dir"])
    ensure_dir(out_cfg["logs_dir"])
    ensure_dir(out_cfg["figures_dir"])

    data_cfg = cfg["data"]
    eval_tf = get_eval_transforms(data_cfg["image_size"])

    label_map_path = out_cfg.get("label_map_path")
    if label_map_path and Path(label_map_path).exists():
        logger.info("\nLoading label_map from: %s\n", label_map_path)
        label_map = load_label_map(label_map_path)
    else:
        label_map = None

    split_dir = data_cfg.get("split_dir")
    
    if split_dir:
        split_dir_path = Path(split_dir)
        split_subdir = {
            "train": split_dir_path / "train",
            "val": split_dir_path / "val",
            "test": split_dir_path / "test"
        }.get(args.split)
        
        if not split_subdir:
            raise ValueError(f"Invalid split '{args.split}'. Must be 'train', 'val', or 'test'.")
        
        npy_file = split_subdir / f"{args.split}_annotations.npy"
        pkl_file = split_subdir / f"{args.split}_annotations.pkl"
        
        ds = RDDBboxCropDataset(
            npy_path=str(npy_file) if npy_file.exists() else None,
            pkl_path=str(pkl_file) if pkl_file.exists() else None,
            split=None, 
            transform=eval_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=data_cfg.get("allowed_labels"),
            label_map=label_map,
        )
    else:
        ds = RDDBboxCropDataset(
            csv_path=data_cfg["csv_path"],
            split=args.split,
            transform=eval_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=data_cfg.get("allowed_labels"),
            label_map=label_map,
        )

    if label_map is None:
        label_map = ds.label_map
    id_to_label = ds.id_to_label
    class_names = [id_to_label[i] for i in range(len(id_to_label))]

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )

    model_cfg = cfg["model"]
    model = create_model(
        model_name=model_cfg["name"],
        num_classes=len(label_map),
        pretrained=model_cfg.get("pretrained", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    ckpt_meta = load_checkpoint(args.checkpoint, model=model, optimizer=None, map_location=device)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, criterion, device)

    print("-" * 100)
    print(f"Config:              {args.config}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Split:               {args.split}")
    print(f"Loaded epoch:        {ckpt_meta.get('epoch', 'N/A')}")
    print(f"Best metric in ckpt: {ckpt_meta.get('best_metric', 'N/A')}")
    print(f"Samples:             {len(ds)}")
    print("-" * 100)
    print(f"Loss:                {metrics['loss']:.4f}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Macro-F1:            {metrics['macro_f1']:.4f}")
    print("-" * 100)

    eval_out = {
        "config": cfg,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_f1": metrics["per_class_f1"],
        "per_class_support": metrics["per_class_support"],
        "class_names": class_names,
        "checkpoint_meta": ckpt_meta,
    }

    metrics_path = Path(out_cfg["logs_dir"]) / f"eval_{args.split}.json"
    save_json(eval_out, metrics_path)

    fig_dir = Path(out_cfg["figures_dir"])
    cm_path = fig_dir / f"confusion_matrix_{args.split}.png"
    cm_norm_path = fig_dir / f"confusion_matrix_{args.split}_normalized.png"

    save_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names, cm_path, normalize=None)
    save_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names, cm_norm_path, normalize="true")

    print(f"Saved metrics to:            {metrics_path}")
    print(f"Saved confusion matrix to:   {cm_path}")
    print(f"Saved normalized matrix to:  {cm_norm_path}")
    
    logger.info("\nEvaluation completed successfully. Metrics saved to %s\n", metrics_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        logger.error("\nEvaluation failed with error: %s\n", e, exc_info=True)
        raise