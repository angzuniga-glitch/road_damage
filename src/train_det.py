from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_det import RDDDetectionDataset, DetectionTransform, detection_collate_fn
from src.models.detection_factory import create_detection_model, count_trainable_parameters
from src.utils import ensure_dir, get_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train object detector on RDD2022.")
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    data_cfg = cfg["data"]

    allowed_labels = data_cfg["allowed_labels"]

    train_ds = RDDDetectionDataset(
        rdd_root=data_cfg["rdd_root"],
        split=data_cfg["train_split"],
        allowed_labels=allowed_labels,
        transform=DetectionTransform(train=True, hflip_prob=data_cfg.get("hflip_prob", 0.5)),
        countries=data_cfg.get("countries"),
        split_mode=data_cfg.get("split_mode", "random"),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 1337),
        xml_glob=data_cfg.get("xml_glob", "**/annotations/xmls/*.xml"),
        image_dir_hint=data_cfg.get("image_dir_hint", "images"),
    )

    val_ds = RDDDetectionDataset(
        rdd_root=data_cfg["rdd_root"],
        split=data_cfg["val_split"],
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

    num_workers = cfg["train"].get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )

    return train_loader, val_loader, train_ds.label_map


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    train_cfg = cfg["train"]
    opt_name = train_cfg.get("optimizer", "sgd").lower()

    if opt_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

    return torch.optim.SGD(
        params,
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    # Detection models only produce losses in train mode when targets are passed.
    # So we temporarily keep train mode, but disable grads.
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


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 1337))
    device = get_device()

    out_cfg = cfg["outputs"]
    ensure_dir(out_cfg["root_dir"])
    ensure_dir(out_cfg["checkpoints_dir"])
    ensure_dir(out_cfg["logs_dir"])

    train_loader, val_loader, label_map = build_dataloaders(cfg)
    num_classes = len(label_map) + 1  # + background

    model_cfg = cfg["model"]
    model = create_detection_model(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    optimizer = build_optimizer(cfg, model)

    print("=" * 100)
    print(f"Config:            {args.config}")
    print(f"Device:            {device}")
    print(f"Model:             {model_cfg['name']}")
    print(f"Pretrained:        {model_cfg.get('pretrained', True)}")
    print(f"Freeze backbone:   {model_cfg.get('freeze_backbone', False)}")
    print(f"Num classes:       {num_classes} (including background)")
    print(f"Train samples:     {len(train_loader.dataset)}")
    print(f"Val samples:       {len(val_loader.dataset)}")
    print(f"Trainable params:  {count_trainable_parameters(model):,}")
    print("=" * 100)

    best_val_loss = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "label_map": label_map, "config": cfg}

    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("early_stopping_patience", 5)
    no_improve = 0

    best_ckpt = str(Path(out_cfg["checkpoints_dir"]) / out_cfg["best_checkpoint_name"])
    history_path = str(Path(out_cfg["logs_dir"]) / out_cfg["history_name"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        dt = time.time() - t0
        history["train_loss"].append({"epoch": epoch, "loss": train_loss})
        history["val_loss"].append({"epoch": epoch, "loss": val_loss})
        save_json(history, history_path)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"time={dt:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=-best_val_loss,  # keep interface compatible
                config=cfg,
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": best_ckpt,
        "history_path": history_path,
    }
    save_json(summary, Path(out_cfg["logs_dir"]) / "summary.json")

    print("=" * 100)
    print("Detection training complete")
    print(f"Best epoch:        {best_epoch}")
    print(f"Best val loss:     {best_val_loss:.4f}")
    print(f"Checkpoint:        {best_ckpt}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
