from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from src.data.dataset import RDDBboxCropDataset, save_label_map
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.factory import create_model, count_trainable_parameters
from src.utils import (
    AverageMeter,
    compute_classification_metrics,
    ensure_dir,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train road-damage crop classifier.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return p.parse_args()

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    data_cfg = cfg["data"]
    train_tf = get_train_transforms(data_cfg["image_size"])
    eval_tf = get_eval_transforms(data_cfg["image_size"])

    label_map_path = cfg["outputs"]["label_map_path"]
    cache_images = data_cfg.get("cache_images", False)
    split_dir = data_cfg.get("split_dir")

    if split_dir:
        split_dir = Path(split_dir)
        train_npy = split_dir / "train" / "train_annotations.npy"
        val_npy = split_dir / "val" / "val_annotations.npy"
        train_pkl = split_dir / "train" / "train_annotations.pkl"
        val_pkl = split_dir / "val" / "val_annotations.pkl"

        train_ds = RDDBboxCropDataset(
            npy_path=str(train_npy) if train_npy.exists() else None,
            pkl_path=str(train_pkl) if train_pkl.exists() else None,
            split=None,
            transform=train_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=data_cfg.get("allowed_labels"),
            cache_images=cache_images,
        )
        val_ds = RDDBboxCropDataset(
            npy_path=str(val_npy) if val_npy.exists() else None,
            pkl_path=str(val_pkl) if val_pkl.exists() else None,
            split=None,
            transform=eval_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=data_cfg.get("allowed_labels"),
            cache_images=cache_images,
        )
    else:
        csv_path = data_cfg["csv_path"]
        train_ds = RDDBboxCropDataset(
            csv_path=csv_path,
            split=data_cfg["train_split"],
            transform=train_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=data_cfg.get("allowed_labels"),
            cache_images=cache_images,
        )
        val_ds = RDDBboxCropDataset(
            csv_path=csv_path,
            split=data_cfg["val_split"],
            transform=eval_tf,
            countries=data_cfg.get("countries"),
            allowed_labels=data_cfg.get("allowed_labels"),
            cache_images=cache_images,
        )
    all_labels =sorted(set(row["label"] for row in train_ds.data))
    label_map = {label:idx for idx, label in enumerate(all_labels)}
    save_label_map(label_map, label_map_path)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        prefetch_factor=cfg["train"].get("prefetch_factor", 2),
        persistent_workers=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        prefetch_factor=cfg["train"].get("prefetch_factor", 2),
        persistent_workers=True,
    )

    return train_dl, val_dl, label_map

def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    opt_name = cfg["train"]["optimizer"].lower()
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"].get("weight_decay", 0.0)

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = cfg["train"].get("momentum", 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {opt_name}")

def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    sched_name = cfg["train"].get("scheduler", "none").lower()

    if sched_name == "none":
        return None
    if sched_name == "steplr":
        step_size = cfg["train"].get("step_size", 10)
        gamma = cfg["train"].get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_name == "cosine":
        epochs = cfg["train"]["epochs"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    raise ValueError(f"Unsupported scheduler: {sched_name}")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, float]:
    model.train()

    loss_meter = AverageMeter()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=images.size(0))

        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)

        loss_meter.update(loss.item(), n=images.size(0))

        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = loss_meter.avg
    return metrics


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    torch.set_float32_matmul_precision('high')
    logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
    logger.info("\nStarting training with config: %s\n", args.config)

    set_seed(cfg.get("seed", 1337))
    device = get_device()

    out_cfg = cfg["outputs"]
    ensure_dir(out_cfg["root_dir"])
    ensure_dir(out_cfg["checkpoints_dir"])
    ensure_dir(out_cfg["logs_dir"])
    ensure_dir(out_cfg["figures_dir"])

    train_loader, val_loader, label_map = build_dataloaders(cfg)
    num_classes = len(label_map)

    model_cfg = cfg["model"]
    model = create_model(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)
    model = torch.compile(model)

    optimizer = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler()
    scheduler = build_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss()

    print("-" * 100)
    print(f"Config:              {args.config}")
    print(f"Device:              {device}")
    print(f"Model:               {model_cfg['name']}")
    print(f"Pretrained:          {model_cfg.get('pretrained', False)}")
    print(f"Freeze backbone:     {model_cfg.get('freeze_backbone', False)}")
    print(f"Num classes:         {num_classes}")
    print(f"Trainable params:    {count_trainable_parameters(model):,}")
    print(f"Train samples:       {len(train_loader.dataset)}")
    print(f"Val samples:         {len(val_loader.dataset)}")
    print("-" * 100)

    history = {
        "config": cfg,
        "label_map": label_map,
        "train": [],
        "val": [],
    }

    best_metric = -1.0
    best_epoch = -1
    patience = cfg["train"].get("early_stopping_patience", 10)
    min_delta = cfg["train"].get("min_delta", 0.0)
    epochs_no_improve = 0

    ckpt_path = str(Path(out_cfg["checkpoints_dir"]) / out_cfg["best_checkpoint_name"])
    history_path = str(Path(out_cfg["logs_dir"]) / out_cfg["history_name"])

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        start = time.time()

        # profiling block
        if epoch == 1:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                acc_events=True,
            ) as prof:
                with record_function("train_epoch"):
                    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

            prof_dir = Path(out_cfg["logs_dir"]) / "profiler"
            prof_dir.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(prof_dir / "trace.json"))
            with open(prof_dir / "profiler_results.txt", 'w', encoding="utf-8") as f:
                f.write(prof.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=90
                ))
        else:
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        train_metrics["epoch_time_sec"] = epoch_time
        val_metrics["epoch_time_sec"] = epoch_time

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        current_metric = val_metrics["macro_f1"]
        if current_metric > best_metric + min_delta:
            best_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0

            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                config=cfg,
            )
        else:
            epochs_no_improve += 1

        save_json(history, history_path)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}, (patience: {patience}, min delta: {min_delta})")
            print(f"Best epoch: {best_epoch}, best val macro-F1: {best_metric:.4f}")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "num_classes": num_classes,
        "trainable_parameters": count_trainable_parameters(model),
        "checkpoint_path": ckpt_path,
        "history_path": history_path,
    }
    save_json(summary, Path(out_cfg["logs_dir"]) / "summary.json")

    print("-" * 100)
    logger.info("Training complete")
    print(f"Best epoch:          {best_epoch}")
    print(f"Best val macro-F1:   {best_metric:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"History saved to:    {history_path}")
    print("-" * 100)
    
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        raise