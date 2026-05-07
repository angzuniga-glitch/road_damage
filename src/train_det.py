from __future__ import annotations
import os
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = False
torch._dynamo.config.verbose = False  

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
    train_cfg = cfg["train"]
    allowed_labels = data_cfg["allowed_labels"]

    common_ds_kwargs = dict(
        rdd_root = data_cfg["rdd_root"],
        allowed_labels = allowed_labels,
        countries = data_cfg.get("countries"),
        split_mode = data_cfg.get("split_mode", "random"),
        train_ratio = data_cfg.get("train_ratio", 0.8),
        val_ratio = data_cfg.get("val_ratio", 0.1),
        seed = cfg.get("seed", 1337),
        xml_glob = data_cfg.get("xml_glob", "**/annotations/xmls/*.xml"),
        image_dir_hint = data_cfg.get("image_dir_hint", "images"),
        )

    train_ds = RDDDetectionDataset(
        split = data_cfg["train_split"],
        transform = DetectionTransform(
            train=True, 
            hflip_prob=data_cfg.get("hflip_prob", 0.5),
            min_sizes=tuple(data_cfg.get("multiscale_min_sizes", [480, 512, 544, 576, 608, 640])),
            max_size=data_cfg.get("multiscale_max_size", 1333),
        ),
        **common_ds_kwargs,
    )

    val_ds = RDDDetectionDataset(
        split = data_cfg["val_split"],
        transform = DetectionTransform(train=False),
        **common_ds_kwargs,
    )

    num_workers = cfg["train"].get("num_workers", 4)
    use_cuda = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size = train_cfg["batch_size"],
        num_workers = num_workers,
        pin_memory = use_cuda,
        collate_fn = detection_collate_fn,
        persistent_workers = num_workers > 0,
        prefetch_factor = 2 if num_workers > 0 else None,
    )


    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, train_ds.label_map


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    train_cfg = cfg["train"]
    opt_name = train_cfg.get("optimizer", "sgd").lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
    else: 
        optimizer = torch.optim.SGD(
            params,
            lr=train_cfg["lr"],
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
        )
    return optimizer

def build_scheduler(cfg: Dict[str, Any], optimizer, steps_per_epoch: int):
    train_cfg = cfg["train"]
    sched_name = train_cfg.get("scheduler", "onecycle").lower()
    total_steps = train_cfg["epochs"] * steps_per_epoch
    if sched_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = train_cfg["lr"],
            total_steps = total_steps,
            pct_start = 0.3,
            anneal_strategy = "cos",
        )
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = train_cfg["epochs"]
        )
    return None


def train_one_epoch(model, loader, optimizer, device, scaler, 
                    grad_clip: float = 5.0, accum_steps: int = 1) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    optimizer.zero_grad(set_to_none = True)

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking = True) for img in images]
        targets = [{k: v.to(device, non_blocking = True) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none = True)

        total_loss += float(loss.item()) * accum_steps
        total_batches += 1
    
    if total_batches % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none = True)

    return total_loss / max(total_batches, 1)

@torch.no_grad()
def validate_one_epoch(model, loader, device) -> float:
    # Detection models only produce losses in train mode when targets are passed.
    # So we temporarily keep train mode, but disable grads.
    model.train()
    total_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        images = [img.to(device, non_blocking = True) for img in images]
        targets = [{k: v.to(device, non_blocking = True) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled = device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 1337))
    run_start = time.time()
    log_path = Path(cfg["outputs"]["logs_dir"]) / "train.log"
    ensure_dir(cfg["outputs"]["logs_dir"])

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    logging.getLogger("torchvision").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
    logging.getLogger("torch._inductor").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
    logger = logging.getLogger("train_det")
    logger.setLevel(logging.INFO)

    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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

    model.backbone = torch.compile(model.backbone, mode="default")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch = len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled = device.type == "cuda")

    train_cfg = cfg["train"]
    accum_steps = train_cfg.get("grad_accum_steps", 1)
    grad_clip = train_cfg.get("grad_clip", 5.0)

    print(datetime.now().isoformat())
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
    print(f"Batch size        : {train_cfg['batch_size']}  ×  accum {accum_steps}  =  effective {train_cfg['batch_size'] * accum_steps}")
    print(f"AMP               : {'enabled' if device.type == 'cuda' else 'disabled (CPU)'}")
    print(f"Scheduler         : {train_cfg.get('scheduler', 'onecycle')}")
    print("=" * 100)

    best_val_loss = float("inf")
    best_epoch    = -1
    history       = {"train_loss": [], "val_loss": [], "label_map": label_map, "config": cfg}
    epochs        = train_cfg["epochs"]
    patience      = train_cfg.get("early_stopping_patience", 5)
    no_improve    = 0

    best_ckpt = str(Path(out_cfg["checkpoints_dir"]) / out_cfg["best_checkpoint_name"])
    history_path = str(Path(out_cfg["logs_dir"]) / out_cfg["history_name"])

    logger.info(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
 
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler, grad_clip=grad_clip, 
            accum_steps=accum_steps,
        )
        val_loss = validate_one_epoch(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0
 
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append({"epoch": epoch, "loss": train_loss})
        history["val_loss"].append({"epoch": epoch, "loss": val_loss})
        save_json(history, history_path)

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - run_start))
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.2e} | epoch_time={dt:.1f}s | "
            f"elapsed={elapsed} | "
            f"LR post step: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.2e} | epoch_time={dt:.1f}s | "
            f"elapsed={elapsed} | "
            f"LR post step: {optimizer.param_groups[0]['lr']:.2e}"
        )
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            save_checkpoint(
                path        = best_ckpt,
                model       = model,
                optimizer   = optimizer,
                epoch       = epoch,
                best_metric = -best_val_loss,
                config      = cfg,
            )
            logger.info(f"  ✓ new best checkpoint saved  (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}"
                )
                break
 
    summary = {
        "best_epoch"   : best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint"   : best_ckpt,
        "history_path" : history_path,
    }
    save_json(summary, Path(out_cfg["logs_dir"]) / "summary.json")
 
    print("=" * 100)
    logger.info(f"Training complete  |  best epoch {best_epoch}  |  val_loss {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {best_ckpt}")

    print(f"Training complete  |  best epoch {best_epoch}  |  val_loss {best_val_loss:.4f}")
    print(f"Checkpoint: {best_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
