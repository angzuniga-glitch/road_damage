from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.data.dataset_yolo import convert_rdd_to_yolo
from src.utils import ensure_dir, set_seed, setup_logging

SEP = "-" * 100
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on RDD2022.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip VOC-YOLO dataset conversion.",
    )
    return p.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> int:

    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(Path(cfg["outputs"]["root_dir"]) / "logs" / "train_yolo.log")
    set_seed(cfg.get("seed", 1337))

    # Forces project root  as base for output paths
    SETTINGS.update({"runs_dir": str(Path.cwd())})

    logging.getLogger("ultralytics.utils.checks").setLevel(logging.WARNING)
    logging.getLogger("ultralytics.utils.downloads").setLevel(logging.WARNING)
    logging.getLogger("ultralytics.engine.validator").setLevel(logging.WARNING)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    out_cfg = cfg["outputs"]

    ensure_dir(out_cfg["root_dir"])

    # Convert dataset
    yolo_data_dir = Path(cfg.get("yolo_data_dir", "yolo_dataset"))
    if not args.skip_convert:
        dataset_yaml = convert_rdd_to_yolo(cfg, out_dir=yolo_data_dir)
    else:
        dataset_yaml = yolo_data_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"dataset.yaml not found at {dataset_yaml}. "
                "Run without --skip-convert first."
            )

    # Build model
    model_name = model_cfg["name"]
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)

    if pretrained:
        # Load pretrained weights
        model = YOLO(model_name)
        logger.info("Loaded pretrained model: %s", model_name)
    else:
        # Strip .pt extension
        arch_yaml = model_name.replace(".pt", ".yaml")
        model = YOLO(arch_yaml)
        logger.info("Initialized model from scratch: %s", arch_yaml)

    # Freeze backbone
    freeze_layers = 0
    if freeze_backbone:
        freeze_layers = 10
        logger.info("Freezing first %s layers (backbone)", freeze_layers)

    # 4. Train
    model.train(
        data=str(dataset_yaml),
        optimizer=train_cfg.get("optimizer", "SGD"),
        epochs=train_cfg.get("epochs", 50),
        imgsz=train_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch_size", 16),
        lr0=train_cfg.get("lr", 0.01),
        lrf=train_cfg.get("lrf", 0.01),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        patience=train_cfg.get("early_stopping_patience", 10),
        freeze=freeze_layers if freeze_layers > 0 else None,
        device=train_cfg.get("device", 0),
        workers=train_cfg.get("num_workers", 4),
        project=str(Path(out_cfg["root_dir"]).resolve()),
        name=out_cfg.get("run_name", "train"),
        exist_ok=True,
        verbose=cfg.get("verbose", True),
        seed=cfg.get("seed", 1337),
        fliplr=train_cfg.get("hflip_prob", 0.5),
        hsv_h=train_cfg.get("hsv_h", 0.015),
        hsv_s=train_cfg.get("hsv_s", 0.7),
        hsv_v=train_cfg.get("hsv_v", 0.4),
        degrees=train_cfg.get("degrees", 0.0),
        translate=train_cfg.get("translate", 0.1),
        scale=train_cfg.get("scale", 0.5),
        mosaic=train_cfg.get("mosaic", 1.0),
        close_mosaic=train_cfg.get("close_mosaic", 10),
    )

    metrics = model.val(
        data=str(dataset_yaml),
        project=str(Path(out_cfg["root_dir"]).resolve()),
        name="val_final",
        exist_ok=True,
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    logger.info(
        "\n%s\n"
        "Config:       %s\n"
        "Model:        %s  pretrained=%s  freeze=%s\n"
        "mAP50:        %.4f\n"
        "mAP50-95:     %.4f\n"
        "Results dir:  %s/train/\n"
        "%s",
        SEP,
        args.config,
        model_name,
        pretrained,
        freeze_backbone,
        map50,
        map50_95,
        out_cfg["root_dir"],
        SEP,
    )  # pylint: disable=logging-too-many-args

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Training failed with error: {e}", flush=True)
        raise
