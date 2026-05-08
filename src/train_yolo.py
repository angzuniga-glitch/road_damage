from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.data.dataset_yolo import convert_rdd_to_yolo
from src.utils import ensure_dir, set_seed

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on RDD2022.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--skip-convert", action="store_true",
                   help="Skip VOC→YOLO dataset conversion (use if already done).")
    return p.parse_args()

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main() -> int:

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics"
        )

    args   = parse_args()
    cfg    = load_config(args.config)

    set_seed(cfg.get("seed", 1337))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    train_cfg  = cfg["train"]
    model_cfg  = cfg["model"]
    out_cfg    = cfg["outputs"]

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
    model_name  = model_cfg["name"]       
    pretrained  = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)

    if pretrained:
        # Load pretrained weights 
        model = YOLO(model_name)
        logger.info(f"Loaded pretrained model: {model_name}")
    else:
        # Strip .pt extension
        arch_yaml = model_name.replace(".pt", ".yaml")
        model = YOLO(arch_yaml)
        logger.info(f"Initialised model from scratch: {arch_yaml}")

    # Freeze backbone 
    freeze_layers = 0
    if freeze_backbone:
        freeze_layers = 10
        logger.info(f"Freezing first {freeze_layers} layers (backbone)")


    # 4. Train 
    results = model.train(
        data        = str(dataset_yaml),
        epochs      = train_cfg.get("epochs", 50),
        imgsz       = train_cfg.get("imgsz", 640),
        batch       = train_cfg.get("batch_size", 16),
        lr0         = train_cfg.get("lr", 0.01),
        lrf         = train_cfg.get("lrf", 0.01),    
        momentum    = train_cfg.get("momentum", 0.937),
        weight_decay= train_cfg.get("weight_decay", 0.0005),
        warmup_epochs= train_cfg.get("warmup_epochs", 3),
        patience    = train_cfg.get("early_stopping_patience", 10),
        freeze      = freeze_layers if freeze_layers > 0 else None,
        device      = 0,                          
        workers     = train_cfg.get("num_workers", 4),
        project     = str(Path(out_cfg["root_dir"]).resolve()),
        name        = "train",
        exist_ok    = True,                            
        verbose     = True,
        seed        = cfg.get("seed", 1337),
        fliplr      = train_cfg.get("hflip_prob", 0.5),
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 0.0,
        translate   = 0.1,
        scale       = 0.5,
        mosaic      = 1.0,
        close_mosaic= train_cfg.get("close_mosaic", 10),
    )

    metrics = model.val(data=str(dataset_yaml))

    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print("=" * 80)
    print(f"Config:       {args.config}")
    print(f"Model:        {model_name}  pretrained={pretrained}  freeze={freeze_backbone}")
    print(f"mAP50:        {map50:.4f}")
    print(f"mAP50-95:     {map50_95:.4f}")
    print(f"Results dir:  {out_cfg['root_dir']}/train/")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())