from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.data.xml_utils import load_detection_records, parse_voc_xml

logger = logging.getLogger(__name__)


class DetectionTransform:
    """
    Minimal detection transform:
    - PIL image -> tensor
    - optional random horizontal flip
    """

    def __init__(
        self,
        train: bool = False,
        hflip_prob: float = 0.5,
        min_sizes: tuple = (480, 512, 544, 576, 608, 640),
        max_size: int = 1333,
    ) -> None:
        self.train = train
        self.hflip_prob = hflip_prob
        self.min_sizes = min_sizes
        self.max_size = max_size

    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:

        w, h = image.size
        if self.train:
            target_min = random.choice(self.min_sizes)

            scale = target_min / min(h, w)
            if scale * max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)

            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            if "boxes" in target and len(target["boxes"]) > 0:
                target["boxes"] = target["boxes"].clone() * scale

        image = F.to_tensor(image)

        if self.train and torch.rand(1).item() < self.hflip_prob:
            _, h, w = image.shape
            image = torch.flip(image, dims=[2])
            boxes = target["boxes"].clone()
            xmin = boxes[:, 0].clone()
            xmax = boxes[:, 2].clone()
            boxes[:, 0] = w - xmax
            boxes[:, 2] = w - xmin
            target["boxes"] = boxes

        return image, target


class RDDDetectionDataset(Dataset):
    def __init__(
        self,
        rdd_root: str,
        split: str,
        allowed_labels: Sequence[str],
        transform: Optional[Callable] = None,
        countries: Optional[Sequence[str]] = None,
        split_mode: str = "random",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 1337,
        xml_glob: str = "**/annotations/xmls/*.xml",
        image_dir_hint: str = "images",
        cache_annotations: bool = True,
        cache_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.rdd_root = rdd_root
        self.split = split
        self.transform = transform
        self.allowed_labels = list(allowed_labels)
        self.label_map = {
            lab: i + 1 for i, lab in enumerate(self.allowed_labels)
        }  # detection labels start at 1

        self.records = load_detection_records(
            rdd_root=rdd_root,
            allowed_labels=allowed_labels,
            countries=countries,
            split_mode=split_mode,
            split=split,
            xml_glob=xml_glob,
            image_dir_hint=image_dir_hint,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        if len(self.records) == 0:
            raise ValueError(f"No detection records found for split='{split}'")

        self._ann_cache: Optional[List[Dict]] = None
        if cache_annotations:
            _cache_path = (
                Path(cache_path)
                if cache_path
                else Path("outputs") / f".ann_cache_{split}.pkl"
            )

            if _cache_path.exists():
                logger.info("[%s] Loading annotation cache: %s", split, _cache_path)
                with open(_cache_path, "rb") as f:
                    self._ann_cache = pickle.load(f)
            else:
                logger.info(
                    "[%s] Building annotation cache for %s images ...",
                    split,
                    len(self.records),
                )
                self._ann_cache = [
                    self._parse_target(rec, idx) for idx, rec in enumerate(self.records)
                ]
                _cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_cache_path, "wb") as f:
                    pickle.dump(self._ann_cache, f)
                logger.info("[%s] Cache saved: %s", split, _cache_path)

    def _parse_target(self, rec: Dict, idx: int) -> Dict:

        ann_path = rec["ann_path"]
        _width, _height, objects = parse_voc_xml(ann_path)

        boxes: List[List[float]] = []
        labels: List[int] = []

        for label_str, (xmin, ymin, xmax, ymax) in objects:
            if label_str not in self.label_map:
                continue
            boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            labels.append(self.label_map[label_str])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_path = rec["image_path"]

        image = Image.open(image_path).convert("RGB")

        if self._ann_cache is not None:
            raw = self._ann_cache[idx]
            target = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }
        else:
            target = self._parse_target(rec, idx)

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = F.to_tensor(image)

        return image, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))
