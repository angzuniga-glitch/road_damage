from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.data.xml_utils import load_detection_records, parse_voc_xml


class DetectionTransform:
    """
    Minimal detection transform:
    - PIL image -> tensor
    - optional random horizontal flip
    """
    def __init__(self, train: bool = False, hflip_prob: float = 0.5) -> None:
        self.train = train
        self.hflip_prob = hflip_prob

    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
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
    ) -> None:
        super().__init__()
        self.rdd_root = rdd_root
        self.split = split
        self.transform = transform
        self.allowed_labels = list(allowed_labels)
        self.label_map = {lab: i + 1 for i, lab in enumerate(self.allowed_labels)}  # detection labels start at 1

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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image_path = rec["image_path"]
        ann_path = rec["ann_path"]

        image = Image.open(image_path).convert("RGB")
        width, height, objects = parse_voc_xml(ann_path)

        boxes: List[List[float]] = []
        labels: List[int] = []

        for label_str, (xmin, ymin, xmax, ymax) in objects:
            if label_str not in self.label_map:
                continue
            boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            labels.append(self.label_map[label_str])

        if len(boxes) == 0:
            # Should already be filtered out, but keep this defensive.
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = F.to_tensor(image)

        return image, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))
