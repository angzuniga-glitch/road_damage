from __future__ import annotations
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from torchvision import transforms as T
import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CropSample:
    image_path: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    split: str
    country: str

def build_label_map(labels: Sequence[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {lab: i for i, lab in enumerate(uniq)}

def save_label_map(label_map: Dict[str, int], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

def load_label_map(path: str | Path) -> Dict[str, int]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {k: int(v) for k, v in obj.items()}

def load_and_crop(s: CropSample) -> Image.Image:
    with Image.open(s.image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img.crop((s.xmin, s.ymin, s.xmax, s.ymax)).copy()

def cache_crop(args):
    idx, sample = args
    return idx, load_and_crop(sample)

class RDDBboxCropDataset(Dataset):
    def __init__(
            self,
            csv_path: str = None,
            pkl_path: str = None,
            npy_path: str = None,
            split: str = "train",
            transform: Optional[Callable] = None,
            countries: Optional[List[str]] = None,
            allowed_labels: Optional[List[str]] = None,
            cache_images: bool = False,
            label_map: Optional[Dict[str, int]] = None,
            image_size: int = 224, # Added 05.05.26
    ):
        if npy_path and Path(npy_path).exists():
            logger.info("\nLoading from numpy: %s", npy_path)
            arr = np.load(npy_path, allow_pickle=True)
            rows = []
            for record in arr:
                rows.append({
                    'image_path': str(record['image_path']),
                    'ann_path': str(record['ann_path']),
                    'xmin': int(record['xmin']),
                    'ymin': int(record['ymin']),
                    'xmax': int(record['xmax']),
                    'ymax': int(record['ymax']),
                    'label': str(record['label']),
                    'split': str(record['split']),
                    'country': str(record['country']),
                })
            self.data = rows
        elif pkl_path and Path(pkl_path).exists():
            logger.info("\nLoading from pickle: %s\n", pkl_path)
            with open(pkl_path, 'rb') as f:
                image_to_boxes = pickle.load(f)

            rows = []
            for img_path, boxes_array in image_to_boxes.items():
                for box in boxes_array:
                    rows.append({
                        'image_path': str(box['image_path']),
                        'ann_path': str(box['ann_path']),
                        'xmin': int(box['xmin']),
                        'ymin': int(box['ymin']),
                        'xmax': int(box['xmax']),
                        'ymax': int(box['ymax']),
                        'label': str(box['label']),
                        'split': str(box['split']),
                        'country': str(box['country']),
                    })
            self.data = rows
        elif csv_path:
            logger.info("\nLoading from CSV: %s", csv_path)
            df = pd.read_csv(csv_path)
            if split:
                df = df[df["split"] == split]
            if countries:
                df = df[df["country"].isin(countries)]
            if allowed_labels:
                df = df[df["label"].isin(allowed_labels)]
            self.data = df.to_dict('records')
        else:
            raise ValueError("Either csv_path, pkl_path, or npy_path must be provided.")

        if countries:
            self.data = [row for row in self.data if row['country'] in countries]
        if allowed_labels:
            self.data = [row for row in self.data if row['label'] in allowed_labels]

        if len(self.data) == 0:
            raise ValueError("No samples after filtering.")

        all_labels = [row['label'] for row in self.data]
        
        # Added 05.05.26, logs extra labels as an warning, raises error for missing labels
        if label_map is not None:
            current_label_set = set(all_labels)
            map_label_set = set(label_map.keys())
            
            extra_in_map = map_label_set - current_label_set
            missing_from_map = current_label_set - map_label_set

            if extra_in_map:
                logger.warning("label_map contains unused labels: %s", extra_in_map)
            if missing_from_map:
                raise ValueError(
                    f"Dataset has labels not in label_map: {missing_from_map}. "
                    f"Current data labels: {sorted(current_label_set)}, "
                    f"label_map keys: {sorted(map_label_set)}"
                )
            self.label_map = dict(label_map)
        else:
            self.label_map = build_label_map(all_labels)
        
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        # End add

        self.samples: List[CropSample] = []
        for row in self.data:
            self.samples.append(
                CropSample(
                    image_path = str(row['image_path']),
                    xmin = int(row['xmin']),
                    ymin = int(row['ymin']),
                    xmax = int(row['xmax']),
                    ymax = int(row['ymax']),
                    label = str(row['label']),
                    split = str(row['split']),
                    country = str(row['country']),
                )
            )
        self.transform = transform
        self.image_size = image_size # added 05.05.26
        self.cache_images = cache_images

        self.cached_crops: Optional[List[Image.Image]] = None
        if self.cache_images:
            split_name = split if split else "data"
            print(f"[{split_name}] Caching {len(self.samples)} crops in memory...")

            self.cached_crops = [None] * len(self.samples)
            with ProcessPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(
                    executor.map(cache_crop, enumerate(self.samples)),
                    total=len(self.samples),
                    desc=f"[{split_name}] Caching"
                ))
                for idx, crop in results:
                    self.cached_crops[idx] = crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]

        if self.cached_crops is not None:
            img = self.cached_crops[idx]
        else:
            img = Image.open(s.image_path).convert("RGB")
            w, h = img.size
            left = max(0, min(s.xmin, w - 1))
            upper = max(0, min(s.ymin, h - 1))
            right = max(left + 1, min(s.xmax, w))
            lower = max(upper + 1, min(s.ymax, h))
            img = img.crop((left, upper, right, lower))

        # if self.transform is not None:
        #     img = self.transform(img)
        # else:
        #     img = torch.from_numpy(
        #         (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        #          .view(img.size[1], img.size[0], 3)
        #          .numpy())
        #     ).permute(2, 0, 1).float() / 255.0

        # Added 05.05.26, Removed manual conversion, switched to resize image
        # and then convert ToTensor() for PIL image to [0, 1] float tensor
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor()
            ])(img)
        # End add

        label_id = self.label_map[s.label]
        return img, label_id