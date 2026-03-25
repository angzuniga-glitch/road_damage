from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


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


def _load_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"crops.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"image_path", "xmin", "ymin", "xmax", "ymax", "label", "split", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"crops.csv missing columns: {sorted(missing)}")
    return df


def build_label_map(labels: Sequence[str]) -> Dict[str, int]:
    """
    Stable label map: sorted unique labels.
    """
    uniq = sorted(set(labels))
    return {lab: i for i, lab in enumerate(uniq)}


def save_label_map(label_map: Dict[str, int], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)


def load_label_map(path: str | Path) -> Dict[str, int]:
    path = Path(path)
    with path.open("r") as f:
        obj = json.load(f)
    # Ensure int values
    return {k: int(v) for k, v in obj.items()}


class RDDBboxCropDataset(Dataset):
    """
    Crop-level classification dataset built from crops.csv.

    Each item:
      - loads the image
      - crops bbox (xmin,ymin,xmax,ymax)
      - converts to RGB
      - applies transform
      - returns (image_tensor, label_id)

    Notes:
    - Coordinates are assumed to already include any padding you want.
    - Image reading uses PIL.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        countries: Optional[Sequence[str]] = None,
        allowed_labels: Optional[Sequence[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
        label_map_out: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform

        df = _load_csv(self.csv_path)

        # Split filter
        df = df[df["split"] == split].copy()

        # Optional filters
        if countries is not None:
            countries_set = set(countries)
            df = df[df["country"].isin(countries_set)].copy()

        if allowed_labels is not None:
            allowed_set = set(allowed_labels)
            df = df[df["label"].isin(allowed_set)].copy()

        if len(df) == 0:
            raise ValueError(
                f"No samples after filtering. split={split}, countries={countries}, allowed_labels={allowed_labels}"
            )

        # Create label map if not provided
        if label_map is None:
            label_map = build_label_map(df["label"].tolist())
            if label_map_out is not None:
                save_label_map(label_map, label_map_out)

        self.label_map = label_map
        self.id_to_label = {v: k for k, v in self.label_map.items()}

        # Build list of samples
        self.samples: List[CropSample] = []
        for row in df.itertuples(index=False):
            self.samples.append(
                CropSample(
                    image_path=str(row.image_path),
                    xmin=int(row.xmin),
                    ymin=int(row.ymin),
                    xmax=int(row.xmax),
                    ymax=int(row.ymax),
                    label=str(row.label),
                    split=str(row.split),
                    country=str(row.country),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]

        # Load image
        img = Image.open(s.image_path).convert("RGB")

        # Crop bbox (PIL crop box is (left, upper, right, lower))
        # Ensure bounds are valid (defensive)
        w, h = img.size
        left = max(0, min(s.xmin, w - 1))
        upper = max(0, min(s.ymin, h - 1))
        right = max(left + 1, min(s.xmax, w))
        lower = max(upper + 1, min(s.ymax, h))

        img = img.crop((left, upper, right, lower))

        # Transform -> tensor
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Fallback: minimal conversion to tensor
            img = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                 .view(img.size[1], img.size[0], 3)
                 .numpy())
            ).permute(2, 0, 1).float() / 255.0

        label_id = self.label_map[s.label]
        return img, label_id