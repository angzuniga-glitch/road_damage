from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def derive_country_from_path(path: str) -> str:
    parts = Path(path).parts
    known = {"China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"}
    for name in parts:
        if name in known:
            return name
    return "unknown"


def derive_split_from_path(path: str) -> str:
    parts = [p.lower() for p in Path(path).parts]
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "unknown"


def parse_voc_xml(xml_path: str | Path) -> Tuple[int, int, List[Tuple[str, Tuple[int, int, int, int]]]]:
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> node in {xml_path}")

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    objects: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        label = (obj.findtext("name", default="") or "").strip()
        bbox = obj.find("bndbox")
        if not label or bbox is None:
            continue

        xmin = int(float(bbox.findtext("xmin", default="0")))
        ymin = int(float(bbox.findtext("ymin", default="0")))
        xmax = int(float(bbox.findtext("xmax", default="0")))
        ymax = int(float(bbox.findtext("ymax", default="0")))

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append((label, (xmin, ymin, xmax, ymax)))

    return width, height, objects


def build_xml_index(
    rdd_root: str | Path,
    xml_glob: str = "**/annotations/xmls/*.xml",
) -> Dict[str, Path]:
    root = Path(rdd_root)
    xml_index: Dict[str, Path] = {}
    for xp in root.glob(xml_glob):
        if xp.is_file() and xp.suffix.lower() == ".xml":
            stem = xp.stem
            if stem not in xml_index:
                xml_index[stem] = xp
    return xml_index


def collect_images(
    rdd_root: str | Path,
    image_dir_hint: str = "images",
) -> List[Path]:
    root = Path(rdd_root)
    hint = image_dir_hint.lower().strip()
    images: List[Path] = []

    for dirpath, _, filenames in os.walk(root):
        if hint and hint not in dirpath.lower():
            continue
        for fn in filenames:
            low = fn.lower()
            if low.endswith(IMG_EXTS):
                images.append(Path(dirpath) / fn)

    return images


def discover_rdd_pairs(
    rdd_root: str | Path,
    xml_glob: str = "**/annotations/xmls/*.xml",
    image_dir_hint: str = "images",
) -> List[Dict]:
    xml_index = build_xml_index(rdd_root, xml_glob=xml_glob)
    images = collect_images(rdd_root, image_dir_hint=image_dir_hint)

    pairs: List[Dict] = []
    for img_path in images:
        xml_path = xml_index.get(img_path.stem)
        if xml_path is None:
            continue

        pairs.append(
            {
                "image_path": str(img_path),
                "ann_path": str(xml_path),
                "country": derive_country_from_path(str(img_path)),
                "folder_split": derive_split_from_path(str(img_path)),
            }
        )

    return pairs


def assign_random_image_splits(
    records: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []

    # split at image level, not object level
    shuffled = list(records)
    shuffled.sort(key=lambda x: x["image_path"])
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for i, rec in enumerate(shuffled):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        rec2 = dict(rec)
        rec2["split"] = split
        out.append(rec2)

    return out


def load_detection_records(
    rdd_root: str | Path,
    allowed_labels: Optional[Sequence[str]] = None,
    countries: Optional[Sequence[str]] = None,
    split_mode: str = "random",
    split: str = "train",
    xml_glob: str = "**/annotations/xmls/*.xml",
    image_dir_hint: str = "images",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> List[Dict]:
    records = discover_rdd_pairs(rdd_root, xml_glob=xml_glob, image_dir_hint=image_dir_hint)

    if countries is not None:
        keep = set(countries)
        records = [r for r in records if r["country"] in keep]

    if split_mode == "folder":
        for r in records:
            r["split"] = r["folder_split"]
    else:
        records = assign_random_image_splits(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    records = [r for r in records if r["split"] == split]

    # Filter out images that contain no allowed objects
    allowed_set = set(allowed_labels) if allowed_labels is not None else None
    filtered: List[Dict] = []

    for r in records:
        _, _, objects = parse_voc_xml(r["ann_path"])
        if allowed_set is not None:
            objects = [obj for obj in objects if obj[0] in allowed_set]
        if len(objects) == 0:
            continue
        filtered.append(r)

    return filtered
