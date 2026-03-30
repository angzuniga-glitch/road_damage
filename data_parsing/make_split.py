# Scans the RDD dataset, parses XMLs, extracts annotations, and outputs:
# - train_metadata.csv, train_annotations.npy, train_annotations.pkl
# - val_metadata.csv, val_annotations.npy, val_annotations.pkl
# - test_metadata.csv, test_annotations.npy, test_annotations.pkl
#
# Command that generated current train/val/test split:
# python data_parsing/make_split.py \
#     --rdd_root /nfshome/data \         # Path to dataset
#     --out data/crops_80.10.10_split \  # Output directory for train/val/test folders
#     --image_dir_hint '' \              # Empty = scan all dirs, or specify substring to filter
#     --train_ratio 0.8 \                # 80% of annotated images are train
#     --val_ratio 0.1 \                  # 10% of annotated images are val, leftover test
#     --split_mode random \              # Use random splitting 'folder' for path-based
#     --seed 42                          # Seed for random split
#
# Example Command:
# python data_parsing/make_split.py \
#     --rdd_root /nfshome/data \     # Path to dataset
#     --output_dir data/crops \      # default='data/'  Output directory for train/val/test folders
#     --split_mode random \          # default='folder' Use random splitting 'folder' for path-based or random.
#     --train_ratio 0.8 \            # default='0.8'    80% of annotated images → train
#     --val_ratio 0.1 \              # default='0.1'    10% of annotated images → val (rest → test)
#     --seed 42 \                    # default='1337'   Seed for random split
#     --image_dir_hint '' \          # default=''       Empty = scan all dirs, or specify substring to filter
#     --xml_glob '**/xmls/*.xml' \   # default='**/annotations/xmls/*.xml' Pattern to find XML annotations
#     --pad_ratio 0.10 \             # default='0.1'    Add 10% padding to each side of bbox
#     --min_box_size 5 \             # default='5'      Drop boxes smaller than 5 pixels
#     --allowed_labels '' \          # default=''       Empty = all labels, or 'D00,D10' for specific

from __future__ import annotations
import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import pickle
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

@dataclass(frozen=True)
class ObjRow:
    image_path: str
    ann_path: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    split: str
    country: str

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train/val/test splits from dataset.")

    p.add_argument("--rdd_root", type=str, required=True,
                   help="Path to dataset (/nfshome/data).")
    p.add_argument("--output_dir", type=str, default="data",
                   help="Output directory for splits (default: data/).")
    p.add_argument("--pad_ratio", type=float, default=0.10,
                   help="Padding fraction added to each side of bbox.")
    p.add_argument("--min_box_size", type=int, default=5,
                   help="Drop boxes with width/height smaller than this.")
    p.add_argument("--allowed_labels", type=str, default="",
                   help="Comma-separated whitelist of labels to include.")
    p.add_argument("--split_mode", type=str, default="folder", choices=["folder", "random"],
                   help="Split mode: 'folder' for path-based, 'random' for random split.")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Ratio of images to include in the training split.")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Ratio of images to include in the validation split.")
    p.add_argument("--seed", type=int, default=1337,
                   help="Random seed for random split.")
    p.add_argument("--xml_glob", type=str, default="**/annotations/xmls/*.xml",
                   help="Glob (relative to rdd_root) to find annotation XMLs.")
    p.add_argument("--image_dir_hint", type=str, default="",
                   help="Empty = scan all dirs, or specify substring to filter")
    return p.parse_args()

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def derive_split_from_path(path: str) -> str:
    parts = [p.lower() for p in Path(path).parts]
    if any(x in parts for x in ["train", "training"]):
        return "train"
    if any(x in parts for x in ["val", "valid", "validation"]):
        return "val"
    if any(x in parts for x in ["test", "testing"]):
        return "test"
    return "unknown"

def derive_country_from_path(path: str) -> str:
    parts = Path(path).parts
    for i, name in enumerate(parts):
        if name.lower() == "data" and i + 1 < len(parts):
            return parts[i + 1]
    known = {"China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"}
    for name in parts:
        if name in known:
            return name
    return "unknown"

def assign_random_split(rng: random.Random, train_ratio: float, val_ratio: float) -> str:
    r = rng.random()
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"

def parse_voc_xml(xml_path: Path) -> Tuple[Tuple[int, int], List[Tuple[str, Tuple[int, int, int, int]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError("No <size> tag")
    w = int(size.findtext("width", default="0"))
    h = int(size.findtext("height", default="0"))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size w={w}, h={h}")

    objs: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name", default="") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        xmin = int(float(bnd.findtext("xmin", default="0")))
        ymin = int(float(bnd.findtext("ymin", default="0")))
        xmax = int(float(bnd.findtext("xmax", default="0")))
        ymax = int(float(bnd.findtext("ymax", default="0")))
        objs.append((name, (xmin, ymin, xmax, ymax)))

    return (w, h), objs

def pad_box(box: Tuple[int, int, int, int], img_w: int, img_h: int, pad_ratio: float) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))

    xmin2 = clamp(xmin - pad_x, 0, img_w - 1)
    ymin2 = clamp(ymin - pad_y, 0, img_h - 1)
    xmax2 = clamp(xmax + pad_x, 1, img_w)
    ymax2 = clamp(ymax + pad_y, 1, img_h)
    if xmax2 <= xmin2:
        xmax2 = min(img_w, xmin2 + 1)
    if ymax2 <= ymin2:
        ymax2 = min(img_h, ymin2 + 1)
    return xmin2, ymin2, xmax2, ymax2

def build_xml_index(root: Path, xml_glob: str) -> Dict[str, Path]:
    xml_index: Dict[str, Path] = {}
    xml_paths = list(root.glob(xml_glob))
    if not xml_paths:
        return xml_index

    for xp in xml_paths:
        if xp.is_file() and xp.suffix.lower() == ".xml":
            stem = xp.stem
            if stem not in xml_index:
                xml_index[stem] = xp
    return xml_index

def collect_images(root: Path, image_dir_hint: str) -> List[Path]:
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

def create_npy(rows: List[ObjRow]) -> np.ndarray:
    if not rows:
        return np.array(
            [],
            dtype=[
                ("image_path", "U256"),
                ("ann_path", "U256"),
                ("xmin", "i4"),
                ("ymin", "i4"),
                ("xmax", "i4"),
                ("ymax", "i4"),
                ("label", "U32"),
                ("split", "U16"),
                ("country", "U32"),
            ],
        )

    # Define structured dtype
    dtype = [
        ("image_path", "U256"),  # Unicode string, max 256 chars
        ("ann_path", "U256"),
        ("xmin", "i4"),          # 32-bit integer
        ("ymin", "i4"),
        ("xmax", "i4"),
        ("ymax", "i4"),
        ("label", "U32"),        # Unicode string, max 32 chars
        ("split", "U16"),        # Unicode string, max 16 chars
        ("country", "U32"),      # Unicode string, max 32 chars
    ]

    arr = np.zeros(len(rows), dtype=dtype)

    for i, row in enumerate(rows):
        arr[i] = (
            row.image_path,
            row.ann_path,
            row.xmin,
            row.ymin,
            row.xmax,
            row.ymax,
            row.label,
            row.split,
            row.country,
        )

    return arr

def create_pkl (rows: List[ObjRow]) -> Dict[str, np.ndarray]:
    if not rows:
        return{}

    arr = create_npy(rows)

    unique_images = np.unique(arr['image_path'])
    image_to_boxes = {img: arr[arr['image_path'] == img]
                      for img in unique_images}

    return image_to_boxes

def write_split_files(rows: List[ObjRow], output_dir: Path, split_name: str) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    csv_path = split_dir / f"{split_name}_metadata.csv"
    npy_path = split_dir / f"{split_name}_annotations.npy"
    plk_path = split_dir / f"{split_name}_annotations.pkl"

    # CSV
    fieldnames = ["image_path", "ann_path", "xmin", "ymin", "xmax", "ymax", "label", "split", "country"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    # NPY
    arr =create_npy(rows)
    np.save(npy_path, arr)

    # PKL
    image_to_boxes = create_pkl(rows)
    with plk_path.open("wb") as f:
        pickle.dump(image_to_boxes, f)

def main() -> int:
    args = parse_args()
    root = Path(args.rdd_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    allowed = set(x.strip() for x in args.allowed_labels.split(",") if x.strip())
    rng = random.Random(args.seed)

    print(f"\nBuilding XML index with glob: {args.xml_glob}")
    xml_index = build_xml_index(root, args.xml_glob)
    print(f"XML files indexed: {len(xml_index)}")

    if len(xml_index) == 0:
        print("ERROR: Found 0 XML files. Check --xml_glob and dataset layout.", file=sys.stderr)
        print(f"Try: find {root} -type f -path '*annotations*xmls*' -name '*.xml' | head", file=sys.stderr)
        return 2

    print(f"Collecting images: '{args.image_dir_hint}'")
    images = collect_images(root, args.image_dir_hint)
    print(f"Images found: {len(images)}")

    if not images:
        print("ERROR: Found 0 images. Try --image_dir_hint '' to scan all.", file=sys.stderr)
        return 2

    rows: List[ObjRow] = []
    missing_xml = parse_fail = no_objects = dropped_small = dropped_label = 0

    for i, img_path in enumerate(tqdm(images, desc="Processing images", unit="img"), start=1):

        stem = img_path.stem
        xml_path = xml_index.get(stem)
        if xml_path is None:
            missing_xml += 1
            continue

        try:
            (w, h), objs = parse_voc_xml(xml_path)
        except Exception:
            parse_fail += 1
            continue

        if not objs:
            no_objects += 1
            continue

        split = (
            derive_split_from_path(str(img_path))
            if args.split_mode == "folder"
            else assign_random_split(rng, args.train_ratio, args.val_ratio)
        )
        if split == "unknown" and args.split_mode == "folder":
            split = assign_random_split(rng, args.train_ratio, args.val_ratio)

        country = derive_country_from_path(str(img_path))

        for label, box in objs:
            if allowed and label not in allowed:
                dropped_label += 1
                continue
            xmin, ymin, xmax, ymax = pad_box(box, w, h, args.pad_ratio)
            if (xmax - xmin) < args.min_box_size or (ymax - ymin) < args.min_box_size:
                dropped_small += 1
                continue

            rows.append(
                ObjRow(
                    image_path=str(img_path),
                    ann_path=str(xml_path),
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    label=label,
                    split=split,
                    country=country,
                )
            )

    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        if row.split in splits:
            splits[row.split].append(row)

    print(f"\nWriting split files to: {output_dir}")
    for split_name in ["train", "val", "test"]:
        write_split_files(splits[split_name], output_dir, split_name)

    print(f"{'-' *10}Summary{'-' *10}")
    print(f"Root:                 {root}")
    print(f"Images found:         {len(images)}")
    print(f"XMLs indexed:         {len(xml_index)}")
    print(f"Total rows:           {len(rows)}")
    print(f"      Train: {len(splits['train'])}")
    print(f"      Val:   {len(splits['val'])}")
    print(f"      Test:  {len(splits['test'])}")
    print(f"Missing XMLs:         {missing_xml}") # images do not have matching XML files
    print(f"XML parse failures:   {parse_fail}")
    print(f"No objects in XML:    {no_objects}")
    print(f"Dropped small boxes:  {dropped_small}") # boxes that are smaller than {args.min_box_size} pixels.
    print(f"Dropped label filter: {dropped_label}")

    if len(rows) == 0:
        print("\nERROR: 0 rows written. Likely image stems don't match XML stems.", file=sys.stderr)
        return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
