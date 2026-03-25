#!/usr/bin/env python3
"""
Build crop-level classification index from RDD-style VOC XML annotations.

Dataset layout:
  /nfshome/data/<Country>/<split>/annotations/xmls/<stem>.xml
and images live somewhere else (often .../<split>/images/<stem>.jpg)

Key fix vs earlier version:
- Build a global index of all XMLs: stem -> xml_path
- Then match each image by stem lookup (fast + reliable on NFS)

Output: CSV with one row per annotated object (bbox crop sample).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

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
    p = argparse.ArgumentParser(description="Build crop index CSV for RDD bbox classification (XML-index matching).")
    p.add_argument("--rdd_root", type=str, required=True, help="Path to dataset root (e.g., /nfshome/data).")
    p.add_argument("--out", type=str, required=True, help="Output CSV path (e.g., data/crops.csv).")

    p.add_argument("--pad_ratio", type=float, default=0.10, help="Padding fraction added to each side of bbox.")
    p.add_argument("--min_box_size", type=int, default=5, help="Drop boxes with width/height smaller than this.")
    p.add_argument("--allowed_labels", type=str, default="", help="Comma-separated whitelist of labels to include.")

    p.add_argument("--split_mode", type=str, default="folder", choices=["folder", "random"])
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)

    # Layout hints
    p.add_argument(
        "--xml_glob",
        type=str,
        default="**/annotations/xmls/*.xml",
        help="Glob (relative to rdd_root) to find annotation XMLs.",
    )
    p.add_argument(
        "--image_dir_hint",
        type=str,
        default="images",
        help="Only treat image files under directories whose path contains this substring (case-insensitive). "
             "Set empty '' to accept all images.",
    )
    p.add_argument("--print_every", type=int, default=2000, help="Print progress every N images.")
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
    """
    For your layout:
      /nfshome/data/<Country>/<split>/images/...
    we want <Country>.
    """
    parts = Path(path).parts
    # Find the "data" directory and take the next component as country
    for i, name in enumerate(parts):
        if name.lower() == "data" and i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: guess from known country folders if present
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
    """
    Build stem -> xml_path map. If duplicates occur, prefer the first encountered.
    """
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


def main() -> int:
    args = parse_args()
    root = Path(args.rdd_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    allowed = set(x.strip() for x in args.allowed_labels.split(",") if x.strip())
    rng = random.Random(args.seed)

    print(f"[info] building XML index with glob: {args.xml_glob}")
    xml_index = build_xml_index(root, args.xml_glob)
    print(f"[info] XML files indexed: {len(xml_index)}")

    if len(xml_index) == 0:
        print("ERROR: Found 0 XML files. Check --xml_glob and dataset layout.", file=sys.stderr)
        print(f"Try: find {root} -type f -path '*annotations*xmls*' -name '*.xml' | head", file=sys.stderr)
        return 2

    print(f"[info] collecting images (hint='{args.image_dir_hint}')")
    images = collect_images(root, args.image_dir_hint)
    print(f"[info] images found: {len(images)}")

    if not images:
        print("ERROR: Found 0 images. Try --image_dir_hint '' to scan all.", file=sys.stderr)
        return 2

    rows: List[ObjRow] = []
    missing_xml = parse_fail = no_objects = dropped_small = dropped_label = 0

    for i, img_path in enumerate(images, start=1):
        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[progress] scanned {i}/{len(images)} images, rows so far={len(rows)}")

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

        split = derive_split_from_path(str(img_path)) if args.split_mode == "folder" else assign_random_split(
            rng, args.train_ratio, args.val_ratio
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

    # Write CSV
    fieldnames = ["image_path", "ann_path", "xmin", "ymin", "xmax", "ymax", "label", "split", "country"]
    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r.__dict__)

    print("=== build_crop_index summary ===")
    print(f"Root:                 {root}")
    print(f"Images found:         {len(images)}")
    print(f"XML indexed:          {len(xml_index)}")
    print(f"Rows written:         {len(rows)}")
    print(f"CSV:                  {out}")
    print(f"Missing XML (by stem):{missing_xml}")
    print(f"XML parse failures:   {parse_fail}")
    print(f"No objects in XML:    {no_objects}")
    print(f"Dropped small boxes:  {dropped_small}")
    print(f"Dropped label filter: {dropped_label}")

    if len(rows) == 0:
        print("\nERROR: 0 rows written. Likely image stems don't match XML stems.", file=sys.stderr)
        print("Check one image stem and see if XML exists with same stem:", file=sys.stderr)
        print("  ls /nfshome/data/Japan/train/annotations/xmls | head", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())