"""Convert Pascal VOC annotations to YOLO labels and tag each image as damaged, clean or a hard negative.

Standard library only, so CI can test it. Used for the original UAV-PDD2023 release and for the
RDD2022 China_Drone images that the unified dataset dropped.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

DAMAGED, CLEAN, HARD_NEGATIVE = "damaged", "clean", "hard_negative"


@dataclass(frozen=True)
class VocBox:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocImage:
    filename: str
    width: int
    height: int
    boxes: list[VocBox] = field(default_factory=list)


def parse_voc(xml_path: Path) -> VocImage:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes = []
    for obj in root.iter("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append(
            VocBox(
                obj.findtext("name", "").strip(),
                *(float(bnd.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")),
            )
        )
    return VocImage(root.findtext("filename", xml_path.stem).strip(), width, height, boxes)


def tag_image(image: VocImage, class_map: Mapping[str, int]) -> str:
    """damaged: at least one mapped box. clean: no objects at all. hard_negative: only unmapped objects."""
    if any(b.name in class_map for b in image.boxes):
        return DAMAGED
    return HARD_NEGATIVE if image.boxes else CLEAN


def to_yolo_lines(image: VocImage, class_map: Mapping[str, int]) -> tuple[list[str], Counter]:
    """Return YOLO label lines plus counts of dropped boxes by reason. Boxes are clipped to the image."""
    lines, dropped = [], Counter()
    w, h = image.width, image.height
    if w <= 0 or h <= 0:
        raise ValueError(f"{image.filename}: missing image size")
    for b in image.boxes:
        if b.name not in class_map:
            dropped[f"unmapped:{b.name}"] += 1
            continue
        x0, y0 = max(0.0, min(b.xmin, w)), max(0.0, min(b.ymin, h))
        x1, y1 = max(0.0, min(b.xmax, w)), max(0.0, min(b.ymax, h))
        if (x0, y0, x1, y1) != (b.xmin, b.ymin, b.xmax, b.ymax):
            dropped["clipped_kept"] += 1  # a note, not a drop: the clipped box is kept
        if x1 - x0 < 1 or y1 - y0 < 1:
            dropped["degenerate"] += 1
            continue
        cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"{class_map[b.name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, +dropped


def convert_voc_dir(
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    class_map: Mapping[str, int],
    rename: Callable[[Path], str],
) -> list[dict]:
    """Copy images to out_dir/images and write YOLO labels to out_dir/labels. Returns one record per image."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    records = []
    for image_path, xml_path in pairs:
        image = parse_voc(xml_path)
        lines, dropped = to_yolo_lines(image, class_map)
        stem = rename(image_path)
        target = out_dir / "images" / f"{stem}{image_path.suffix.lower()}"
        if not target.exists() or target.stat().st_size != image_path.stat().st_size:
            shutil.copyfile(image_path, target)
        (out_dir / "labels" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))
        records.append(
            {
                "stem": stem,
                "source_file": image_path.name,
                "width": image.width,
                "height": image.height,
                "tag": tag_image(image, class_map),
                "boxes": len(lines),
                "raw_classes": dict(Counter(b.name for b in image.boxes)),
                "dropped": dict(dropped),
            }
        )
    return records
