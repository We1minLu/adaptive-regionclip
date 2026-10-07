#!/usr/bin/env python3
"""Create lightweight Cityscapes/Foggy VOC test sets for Exp1.

Images are referenced with absolute symlinks.  XML annotations are derived from
the clear Cityscapes gtFine polygons; the three fog variants reuse the matching
clear-scene objects.  No image data is copied.
"""

import argparse
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASS_NAMES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


def add_text(parent, tag, value):
    child = ET.SubElement(parent, tag)
    child.text = str(value)
    return child


def indent(element, level=0):
    padding = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = padding + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = padding
    if level and (not element.tail or not element.tail.strip()):
        element.tail = padding


def canonical_label(label):
    is_group = label.endswith("group")
    base = label[:-5] if is_group else label
    if base not in CLASS_NAMES:
        return None, is_group
    return base, is_group


def polygon_box(polygon, width, height):
    if len(polygon) < 3:
        return None
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    xmin = max(0.0, min(xs))
    ymin = max(0.0, min(ys))
    xmax = min(float(width), max(xs))
    ymax = min(float(height), max(ys))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def read_annotations(gt_root):
    records = {}
    class_counts = Counter()
    group_counts = Counter()
    invalid_polygons = 0
    for path in sorted(gt_root.glob("*/*_gtFine_polygons.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        scene_id = path.name.replace("_gtFine_polygons.json", "_leftImg8bit")
        width = int(data["imgWidth"])
        height = int(data["imgHeight"])
        objects = []
        for item in data.get("objects", []):
            label, is_group = canonical_label(item.get("label", ""))
            if label is None:
                continue
            if is_group:
                group_counts[label] += 1
                continue
            box = polygon_box(item.get("polygon", []), width, height)
            if box is None:
                invalid_polygons += 1
                continue
            objects.append({"name": label, "bbox": box, "is_group": is_group})
            class_counts[label] += 1
        records[scene_id] = {"width": width, "height": height, "objects": objects}
    return records, class_counts, group_counts, invalid_polygons


def make_xml(record, image_id, database):
    root = ET.Element("annotation")
    add_text(root, "folder", "JPEGImages")
    add_text(root, "filename", image_id + ".png")
    add_text(root, "path", image_id + ".png")
    source = ET.SubElement(root, "source")
    add_text(source, "database", database)
    size = ET.SubElement(root, "size")
    add_text(size, "width", record["width"])
    add_text(size, "height", record["height"])
    add_text(size, "depth", 3)
    add_text(root, "segmented", 0)
    for item in record["objects"]:
        obj = ET.SubElement(root, "object")
        add_text(obj, "name", item["name"])
        add_text(obj, "pose", "Unspecified")
        add_text(obj, "truncated", 0)
        add_text(obj, "difficult", int(item["is_group"]))
        box = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), item["bbox"]):
            add_text(box, tag, int(round(value)))
    indent(root)
    return ET.ElementTree(root)


def collect_clear_images(clear_root):
    return {path.stem: path.resolve() for path in sorted(clear_root.glob("*/*_leftImg8bit.png"))}


def clear_id_for_fog(image_id):
    marker = "_foggy_beta_"
    if marker not in image_id:
        raise ValueError("Fog image has no beta marker: {}".format(image_id))
    return image_id.split(marker, 1)[0]


def collect_fog_images(fog_root):
    return {path.stem: path.resolve() for path in sorted(fog_root.glob("*/*_foggy_beta_*.png"))}


def validate_inputs(records, clear_images, fog_images):
    clear_missing_annotations = sorted(set(clear_images) - set(records))
    annotation_missing_images = sorted(set(records) - set(clear_images))
    fog_missing_annotations = sorted(
        image_id for image_id in fog_images if clear_id_for_fog(image_id) not in records
    )
    if clear_missing_annotations:
        raise FileNotFoundError("Clear images without annotations: {}".format(clear_missing_annotations[:3]))
    if annotation_missing_images:
        raise FileNotFoundError("Annotations without clear images: {}".format(annotation_missing_images[:3]))
    if fog_missing_annotations:
        raise FileNotFoundError("Fog images without clear annotations: {}".format(fog_missing_annotations[:3]))


def write_voc(root, images, records, database, foggy=False):
    jpeg_dir = root / "JPEGImages"
    annotation_dir = root / "Annotations"
    split_dir = root / "ImageSets" / "Main"
    jpeg_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    image_ids = []
    object_count = 0
    for image_id, source in sorted(images.items()):
        record_id = clear_id_for_fog(image_id) if foggy else image_id
        record = records[record_id]
        os.symlink(str(source), str(jpeg_dir / (image_id + ".png")))
        make_xml(record, image_id, database).write(
            str(annotation_dir / (image_id + ".xml")), encoding="utf-8", xml_declaration=False
        )
        image_ids.append(image_id)
        object_count += len(record["objects"])
    (split_dir / "test.txt").write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    return {"images": len(image_ids), "objects": object_count}


def build_report(args, records, clear_images, fog_images, class_counts, group_counts, invalid_polygons):
    beta_counts = Counter()
    for image_id in fog_images:
        beta_counts[image_id.split("_foggy_beta_", 1)[1]] += 1
    clear_objects = sum(len(record["objects"]) for record in records.values())
    fog_objects = sum(len(records[clear_id_for_fog(image_id)]["objects"]) for image_id in fog_images)
    return {
        "clear_root": str(args.clear_root.resolve()),
        "foggy_root": str(args.foggy_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "clear_images": len(clear_images),
        "foggy_images": len(fog_images),
        "clear_objects": clear_objects,
        "foggy_objects": fog_objects,
        "class_counts_clear": dict(class_counts),
        "group_counts_clear": dict(group_counts),
        "fog_beta_counts": dict(sorted(beta_counts.items())),
        "invalid_polygons": invalid_polygons,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Exp1 City/Foggy VOC test datasets with symlinked images.")
    parser.add_argument("--clear-root", type=Path, default=Path("/root/autodl-tmp/data/cityspace"))
    parser.add_argument("--foggy-root", type=Path, default=Path("/root/autodl-tmp/data/foggycity"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records, class_counts, group_counts, invalid_polygons = read_annotations(args.clear_root / "gtFine" / "val")
    clear_images = collect_clear_images(args.clear_root / "leftImg8bit" / "val")
    fog_images = collect_fog_images(args.foggy_root / "leftImg8bit_foggy" / "val")
    validate_inputs(records, clear_images, fog_images)
    report = build_report(
        args, records, clear_images, fog_images, class_counts, group_counts, invalid_polygons
    )
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    output_root = args.output_root.resolve()
    clear_target = output_root / "cityscapes_voc" / "VOC2007"
    fog_target = output_root / "foggy_cityscapes_voc" / "VOC2007"
    if clear_target.exists() or clear_target.is_symlink() or fog_target.exists() or fog_target.is_symlink():
        raise FileExistsError("Refusing to replace an existing City/Foggy VOC dataset")
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".exp1-voc-stage-", dir=str(output_root)))
    try:
        clear_stage = stage / "cityscapes_voc" / "VOC2007"
        fog_stage = stage / "foggy_cityscapes_voc" / "VOC2007"
        clear_stats = write_voc(clear_stage, clear_images, records, "Cityscapes", foggy=False)
        fog_stats = write_voc(fog_stage, fog_images, records, "Foggy Cityscapes", foggy=True)
        clear_target.parent.mkdir(parents=True, exist_ok=True)
        fog_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(clear_stage), str(clear_target))
        os.replace(str(fog_stage), str(fog_target))
        report["clear_written"] = clear_stats
        report["foggy_written"] = fog_stats
        report_path = output_root / "exp1_cityscapes_voc_build_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        if stage.exists():
            shutil.rmtree(str(stage))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
