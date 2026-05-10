#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = {
    1: "person",
    2: "rider",
    3: "car",
    4: "truck",
    5: "bus",
    6: "train",
    7: "motorcycle",
    8: "bicycle",
}


def indent(elem, level=0):
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def add_text(parent, tag, text):
    elem = ET.SubElement(parent, tag)
    elem.text = str(text)
    return elem


def image_id_from_file_name(file_name):
    return Path(file_name).stem


def voc_xml(image, annotations, image_id):
    root = ET.Element("annotation")
    add_text(root, "folder", "JPEGImages")
    add_text(root, "filename", image_id + ".png")
    add_text(root, "path", image_id + ".png")

    source = ET.SubElement(root, "source")
    add_text(source, "database", "Foggy Cityscapes")

    size = ET.SubElement(root, "size")
    width = int(image["width"])
    height = int(image["height"])
    add_text(size, "width", width)
    add_text(size, "height", height)
    add_text(size, "depth", 3)
    add_text(root, "segmented", 0)

    for ann in annotations:
        category_id = int(ann["category_id"])
        if category_id not in CLASS_NAMES:
            continue

        x, y, w, h = ann["bbox"]
        xmin = max(0.0, x)
        ymin = max(0.0, y)
        xmax = min(float(width), x + w)
        ymax = min(float(height), y + h)
        if xmax <= xmin or ymax <= ymin:
            continue

        obj = ET.SubElement(root, "object")
        add_text(obj, "name", CLASS_NAMES[category_id])
        add_text(obj, "pose", "Unspecified")
        add_text(obj, "truncated", 0)
        add_text(obj, "difficult", int(ann.get("iscrowd", 0)))
        box = ET.SubElement(obj, "bndbox")
        add_text(box, "xmin", int(round(xmin)))
        add_text(box, "ymin", int(round(ymin)))
        add_text(box, "xmax", int(round(xmax)))
        add_text(box, "ymax", int(round(ymax)))

    indent(root)
    return ET.ElementTree(root)


def convert_split(coco_json, image_root, out_root, split_name):
    with open(coco_json, "r") as handle:
        data = json.load(handle)

    annotations_by_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    jpeg_dir = out_root / "JPEGImages"
    anno_dir = out_root / "Annotations"
    split_dir = out_root / "ImageSets" / "Main"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    image_ids = []
    missing_images = []
    for image in sorted(data["images"], key=lambda item: item["file_name"]):
        image_id = image_id_from_file_name(image["file_name"])
        src = image_root / image["file_name"]
        dst = jpeg_dir / (image_id + ".png")
        if not src.exists():
            missing_images.append(str(src))
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

        xml_path = anno_dir / (image_id + ".xml")
        voc_xml(image, annotations_by_image[int(image["id"])], image_id).write(
            xml_path, encoding="utf-8", xml_declaration=False
        )
        image_ids.append(image_id)

    split_path = split_dir / (split_name + ".txt")
    split_path.write_text("\n".join(image_ids) + "\n")
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images referenced by {coco_json} were not found; first: {missing_images[0]}"
        )
    return image_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        default="/home/gwb/labProject/datasets/cityscapes_foggy",
        help="Foggy Cityscapes root containing annotations and leftImg8bit_foggy.",
    )
    parser.add_argument(
        "--out-root",
        default="datasets/foggy_cityscapes_voc/VOC2007",
        help="Output VOC2007 directory.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"{out_root} already exists; pass --force to replace it")
        shutil.rmtree(out_root)

    train_ids = convert_split(
        src_root / "annotations" / "instances_cityscapes_foggy_train_mixed.json",
        src_root / "leftImg8bit_foggy" / "train",
        out_root,
        "target_trainval",
    )
    test_ids = convert_split(
        src_root / "annotations" / "instances_cityscapes_foggy_val_mixed.json",
        src_root / "leftImg8bit_foggy" / "val",
        out_root,
        "test",
    )

    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"train/test overlap: {len(overlap)} ids, first: {overlap[0]}")

    print(f"output: {out_root}")
    print(f"target_trainval: {len(train_ids)}")
    print(f"test: {len(test_ids)}")
    print("overlap: 0")


if __name__ == "__main__":
    main()
