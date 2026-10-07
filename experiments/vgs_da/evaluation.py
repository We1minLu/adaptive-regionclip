"""CPU VOC07 metrics and online, annotation-free evaluation for the formal run.

Native repository conventions are deliberate: prediction x/y minima +1,
scores serialized to 3 decimals, boxes to 1 decimal, inclusive IoU, strict
IoU > threshold, and eleven-point AP. Mixed AP pools all density-specific
image IDs; it is not the arithmetic mean of the three density AP values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
DENSITIES = ("0.005", "0.01", "0.02")
THRESHOLDS = tuple(range(50, 100, 5))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(manifest_or_records):
    if isinstance(manifest_or_records, (str, Path)):
        value = json.loads(Path(manifest_or_records).read_text(encoding="utf-8"))
    else:
        value = manifest_or_records
    if isinstance(value, dict):
        for key in ("records", "items", "images"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or not value:
        raise ValueError("Evaluation manifest must contain a nonempty list of records")
    ids = [str(row["image_id"]) for row in value]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation image_id; retain the fog density suffix")
    for row in value:
        if "annotations" not in row:
            raise ValueError("Evaluation GT annotations are required in the CPU manifest")
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise ValueError("Invalid original image dimensions")
        for obj in row["annotations"]:
            box = np.asarray(obj["bbox"], dtype=np.float64)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or np.any(box != np.round(box)) or not 0 <= int(obj["category_id"]) < 8):
                raise ValueError("VOC GT must have integer raw XML XYXY boxes and eight-class labels")
    return value


def _array(value, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def prediction_record(image_id, result):
    """Convert wrapper output or plain arrays to unrounded original-pixel data."""
    if "instances" in result:
        instances = result["instances"].to("cpu")
        boxes, scores, classes = instances.pred_boxes.tensor, instances.scores, instances.pred_classes
    else:
        boxes, scores, classes = result["boxes"], result["scores"], result["classes"]
    boxes = _array(boxes, np.float64).reshape(-1, 4)
    scores = _array(scores, np.float64).reshape(-1)
    raw_classes = _array(classes, np.float64).reshape(-1)
    if len(boxes) != len(scores) or len(scores) != len(raw_classes):
        raise ValueError("Prediction array length mismatch")
    if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or not np.isfinite(raw_classes).all() or np.any(raw_classes != np.round(raw_classes))
            or np.any(raw_classes < 0) or np.any(raw_classes >= 8)):
        raise ValueError("Non-finite predictions or invalid class IDs")
    return dict(image_id=str(image_id), boxes=boxes.tolist(), scores=scores.tolist(),
                classes=raw_classes.astype(np.int64).tolist())


def _serialized_predictions(records, predictions):
    by_id = {}
    for pred in predictions:
        key = str(pred["image_id"])
        if key in by_id:
            raise ValueError("Repeated prediction image_id: " + key)
        by_id[key] = prediction_record(key, pred)
    expected = {str(row["image_id"]) for row in records}
    if set(by_id) != expected:
        raise ValueError("Prediction image IDs differ from the evaluation manifest")
    output = [[] for _ in CLASSES]
    for row in records:
        image_id = str(row["image_id"])
        pred = by_id[image_id]
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["classes"]):
            serialized_box = [float(f"{box[0] + 1:.1f}"), float(f"{box[1] + 1:.1f}"),
                              float(f"{box[2]:.1f}"), float(f"{box[3]:.1f}")]
            output[label].append((image_id, float(f"{score:.3f}"), serialized_box))
    return output


def voc_ap_07(recall, precision):
    return float(sum(float(precision[recall >= threshold].max())
                     if np.any(recall >= threshold) else 0.0
                     for threshold in np.arange(0.0, 1.1, 0.1)) / 11.0)


def _class_result(records, detections, label, threshold):
    gt = {}
    npos = 0
    for row in records:
        objects = [obj for obj in row["annotations"] if int(obj["category_id"]) == label]
        boxes = np.asarray([obj["bbox"] for obj in objects], dtype=np.float64).reshape(-1, 4)
        difficult = np.asarray([bool(obj.get("difficult", False)) for obj in objects], dtype=bool)
        gt[str(row["image_id"])] = (boxes, difficult, np.zeros(len(objects), dtype=bool))
        npos += int((~difficult).sum())
    # Match numpy's default argsort exactly, including rounded-score ties.
    order = np.argsort(-np.asarray([row[1] for row in detections], dtype=np.float64))
    tp = np.zeros(len(detections), dtype=np.float64)
    fp = np.zeros(len(detections), dtype=np.float64)
    ignored = duplicates = 0
    for rank, index in enumerate(order):
        image_id, _, predicted = detections[int(index)]
        boxes, difficult, detected = gt[image_id]
        best_iou, best = -np.inf, -1
        if len(boxes):
            predicted = np.asarray(predicted, dtype=np.float64)
            wh = np.maximum(np.minimum(boxes[:, 2:], predicted[2:])
                            - np.maximum(boxes[:, :2], predicted[:2]) + 1.0, 0.0)
            intersection = wh[:, 0] * wh[:, 1]
            areas = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
            predicted_area = (predicted[2] - predicted[0] + 1.0) * (predicted[3] - predicted[1] + 1.0)
            union = areas + predicted_area - intersection
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            best = int(overlap.argmax())
            best_iou = float(overlap[best])
        if best_iou > threshold:
            if difficult[best]:
                ignored += 1
            elif not detected[best]:
                tp[rank] = 1
                detected[best] = True
            else:
                fp[rank] = 1
                duplicates += 1
        else:
            fp[rank] = 1
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cumulative_tp / npos if npos else np.zeros_like(cumulative_tp)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
    return dict(label=label, name=CLASSES[label], gt=npos, AP=100 * voc_ap_07(recall, precision),
                tp=int(tp.sum()), fp=int(fp.sum()), duplicate_fp=duplicates,
                ignored=ignored, detections=len(detections))


def _metrics(records, serialized):
    ids = {str(row["image_id"]) for row in records}
    selected = [[row for row in detections if row[0] in ids] for detections in serialized]
    by_iou = {}
    for integer in THRESHOLDS:
        rows = [_class_result(records, selected[label], label, integer / 100.0) for label in range(8)]
        by_iou[str(integer)] = dict(AP=float(np.mean([row["AP"] for row in rows])), per_class=rows,
                                    tp=sum(row["tp"] for row in rows), fp=sum(row["fp"] for row in rows))
    return dict(images=len(records), AP=float(np.mean([row["AP"] for row in by_iou.values()])),
                AP50=by_iou["50"]["AP"], AP75=by_iou["75"]["AP"], by_iou=by_iou)


def evaluate_predictions(manifest_or_records, predictions):
    records = load_records(manifest_or_records)
    serialized = _serialized_predictions(records, predictions)
    density = {str(row.get("beta", "clear")) for row in records}
    groups = {beta: [row for row in records if str(row.get("beta", "clear")) == beta]
              for beta in sorted(density)}
    scenes = {}
    for row in records:
        scene = str(row.get("scene_id", row.get("scene", row["image_id"])))
        scenes.setdefault(scene, []).append(str(row.get("beta", "clear")))
    balanced = density == set(DENSITIES) and all(sorted(x) == sorted(DENSITIES) for x in scenes.values())
    return dict(complete=True, metric="repository_native_VOC2007_11point", units="AP percent",
                images=len(records), scenes=len(scenes), density_counts={k: len(v) for k, v in groups.items()},
                complete_three_density_pairing=balanced, pooled=_metrics(records, serialized),
                per_density={beta: _metrics(rows, serialized) for beta, rows in groups.items()},
                conventions=dict(prediction_minima_offset=1, score_decimal_places=3,
                                 box_decimal_places=1, inclusive_IoU=True, match_rule="strict >",
                                 empty_GT_class_AP=0, class_count=8, numpy_version=np.__version__),
                target_used_for_training=False, target_used_for_selection=False,
                note="Pooled AP ranks all unique density image IDs together; per-density AP averages are different metrics.")


def _default_mapper(record):
    from PIL import Image
    import torch
    with Image.open(record["file_name"]) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if tuple(array.shape[:2]) != (int(record["height"]), int(record["width"])):
        raise ValueError("Unexpected image dimensions; evaluation does not silently resize")
    return dict(image=torch.from_numpy(array).permute(2, 0, 1),
                height=int(record["height"]), width=int(record["width"]), image_id=str(record["image_id"]))


def evaluate_model(model, manifest_or_records, output_dir, mapper=None, max_images=None):
    """Run model.predict on RGB images only, then evaluate on CPU GT annotations.

    Intended for the synchronous periodic-evaluation hook. Does not select or
    modify checkpoints. A capped smoke evaluation is marked partial.
    """
    import torch
    all_records = load_records(manifest_or_records)
    records = all_records if max_images is None else all_records[:int(max_images)]
    if not records:
        raise ValueError("Evaluation contains no images")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start, predictions = time.time(), []
    was_training = model.training
    model.eval()
    temporary = output_dir / "predictions.jsonl.tmp"
    try:
        with torch.inference_mode(), temporary.open("w", encoding="utf-8") as stream:
            for index, record in enumerate(records):
                # Even a custom mapper must never receive evaluation annotations.
                clean = {key: value for key, value in record.items()
                         if key not in ("annotations", "gt", "labels", "image_labels", "instances", "annotation_file")}
                mapped = (mapper or _default_mapper)(clean)
                if any(key in mapped for key in ("instances", "annotations", "gt", "labels", "image_labels")):
                    raise ValueError("Evaluation mapper leaked labels into prediction")
                outputs = model.predict([mapped])
                if len(outputs) != 1:
                    raise ValueError("Expected one prediction per image")
                prediction = prediction_record(record["image_id"], outputs[0])
                predictions.append(prediction)
                stream.write(json.dumps(prediction, separators=(",", ":"), allow_nan=False) + "\n")
                if (index + 1) % 25 == 0 or index + 1 == len(records):
                    stream.flush()
                    atomic_json(output_dir / "status.json", dict(status="running", complete=False,
                                images=index + 1, expected=len(records), seconds=time.time() - start))
        temporary.replace(output_dir / "predictions.jsonl")
        result = evaluate_predictions(records, predictions)
        result["complete"] = len(records) == len(all_records)
        result["expected_images"] = len(all_records)
        result["seconds"] = time.time() - start
        result["predictions_sha256"] = sha256(output_dir / "predictions.jsonl")
        if isinstance(manifest_or_records, (str, Path)):
            result["manifest_sha256"] = sha256(manifest_or_records)
        atomic_json(output_dir / "metrics.json", result)
        atomic_json(output_dir / "status.json", dict(status="complete" if result["complete"] else "partial",
                    complete=result["complete"], images=len(records), expected=len(all_records),
                    seconds=result["seconds"], metrics_sha256=sha256(output_dir / "metrics.json")))
        return result
    finally:
        model.train(was_training)


def self_test():
    def record(key, beta="clear", annotations=None, scene=None):
        return dict(image_id=key, scene_id=scene or key, beta=beta, width=100, height=100,
                    annotations=annotations if annotations is not None else
                    [dict(bbox=[1, 1, 10, 10], category_id=2, difficult=False)])
    def prediction(key, boxes, scores=None, labels=None):
        return dict(image_id=key, boxes=boxes, scores=scores or [.9] * len(boxes),
                    classes=labels or [2] * len(boxes))
    rows = [record("one")]
    perfect = [prediction("one", [[0, 0, 10, 10]])]
    metric = evaluate_predictions(rows, perfect)
    assert metric["pooled"]["AP50"] == 12.5
    assert metric["pooled"]["by_iou"]["50"]["per_class"][2]["AP"] == 100
    duplicate = evaluate_predictions(rows, [prediction("one", [[0, 0, 10, 10]] * 2, [.9, .8])])
    assert duplicate["pooled"]["by_iou"]["50"]["per_class"][2]["duplicate_fp"] == 1
    difficult = [record("one", annotations=[dict(bbox=[1, 1, 10, 10], category_id=2, difficult=True)])]
    assert evaluate_predictions(difficult, perfect)["pooled"]["by_iou"]["50"]["per_class"][2]["ignored"] == 1
    # A half-width native serialized box has exactly .5 IoU and must fail >.5.
    exact_half = evaluate_predictions(rows, [prediction("one", [[0, 0, 5, 10]])])
    assert exact_half["pooled"]["by_iou"]["50"]["tp"] == 0
    mixed = [record("scene_foggy_beta_" + beta, beta, scene="scene") for beta in DENSITIES]
    mp = [prediction(row["image_id"], [[0, 0, 10, 10]]) for row in mixed]
    result = evaluate_predictions(mixed, mp)
    assert result["complete_three_density_pairing"] and result["pooled"]["by_iou"]["50"]["tp"] == 3
    try:
        load_records([rows[0], rows[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate identity accepted")
    assert _serialized_predictions(rows, [prediction("one", [[0, 0, 10, 10]], [.90049])])[2][0][1] == .9
    print(json.dumps(dict(passed=True, tests=7, GPU_used=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--predictions", help="JSONL records with image_id, boxes, scores, classes")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.manifest, args.predictions, args.out)):
        parser.error("--manifest, --predictions and --out are required")
    with Path(args.predictions).open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream if line.strip()]
    result = evaluate_predictions(args.manifest, predictions)
    result.update(manifest_sha256=sha256(args.manifest), predictions_sha256=sha256(args.predictions))
    atomic_json(args.out, result)
    print(json.dumps(dict(images=result["images"], pooled=result["pooled"]["AP50"], metric=result["metric"])))


if __name__ == "__main__":
    main()
