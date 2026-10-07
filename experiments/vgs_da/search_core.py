"""Cached-feature semantic-conditioned supplementary proposal head.

Map construction reads only initial proposals, RPN logits and ROI probabilities.
Ground truth is confined to matching/loss/evaluation functions. Coordinates are
continuous original-image XYXY; class probabilities contain eight foreground
classes followed by background. This module has no RegionCLIP dependency.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms


ARMS = ("v", "vg", "vgs")
STRIDES = (8, 16, 32)
ANCHOR_SIZES = ((8, 16, 32), (64, 128), (256, 512))
ANCHOR_RATIOS = (0.5, 1.0, 2.0)  # height / width
MAP_CHANNELS = 15


def box_iou(a, b):
    a, b = a.float(), b.float()
    if not len(a) or not len(b):
        return a.new_zeros((len(a), len(b)))
    low = torch.maximum(a[:, None, :2], b[None, :, :2])
    high = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (high - low).clamp(min=0).prod(dim=2)
    aa = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(dim=1)
    bb = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(dim=1)
    return inter / (aa[:, None] + bb[None, :] - inter).clamp(min=1e-8)


def encode_boxes(anchors, gt):
    awh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-6)
    ac = (anchors[:, 2:] + anchors[:, :2]) * 0.5
    gwh = (gt[:, 2:] - gt[:, :2]).clamp(min=1e-6)
    gc = (gt[:, 2:] + gt[:, :2]) * 0.5
    return torch.cat(((gc - ac) / awh, torch.log(gwh / awh)), dim=1)


def decode_boxes(anchors, deltas):
    # Float32 is intentional: large boxes/log-scale transforms overflow in fp16.
    anchors, deltas = anchors.float(), deltas.float()
    awh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-6)
    ac = (anchors[:, 2:] + anchors[:, :2]) * 0.5
    pc = ac + deltas[:, :2] * awh
    pwh = awh * torch.exp(deltas[:, 2:].clamp(max=math.log(1000.0 / 16)))
    return torch.cat((pc - pwh * 0.5, pc + pwh * 0.5), dim=1)


def _rect_sum(values, boxes_grid, height, width):
    """Rectangle splat with O(N*C + H*W*C) rather than N*H*W memory."""
    values = np.asarray(values, dtype=np.float32)
    diff = np.zeros((height + 1, width + 1, values.shape[1]), np.float32)
    if len(values):
        x0, y0, x1, y1 = boxes_grid.T
        np.add.at(diff, (y0, x0), values)
        np.add.at(diff, (y1, x0), -values)
        np.add.at(diff, (y0, x1), -values)
        np.add.at(diff, (y1, x1), values)
    return diff.cumsum(axis=0).cumsum(axis=1)[:height, :width]


def build_maps(boxes, rpn_logits, probs, image_size, grid_size, k0=300):
    """Return full 15-channel G+S map; no GT arguments are accepted.

    G: normalized log count, validity, mean sigmoid RPN logit, mean relative
       width and height. S: RPN-confidence-weighted 9-class probability means
       and normalized weighted Jensen-Shannon divergence.
    """
    boxes = np.asarray(boxes, np.float32)[:k0]
    logits = np.asarray(rpn_logits, np.float32).reshape(-1)[:len(boxes)]
    probs = np.asarray(probs, np.float32)[:len(boxes)]
    if probs.shape != (len(boxes), 9) or len(logits) != len(boxes):
        raise ValueError("B0 boxes/logits/probs disagree; probs must be [N_B0,9]")
    ih, iw = (int(x) for x in image_size)
    gh, gw = (int(x) for x in grid_size)
    if min(ih, iw, gh, gw) <= 0:
        raise ValueError("Nonpositive image or grid size")
    if not np.isfinite(boxes).all() or not np.isfinite(logits).all():
        raise ValueError("Nonfinite boxes or RPN logits")
    if not np.isfinite(probs).all() or (probs < -1e-6).any():
        raise ValueError("Invalid class probabilities")
    if len(probs) and not np.allclose(probs.sum(1), 1, atol=2e-3):
        raise ValueError("Expected softmax probabilities including background")
    clipped = boxes.copy()
    clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, iw)
    clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, ih)
    valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    clipped, logits, probs = clipped[valid], logits[valid], probs[valid]
    q = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    # Pixel rectangles use floor/ceil so every nonempty box covers a grid cell.
    xy0 = np.floor(clipped[:, :2] * [gw / iw, gh / ih]).astype(np.int64)
    xy1 = np.ceil(clipped[:, 2:] * [gw / iw, gh / ih]).astype(np.int64)
    xy0 = np.minimum(np.maximum(xy0, 0), [gw - 1, gh - 1])
    xy1 = np.minimum(np.maximum(xy1, xy0 + 1), [gw, gh])
    grid_boxes = np.concatenate((xy0, xy1), axis=1)
    wh = (clipped[:, 2:] - clipped[:, :2]) / np.array([iw, ih])
    p = np.clip(probs, 0, 1)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    # count + sum(q) + sum(w,h) + sum(q*p[9]) + sum(q*entropy)
    values = np.concatenate((np.ones((len(p), 1)), q[:, None], wh,
                             q[:, None] * p, (q * ent)[:, None]), axis=1)
    summed = _rect_sum(values, grid_boxes, gh, gw)
    count = np.maximum(summed[..., 0], 0)
    covered = count > 0.5
    n = np.maximum(count, 1)
    qsum = np.maximum(summed[..., 1], 0)
    means = np.maximum(summed[..., 4:13], 0) / np.maximum(qsum[..., None], 1e-8)
    means = means / np.maximum(means.sum(axis=2, keepdims=True), 1e-8)
    mean_ent = np.maximum(summed[..., 13], 0) / np.maximum(qsum, 1e-8)
    entropy_of_mean = -(means * np.log(np.maximum(means, 1e-12))).sum(axis=2)
    disagreement = np.clip((entropy_of_mean - mean_ent) / math.log(9), 0, 1)
    geom = np.stack((np.log1p(count) / math.log1p(k0), covered.astype(np.float32),
                     qsum / n, summed[..., 2] / n, summed[..., 3] / n), axis=2)
    out = np.concatenate((geom, means, disagreement[..., None]), axis=2)
    out[~covered] = 0
    return np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32)


def mask_maps(maps, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm: " + arm)
    if arm == "vgs":
        return maps
    out = maps.clone()
    out[:, (0 if arm == "v" else 5):] = 0
    return out


def load_cache(path, k0=300, include_gt=True):
    """Load an NPZ image. Maps are built before and independently of GT reads."""
    with np.load(str(path), allow_pickle=False) as cache:
        f3 = np.array(cache["f3"], copy=True)
        f4 = np.array(cache["f4"], copy=True)
        if f3.ndim == 4 and f3.shape[0] == 1:
            f3 = f3[0]
        if f4.ndim == 4 and f4.shape[0] == 1:
            f4 = f4[0]
        if f3.ndim != 3 or f4.ndim != 3:
            raise ValueError("Expected CHW features in " + str(path))
        boxes = np.array(cache["boxes"], np.float32, copy=True).reshape(-1, 4)
        logits = np.array(cache["rpn_logits"], np.float32, copy=True).reshape(-1)
        probs = np.array(cache["probs"], np.float32, copy=True)
        image_size = tuple(int(x) for x in cache["image_size"].reshape(-1))
        maps = build_maps(boxes[:k0], logits[:k0], probs[:k0], image_size,
                          f3.shape[-2:], k0=k0)
        result = {"f3": torch.from_numpy(f3), "f4": torch.from_numpy(f4),
                  "maps": torch.from_numpy(maps),
                  "boxes": torch.from_numpy(boxes[:k0]), "image_size": image_size,
                  "cache_path": str(path)}
        if include_gt:
            result["gt"] = torch.from_numpy(np.array(cache["gt"], np.float32, copy=True).reshape(-1, 4))
            result["labels"] = torch.from_numpy(np.array(cache["labels"], np.int64, copy=True).reshape(-1))
            if len(result["gt"]) != len(result["labels"]):
                raise ValueError("GT/label count mismatch")
    return result


class CachedImages(torch.utils.data.Dataset):
    def __init__(self, entries, root, k0=300, include_gt=True):
        self.entries, self.root = entries, Path(root)
        self.k0, self.include_gt = k0, include_gt
        self.memory_cache = {}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if index in self.memory_cache:
            return self.memory_cache[index]
        return self._read_sample(index)

    def _read_sample(self, index):
        item = self.entries[index]
        path = Path(item["cache_path"])
        if not path.is_absolute():
            path = self.root / path
        sample = load_cache(path, self.k0, self.include_gt)
        sample["key"] = item.get("key", path.stem)
        sample["scene"] = item.get("scene", sample["key"])
        return sample

    def warm_cache(self, image_count, seed=0):
        """Static read-only cache; LRU is ineffective for shuffled full epochs.

        On Linux, warm this in the parent before DataLoader forks its workers.
        The tensor backing storage is then shared through copy-on-write pages.
        No sample, ordering, or numerical operation is changed by this option.
        """
        indices = np.random.RandomState(seed).permutation(len(self))[:image_count]
        for index in indices:
            self.memory_cache[int(index)] = self._read_sample(int(index))
        byte_count = sum(value.numel() * value.element_size()
                         for sample in self.memory_cache.values()
                         for value in sample.values() if torch.is_tensor(value))
        return {"images": len(self.memory_cache), "tensor_bytes": byte_count,
                "keys": [self.memory_cache[int(i)]["key"] for i in indices]}


def collate_samples(samples):
    # Fixed original Cityscapes resolution permits direct stacking.
    return {"f3": torch.stack([s["f3"] for s in samples]),
            "f4": torch.stack([s["f4"] for s in samples]),
            "maps": torch.stack([s["maps"] for s in samples]),
            # Do not send the full feature tensors a second time through IPC/
            # pinned-memory queues. Loss and decoding use metadata only.
            "samples": [{k: v for k, v in s.items() if k not in ("f3", "f4", "maps")}
                        for s in samples]}


def _conv_gn(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(8 if cout % 8 == 0 else 4, cout), nn.ReLU(inplace=True))


class SearchHead(nn.Module):
    def __init__(self, f3_channels=512, f4_channels=1024, channels=64, arm="vgs"):
        super().__init__()
        self.arm = arm
        self.model_config = {"f3_channels": f3_channels, "f4_channels": f4_channels,
                             "channels": channels, "arm": arm}
        self.lateral3 = _conv_gn(f3_channels, channels, 1)
        self.lateral4 = _conv_gn(f4_channels, channels, 1)
        self.map_projection = _conv_gn(MAP_CHANNELS, 16, 1)
        self.fusion = _conv_gn(channels + 16, channels, 1)
        self.tower = nn.Sequential(_conv_gn(channels, channels), _conv_gn(channels, channels))
        self.obj = nn.ModuleList()
        self.miss = nn.ModuleList()
        self.delta = nn.ModuleList()
        for sizes in ANCHOR_SIZES:
            n = len(sizes) * len(ANCHOR_RATIOS)
            self.obj.append(nn.Conv2d(channels, n, 1))
            self.miss.append(nn.Conv2d(channels, n, 1))
            self.delta.append(nn.Conv2d(channels, n * 4, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for layer in self.obj:
            nn.init.constant_(layer.bias, -math.log(99))
        self._anchor_cache = {}

    def forward(self, f3, f4, maps):
        # GN supplies trainable channel normalization; no target-derived stats.
        p4 = self.lateral4(f4)
        p3 = self.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest")
        p5 = F.avg_pool2d(p4, kernel_size=2, stride=2, ceil_mode=True)
        maps = mask_maps(maps, self.arm)
        outputs = []
        for level, feature in enumerate((p3, p4, p5)):
            m = F.interpolate(maps, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            h = self.tower(self.fusion(torch.cat((feature, self.map_projection(m)), dim=1)))
            obj = self.obj[level](h)
            miss = self.miss[level](h)
            delta = self.delta[level](h)
            batch, count, height, width = obj.shape
            outputs.append({"obj": obj.permute(0, 2, 3, 1).reshape(batch, -1),
                            "miss": miss.permute(0, 2, 3, 1).reshape(batch, -1),
                            "delta": delta.reshape(batch, count, 4, height, width)
                                      .permute(0, 3, 4, 1, 2).reshape(batch, -1, 4),
                            "shape": (height, width)})
        return outputs

    def anchors(self, outputs):
        device = outputs[0]["obj"].device
        shapes = tuple(o["shape"] for o in outputs)
        key = (str(device), shapes)
        if key not in self._anchor_cache:
            result = []
            for shape, stride, sizes in zip(shapes, STRIDES, ANCHOR_SIZES):
                height, width = shape
                yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                        torch.arange(width, device=device, dtype=torch.float32))
                centers = torch.stack(((xx + 0.5) * stride, (yy + 0.5) * stride), dim=-1).reshape(-1, 2)
                wh = torch.tensor([[size / math.sqrt(ratio), size * math.sqrt(ratio)]
                                   for size in sizes for ratio in ANCHOR_RATIOS],
                                  device=device, dtype=torch.float32)
                low = centers[:, None] - wh[None] * 0.5
                high = centers[:, None] + wh[None] * 0.5
                result.append(torch.cat((low, high), dim=2).reshape(-1, 4))
            self._anchor_cache[key] = torch.cat(result)
        return self._anchor_cache[key]


def flatten_outputs(outputs):
    return {key: torch.cat([o[key] for o in outputs], dim=1) for key in ("obj", "miss", "delta")}


@torch.no_grad()
def match_anchors(anchors, gt, base_boxes, positive_iou=0.5, negative_iou=0.3):
    n = len(anchors)
    labels = torch.full((n,), -1, dtype=torch.int64, device=anchors.device)
    matched = torch.zeros(n, dtype=torch.int64, device=anchors.device)
    if not len(gt):
        labels.fill_(0)
        return labels, matched, anchors.new_zeros(0), {"gt": 0, "missed_gt": 0,
            "gt_with_positive": 0, "forced_collisions": 0, "forced_anchors": []}
    overlaps = box_iou(anchors, gt)
    best_iou, matched = overlaps.max(dim=1)
    labels[best_iou < negative_iou] = 0
    labels[best_iou >= positive_iou] = 1
    # A unique forced match per GT prevents two tiny adjacent GT boxes sharing
    # their sole best anchor. Conflicts use the next-best available anchor.
    values, indices = overlaps.topk(min(len(gt), n), dim=0)
    order = values[0].argsort().cpu().tolist()
    index_lists = indices.t().cpu().tolist()
    reserved, forced, collisions = set(), [], 0
    for g in order:
        choices = index_lists[g]
        collisions += int(choices[0] in reserved)
        chosen = next((a for a in choices if a not in reserved), None)
        if chosen is None:
            continue
        reserved.add(chosen)
        forced.append(chosen)
        labels[chosen], matched[chosen] = 1, g
    base_quality = box_iou(base_boxes, gt).max(dim=0)[0] if len(base_boxes) else gt.new_zeros(len(gt))
    uncovered = (base_quality < positive_iou).float()
    covered_gts = matched[labels == 1].unique().numel()
    stats = {"gt": len(gt), "missed_gt": int(uncovered.sum().item()),
             "gt_with_positive": covered_gts, "forced_collisions": collisions,
             "forced_anchors": forced}
    return labels, matched, uncovered, stats


def _sample_anchors(labels, forced, sample_count=256, positive_fraction=0.5):
    positives = torch.where(labels == 1)[0]
    negatives = torch.where(labels == 0)[0]
    limit = min(int(sample_count * positive_fraction), len(positives))
    forced = torch.tensor(forced, dtype=torch.long, device=labels.device)
    if len(forced) > limit:
        forced = forced[torch.randperm(len(forced), device=labels.device)[:limit]]
    if len(forced):
        occupied = torch.zeros_like(labels, dtype=torch.bool)
        occupied[forced] = True
        remaining = positives[~occupied[positives]]
    else:
        remaining = positives
    remaining = remaining[torch.randperm(len(remaining), device=labels.device)[:limit - len(forced)]]
    positives = torch.cat((forced, remaining))
    negatives = negatives[torch.randperm(len(negatives), device=labels.device)[:sample_count - len(positives)]]
    return positives, negatives


def training_loss(outputs, anchors, samples, sample_count=256, miss_weight=1.0, box_weight=1.0):
    pred = flatten_outputs(outputs)
    losses = {k: pred["obj"].float().sum() * 0 for k in ("obj", "miss", "box")}
    stats = {"gt": 0, "missed_gt": 0, "gt_with_positive": 0, "forced_collisions": 0,
             "sampled_pos": 0, "sampled_neg": 0, "sampled_miss_pos": 0}
    for batch, sample in enumerate(samples):
        gt = sample["gt"].to(anchors.device).float()
        base = sample["boxes"].to(anchors.device).float()
        labels, matched, uncovered, match_stats = match_anchors(anchors, gt, base)
        pos, neg = _sample_anchors(labels, match_stats["forced_anchors"], sample_count)
        selected = torch.cat((pos, neg))
        if len(selected):
            losses["obj"] = losses["obj"] + F.binary_cross_entropy_with_logits(
                pred["obj"][batch, selected].float(), labels[selected].float())
        if len(pos):
            target_miss = uncovered[matched[pos]]
            losses["miss"] = losses["miss"] + F.binary_cross_entropy_with_logits(
                pred["miss"][batch, pos].float(), target_miss)
            target_delta = encode_boxes(anchors[pos], gt[matched[pos]])
            error = (pred["delta"][batch, pos].float() - target_delta).abs()
            beta = 1.0 / 9.0
            smooth = torch.where(error < beta, 0.5 * error.square() / beta, error - 0.5 * beta)
            losses["box"] = losses["box"] + (smooth.sum(1) * (1 + 2 * target_miss)).sum() / max(len(selected), 1)
            stats["sampled_miss_pos"] += int(target_miss.sum().item())
        for name in ("gt", "missed_gt", "gt_with_positive", "forced_collisions"):
            stats[name] += match_stats[name]
        stats["sampled_pos"] += len(pos)
        stats["sampled_neg"] += len(neg)
    losses = {key: value / len(samples) for key, value in losses.items()}
    losses["total"] = losses["obj"] + miss_weight * losses["miss"] + box_weight * losses["box"]
    return losses, stats


@torch.no_grad()
def select_candidates(outputs, anchors, sample, batch_index=0, kplus=200,
                      pre_nms=2000, nms_threshold=0.7, duplicate_iou=0.95,
                      score_mode="residual"):
    pred = flatten_outputs(outputs)
    score = pred["obj"][batch_index].float().sigmoid()
    if score_mode == "residual":
        score = score * pred["miss"][batch_index].float().sigmoid()
    elif score_mode != "objectness":
        raise ValueError("Unknown score mode")
    scores, indices = score.topk(min(pre_nms, len(score)), sorted=True)
    boxes = decode_boxes(anchors[indices], pred["delta"][batch_index, indices])
    ih, iw = sample["image_size"]
    boxes[:, 0::2].clamp_(min=0, max=iw)
    boxes[:, 1::2].clamp_(min=0, max=ih)
    wh = boxes[:, 2:] - boxes[:, :2]
    valid = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores) & (wh.min(dim=1)[0] >= 1)
    boxes, scores = boxes[valid], scores[valid]
    before_duplicates = len(boxes)
    base = sample["boxes"].to(boxes.device).float()
    if len(boxes) and len(base):
        keep = box_iou(boxes, base).max(dim=1)[0] <= duplicate_iou
        boxes, scores = boxes[keep], scores[keep]
    duplicate_removed = before_duplicates - len(boxes)
    # Filter duplicate base boxes before NMS/top-K so lower-ranked surviving
    # candidates refill the budget within the fixed pre-NMS pool.
    keep = nms(boxes, scores, nms_threshold)[:kplus]
    info = {"selected": len(keep), "pre_nms_valid": before_duplicates,
            "base_duplicates_removed": duplicate_removed, "underfilled": len(keep) < kplus}
    return boxes[keep], scores[keep], info


@torch.no_grad()
def recall_counts(boxes, gt, thresholds=(0.5, 0.75)):
    if not len(gt):
        return [0 for _ in thresholds]
    quality = box_iou(boxes, gt).max(dim=0)[0] if len(boxes) else gt.new_zeros(len(gt))
    return [int((quality >= threshold).sum().item()) for threshold in thresholds]


def load_model(path, device="cpu"):
    checkpoint = torch.load(str(path), map_location=device)
    model = SearchHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
