#!/usr/bin/env python
import argparse
import json
import statistics as st
from collections import defaultdict

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.modeling.meta_arch.clip_rcnn import Boxes

from train_net import get_trainer_class


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Compare C3 quality scatter modes without training.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--feature", choices=["res3", "res4"], default="res3")
    parser.add_argument(
        "--modes",
        default="zero_filled_mean_max_norm,zero_filled_mean",
        help="Comma-separated C3_ADAPTER.SCATTER_MODE values.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.weights
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def load_model(cfg):
    trainer_cls = get_trainer_class(cfg)
    model = trainer_cls.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    if (
        cfg.MODEL.META_ARCHITECTURE in ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN"
    ):
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )
    model.eval()
    return model


def compute_features(model, images):
    x = images.tensor.type(model.backbone.conv1.weight.dtype)
    x = model.backbone.relu(model.backbone.bn1(model.backbone.conv1(x)))
    x = model.backbone.relu(model.backbone.bn2(model.backbone.conv2(x)))
    x = model.backbone.relu(model.backbone.bn3(model.backbone.conv3(x)))
    x = model.backbone.avgpool(x)
    x = model.backbone.layer1(x)
    res3 = model.backbone.layer2(x)
    res4 = model.backbone.layer3(res3)
    return {"res3": res3, "res4": res4}


def get_rpn_proposals(model, batched_inputs):
    if model.clip_crop_region_type != "RPN":
        raise ValueError("This diagnostic expects MODEL.CLIP.CROP_REGION_TYPE == 'RPN'.")
    offline_images = model.offline_preprocess_image(batched_inputs)
    offline_features = model.offline_backbone(offline_images.tensor)
    proposals, _ = model.offline_proposal_generator(offline_images, offline_features, None)
    return proposals


def compute_effective_classes(model, feature, boxes, feature_name):
    if sum(len(x) for x in boxes) == 0:
        return feature.new_zeros((0,))
    if feature_name == "res3":
        pooler = model.c3_quality_pooler
        projector = model.c3_quality_proj
    elif feature_name == "res4":
        pooler = model.c4_quality_pooler
        projector = model.c4_quality_proj
    else:
        raise ValueError("Unknown feature: {}".format(feature_name))

    roi = pooler([feature], boxes)
    roi_vec = roi.mean(dim=[2, 3])
    emb = projector(roi_vec)
    text_weight = model.roi_heads.box_predictor.cls_score.weight[:model.roi_heads.num_classes]
    logits = torch.nn.functional.normalize(emb, dim=1) @ torch.nn.functional.normalize(text_weight, dim=1).t()
    logits = logits / model.c3_adapter_quality_logit_temperature
    probs = torch.nn.functional.softmax(logits, dim=1)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    return entropy.exp()


def compute_quality(model, feature, proposals, feature_name):
    boxes = [p.proposal_boxes for p in proposals]
    eff = compute_effective_classes(model, feature, boxes, feature_name)
    aug_boxes = [
        Boxes(model.jitter_boxes(p.proposal_boxes.tensor, p.image_size, model.c3_adapter_perturb_scale))
        for p in proposals
    ]
    eff_aug = compute_effective_classes(model, feature, aug_boxes, feature_name)
    num_classes = model.roi_heads.num_classes
    certainty = (float(num_classes) - eff) / max(float(num_classes - 1), 1.0)
    stability = torch.exp(-(eff_aug - eff).abs() / model.c3_adapter_quality_tau)
    quality = certainty * stability
    if model.c3_adapter_clamp_quality:
        quality = quality.clamp(0.0, 1.0)
    return eff, eff_aug, quality


def gt_mask_for_feature(instances, h, w, stride):
    mask = torch.zeros((h, w), dtype=torch.bool, device=instances.gt_boxes.tensor.device)
    for box in instances.gt_boxes.tensor:
        x1 = int(torch.floor(box[0] / stride).clamp(0, w - 1).item())
        y1 = int(torch.floor(box[1] / stride).clamp(0, h - 1).item())
        x2 = int(torch.ceil(box[2] / stride).clamp(x1 + 1, w).item())
        y2 = int(torch.ceil(box[3] / stride).clamp(y1 + 1, h).item())
        mask[y1:y2, x1:x2] = True
    return mask


def scatter_region_quality_to_feature(model, quality, proposals, spatial_size, stride, mode):
    h, w = spatial_size
    quality_maps = []
    offset = 0
    for proposals_per_image in proposals:
        boxes = proposals_per_image.proposal_boxes.tensor
        qualities = quality[offset: offset + len(proposals_per_image)]
        offset += len(proposals_per_image)
        if len(proposals_per_image) == 0:
            quality_maps.append(quality.new_zeros((1, h, w)))
            continue
        if mode == "max":
            yy, xx = torch.meshgrid(
                torch.arange(h, device=quality.device),
                torch.arange(w, device=quality.device),
            )
            region_maps = []
            for box, q in zip(boxes, qualities):
                x1 = int(torch.floor(box[0] / stride).clamp(0, w - 1).item())
                y1 = int(torch.floor(box[1] / stride).clamp(0, h - 1).item())
                x2 = int(torch.ceil(box[2] / stride).clamp(x1 + 1, w).item())
                y2 = int(torch.ceil(box[3] / stride).clamp(y1 + 1, h).item())
                mask = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
                region_maps.append(torch.where(mask, q.expand(h, w), quality.new_zeros((h, w))))
            quality_maps.append(torch.stack(region_maps, dim=0).max(dim=0).values.unsqueeze(0))
        elif mode in ("zero_filled_mean", "zero_filled_mean_max_norm"):
            sum_map = quality.new_zeros((h, w))
            count_map = quality.new_zeros((h, w))
            for box, q in zip(boxes, qualities):
                x1 = int(torch.floor(box[0] / stride).clamp(0, w - 1).item())
                y1 = int(torch.floor(box[1] / stride).clamp(0, h - 1).item())
                x2 = int(torch.ceil(box[2] / stride).clamp(x1 + 1, w).item())
                y2 = int(torch.ceil(box[3] / stride).clamp(y1 + 1, h).item())
                sum_map[y1:y2, x1:x2] += q
                count_map[y1:y2, x1:x2] += 1
            quality_map = sum_map / count_map.max().clamp_min(1.0)
            if mode == "zero_filled_mean_max_norm":
                quality_map = quality_map / quality_map.max().clamp_min(1e-6)
            quality_maps.append(quality_map.unsqueeze(0))
        else:
            raise ValueError("Unknown scatter mode: {}".format(mode))
    return torch.stack(quality_maps, dim=0)


def add_stats(stats, mode, qmap, gt_mask):
    q = qmap.squeeze(0)
    bg_mask = ~gt_mask
    covered = q > 0
    gt_vals = q[gt_mask]
    bg_vals = q[bg_mask]
    bg_covered_vals = q[bg_mask & covered]
    gt_mean = gt_vals.mean().item() if gt_vals.numel() else 0.0
    bg_mean = bg_vals.mean().item() if bg_vals.numel() else 0.0
    bg_cov_mean = bg_covered_vals.mean().item() if bg_covered_vals.numel() else 0.0
    stats[mode]["gt_mean"].append(gt_mean)
    stats[mode]["bg_mean"].append(bg_mean)
    stats[mode]["bg_covered_mean"].append(bg_cov_mean)
    stats[mode]["gap"].append(gt_mean - bg_mean)
    stats[mode]["ratio"].append(gt_mean / max(bg_mean, 1e-8))
    stats[mode]["covered_frac"].append(covered.float().mean().item())
    stats[mode]["qmax"].append(q.max().item())
    stats[mode]["qmean"].append(q.mean().item())
    stats[mode]["high_0_5_frac"].append((q > 0.5).float().mean().item())
    stats[mode]["high_0_8_frac"].append((q > 0.8).float().mean().item())
    flat = q.flatten()
    total = flat.sum().clamp_min(1e-12)
    sorted_vals = torch.sort(flat, descending=True).values
    for pct in (0.01, 0.05, 0.10):
        k = max(int(flat.numel() * pct), 1)
        stats[mode]["top_{}_mass_frac".format(int(pct * 100))].append(
            (sorted_vals[:k].sum() / total).item()
        )


def summarize(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": st.mean(values), "min": min(values), "max": max(values)}


def main():
    args = build_arg_parser().parse_args()
    cfg = setup(args)
    if not cfg.MODEL.C3_ADAPTER.ENABLED:
        raise ValueError("MODEL.C3_ADAPTER.ENABLED must be true.")
    dataset_name = args.dataset or cfg.DATASETS.TEST[0]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    model = load_model(cfg)
    data_loader = build_detection_test_loader(cfg, dataset_name)
    stats = defaultdict(lambda: defaultdict(list))
    q_stats = defaultdict(list)
    old_mode = model.c3_adapter_scatter_mode

    with torch.no_grad():
        num_images = 0
        for batch in data_loader:
            proposals = get_rpn_proposals(model, batch)
            images = model.preprocess_image(batch)
            features = compute_features(model, images)
            feature = features[args.feature]
            stride = 8.0 if args.feature == "res3" else 16.0
            eff, eff_aug, quality = compute_quality(model, feature, proposals, args.feature)
            q_stats["effective_classes"].append(eff.mean().item())
            q_stats["effective_classes_delta"].append((eff_aug - eff).abs().mean().item())
            q_stats["region_q_mean"].append(quality.mean().item())

            for mode in modes:
                model.c3_adapter_scatter_mode = mode
                qmaps = scatter_region_quality_to_feature(
                    model, quality, proposals, feature.shape[-2:], stride, mode
                )
                for i, sample in enumerate(batch):
                    instances = sample["instances"].to(qmaps.device)
                    gt_mask = gt_mask_for_feature(instances, feature.shape[-2], feature.shape[-1], stride)
                    add_stats(stats, mode, qmaps[i], gt_mask)

            num_images += len(batch)
            if num_images >= args.num_images:
                break

    model.c3_adapter_scatter_mode = old_mode

    output = {
        "dataset": dataset_name,
        "feature": args.feature,
        "weights": args.weights,
        "num_images": num_images,
        "region_quality": {k: summarize(v) for k, v in q_stats.items()},
        "modes": {
            mode: {metric: summarize(vals) for metric, vals in metrics.items()}
            for mode, metrics in stats.items()
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
