#!/usr/bin/env python
import argparse
import json
import statistics as st
from collections import defaultdict

import torch
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, pairwise_iou

from train_net import get_trainer_class


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Diagnose C5 ROI-level quality without training.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--max-proposals", type=int, default=300)
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


def get_rpn_proposals(model, batched_inputs, max_proposals):
    if model.clip_crop_region_type != "RPN":
        raise ValueError("This diagnostic expects MODEL.CLIP.CROP_REGION_TYPE == 'RPN'.")
    offline_images = model.offline_preprocess_image(batched_inputs)
    offline_features = model.offline_backbone(offline_images.tensor)
    proposals, _ = model.offline_proposal_generator(offline_images, offline_features, None)
    if max_proposals > 0:
        for p in proposals:
            keep = min(len(p), max_proposals)
            p._fields["proposal_boxes"] = Boxes(p.proposal_boxes.tensor[:keep])
            p._fields["objectness_logits"] = p.objectness_logits[:keep]
    return proposals


def c5_region_embeddings(model, features, boxes):
    if sum(len(x) for x in boxes) == 0:
        device = features[model.roi_heads.in_features[0]].device
        return torch.empty((0, model.roi_heads.box_predictor.cls_score.weight.shape[1]), device=device)
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features],
        boxes,
        model.backbone.layer4,
    )
    if model.use_clip_attpool:
        return model.backbone.attnpool(box_features)
    return box_features.mean(dim=[2, 3])


def effective_classes_from_embedding(model, emb):
    if emb.numel() == 0:
        return emb.new_zeros((0,)), emb.new_zeros((0, model.roi_heads.num_classes))
    text_weight = model.roi_heads.box_predictor.cls_score.weight[: model.roi_heads.num_classes]
    logits = F.normalize(emb, dim=1) @ F.normalize(text_weight, dim=1).t()
    logits = logits / model.c3_adapter_quality_logit_temperature
    probs = F.softmax(logits, dim=1)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    return entropy.exp(), probs


def compute_c5_quality(model, features, proposals):
    boxes = [p.proposal_boxes for p in proposals]
    emb = c5_region_embeddings(model, features, boxes)
    eff, probs = effective_classes_from_embedding(model, emb)
    aug_boxes = [
        Boxes(model.jitter_boxes(p.proposal_boxes.tensor, p.image_size, model.c3_adapter_perturb_scale))
        for p in proposals
    ]
    emb_aug = c5_region_embeddings(model, features, aug_boxes)
    eff_aug, _ = effective_classes_from_embedding(model, emb_aug)
    num_classes = model.roi_heads.num_classes
    certainty = (float(num_classes) - eff) / max(float(num_classes - 1), 1.0)
    stability = torch.exp(-(eff_aug - eff).abs() / model.c3_adapter_quality_tau)
    quality = certainty * stability
    if model.c3_adapter_clamp_quality:
        quality = quality.clamp(0.0, 1.0)
    return eff, eff_aug, quality, probs


def proposal_ious(proposals, batch):
    values = []
    for p, sample in zip(proposals, batch):
        gt = sample["instances"].to(p.proposal_boxes.tensor.device).gt_boxes
        if len(p) == 0:
            continue
        if len(gt) == 0:
            values.append(p.proposal_boxes.tensor.new_zeros((len(p),)))
            continue
        values.append(pairwise_iou(p.proposal_boxes, gt).max(dim=1).values)
    if not values:
        return torch.empty((0,))
    return torch.cat(values, dim=0)


def summarize(values):
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": st.mean(values), "min": min(values), "max": max(values)}


def add_group(stats, name, mask, eff, eff_aug, q, probs, ious):
    if mask.sum().item() == 0:
        return
    m = mask.bool()
    stats[name]["count"].append(float(m.sum().item()))
    stats[name]["iou"].extend(ious[m].detach().cpu().tolist())
    stats[name]["effective_classes"].extend(eff[m].detach().cpu().tolist())
    stats[name]["effective_delta"].extend((eff_aug[m] - eff[m]).abs().detach().cpu().tolist())
    stats[name]["q"].extend(q[m].detach().cpu().tolist())
    stats[name]["top1_prob"].extend(probs[m].max(dim=1).values.detach().cpu().tolist())


def main():
    args = build_arg_parser().parse_args()
    cfg = setup(args)
    if not cfg.MODEL.C3_ADAPTER.ENABLED:
        raise ValueError("MODEL.C3_ADAPTER.ENABLED must be true for q hyperparameters.")
    dataset_name = args.dataset or cfg.DATASETS.TEST[0]
    model = load_model(cfg)
    loader = build_detection_test_loader(cfg, dataset_name)
    stats = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        num_images = 0
        for batch in loader:
            proposals = get_rpn_proposals(model, batch, args.max_proposals)
            images = model.preprocess_image(batch)
            features = model.recognition_features(images, proposals, is_source=False)
            eff, eff_aug, q, probs = compute_c5_quality(model, features, proposals)
            ious = proposal_ious(proposals, batch).to(q.device)
            add_group(stats, "all", torch.ones_like(q, dtype=torch.bool), eff, eff_aug, q, probs, ious)
            add_group(stats, "high_iou_ge_0_5", ious >= 0.5, eff, eff_aug, q, probs, ious)
            add_group(stats, "mid_iou_0_1_0_5", (ious >= 0.1) & (ious < 0.5), eff, eff_aug, q, probs, ious)
            add_group(stats, "low_iou_lt_0_1", ious < 0.1, eff, eff_aug, q, probs, ious)
            num_images += len(batch)
            if num_images >= args.num_images:
                break

    output = {
        "dataset": dataset_name,
        "weights": args.weights,
        "num_images": num_images,
        "max_proposals_per_image": args.max_proposals,
        "groups": {
            name: {metric: summarize(vals) for metric, vals in metrics.items()}
            for name, metrics in stats.items()
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
