"""Native RPN computation with detached selection and live selected logits.

The historical RPN's predict_proposals is no-grad. DeST additionally needs the
raw anchor logits for spatial guidance and H2FA. Index selection remains
discrete; coordinates remain detached, exactly as in ordinary Faster R-CNN.
"""
import torch


def flatten_rpn_outputs(scores, deltas, box_dim=4):
    flat_scores = [s.permute(0, 2, 3, 1).flatten(1) for s in scores]
    flat_deltas = [d.view(d.shape[0], -1, box_dim, d.shape[-2], d.shape[-1])
                   .permute(0, 3, 4, 1, 2).flatten(1, -2) for d in deltas]
    return flat_scores, flat_deltas


@torch.no_grad()
def _select_indices(decoded, scores, image_sizes, nms_thresh, pre_topk,
                    post_topk, min_size):
    from detectron2.layers import batched_nms
    from detectron2.structures import Boxes
    # Retain global anchor indices through exactly the native top-k/filter/NMS
    # sequence. Gather differentiable logits only after leaving this function.
    boxes_parts, score_parts, index_parts, level_parts = [], [], [], []
    offset = 0
    for level, (boxes, logits) in enumerate(zip(decoded, scores)):
        count = min(int(pre_topk), logits.shape[1])
        values, indices = logits.detach().sort(dim=1, descending=True)
        indices, values = indices[:, :count], values[:, :count]
        batch = torch.arange(boxes.shape[0], device=boxes.device)[:, None]
        boxes_parts.append(boxes[batch, indices])
        score_parts.append(values)
        index_parts.append(indices + offset)
        level_parts.append(torch.full((count,), level, device=boxes.device, dtype=torch.long))
        offset += logits.shape[1]
    all_boxes = torch.cat(boxes_parts, dim=1)
    all_scores = torch.cat(score_parts, dim=1)
    all_indices = torch.cat(index_parts, dim=1)
    all_levels = torch.cat(level_parts)
    selected = []
    for image_index, image_size in enumerate(image_sizes):
        boxes, scores_i = Boxes(all_boxes[image_index]), all_scores[image_index]
        indices, levels = all_indices[image_index], all_levels
        if not (torch.isfinite(boxes.tensor).all() and torch.isfinite(scores_i).all()):
            raise FloatingPointError("Nonfinite DeST RPN outputs")
        boxes.clip(image_size)
        keep = boxes.nonempty(threshold=min_size)
        boxes, scores_i, indices, levels = boxes[keep], scores_i[keep], indices[keep], levels[keep]
        keep = batched_nms(boxes.tensor, scores_i, levels, nms_thresh)[:post_topk]
        selected.append((boxes[keep], indices[keep]))
    return selected


def run_rpn(rpn, images, features, source_targets=None, proposal_budget=300,
            pre_nms_topk=None):
    """Return proposals, source-only losses, raw logits, and anchor centers.

    source_targets describes the first source images in the combined batch.
    No target anchor labels, negatives, or pseudo-box regression are created.
    """
    from detectron2.structures import Instances
    if type(proposal_budget) is not int or proposal_budget <= 0:
        raise ValueError("proposal_budget must be a positive integer")
    pre_nms_topk = rpn.pre_nms_topk[False] if pre_nms_topk is None else pre_nms_topk
    if type(pre_nms_topk) is not int or pre_nms_topk <= 0:
        raise ValueError("pre_nms_topk must be a positive integer")
    feature_list = [features[name].float() for name in rpn.in_features]
    with torch.cuda.amp.autocast(enabled=False):
        anchors = rpn.anchor_generator(feature_list)
        scores, deltas = rpn.rpn_head(feature_list)
        scores, deltas = flatten_rpn_outputs(scores, deltas, rpn.anchor_generator.box_dim)
        raw_logits = torch.cat(scores, dim=1)
        all_anchors = torch.cat([a.tensor for a in anchors])
        centers = (all_anchors[:, :2] + all_anchors[:, 2:]) * .5
        losses = {}
        if source_targets is not None:
            count = len(source_targets)
            if not 0 < count <= raw_logits.shape[0]:
                raise ValueError("Invalid source target count")
            labels, boxes = rpn.label_and_sample_anchors(anchors, source_targets)
            losses = rpn.losses(anchors, [s[:count] for s in scores], labels,
                                [d[:count] for d in deltas], boxes)
        with torch.no_grad():
            decoded = rpn._decode_proposals(anchors, deltas)
            selected = _select_indices(decoded, scores, images.image_sizes,
                rpn.nms_thresh, pre_nms_topk,
                proposal_budget, rpn.min_box_size)
        proposals = []
        for index, (boxes, anchor_indices) in enumerate(selected):
            proposal = Instances(images.image_sizes[index])
            proposal.proposal_boxes = boxes
            proposal.objectness_logits = raw_logits[index, anchor_indices].detach()
            proposal.aggregation_objectness_logits = raw_logits[index, anchor_indices]
            proposal.anchor_indices = anchor_indices
            proposals.append(proposal)
    return proposals, losses, raw_logits, centers
