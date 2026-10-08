"""Online RegionCLIP + VGS supplement learning and two domain adversaries.

The fixed SourceB teacher supplies B0 semantics only. Source detection gradients
reach the student's actual C3/C4/res5/attention-pool and native box regressor.
Proposal coordinates, maps and conditional labels are stop-gradient; the search
head receives its own supervised anchor losses and conditional domain loss.
The optional EMA experiment adds AT-style paired views and H2FA image-level
soft consistency, never target pseudo-box classification/regression losses.
"""
import contextlib
import copy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import torch
from checkpoint_identity import same_sources
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
try:
    from search_core import SearchHead, build_maps, training_loss, select_candidates, flatten_outputs
    from domain_core import ConditionalDomainAdapter, DomainFeatureTap, gradient_reverse
except ImportError:
    # Local read-only development fallback; deployed code includes both files.
    for folder in (HERE.parent / "learned_search", HERE.parent / "learned_search_da"):
        sys.path.insert(0, str(folder))
    from search_core import SearchHead, build_maps, training_loss, select_candidates, flatten_outputs
    from domain_core import ConditionalDomainAdapter, DomainFeatureTap, gradient_reverse

VERSION = "formal-regionclip-vgs-da-v1"
EMA_VERSION = "formal-regionclip-vgs-da-ema-image-v2"
IMAGE_SIZE = (1024, 2048)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ramp(progress):
    progress = float(progress)
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("Training progress must be in [0,1]")
    return 2. / (1. + math.exp(-10. * progress)) - 1.


def target_image_labels_allowed(config):
    """Default to the original protocol; reject truthy strings/integers."""
    allowed = config.get("target_image_labels_allowed", True)
    if type(allowed) is not bool:
        raise ValueError("target_image_labels_allowed must be a boolean")
    return allowed


def _native_logits(raw, classifier):
    """Exact native fixed-text path; omits unused DA/EMA prompt branches."""
    normalized = F.normalize(raw, p=2., dim=1)
    foreground = normalized @ F.normalize(classifier.cls_score.weight, p=2., dim=1).t()
    if classifier.use_bias:
        foreground = foreground + classifier.cls_score.bias
    background = classifier.cls_bg_score(normalized)
    if classifier.use_bias:
        # Match the existing native implementation, including its bias branch.
        background = background + classifier.cls_bg_score.bias
    return torch.cat((foreground, background), dim=1) / classifier.temperature


class FixedSemanticClassifier(nn.Module):
    """Copy only the native text/background layers, not unused full CLIP models."""
    def __init__(self, native):
        super().__init__()
        self.cls_score = copy.deepcopy(native.cls_score)
        self.cls_bg_score = copy.deepcopy(native.cls_bg_score)
        self.use_bias = bool(native.use_bias)
        self.temperature = float(native.temperature)
        self.requires_grad_(False)

    def forward(self, raw):
        return _native_logits(raw, self)


class TraditionalDomainDiscriminators(nn.Module):
    """Spatial source/target BCE on C3 and C4, balanced over domains/layers."""
    def __init__(self, hidden=128):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False), nn.Conv2d(hidden, 1, 1)) for name, channels in (("res3", 512), ("res4", 1024))})
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=.01)
                nn.init.zeros_(layer.bias)

    def forward(self, source, target, coefficient):
        losses, stats = [], {}
        with torch.cuda.amp.autocast(enabled=False):
            for name, discriminator in self.heads.items():
                s = discriminator(gradient_reverse(source[name].float(), coefficient))
                t = discriminator(gradient_reverse(target[name].float(), coefficient))
                loss = .5 * (F.binary_cross_entropy_with_logits(s, torch.zeros_like(s))
                             + F.binary_cross_entropy_with_logits(t, torch.ones_like(t)))
                losses.append(loss)
                stats[name] = {"bce": float(loss.detach()),
                    "accuracy": float(.5 * ((s.detach() < 0).float().mean() + (t.detach() >= 0).float().mean()))}
        return torch.stack(losses).mean(), stats


class FormalVGSDA(nn.Module):
    """Construct from the root protocol's config dictionary.

    Required paths: source_cfg, source_checkpoint, rpn_checkpoint,
    text_embeddings, source_search_checkpoint (None explicitly allows random).
    Input dictionaries contain RGB CHW image tensors at1024x2048. Source also
    contains Instances GT. Target contains eight binary image_labels only when
    permitted by config; UDA target records must omit that key entirely.
    """
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.target_image_labels_allowed = target_image_labels_allowed(config)
        self.image_consistency_enabled = config.get("image_consistency_enabled", False)
        if type(self.image_consistency_enabled) is not bool:
            raise ValueError("image_consistency_enabled must be a boolean")
        self.version = EMA_VERSION if self.image_consistency_enabled else VERSION
        self.ema_updates = 0
        self.ema_decay = float(config.get("ema_decay", .9996))
        self.image_consistency_weight = float(config.get("image_consistency_weight", 1.))
        if self.image_consistency_enabled:
            if self.target_image_labels_allowed or config.get("strong_weak_enabled") is not True:
                raise ValueError("EMA image consistency requires unlabeled target and strong/weak views")
            if (self.ema_decay, self.image_consistency_weight) != (.9996, 1.):
                raise ValueError("Requested EMA/image loss coefficients are 0.9996 and 1.0")
            if (config.get("image_aggregation", "h2fa_iir"),
                config.get("semantic_teacher_mode", "ema"),
                config.get("source_view", "strong"),
                config.get("evaluation_model", "student")) != ("h2fa_iir", "ema", "strong", "student"):
                raise ValueError("Unsupported EMA image-consistency protocol")
        elif config.get("strong_weak_enabled", False):
            raise ValueError("Paired augmentation requires the explicit EMA consistency experiment")
        if config.get("repo_root") and str(config["repo_root"]) not in sys.path:
            sys.path.insert(0, str(config["repo_root"]))
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        cfg = get_cfg(); cfg.merge_from_file(str(config["source_cfg"]))
        cfg.MODEL.WEIGHTS = str(config["source_checkpoint"])
        cfg.MODEL.CLIP.BB_RPN_WEIGHTS = str(config["rpn_checkpoint"])
        cfg.MODEL.CLIP.TEXT_EMB_PATH = str(config["text_embeddings"])
        cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG = str(Path(config.get("repo_root", "/root/regionclip/DA-Pro")) / "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4"]
        cfg.MODEL.DEVICE = config.get("device", "cuda")
        cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = False
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(config.get("source_roi_batch", 256))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(config.get("score_threshold", .001))
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(config.get("nms_threshold", .5))
        cfg.TEST.DETECTIONS_PER_IMAGE = int(config.get("detections_per_image", 100))
        cfg.MODEL.CLIP.MULTIPLY_RPN_SCORE = False
        cfg.DATALOADER.NUM_WORKERS = 0
        if cfg.MODEL.DA_PRO.ENABLED or cfg.LEARNABLE_PROMPT.TUNING or cfg.MODEL.MASK_ON:
            raise ValueError("Expected SourceB with original prompt/DA/mask extensions disabled")
        cfg.freeze(); self.cfg = cfg
        self.student = build_model(cfg)
        initial = torch.load(str(config["source_checkpoint"]), map_location="cpu")
        self.student.load_state_dict(initial["model"], strict=True)
        iteration = initial.get("iteration")
        del initial
        if self.student.c3_adapter_enabled or self.student.c4_semantic_reshaper_enabled or self.student.c5_adapter_apply_residual or self.student.active_region_reshaper_fn() is not None:
            raise ValueError("SourceB must not enable older adaptation/reshaping modules")
        predictor = self.student.roi_heads.box_predictor
        if not predictor.use_clip_cls_emb or predictor.is_prompt_tuning or predictor.num_classes != 8 or predictor.test_cls_score is not None:
            raise ValueError("Expected closed-set fixed eight-text native RegionCLIP predictor")
        if predictor.no_box_delta or float(predictor.temperature) != .01:
            raise ValueError("Expected native box regression and fixed temperature0.01")
        predictor.multiply_rpn_score = False
        self.student.roi_heads.proposal_append_gt = False
        self.teacher_backbone = copy.deepcopy(self.student.backbone).requires_grad_(False)
        self.teacher_classifier = FixedSemanticClassifier(predictor)
        # Preserve native freeze_at for recognition backbone; everything else
        # except its actual bbox regressor remains frozen, including DAPromptHead.
        recognition_flags = {name: p.requires_grad for name, p in self.student.backbone.named_parameters()}
        self.student.requires_grad_(False)
        for name, parameter in self.student.backbone.named_parameters():
            parameter.requires_grad_(recognition_flags[name])
        predictor.bbox_pred.requires_grad_(True)
        if not any(p.requires_grad for p in self.student.backbone.layer4.parameters()) or not any(p.requires_grad for p in self.student.backbone.attnpool.parameters()):
            raise ValueError("This experiment requires trainable res5 and attention pool")
        search_path = config.get("source_search_checkpoint")
        if search_path:
            search_state = torch.load(str(search_path), map_location="cpu")
            if search_state["model_config"].get("arm") != "vgs":
                raise ValueError("Source search initialization must be the VGS arm")
            self.search = SearchHead(**search_state["model_config"])
            self.search.load_state_dict(search_state["model_state"], strict=True)
            del search_state
        else:
            self.search = SearchHead(arm="vgs")
        if self.image_consistency_enabled:
            self.teacher_search = copy.deepcopy(self.search).requires_grad_(False)
            self.teacher_bbox = copy.deepcopy(predictor.bbox_pred).requires_grad_(False)
        self.global_D = TraditionalDomainDiscriminators()
        self.conditional_D = ConditionalDomainAdapter(channels=self.search.model_config["channels"])
        self.teacher_roi_chunk = int(config.get("teacher_roi_chunk", 64))
        self.inference_roi_chunk = int(config.get("inference_roi_chunk", 64))
        self.train_roi_chunk = int(config.get("train_roi_chunk", 64))
        self.checkpoint_res5 = bool(config.get("checkpoint_res5", True))
        if min(self.teacher_roi_chunk, self.inference_roi_chunk, self.train_roi_chunk) < 1:
            raise ValueError("ROI chunk sizes must be positive")
        self.source_loss_weight = float(config.get("source_loss_weight", 1.))
        self.search_loss_weight = float(config.get("search_loss_weight", 1.))
        self.traditional_domain_weight = float(config.get("traditional_domain_weight", .1))
        self.conditional_domain_weight = float(config.get("conditional_domain_weight", 1.))
        self.conditional_grl_max = float(config.get("conditional_grl_max", .05))
        if (self.source_loss_weight, self.search_loss_weight, self.traditional_domain_weight,
                self.conditional_domain_weight, self.conditional_grl_max) != (1., 1., .1, 1., .05):
            raise ValueError("Loss/GRL weights must match the locked formal design")
        self.source_identity = {name: {"path": str(Path(config[name]).resolve()), "sha256": file_sha(config[name])}
            for name in ("source_cfg", "source_checkpoint", "rpn_checkpoint", "text_embeddings")}
        self.source_identity["source_search_checkpoint"] = ({"path": str(Path(search_path).resolve()), "sha256": file_sha(search_path)} if search_path else None)
        self.initialization_audit = {"version": self.version, "source_iteration": iteration, "sources": self.source_identity,
            "trainable_detector": [name for name, p in self.student.named_parameters() if p.requires_grad],
            "source_roi_batch": self.student.roi_heads.batch_size_per_image,
            "source_roi_positive_fraction": self.student.roi_heads.positive_fraction,
            "native_classification_focal_gamma": predictor.focal_scaled_loss,
            "teacher": "fixed SourceB recognition backbone plus native text/background classifier; shared parameter-free ROIAlign",
            "proposal_append_GT": False, "final_scores": "native ROI probabilities only; no objectness fusion",
            "domain_gradient_scope": "traditional C3/C4; conditional search lateral3/4 and online recognition backbone",
            "target_image_labels_allowed": self.target_image_labels_allowed,
            "target_image_labels_used": self.target_image_labels_allowed,
            "target_presence_filter_enabled": self.target_image_labels_allowed,
            "source_presence_origin": "source_GT_class_presence",
            "maps": "original 15 channels, detached teacher B0 semantics and RPN geometry; no new entropy reranking"}
        if self.image_consistency_enabled:
            self.initialization_audit.update(
                teacher="EMA recognition backbone, VGS and bbox; fixed native classifier and shared frozen RPN",
                ema_decay=self.ema_decay, image_consistency_weight=self.image_consistency_weight,
                image_aggregation="H2FA IIR foreground-softmax times class-routed raw-objectness proposal-softmax",
                consistency_target="detached weak-view teacher soft image probabilities; no target GT",
                semantic_teacher_mode="EMA weak-view teacher also supplies S maps and conditional-DA classes",
                source_view="strong only; existing source GT supervision weight unchanged",
                evaluation_model="student", burn_in_steps=0,
                consistency_gradient_scope="student ROI/backbone and selected VGS raw objectness; no box-coordinate loss")
        self.to(torch.device(config.get("device", "cuda")))
        self.train(True)
        detector_ids = {id(p) for p in self.detector_parameters()}
        aux_ids = {id(p) for p in self.aux_parameters()}
        if detector_ids & aux_ids or detector_ids | aux_ids != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError("Optimizer groups omit or overlap trainable parameters")

    @property
    def device(self):
        return next(self.student.backbone.parameters()).device

    def detector_parameters(self):
        return [p for p in self.student.parameters() if p.requires_grad]

    def aux_parameters(self):
        return [p for module in (self.search, self.global_D, self.conditional_D) for p in module.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "teacher_backbone"):
            self.teacher_backbone.eval(); self.teacher_classifier.eval()
            if getattr(self, "image_consistency_enabled", False):
                self.teacher_search.eval(); self.teacher_bbox.eval()
            self.student.offline_backbone.eval(); self.student.offline_proposal_generator.eval()
            # Preserve source normalization statistics even if a source module
            # contains ordinary BatchNorm instead of FrozenBatchNorm.
            for module in self.student.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    @torch.no_grad()
    def update_teacher(self):
        """Called once by the trainer, only after both student optimizers commit."""
        if not self.image_consistency_enabled:
            return
        from ema import update_ema
        for teacher, student in ((self.teacher_backbone, self.student.backbone),
                                 (self.teacher_search, self.search),
                                 (self.teacher_bbox, self.student.roi_heads.box_predictor.bbox_pred)):
            update_ema(teacher, student, self.ema_decay)
        self.ema_updates += 1

    @staticmethod
    def _weak_inputs(inputs):
        result = []
        for record in inputs:
            weak = record.get("image_weak")
            if weak is None or weak.shape != record["image"].shape or weak.dtype != record["image"].dtype:
                raise ValueError("EMA training requires paired weak/strong images with shared geometry")
            result.append(dict(record, image=weak))
        return result

    @staticmethod
    def _validate_inputs(inputs, source=False, target_image_labels_allowed=True):
        if type(target_image_labels_allowed) is not bool:
            raise ValueError("target_image_labels_allowed must be a boolean")
        if not inputs:
            raise ValueError("Each domain batch must contain at least one image")
        forbidden = {"instances", "annotations", "gt", "gt_boxes", "gt_classes", "labels", "targets"}
        for record in inputs:
            if record["image"].ndim != 3 or tuple(record["image"].shape) != (3,) + IMAGE_SIZE:
                raise ValueError("Initial formal protocol requires RGB full1024x2048 without padding/resize")
            if int(record.get("height", IMAGE_SIZE[0])) != IMAGE_SIZE[0] or int(record.get("width", IMAGE_SIZE[1])) != IMAGE_SIZE[1]:
                raise ValueError("Image and original coordinate systems must agree")
            if not source and forbidden & set(record):
                raise ValueError("Target/inference model input must not contain region annotations")
            if not source and not target_image_labels_allowed and "image_labels" in record:
                raise ValueError("UDA target/inference input must not contain image_labels, including None")
            if source and "instances" not in record:
                raise ValueError("Source detection requires source Instances GT")

    @torch.no_grad()
    def _base_proposals(self, inputs):
        with torch.cuda.amp.autocast(enabled=False):
            images = self.student.offline_preprocess_image(inputs)
            features = self.student.offline_backbone(images.tensor)
            proposals, _ = self.student.offline_proposal_generator(images, features, None)
            return [p[p.objectness_logits.argsort(descending=True)[:300]] for p in proposals]

    def _raw_roi_features(self, feature, boxes, backbone):
        from detectron2.structures import Boxes
        pooled = self.student.roi_heads._shared_roi_transform([feature], [Boxes(boxes)], backbone.layer4)
        return backbone.attnpool(pooled)

    def _native_predictions(self, raw):
        predictor = self.student.roi_heads.box_predictor
        logits = _native_logits(raw, predictor)
        scores = torch.cat((logits, logits), dim=1)
        deltas = predictor.bbox_pred(raw)
        # Fixed-prompt native losses/inference never use the last two entries.
        return scores, deltas, scores, scores

    @torch.no_grad()
    def _teacher_semantics(self, images, proposals, features=None):
        probabilities = []
        with torch.cuda.amp.autocast(enabled=False):
            if features is None:
                features = self.teacher_backbone(images.tensor.float())
            for image_index, proposal in enumerate(proposals):
                parts = []
                boxes = proposal.proposal_boxes.tensor.detach()
                for start in range(0, len(boxes), self.teacher_roi_chunk):
                    raw = self._raw_roi_features(features["res4"][image_index:image_index + 1], boxes[start:start + self.teacher_roi_chunk], self.teacher_backbone)
                    parts.append(self.teacher_classifier(raw).softmax(dim=1))
                probabilities.append(torch.cat(parts) if parts else boxes.new_empty((0, 9)))
        return probabilities

    def _bundle(self, inputs, teacher_features=None):
        base = self._base_proposals(inputs)
        images = self.student.preprocess_image(inputs)
        if tuple(images.tensor.shape[-2:]) != IMAGE_SIZE or any(tuple(size) != IMAGE_SIZE for size in images.image_sizes):
            raise ValueError("Padded/mismatched image geometry cannot use the original map construction")
        # Paired views share coordinates, so weak teacher features may score
        # strong-view RPN boxes without any box transform or label transfer.
        probabilities = self._teacher_semantics(images, base, features=teacher_features)
        features = self.student.recognition_features(images, base, is_source=False)
        grid = tuple(features["res3"].shape[-2:])
        if grid != (128, 256) or tuple(features["res4"].shape[-2:]) != (64, 128):
            raise ValueError("Expected native C3/C4 stride8/16")
        maps = [torch.from_numpy(build_maps(p.proposal_boxes.tensor.detach().cpu().numpy(),
                 p.objectness_logits.detach().cpu().numpy(), q.detach().cpu().numpy(), IMAGE_SIZE, grid))
                for p, q in zip(base, probabilities)]
        maps = torch.stack(maps).to(device=self.device, dtype=torch.float32)
        return images, features, base, probabilities, maps

    def _samples(self, inputs, base, probabilities, source_count):
        samples = []
        for i, (record, proposal, probs) in enumerate(zip(inputs, base, probabilities)):
            sample = {"boxes": proposal.proposal_boxes.tensor.detach(), "image_size": IMAGE_SIZE,
                      "rpn_logits": proposal.objectness_logits.detach(), "probs": probs.detach()}
            if i < source_count:
                targets = record["instances"].to(self.device)
                sample.update(gt=targets.gt_boxes.tensor, labels=targets.gt_classes)
                presence = torch.zeros(8, device=self.device)
                presence[targets.gt_classes.unique()] = 1
            elif self.target_image_labels_allowed:
                if "image_labels" not in record:
                    raise ValueError("Target requires explicitly permitted image-level labels")
                presence = torch.as_tensor(record["image_labels"], device=self.device).reshape(-1)
            else:
                if "image_labels" in record:
                    raise ValueError("UDA target input must not contain image_labels, including None")
                # Explicit sentinel: the conditional adversary skips only the
                # target class-presence filter, retaining every reliability rule.
                presence = None
            if presence is not None and (presence.shape != (8,) or not bool(((presence == 0) | (presence == 1)).all())):
                raise ValueError("Expected eight binary image-level labels")
            sample["image_labels"] = presence.detach() if presence is not None else None
            samples.append(sample)
        return samples

    def _merged(self, outputs, anchors, samples, base):
        from detectron2.structures import Boxes, Instances
        merged, information = [], []
        for i, (sample, initial) in enumerate(zip(samples, base)):
            use_iir = getattr(self, "image_consistency_enabled", False)
            selected = select_candidates(outputs, anchors, sample, batch_index=i, return_indices=use_iir)
            boxes, scores, stats = selected[:3]
            proposal = Instances(initial.image_size)
            proposal.proposal_boxes = Boxes(torch.cat((initial.proposal_boxes.tensor.detach(), boxes.detach())))
            p = scores.detach().clamp(1e-6, 1 - 1e-6)
            proposal.objectness_logits = torch.cat((initial.objectness_logits.detach(), torch.log(p) - torch.log1p(-p)))
            proposal.is_supplement = torch.cat((torch.zeros(len(initial), dtype=torch.bool, device=self.device), torch.ones(len(boxes), dtype=torch.bool, device=self.device)))
            if use_iir:
                # Keep selection exactly obj*miss. IIR instead weights proposals
                # by raw objectness and differentiates the selected VGS logits.
                raw_obj = flatten_outputs(outputs)["obj"][i, selected[3]].float()
                proposal.aggregation_objectness_logits = torch.cat((initial.objectness_logits.detach().float(), raw_obj))
            merged.append(proposal)
            information.append({"base": len(initial), **stats})
        return merged, information

    def _roi_predictions(self, features, proposals, chunk, train=False, teacher=False):
        prediction_parts = []
        backbone = self.teacher_backbone if teacher else self.student.backbone
        for i, proposal in enumerate(proposals):
            boxes = proposal.proposal_boxes.tensor.detach()
            feature = features["res4"][i:i + 1]
            for start in range(0, len(boxes), chunk):
                selected = boxes[start:start + chunk]
                # Bind each chunk's coordinates now; a late-bound loop closure
                # would silently recompute every checkpoint at the final boxes.
                amp_state = torch.is_autocast_enabled()
                def operation(value, roi_boxes=selected, enabled=amp_state):
                    # Explicitly preserve forward precision under the old
                    # PyTorch1.9 reentrant checkpoint backward recomputation.
                    with torch.cuda.amp.autocast(enabled=enabled):
                        return self._raw_roi_features(value, roi_boxes, backbone)
                raw = checkpoint(operation, feature) if train and self.checkpoint_res5 and feature.requires_grad else operation(feature)
                if teacher:
                    logits = self.teacher_classifier(raw)
                    scores = torch.cat((logits, logits), dim=1)
                    prediction_parts.append((scores, self.teacher_bbox(raw), scores, scores))
                else:
                    prediction_parts.append(self._native_predictions(raw))
        if not prediction_parts:
            raw = features["res4"].new_zeros((0, self.student.roi_heads.box_predictor.cls_score.in_features)) + features["res4"].sum() * 0
            return self._native_predictions(raw)
        return tuple(torch.cat([row[j] for row in prediction_parts], dim=0) for j in range(4))

    @staticmethod
    def _event_context():
        from detectron2.utils.events import EventStorage, get_event_storage
        try:
            get_event_storage()
            return contextlib.nullcontext()
        except AssertionError:
            return EventStorage()

    @staticmethod
    def _image_probabilities(predictions, proposals):
        from image_consistency import h2fa_aggregate
        # Native prediction has TWO identical 9-class groups. Use exactly the
        # eight foreground columns of the first group, not all 17 non-last columns.
        rows = predictions[0][:, :8].split([len(p) for p in proposals])
        return torch.stack([h2fa_aggregate(z, p.aggregation_objectness_logits)
                            for z, p in zip(rows, proposals)])

    @torch.no_grad()
    def _teacher_image_predictions(self, weak_inputs, weak_features):
        """Independent weak RPN+EMA-VGS proposals, without any GT/pseudo boxes."""
        with torch.cuda.amp.autocast(enabled=False):
            base = self._base_proposals(weak_inputs)
            probabilities = self._teacher_semantics(None, base, features=weak_features)
            maps = torch.stack([torch.from_numpy(build_maps(
                p.proposal_boxes.tensor.detach().cpu().numpy(),
                p.objectness_logits.detach().cpu().numpy(), q.cpu().numpy(),
                IMAGE_SIZE, tuple(weak_features["res3"].shape[-2:])))
                for p, q in zip(base, probabilities)]).to(self.device, dtype=torch.float32)
            outputs = self.teacher_search(weak_features["res3"], weak_features["res4"], maps)
            samples = [{"boxes": p.proposal_boxes.tensor.detach(), "image_size": IMAGE_SIZE} for p in base]
            merged, selection = self._merged(outputs, self.teacher_search.anchors(outputs), samples, base)
            predictions = self._roi_predictions(weak_features, merged, self.teacher_roi_chunk, teacher=True)
            return self._image_probabilities(predictions, merged), selection

    def _image_consistency_loss(self, target_features, target_merged, weak_inputs, weak_features):
        from image_consistency import image_consistency, h2fa_group_diagnostics
        teacher_prob, teacher_selection = self._teacher_image_predictions(weak_inputs, weak_features)
        student_predictions = self._roi_predictions(target_features, target_merged,
                                                    self.train_roi_chunk, train=True)
        student_prob = self._image_probabilities(student_predictions, target_merged)
        loss = self.image_consistency_weight * image_consistency(student_prob, teacher_prob)
        diagnostic = []
        rows = student_predictions[0][:, :8].detach().split([len(p) for p in target_merged])
        for z, p in zip(rows, target_merged):
            diagnostic.append(h2fa_group_diagnostics(z, p.aggregation_objectness_logits.detach(),
                                                     int((~p.is_supplement).sum())))
        stats = {"weight": self.image_consistency_weight, "ema_decay": self.ema_decay,
                 "teacher_updates_before_step": self.ema_updates,
                 "student_image_probabilities": student_prob.detach().cpu().tolist(),
                 "teacher_image_probabilities": teacher_prob.cpu().tolist(),
                 "absolute_probability_gap": float((student_prob.detach() - teacher_prob).abs().mean()),
                 "teacher_selection": teacher_selection, "student_aggregation": diagnostic,
                 "target_pseudo_box_losses": False}
        return loss, stats

    def training_losses(self, source_inputs, target_inputs, progress):
        if not self.training:
            raise ValueError("training_losses requires model.train()")
        self._validate_inputs(source_inputs, True)
        self._validate_inputs(target_inputs, False, self.target_image_labels_allowed)
        n = len(source_inputs); inputs = list(source_inputs) + list(target_inputs)
        weak_inputs, weak_features = None, None
        if self.image_consistency_enabled:
            weak_inputs = self._weak_inputs(inputs)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                weak_images = self.student.preprocess_image(weak_inputs)
                weak_features = self.teacher_backbone(weak_images.tensor.float())
        images, features, base, probabilities, maps = self._bundle(inputs, teacher_features=weak_features)
        samples = self._samples(inputs, base, probabilities, n)
        with DomainFeatureTap(self.search) as tap:
            outputs = self.search(features["res3"], features["res4"], maps)
            prefusion = tap.pop()
        anchors = self.search.anchors(outputs)
        source_outputs = [{key: value[:n] if torch.is_tensor(value) else value for key, value in level.items()} for level in outputs]
        search_loss, search_stats = training_loss(source_outputs, anchors, samples[:n])
        if self.image_consistency_enabled:
            all_merged, all_selection = self._merged(outputs, anchors, samples, base)
            merged, selection_stats = all_merged[:n], all_selection[:n]
        else:
            merged, selection_stats = self._merged(outputs, anchors, samples[:n], base[:n])
        targets = [record["instances"].to(self.device) for record in source_inputs]
        with self._event_context():
            proposals = self.student.roi_heads.label_and_sample_proposals(merged, targets)
            predictions = self._roi_predictions({key: value[:n] for key, value in features.items() if value is not None}, proposals, self.train_roi_chunk, train=True)
            detection_loss = self.student.roi_heads.box_predictor.losses(predictions, proposals, is_source=True)
        if set(detection_loss) != {"loss_cls", "loss_box_reg"}:
            raise RuntimeError("An unrequested native detector auxiliary loss was enabled")
        coefficient = ramp(progress)
        global_loss, global_stats = self.global_D({k: features[k][:n] for k in ("res3", "res4")},
            {k: features[k][n:] for k in ("res3", "res4")}, coefficient)
        conditional_loss, conditional_stats = self.conditional_D({k: v[:n] for k, v in prefusion.items()}, samples[:n],
            {k: v[n:] for k, v in prefusion.items()}, samples[n:], grl_coeff=self.conditional_grl_max * coefficient)
        # Already weighted exactly once. The trainer sums this dictionary once;
        # deliberately no 'total' entry to accidentally double-count components.
        losses = {"loss_source_cls": self.source_loss_weight * detection_loss["loss_cls"],
            "loss_source_box": self.source_loss_weight * detection_loss["loss_box_reg"],
            "loss_search_obj": self.search_loss_weight * search_loss["obj"],
            "loss_search_miss": self.search_loss_weight * search_loss["miss"],
            "loss_search_box": self.search_loss_weight * search_loss["box"],
            "loss_traditional_domain": self.traditional_domain_weight * global_loss,
            "loss_conditional_domain": self.conditional_domain_weight * conditional_loss}
        image_stats = None
        if self.image_consistency_enabled:
            losses["loss_target_image_consistency"], image_stats = self._image_consistency_loss(
                {k: v[n:] for k, v in features.items() if v is not None}, all_merged[n:], weak_inputs[n:],
                {k: v[n:] for k, v in weak_features.items() if v is not None})
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise FloatingPointError("Nonfinite integrated source/domain loss")
        roi_stats = {"sampled": sum(len(p) for p in proposals),
                     "foreground": sum(int((p.gt_classes < 8).sum()) for p in proposals),
                     "supplement": sum(int(p.is_supplement.sum()) for p in proposals),
                     "supplement_foreground": sum(int((p.is_supplement & (p.gt_classes < 8)).sum()) for p in proposals)}
        stats = {"source_images": n, "target_images": len(target_inputs), "search": search_stats,
                 "selection": selection_stats, "roi": roi_stats, "global_domain": global_stats,
                 "conditional_domain": conditional_stats, "grl_ramp": coefficient,
                 "global_outer_weight": self.traditional_domain_weight,
                 "conditional_grl_coefficient": self.conditional_grl_max * coefficient,
                 "target_box_GT_used": False, "target_image_labels_used": self.target_image_labels_allowed,
                 "target_presence_filter_enabled": self.target_image_labels_allowed,
                 "teacher_fixed": not self.image_consistency_enabled, "source_GT_appended": False}
        if image_stats is not None:
            stats["image_consistency"] = image_stats
        return losses, stats

    @torch.no_grad()
    def predict(self, inputs, return_proposals=False):
        if self.training:
            raise ValueError("predict requires model.eval()")
        self._validate_inputs(inputs, False, self.target_image_labels_allowed)
        images, features, base, probabilities, maps = self._bundle(inputs)
        outputs = self.search(features["res3"], features["res4"], maps)
        samples = [{"boxes": p.proposal_boxes.tensor.detach(), "image_size": IMAGE_SIZE} for p in base]
        merged, stats = self._merged(outputs, self.search.anchors(outputs), samples, base)
        predictions = self._roi_predictions(features, merged, self.inference_roi_chunk)
        instances, _ = self.student.roi_heads.box_predictor.inference(predictions, merged, is_source=False)
        from detectron2.modeling.postprocessing import detector_postprocess
        results = []
        for record, detected, initial, proposal, info in zip(inputs, instances, base, merged, stats):
            row = {"instances": detector_postprocess(detected, int(record.get("height", 1024)), int(record.get("width", 2048)))}
            if return_proposals:
                row.update(base_proposals=initial, supplement_proposals=proposal[proposal.is_supplement], merged_proposals=proposal, proposal_stats=info)
            results.append(row)
        return results

    def forward(self, source_inputs, target_inputs=None, progress=0.):
        return self.training_losses(source_inputs, target_inputs, progress) if self.training else self.predict(source_inputs)

    @torch.no_grad()
    def verify_native_path(self):
        """One-time FP32 smoke audit against the actual unchanged predictor.

        Uses deterministic raw vectors; restores each module mode and RNG.
        It neither queries images nor performs an optimizer update.
        """
        predictor = self.student.roi_heads.box_predictor
        modes = [(module, module.training) for module in predictor.modules()]
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        try:
            predictor.eval()
            with torch.random.fork_rng(devices=devices):
                width = predictor.cls_score.in_features
                raw = torch.linspace(-1., 1., steps=7 * width, device=self.device, dtype=torch.float32).reshape(7, width)
                with torch.cuda.amp.autocast(enabled=False):
                    native = predictor(raw)
                    reduced = self._native_predictions(raw)
                equal_scores = torch.equal(native[0], reduced[0])
                equal_deltas = torch.equal(native[1], reduced[1])
                result = {"passed": bool(equal_scores and equal_deltas), "native_scores18_exact": equal_scores,
                          "native_deltas_exact": equal_deltas, "rows": 7, "precision": "FP32",
                          "scores_max_abs": float((native[0] - reduced[0]).abs().max()),
                          "deltas_max_abs": float((native[1] - reduced[1]).abs().max())}
                if not result["passed"]:
                    raise RuntimeError("Reduced fixed-text path changed native predictions: " + str(result))
                return result
        finally:
            for module, mode in modes:
                module.training = mode

    @staticmethod
    def _cpu_state(module):
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def checkpoint_state(self):
        # Frozen native text/prompt/offline modules are recovered from the exact
        # source checkpoint, not duplicated in every new experiment checkpoint.
        state = {"version": self.version, "sources": self.source_identity, "config": self.config,
                "student_backbone": self._cpu_state(self.student.backbone),
                "student_bbox": self._cpu_state(self.student.roi_heads.box_predictor.bbox_pred),
                "search": self._cpu_state(self.search), "global_D": self._cpu_state(self.global_D),
                "conditional_D": self._cpu_state(self.conditional_D), "initialization_audit": self.initialization_audit}
        if self.image_consistency_enabled:
            state.update(teacher_backbone=self._cpu_state(self.teacher_backbone),
                         teacher_search=self._cpu_state(self.teacher_search),
                         teacher_bbox=self._cpu_state(self.teacher_bbox), ema_updates=self.ema_updates)
        return state

    def load_checkpoint_state(self, state):
        if state.get("version") != self.version or not same_sources(state.get("sources"), self.source_identity):
            raise ValueError("Checkpoint fixed-source identity differs from this experiment")
        for module, name in ((self.student.backbone, "student_backbone"), (self.student.roi_heads.box_predictor.bbox_pred, "student_bbox"),
                             (self.search, "search"), (self.global_D, "global_D"), (self.conditional_D, "conditional_D")):
            module.load_state_dict(state[name], strict=True)
        if self.image_consistency_enabled:
            for module, name in ((self.teacher_backbone, "teacher_backbone"),
                                 (self.teacher_search, "teacher_search"), (self.teacher_bbox, "teacher_bbox")):
                module.load_state_dict(state[name], strict=True)
            updates = state["ema_updates"]
            if type(updates) is not int or updates < 0:
                raise ValueError("Invalid EMA update count")
            self.ema_updates = updates
        self.train(self.training)


def self_test():
    torch.set_num_threads(2); torch.manual_seed(17)
    actual, ordinary = TraditionalDomainDiscriminators(hidden=8), TraditionalDomainDiscriminators(hidden=8)
    ordinary.load_state_dict(actual.state_dict())
    source = {"res3": torch.randn(1, 512, 3, 4, requires_grad=True), "res4": torch.randn(1, 1024, 2, 2, requires_grad=True)}
    target = {k: torch.randn_like(v, requires_grad=True) for k, v in source.items()}
    s0 = {k: v.detach().clone().requires_grad_() for k, v in source.items()}
    t0 = {k: v.detach().clone().requires_grad_() for k, v in target.items()}
    loss, _ = actual(source, target, .3)
    reference = []
    for name, d in ordinary.heads.items():
        a, b = d(s0[name]), d(t0[name])
        reference.append(.5 * (F.binary_cross_entropy_with_logits(a, torch.zeros_like(a)) + F.binary_cross_entropy_with_logits(b, torch.ones_like(b))))
    (.1 * loss).backward(); (.1 * torch.stack(reference).mean()).backward()
    for name in source:
        assert torch.allclose(source[name].grad, -.3 * s0[name].grad, atol=1e-9, rtol=1e-5)
        assert torch.allclose(target[name].grad, -.3 * t0[name].grad, atol=1e-9, rtol=1e-5)
    for a, b in zip(actual.parameters(), ordinary.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-8, rtol=1e-5)
    assert ramp(0) == 0 and 0 < ramp(.5) < ramp(1) < 1
    print("formal model CPU self-test passed: GRL sign, outer0.1 once, discriminator gradient, schedule")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test()
    else: parser.print_help()
