"""DeST: native RegionCLIP initialization, MRSA, and relaxed RPN guidance.

Keeps RegionCLIP's separate localization and recognition encoders. The original
COCO localization stream is initialized from SourceB, then trained using source
RPN losses and target objectness guidance. No VGS, conditional discriminator,
target instance matching, or target box regression is used.
"""
import contextlib
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .core import collect_windows, mrsa_aggregate, relaxed_spatial_loss
    from .rpn_support import run_rpn
except ImportError:
    from core import collect_windows, mrsa_aggregate, relaxed_spatial_loss
    from rpn_support import run_rpn

VERSION = "dest-mrsa-relaxed-spatial-v1"


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_logits(raw, classifier):
    normalized = F.normalize(raw, p=2., dim=1)
    foreground = normalized @ F.normalize(classifier.cls_score.weight, p=2., dim=1).t()
    if classifier.use_bias:
        foreground = foreground + classifier.cls_score.bias
    background = classifier.cls_bg_score(normalized)
    if classifier.use_bias:
        background = background + classifier.cls_bg_score.bias
    return torch.cat((foreground, background), dim=1) / classifier.temperature


def _legacy_helpers():
    # Namespace-qualified imports avoid collision with dest/model.py or train.py.
    from experiments.vgs_da.image_consistency import h2fa_aggregate, image_consistency
    from experiments.vgs_da.ema import update_ema
    return h2fa_aggregate, image_consistency, update_ema


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coefficient * grad, None


class SpatialDomainDiscriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Conv2d(channels, 128, 1),
            nn.ReLU(inplace=False), nn.Conv2d(128, 1, 1))
            for name, channels in (("res3", 512), ("res4", 1024))})
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=.01)
                nn.init.zeros_(module.bias)

    def forward(self, source, target, coefficient):
        losses, diagnostics = [], {}
        with torch.cuda.amp.autocast(enabled=False):
            for name, head in self.heads.items():
                s = head(_Reverse.apply(source[name].float(), coefficient))
                t = head(_Reverse.apply(target[name].float(), coefficient))
                loss = .5 * (F.binary_cross_entropy_with_logits(s, torch.zeros_like(s)) +
                             F.binary_cross_entropy_with_logits(t, torch.ones_like(t)))
                losses.append(loss)
                diagnostics[name] = {"bce": float(loss.detach())}
        return torch.stack(losses).mean(), diagnostics


def _build_components(cfg, offline_cfg):
    """Build only the active native paths; omit unused CUDA-only prompt models."""
    from detectron2.layers import ShapeSpec
    from detectron2.modeling.backbone import build_backbone
    from detectron2.modeling.proposal_generator import build_proposal_generator
    from detectron2.modeling.poolers import ROIPooler
    from detectron2.modeling.roi_heads.roi_heads import ROIHeads
    from detectron2.modeling.roi_heads.clip_roi_heads import CLIPRes5ROIHeads
    from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

    class FixedTextPredictor(FastRCNNOutputLayers):
        def forward(self, raw):
            logits = native_logits(raw, self)
            scores = torch.cat((logits, logits), dim=1)
            return scores, self.bbox_pred(raw), scores, scores

    model = DetectionComponents(cfg, offline_cfg)
    model.backbone = build_backbone(cfg)
    model.offline_backbone = build_backbone(offline_cfg)
    model.offline_proposal_generator = build_proposal_generator(
        offline_cfg, model.offline_backbone.output_shape())
    dim = int(cfg.MODEL.CLIP.TEXT_EMB_DIM)
    arguments = FastRCNNOutputLayers.from_config(cfg, ShapeSpec(channels=dim))
    arguments["clip_cls_emb"] = (False, None)
    predictor = FixedTextPredictor(**arguments)
    predictor.cls_score = nn.Linear(dim, cfg.MODEL.ROI_HEADS.NUM_CLASSES, bias=False)
    predictor.cls_bg_score = nn.Linear(dim, 1, bias=False)
    predictor.use_clip_cls_emb, predictor.use_bias = True, False
    predictor.is_prompt_tuning = False
    predictor.temperature = float(cfg.MODEL.CLIP.CLSS_TEMP)
    predictor.test_cls_score = None
    predictor.cls_score.requires_grad_(False)
    predictor.cls_bg_score.requires_grad_(False)
    in_features = list(cfg.MODEL.ROI_HEADS.IN_FEATURES)
    if in_features != ["res4"]:
        raise ValueError("DeST requires the source C4 ROI path")
    pooler = ROIPooler(output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
        scales=(1. / model.backbone.output_shape()["res4"].stride,),
        sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
        pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE)
    model.roi_heads = CLIPRes5ROIHeads(in_features=in_features, pooler=pooler,
        res5=None, box_predictor=predictor, **ROIHeads.from_config(cfg))
    return model


class DetectionComponents(nn.Module):
    def __init__(self, cfg, offline_cfg):
        super().__init__()
        self.input_format = cfg.INPUT.FORMAT
        self.offline_input_format = offline_cfg.INPUT.FORMAT
        for name, values in (("pixel_mean", cfg.MODEL.PIXEL_MEAN),
                             ("pixel_std", cfg.MODEL.PIXEL_STD),
                             ("offline_pixel_mean", offline_cfg.MODEL.PIXEL_MEAN),
                             ("offline_pixel_std", offline_cfg.MODEL.PIXEL_STD)):
            self.register_buffer(name, torch.tensor(values).view(3, 1, 1), persistent=False)
        self.div_pixel = sum(cfg.MODEL.PIXEL_MEAN) < 3.
        self.offline_div_pixel = sum(offline_cfg.MODEL.PIXEL_MEAN) < 3.

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess(self, inputs, localization=False):
        from detectron2.structures import ImageList
        values = [record["image"].to(self.device).float() for record in inputs]
        if localization and self.input_format != self.offline_input_format:
            values = [image[[2, 1, 0]] for image in values]
        prefix = "offline_" if localization else ""
        mean, std = getattr(self, prefix + "pixel_mean"), getattr(self, prefix + "pixel_std")
        divide = self.offline_div_pixel if localization else self.div_pixel
        values = [((image / 255.) if divide else image).sub(mean).div(std) for image in values]
        backbone = self.offline_backbone if localization else self.backbone
        return ImageList.from_tensors(values, backbone.size_divisibility)


class DeST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config.get("target_image_labels_allowed") is not False:
            raise ValueError("DeST requires explicitly unlabeled target inputs")
        if config.get("strong_weak_enabled") is not True:
            raise ValueError("DeST requires paired strong/weak target views")
        if config.get("evaluation_model", "teacher") != "teacher":
            raise ValueError("DeST evaluates its adapted EMA teacher")
        self.image_consistency_enabled = True
        self.version, self.ema_updates = VERSION, 0
        self.ema_decay = float(config.get("ema_decay", .9996))
        if not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid EMA decay")
        from detectron2.config import get_cfg
        cfg = get_cfg(); cfg.merge_from_file(config["source_cfg"])
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4"]
        cfg.MODEL.BACKBONE.FREEZE_AT = int(config.get("recognition_freeze_at", 2))
        cfg.MODEL.DEVICE = config.get("device", "cuda")
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(config.get("source_roi_batch", 512))
        cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = False
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(config.get("score_threshold", .001))
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(config.get("nms_threshold", .5))
        cfg.TEST.DETECTIONS_PER_IMAGE = int(config.get("detections_per_image", 100))
        cfg.MODEL.CLIP.MULTIPLY_RPN_SCORE = False
        if cfg.MODEL.DA_PRO.ENABLED or cfg.LEARNABLE_PROMPT.TUNING or cfg.MODEL.MASK_ON:
            raise ValueError("Source configuration enables unsupported DA/prompt/mask paths")
        if cfg.MODEL.ROI_HEADS.NUM_CLASSES != 8 or cfg.MODEL.ROI_HEADS.NAME != "CLIPRes5ROIHeads":
            raise ValueError("This C2F implementation requires SourceB's eight-class C4 head")
        if (cfg.MODEL.C3_ADAPTER.ENABLED or cfg.MODEL.C4_SEMANTIC_RESHAPER.ENABLED or
                cfg.MODEL.C5_SEMANTIC_RESHAPER.ENABLED):
            raise ValueError("Legacy adaptation modules must be disabled")
        offline_cfg = get_cfg()
        offline_path = config.get("rpn_cfg") or str(Path(config["repo_root"]) /
            "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
        offline_cfg.merge_from_file(offline_path)
        offline_cfg.MODEL.BACKBONE.FREEZE_AT = int(config.get("rpn_backbone_freeze_at", 2))
        if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
            offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH
        cfg.freeze(); offline_cfg.freeze()
        self.cfg = cfg
        self.student = _build_components(cfg, offline_cfg)
        initial = torch.load(config["source_checkpoint"], map_location="cpu")
        source_state = initial["model"]
        expected = self.student.state_dict()
        missing = sorted(set(expected) - set(source_state))
        ignored = sorted(set(source_state) - set(expected))
        ignored_prefixes = ("roi_heads.box_predictor.clip_model.", "roi_heads.box_predictor.DAHead.")
        unexpected = [key for key in ignored if not key.startswith(ignored_prefixes)]
        if missing or unexpected:
            raise ValueError("SourceB active-state mismatch; missing=%r unexpected=%r" % (missing, unexpected))
        self.student.load_state_dict({key: source_state[key] for key in expected}, strict=True)
        active_exact = all(torch.equal(value.cpu(), source_state[key].cpu())
                           for key, value in self.student.state_dict().items())
        if not active_exact:
            raise AssertionError("An active SourceB parameter was not restored exactly")
        self.teacher = copy.deepcopy(self.student).requires_grad_(False).eval()
        self.global_D = SpatialDomainDiscriminators()
        self.source_identity = {key: {"path": str(Path(config[key]).resolve()), "sha256": file_sha(config[key])}
            for key in ("source_cfg", "source_checkpoint", "rpn_checkpoint", "text_embeddings")}
        expected_sha = config.get("source_checkpoint_sha256")
        if expected_sha and expected_sha != self.source_identity["source_checkpoint"]["sha256"]:
            raise ValueError("Source checkpoint SHA256 differs from the requested SourceB")
        self.initialization_audit = {"version": VERSION, "source_iteration": initial.get("iteration"),
            "sources": self.source_identity, "loaded_active_keys": len(expected),
            "active_source_tensors_exact": active_exact, "omitted_unused_keys": len(ignored),
            "omitted_prefixes": list(ignored_prefixes), "rpn_stream": "native COCO backbone and head restored from SourceB",
            "trainable_detector": [name for name, p in self.student.named_parameters() if p.requires_grad],
            "evaluation_model": "teacher", "target_image_labels_used": False,
            "target_box_regression": False, "vgs": False, "conditional_discriminator": False,
            "windows_path": "teacher semantic evidence only"}
        del source_state, expected, initial
        self.to(torch.device(config.get("device", "cuda")))
        self.train(True)
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        if trainable != {id(p) for p in self.detector_parameters() + self.aux_parameters()}:
            raise AssertionError("Trainable parameter partition mismatch")

    @property
    def device(self):
        return self.student.device

    def detector_parameters(self):
        return [p for p in self.student.parameters() if p.requires_grad]

    def aux_parameters(self):
        return list(self.global_D.parameters())

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "teacher"):
            self.teacher.eval()
        if hasattr(self, "student"):
            for module in self.student.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @torch.no_grad()
    def update_teacher(self):
        _legacy_helpers()[2](self.teacher, self.student, self.ema_decay)
        # Shared fixed text embeddings (and frozen early layers) must remain
        # exactly shared: repeated EMA arithmetic can drift even equal tensors.
        teacher_parameters = dict(self.teacher.named_parameters())
        for name, parameter in self.student.named_parameters():
            if not parameter.requires_grad:
                teacher_parameters[name].copy_(parameter)
        # Normalization statistics are fixed in train(), not learned via EMA.
        teacher_buffers = dict(self.teacher.named_buffers())
        for name, buffer in self.student.named_buffers():
            teacher_buffers[name].copy_(buffer)
        self.ema_updates += 1

    @staticmethod
    def _event_context():
        from detectron2.utils.events import EventStorage, get_event_storage
        try:
            get_event_storage()
            return contextlib.nullcontext()
        except AssertionError:
            return EventStorage()

    @staticmethod
    def _validate(inputs, source=False, weak=False):
        if not inputs:
            raise ValueError("Empty batch")
        for record in inputs:
            image = record["image"]
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Expected RGB CHW input")
            if source and "instances" not in record:
                raise ValueError("Source input lacks GT")
            if not source and set(record) & {"instances", "annotations", "gt", "gt_boxes", "gt_classes", "labels", "targets", "image_labels"}:
                raise ValueError("Target input contains labels")
            if weak and ("image_weak" not in record or record["image_weak"].shape != image.shape):
                raise ValueError("Target strong and weak geometry must agree")

    def _rpn(self, model, inputs, source_targets=None):
        images = model.preprocess(inputs, localization=True)
        features = model.offline_backbone(images.tensor)
        return run_rpn(model.offline_proposal_generator, images, features,
            source_targets=source_targets, proposal_budget=int(self.config.get("initial_proposals", 300)),
            pre_nms_topk=self.config.get("rpn_pre_nms_topk"))

    def _roi_predictions(self, model, features, proposals, chunk, train=False):
        from detectron2.structures import Boxes
        parts = []
        for image_index, proposal in enumerate(proposals):
            boxes = proposal.proposal_boxes.tensor.detach()
            for start in range(0, len(boxes), chunk):
                region_boxes = boxes[start:start + chunk]
                amp_enabled = torch.is_autocast_enabled()
                def operation(feature, selected=region_boxes, enabled=amp_enabled):
                    with torch.cuda.amp.autocast(enabled=enabled):
                        pooled = model.roi_heads._shared_roi_transform([feature], [Boxes(selected)], model.backbone.layer4)
                        return model.backbone.attnpool(pooled)
                value = features["res4"][image_index:image_index + 1]
                use_checkpoint = train and self.config.get("checkpoint_res5", False) and value.requires_grad
                raw = checkpoint(operation, value) if use_checkpoint else operation(value)
                parts.append(model.roi_heads.box_predictor(raw))
        if not parts:
            dim = model.roi_heads.box_predictor.cls_score.in_features
            raw = features["res4"].new_zeros((0, dim)) + features["res4"].sum() * 0
            return model.roi_heads.box_predictor(raw)
        return tuple(torch.cat([part[index] for part in parts]) for index in range(4))

    @staticmethod
    def _image_probabilities(predictions, proposals):
        aggregate = _legacy_helpers()[0]
        rows = predictions[0][:, :8].split([len(p) for p in proposals])
        return torch.stack([aggregate(row, proposal.aggregation_objectness_logits)
                            for row, proposal in zip(rows, proposals)])

    @torch.no_grad()
    def _teacher_targets(self, weak_inputs):
        from detectron2.structures import Boxes, Instances
        with torch.cuda.amp.autocast(enabled=False):
            base, _, _, _ = self._rpn(self.teacher, weak_inputs)
            images = self.teacher.preprocess(weak_inputs)
            features = self.teacher.backbone(images.tensor)
            evidence = []
            for proposal in base:
                windows = collect_windows(proposal.image_size, self.config.get("window_scales", [.125, .25, .5, 1.]),
                                          device=self.device, dtype=torch.float32)
                item = Instances(proposal.image_size)
                item.proposal_boxes = Boxes(torch.cat((proposal.proposal_boxes.tensor, windows)))
                evidence.append(item)
            predictions = self._roi_predictions(self.teacher, features, evidence,
                int(self.config.get("teacher_roi_chunk", 128)))
            counts = [len(item) for item in evidence]
            parts = [field.split(counts) for field in predictions]
            base_predictions = tuple(torch.cat([parts[field][i][:len(base[i])] for i in range(len(base))])
                                     for field in range(4))
            final, _ = self.teacher.roi_heads.box_predictor.inference(base_predictions, base, is_source=False)
            target_probs, supports, diagnostics = [], [], []
            for i, (proposal, item, detected) in enumerate(zip(base, evidence, final)):
                # The duplicated native score groups are NOT 17 foreground classes.
                target, info = mrsa_aggregate(item.proposal_boxes.tensor, parts[0][i][:, :9],
                    rho=float(self.config.get("mrsa_rho", 1.)),
                    max_iter=int(self.config.get("mrsa_max_iter", 100)),
                    tol=float(self.config.get("mrsa_tolerance", 1e-4)))
                keep = detected.scores >= float(self.config.get("spatial_score_threshold", .5))
                supports.append(detected.pred_boxes.tensor[keep].detach())
                target_probs.append(target)
                diagnostics.append(dict(info, rpn_regions=len(proposal), window_regions=len(item)-len(proposal),
                                        final_spatial_hints=int(keep.sum())))
            return torch.stack(target_probs), supports, diagnostics

    def training_losses(self, source_inputs, target_inputs, progress=0.):
        if not self.training:
            raise ValueError("training_losses requires training mode")
        self._validate(source_inputs, source=True)
        self._validate(target_inputs, weak=True)
        n = len(source_inputs)
        inputs = list(source_inputs) + list(target_inputs)
        weak = [dict(record, image=record["image_weak"]) for record in target_inputs]
        teacher_probs, spatial_boxes, teacher_stats = self._teacher_targets(weak)
        targets = [record["instances"].to(self.device) for record in source_inputs]
        with self._event_context():
            proposals, rpn_losses, raw_logits, centers = self._rpn(self.student, inputs, targets)
            images = self.student.preprocess(inputs)
            features = self.student.backbone(images.tensor)
            source_features = {key: value[:n] for key, value in features.items() if value is not None}
            target_features = {key: value[n:] for key, value in features.items() if value is not None}
            sampled = self.student.roi_heads.label_and_sample_proposals(proposals[:n], targets)
            chunk = int(self.config.get("train_roi_chunk", 128))
            source_prediction = self._roi_predictions(self.student, source_features, sampled, chunk, train=True)
            detector_losses = self.student.roi_heads.box_predictor.losses(source_prediction, sampled, is_source=True)
            target_prediction = self._roi_predictions(self.student, target_features, proposals[n:], chunk, train=True)
        student_probs = self._image_probabilities(target_prediction, proposals[n:])
        sem_loss = _legacy_helpers()[1](student_probs, teacher_probs)
        spatial_losses, spatial_stats = [], []
        for i, (record, boxes) in enumerate(zip(target_inputs, spatial_boxes)):
            loss, info = relaxed_spatial_loss(raw_logits[n+i], centers, boxes,
                tuple(record["image"].shape[-2:]), gamma=float(self.config.get("spatial_gamma", 1.5)),
                tau=float(self.config.get("spatial_tau", .7)))
            spatial_losses.append(loss); spatial_stats.append(info)
        spa_loss = torch.stack(spatial_losses).mean()
        if not 0 <= float(progress) <= 1:
            raise ValueError("Training progress must be in [0,1]")
        coefficient = 2. / (1. + math.exp(-10. * float(progress))) - 1.
        dom_loss, dom_stats = self.global_D(source_features, target_features, coefficient)
        source_weight = float(self.config.get("source_loss_weight", 1.))
        losses = {"loss_source_cls": source_weight * detector_losses["loss_cls"],
                  "loss_source_box": source_weight * detector_losses["loss_box_reg"],
                  "loss_source_rpn_cls": source_weight * rpn_losses["loss_rpn_cls"],
                  "loss_source_rpn_box": source_weight * rpn_losses["loss_rpn_loc"],
                  "loss_target_semantic": float(self.config.get("image_consistency_weight", 1.)) * sem_loss,
                  "loss_target_spatial": float(self.config.get("spatial_loss_weight", .1)) * spa_loss,
                  "loss_domain": float(self.config.get("traditional_domain_weight", .1)) * dom_loss}
        if not all(torch.isfinite(value) for value in losses.values()):
            raise FloatingPointError("Nonfinite DeST loss")
        stats = {"source_images": n, "target_images": len(target_inputs),
            "roi": {"sampled": sum(len(p) for p in sampled),
                    "foreground": sum(int((p.gt_classes < 8).sum()) for p in sampled)},
            "global_domain": dom_stats, "grl_ramp": coefficient,
            "teacher_mrsa": teacher_stats, "spatial": spatial_stats,
            "student_image_probabilities": student_probs.detach().cpu().tolist(),
            "teacher_image_probabilities": teacher_probs.detach().cpu().tolist(),
            "semantic_absolute_gap": float((student_probs.detach() - teacher_probs).abs().mean()),
            "unweighted_losses": {"semantic": float(sem_loss.detach()),
                "spatial": float(spa_loss.detach()), "domain": float(dom_loss.detach())},
            "target_image_labels_used": False, "target_box_GT_used": False,
            "teacher_updates_before_step": self.ema_updates, "evaluation_model": "teacher"}
        return losses, stats

    @torch.no_grad()
    def predict(self, inputs, return_proposals=False):
        if self.training:
            raise ValueError("predict requires eval mode")
        self._validate(inputs)
        from detectron2.modeling.postprocessing import detector_postprocess
        with torch.cuda.amp.autocast(enabled=False):
            proposals, _, _, _ = self._rpn(self.teacher, inputs)
            images = self.teacher.preprocess(inputs)
            features = self.teacher.backbone(images.tensor)
            predictions = self._roi_predictions(self.teacher, features, proposals,
                int(self.config.get("inference_roi_chunk", 64)))
            detected, _ = self.teacher.roi_heads.box_predictor.inference(predictions, proposals, is_source=False)
            result = []
            for record, instances, proposal in zip(inputs, detected, proposals):
                height, width = record["image"].shape[-2:]
                row = {"instances": detector_postprocess(instances, int(record.get("height", height)),
                                                          int(record.get("width", width)))}
                if return_proposals:
                    row["base_proposals"] = proposal
                result.append(row)
            return result

    def forward(self, source_inputs, target_inputs=None, progress=0.):
        return self.training_losses(source_inputs, target_inputs, progress) if self.training else self.predict(source_inputs)

    @torch.no_grad()
    def verify_native_path(self):
        from types import SimpleNamespace
        from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
        predictor = self.student.roi_heads.box_predictor
        width = predictor.cls_score.in_features
        raw = torch.linspace(-1., 1., 7 * width, device=self.device).reshape(7, width)
        actual = predictor(raw)
        # Invoke the original repository forward, not this wrapper's formula.
        # Its unused prompt branches still execute, so provide fixed embeddings
        # there; they cannot influence native scores or box deltas when prompt
        # tuning is disabled. This avoids loading an unrelated CUDA CLIP copy.
        repeated = torch.cat((predictor.cls_score.weight, predictor.cls_score.weight))
        predictor.DAHead = SimpleNamespace(get_embedding=lambda: repeated,
                                           get_embedding_ema=lambda: repeated)
        try:
            expected = FastRCNNOutputLayers.forward(predictor, raw)
        finally:
            del predictor.DAHead
        exact = torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
        if not exact:
            raise AssertionError("Native fixed-text prediction mismatch")
        return {"passed": exact, "native_scores18_exact": exact,
                "native_box_deltas_exact": torch.equal(actual[1], expected[1]),
                "active_source_tensors_exact": self.initialization_audit["active_source_tensors_exact"],
                "rows": 7, "unused_prompt_models_loaded": False}

    @staticmethod
    def _cpu_state(module):
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def checkpoint_state(self):
        return {"version": VERSION, "config": self.config, "sources": self.source_identity,
                "student": self._cpu_state(self.student), "teacher": self._cpu_state(self.teacher),
                "global_D": self._cpu_state(self.global_D), "ema_updates": self.ema_updates,
                "initialization_audit": self.initialization_audit}

    def load_checkpoint_state(self, state):
        if state.get("version") != VERSION:
            raise ValueError("Not a DeST checkpoint; historical VGS runs cannot be resumed")
        for key, source in self.source_identity.items():
            if state.get("sources", {}).get(key, {}).get("sha256") != source["sha256"]:
                raise ValueError("Checkpoint source identity mismatch: " + key)
        for name in ("student", "teacher", "global_D"):
            getattr(self, name).load_state_dict(state[name], strict=True)
        count = state["ema_updates"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid EMA update count")
        self.ema_updates = count
        self.train(self.training)
