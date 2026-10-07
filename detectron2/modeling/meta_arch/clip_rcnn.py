# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..poolers import ROIPooler
from .build import META_ARCH_REGISTRY

from PIL import Image
import copy
from ..backbone.fpn import build_resnet_fpn_backbone
from ..backbone.clip_backbone import build_clip_language_encoder
from detectron2.utils.comm import gather_tensors, MILCrossEntropy
from detectron2.layers import get_norm

__all__ = ["CLIPFastRCNN", "PretrainFastRCNN"]


class NonlinearRegionReshaper(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        if self.num_layers <= 1:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, dim),
                    )
                    for _ in range(self.num_layers)
                ]
            )
            self.residual_scales = nn.Parameter(
                torch.full((self.num_layers,), float(residual_scale_init))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_layers <= 1:
            return F.normalize(x + self.net(x), dim=1)
        for scale, block in zip(self.residual_scales, self.blocks):
            x = F.normalize(x + scale.to(dtype=x.dtype) * block(x), dim=1)
        return x


@META_ARCH_REGISTRY.register()
class CLIPFastRCNN(nn.Module):
    """
    Fast R-CNN style where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components: 
    1. Localization branch: pretrained backbone+RPN or equivalent modules, and is able to output object proposals
    2. Recognition branch: is able to recognize zero-shot regions
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
        language_encoder: nn.Module, 
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_c4: False,
        use_clip_attpool: False,
        offline_input_format: Optional[str] = None,
        offline_pixel_mean: Tuple[float],
        offline_pixel_std: Tuple[float],
        da_pro_enabled: bool = False,
        da_pro_loss_weight: float = 10.0,
        c3_adapter_enabled: bool = False,
        c3_adapter_hidden_dim: int = 128,
        c3_adapter_perturb_scale: float = 0.1,
        c3_adapter_quality_tau: float = 0.5,
        c3_adapter_quality_logit_temperature: float = 1.0,
        c3_adapter_quality_mode: str = "certainty_stability",
        c3_adapter_quality_use: str = "feature_map",
        c3_adapter_region_loss_weight_scale: float = 1.0,
        c3_adapter_scatter_mode: str = "max",
        c3_adapter_clamp_quality: bool = True,
        c3_adapter_freeze_backbone: bool = True,
        c3_adapter_observe_period: int = 20,
        c3_adapter_apply_residual: bool = True,
        c4_adapter_apply_residual: bool = False,
        c5_adapter_apply_residual: bool = False,
        c3_adapter_residual_scale: float = 1.0,
        c4_adapter_residual_scale: float = 1.0,
        c5_adapter_residual_scale: float = 1.0,
        c3_adapter_supervised_proj_loss_weight: float = 0.0,
        c3_adapter_supervised_proj_temperature: float = 0.01,
        c3_adapter_supervised_proj_use_proposals: bool = False,
        c3_adapter_supervised_proj_feature: str = "res3",
        c3_adapter_supervised_proj_only: bool = False,
        c3_adapter_detach_quality_map: bool = False,
        c3_adapter_freeze_quality_proj: bool = False,
        c3_adapter_train_adapter_only: bool = False,
        c3_adapter_domain_decoupled: bool = False,
        c3_adapter_domain_shared_scale: float = 5.0,
        c3_adapter_domain_specific_scale: float = 5.0,
        c3_adapter_orth_loss_weight: float = 0.0,
        c3_adapter_inference_domain: str = "target",
        c3_adapter_pooler_resolution: int = 14,
        c3_adapter_pooler_sampling_ratio: int = 0,
        c3_adapter_pooler_type: str = "ROIAlignV2",
        c3_adapter_text_emb_dim: int = 1024,
        region_reshaper_enabled: bool = False,
        region_reshaper_stage_a_only: bool = False,
        region_reshaper_train_only: bool = True,
        region_reshaper_freeze: bool = False,
        region_reshaper_hidden_dim: int = 512,
        region_reshaper_num_layers: int = 1,
        region_reshaper_residual_scale_init: float = 0.1,
        region_reshaper_source_loss_weight: float = 1.0,
        region_reshaper_target_mil_loss_weight: float = 0.2,
        region_reshaper_target_topk: int = 5,
        c5_semantic_reshaper_enabled: bool = False,
        c5_semantic_reshaper_stage_a_only: bool = False,
        c5_semantic_reshaper_train_only: bool = True,
        c5_semantic_reshaper_freeze: bool = False,
        c5_semantic_reshaper_hidden_dim: int = 512,
        c5_semantic_reshaper_num_layers: int = 1,
        c5_semantic_reshaper_residual_scale_init: float = 0.1,
        c5_semantic_reshaper_gamma_init: float = 0.0,
        c5_semantic_reshaper_source_loss_weight: float = 1.0,
        c5_semantic_reshaper_target_mil_loss_weight: float = 0.2,
        c5_semantic_reshaper_target_topk: int = 5,
        c5_semantic_reshaper_adapted_loss_weight: float = 0.2,
        c4_semantic_reshaper_enabled: bool = False,
        c4_semantic_reshaper_stage_a_only: bool = False,
        c4_semantic_reshaper_train_only: bool = True,
        c4_semantic_reshaper_freeze: bool = False,
        c4_semantic_reshaper_hidden_dim: int = 512,
        c4_semantic_reshaper_num_layers: int = 1,
        c4_semantic_reshaper_residual_scale_init: float = 0.1,
        c4_semantic_reshaper_gamma_init: float = 0.0,
        c4_semantic_reshaper_learnable_gamma: bool = True,
        c4_semantic_reshaper_source_loss_weight: float = 1.0,
        c4_semantic_reshaper_target_mil_loss_weight: float = 0.2,
        c4_semantic_reshaper_target_topk: int = 5,
        c4_semantic_reshaper_semantic_aux_loss_weight: float = 0.5,
        offline_rpn_teacher_enabled: bool = False,
        offline_rpn_teacher_loss_weight: float = 1.0,
        offline_rpn_teacher_obj_threshold: float = 0.7,
        offline_rpn_teacher_semantic_threshold: float = 0.2,
        offline_rpn_teacher_max_pseudo_boxes: int = 128,
        offline_rpn_teacher_semantic_filter: bool = True,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.offline_backbone = offline_backbone
        self.backbone = backbone
        self.lang_encoder = language_encoder
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads
        self.offline_rpn_teacher_enabled = offline_rpn_teacher_enabled
        self.offline_rpn_teacher_loss_weight = offline_rpn_teacher_loss_weight
        self.offline_rpn_teacher_obj_threshold = offline_rpn_teacher_obj_threshold
        self.offline_rpn_teacher_semantic_threshold = offline_rpn_teacher_semantic_threshold
        self.offline_rpn_teacher_max_pseudo_boxes = offline_rpn_teacher_max_pseudo_boxes
        self.offline_rpn_teacher_semantic_filter = offline_rpn_teacher_semantic_filter
        self.region_reshaper_enabled = region_reshaper_enabled
        self.region_reshaper_stage_a_only = region_reshaper_stage_a_only
        self.region_reshaper_source_loss_weight = region_reshaper_source_loss_weight
        self.region_reshaper_target_mil_loss_weight = region_reshaper_target_mil_loss_weight
        self.region_reshaper_target_topk = region_reshaper_target_topk
        self.c5_semantic_reshaper_enabled = c5_semantic_reshaper_enabled
        self.c5_semantic_reshaper_stage_a_only = c5_semantic_reshaper_stage_a_only
        self.c5_semantic_reshaper_source_loss_weight = c5_semantic_reshaper_source_loss_weight
        self.c5_semantic_reshaper_target_mil_loss_weight = c5_semantic_reshaper_target_mil_loss_weight
        self.c5_semantic_reshaper_target_topk = c5_semantic_reshaper_target_topk
        self.c5_semantic_reshaper_adapted_loss_weight = c5_semantic_reshaper_adapted_loss_weight
        self.c4_semantic_reshaper_enabled = c4_semantic_reshaper_enabled
        self.c4_semantic_reshaper_stage_a_only = c4_semantic_reshaper_stage_a_only
        self.c4_semantic_reshaper_source_loss_weight = c4_semantic_reshaper_source_loss_weight
        self.c4_semantic_reshaper_target_mil_loss_weight = c4_semantic_reshaper_target_mil_loss_weight
        self.c4_semantic_reshaper_target_topk = c4_semantic_reshaper_target_topk
        self.c4_semantic_reshaper_semantic_aux_loss_weight = c4_semantic_reshaper_semantic_aux_loss_weight
        self.c4_semantic_reshaper_learnable_gamma = c4_semantic_reshaper_learnable_gamma

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        self.c3_adapter_enabled = c3_adapter_enabled
        self.c3_adapter_perturb_scale = c3_adapter_perturb_scale
        self.c3_adapter_quality_tau = c3_adapter_quality_tau
        self.c3_adapter_quality_logit_temperature = c3_adapter_quality_logit_temperature
        self.c3_adapter_quality_mode = c3_adapter_quality_mode
        self.c3_adapter_quality_use = c3_adapter_quality_use
        self.c3_adapter_region_loss_weight_scale = c3_adapter_region_loss_weight_scale
        self.c3_adapter_scatter_mode = c3_adapter_scatter_mode
        self.c3_adapter_clamp_quality = c3_adapter_clamp_quality
        self.c3_adapter_observe_period = c3_adapter_observe_period
        self.c3_adapter_apply_residual = c3_adapter_apply_residual
        self.c4_adapter_apply_residual = c4_adapter_apply_residual
        self.c5_adapter_apply_residual = c5_adapter_apply_residual
        self.c3_adapter_residual_scale = c3_adapter_residual_scale
        self.c4_adapter_residual_scale = c4_adapter_residual_scale
        self.c5_adapter_residual_scale = c5_adapter_residual_scale
        self.c3_adapter_supervised_proj_loss_weight = c3_adapter_supervised_proj_loss_weight
        self.c3_adapter_supervised_proj_temperature = c3_adapter_supervised_proj_temperature
        self.c3_adapter_supervised_proj_use_proposals = c3_adapter_supervised_proj_use_proposals
        self.c3_adapter_supervised_proj_feature = c3_adapter_supervised_proj_feature
        self.c3_adapter_detach_quality_map = c3_adapter_detach_quality_map
        self.c3_adapter_domain_decoupled = c3_adapter_domain_decoupled
        self.c3_adapter_domain_shared_scale = c3_adapter_domain_shared_scale
        self.c3_adapter_domain_specific_scale = c3_adapter_domain_specific_scale
        self.c3_adapter_orth_loss_weight = c3_adapter_orth_loss_weight
        self.c3_adapter_inference_domain = c3_adapter_inference_domain
        self.c3_adapter_orth_losses = {}
        if self.c3_adapter_enabled:
            if not self.use_clip_c4:
                raise ValueError("C3 adapter currently supports CLIP C4 backbones only.")
            if self.c3_adapter_inference_domain not in ("source", "target"):
                raise ValueError("C3_ADAPTER.INFERENCE_DOMAIN must be 'source' or 'target'.")
            if self.c3_adapter_quality_mode not in ("certainty_stability", "certainty_only", "stability_only"):
                raise ValueError(
                    "C3_ADAPTER.QUALITY_MODE must be 'certainty_stability', 'certainty_only', or 'stability_only'."
                )
            if self.c3_adapter_quality_use not in ("feature_map", "region_loss"):
                raise ValueError(
                    "C3_ADAPTER.QUALITY_USE must be 'feature_map' or 'region_loss'."
                )
            if c3_adapter_freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            c3_channels = 512
            c4_channels = 1024
            c5_channels = 2048
            def make_adapter(channels):
                return nn.Sequential(
                    nn.Conv2d(channels, c3_adapter_hidden_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c3_adapter_hidden_dim, channels, kernel_size=1),
                )
            self.c3_adapter = nn.Sequential(
                nn.Conv2d(c3_channels, c3_adapter_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3_adapter_hidden_dim, c3_channels, kernel_size=1),
            )
            self.c4_adapter = nn.Sequential(
                nn.Conv2d(c4_channels, c3_adapter_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3_adapter_hidden_dim, c4_channels, kernel_size=1),
            )
            self.c5_adapter = nn.Sequential(
                nn.Conv2d(c5_channels, c3_adapter_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3_adapter_hidden_dim, c5_channels, kernel_size=1),
            )
            if self.c3_adapter_domain_decoupled:
                self.c3_shared_adapter = make_adapter(c3_channels)
                self.c3_source_adapter = make_adapter(c3_channels)
                self.c3_target_adapter = make_adapter(c3_channels)
                self.c4_shared_adapter = make_adapter(c4_channels)
                self.c4_source_adapter = make_adapter(c4_channels)
                self.c4_target_adapter = make_adapter(c4_channels)
                self.c5_shared_adapter = make_adapter(c5_channels)
                self.c5_source_adapter = make_adapter(c5_channels)
                self.c5_target_adapter = make_adapter(c5_channels)
            else:
                self.c3_shared_adapter = None
                self.c3_source_adapter = None
                self.c3_target_adapter = None
                self.c4_shared_adapter = None
                self.c4_source_adapter = None
                self.c4_target_adapter = None
                self.c5_shared_adapter = None
                self.c5_source_adapter = None
                self.c5_target_adapter = None
            self.c3_quality_pooler = ROIPooler(
                output_size=c3_adapter_pooler_resolution,
                scales=(1.0 / 8.0,),
                sampling_ratio=c3_adapter_pooler_sampling_ratio,
                pooler_type=c3_adapter_pooler_type,
            )
            self.c3_quality_proj = nn.Linear(c3_channels, c3_adapter_text_emb_dim, bias=False)
            self.c4_quality_pooler = ROIPooler(
                output_size=c3_adapter_pooler_resolution,
                scales=(1.0 / 16.0,),
                sampling_ratio=c3_adapter_pooler_sampling_ratio,
                pooler_type=c3_adapter_pooler_type,
            )
            self.c4_quality_proj = nn.Linear(c4_channels, c3_adapter_text_emb_dim, bias=False)
        else:
            self.c5_adapter_apply_residual = False
            self.c3_adapter = None
            self.c4_adapter = None
            self.c5_adapter = None
            self.c3_shared_adapter = None
            self.c3_source_adapter = None
            self.c3_target_adapter = None
            self.c4_shared_adapter = None
            self.c4_source_adapter = None
            self.c4_target_adapter = None
            self.c5_shared_adapter = None
            self.c5_source_adapter = None
            self.c5_target_adapter = None
            self.c3_quality_pooler = None
            self.c3_quality_proj = None
            self.c4_quality_pooler = None
            self.c4_quality_proj = None
        self.da_pro_enabled = da_pro_enabled
        if self.da_pro_enabled:
            da_in_channels = 512 if self.c3_adapter_enabled else 1024
            self.Discriminator = DAFeatDiscriminator(da_in_channels, loss_weight=da_pro_loss_weight)
            self.C4Discriminator = (
                DAFeatDiscriminator(1024, loss_weight=da_pro_loss_weight)
                if self.c3_adapter_enabled and self.c4_adapter_apply_residual
                else None
            )
            self.C5Discriminator = (
                DAFeatDiscriminator(2048, loss_weight=da_pro_loss_weight)
                if self.c3_adapter_enabled and self.c5_adapter_apply_residual
                else None
            )
        else:
            self.Discriminator = None
            self.C4Discriminator = None
            self.C5Discriminator = None
        if c3_adapter_freeze_quality_proj and self.c3_quality_proj is not None:
            for p in self.c3_quality_proj.parameters():
                p.requires_grad = False
            for p in self.c4_quality_proj.parameters():
                p.requires_grad = False
        if c3_adapter_train_adapter_only:
            if not self.c3_adapter_enabled:
                raise ValueError("TRAIN_ADAPTER_ONLY requires C3_ADAPTER.ENABLED=True.")
            for p in self.parameters():
                p.requires_grad = False
            for p in self.c3_adapter.parameters():
                p.requires_grad = True
            for p in self.c4_adapter.parameters():
                p.requires_grad = True
            for p in self.c5_adapter.parameters():
                p.requires_grad = True
            for adapter in [
                self.c3_shared_adapter,
                self.c3_source_adapter,
                self.c3_target_adapter,
                self.c4_shared_adapter,
                self.c4_source_adapter,
                self.c4_target_adapter,
                self.c5_shared_adapter,
                self.c5_source_adapter,
                self.c5_target_adapter,
            ]:
                if adapter is not None:
                    for p in adapter.parameters():
                        p.requires_grad = True
            if self.Discriminator is not None:
                for p in self.Discriminator.parameters():
                    p.requires_grad = True
            if self.C4Discriminator is not None:
                for p in self.C4Discriminator.parameters():
                    p.requires_grad = True
            if self.C5Discriminator is not None:
                for p in self.C5Discriminator.parameters():
                    p.requires_grad = True
        if c3_adapter_supervised_proj_only:
            if not self.c3_adapter_enabled:
                raise ValueError("SUPERVISED_PROJ_ONLY requires C3_ADAPTER.ENABLED=True.")
            for p in self.parameters():
                p.requires_grad = False
            supervised_proj = self.c3_quality_proj
            if self.c3_adapter_supervised_proj_feature == "res4":
                supervised_proj = self.c4_quality_proj
            elif self.c3_adapter_supervised_proj_feature != "res3":
                raise ValueError(
                    "Unknown C3_ADAPTER.SUPERVISED_PROJ_FEATURE: {}".format(
                        self.c3_adapter_supervised_proj_feature
                    )
                )

        self.region_reshaper = (
            NonlinearRegionReshaper(
                c3_adapter_text_emb_dim,
                region_reshaper_hidden_dim,
                num_layers=region_reshaper_num_layers,
                residual_scale_init=region_reshaper_residual_scale_init,
            )
            if self.region_reshaper_enabled
            else None
        )
        self.c5_semantic_projector = None
        self.c5_semantic_reshaper = None
        self.c5_semantic_back_projector = None
        self.c5_semantic_gamma = None
        self.c4_semantic_projector = None
        self.c4_semantic_reshaper = None
        self.c4_semantic_adapter = None
        self.c4_semantic_gamma = None
        self.c4_semantic_pooler = None
        self.c4_semantic_last = {}
        if self.c5_semantic_reshaper_enabled:
            self.c5_semantic_projector = nn.Linear(c3_adapter_text_emb_dim, c3_adapter_text_emb_dim, bias=False)
            self.c5_semantic_reshaper = NonlinearRegionReshaper(
                c3_adapter_text_emb_dim,
                c5_semantic_reshaper_hidden_dim,
                num_layers=c5_semantic_reshaper_num_layers,
                residual_scale_init=c5_semantic_reshaper_residual_scale_init,
            )
            self.c5_semantic_back_projector = nn.Linear(c3_adapter_text_emb_dim, c3_adapter_text_emb_dim, bias=False)
            nn.init.eye_(self.c5_semantic_projector.weight)
            nn.init.eye_(self.c5_semantic_back_projector.weight)
            self.c5_semantic_gamma = nn.Parameter(
                torch.tensor(float(c5_semantic_reshaper_gamma_init))
            )
        if self.c4_semantic_reshaper_enabled:
            c4_channels = 1024
            self.c4_semantic_projector = nn.Linear(c4_channels, c3_adapter_text_emb_dim, bias=False)
            self.c4_semantic_reshaper = NonlinearRegionReshaper(
                c3_adapter_text_emb_dim,
                c4_semantic_reshaper_hidden_dim,
                num_layers=c4_semantic_reshaper_num_layers,
                residual_scale_init=c4_semantic_reshaper_residual_scale_init,
            )
            self.c4_semantic_adapter = nn.Sequential(
                nn.Conv2d(c4_channels, c4_semantic_reshaper_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c4_semantic_reshaper_hidden_dim, c4_channels, kernel_size=1),
            )
            self.c4_semantic_pooler = ROIPooler(
                output_size=c3_adapter_pooler_resolution,
                scales=(1.0 / 16.0,),
                sampling_ratio=c3_adapter_pooler_sampling_ratio,
                pooler_type=c3_adapter_pooler_type,
            )
            gamma = torch.tensor(float(c4_semantic_reshaper_gamma_init))
            if c4_semantic_reshaper_learnable_gamma:
                self.c4_semantic_gamma = nn.Parameter(gamma)
            else:
                del self.c4_semantic_gamma
                self.register_buffer("c4_semantic_gamma", gamma)
        if region_reshaper_train_only and self.region_reshaper_enabled:
            if not self.region_reshaper_enabled:
                raise ValueError("TRAIN_RESHAPER_ONLY requires REGION_RESHAPER.ENABLED=True.")
            for p in self.parameters():
                p.requires_grad = False
            for p in self.region_reshaper.parameters():
                p.requires_grad = True
        elif region_reshaper_freeze and self.region_reshaper is not None:
            for p in self.region_reshaper.parameters():
                p.requires_grad = False
        if c5_semantic_reshaper_train_only and self.c5_semantic_reshaper_enabled:
            for p in self.parameters():
                p.requires_grad = False
            for module in [
                self.c5_semantic_projector,
                self.c5_semantic_reshaper,
                self.c5_semantic_back_projector,
            ]:
                for p in module.parameters():
                    p.requires_grad = True
            self.c5_semantic_gamma.requires_grad = True
        elif c5_semantic_reshaper_freeze and self.c5_semantic_reshaper_enabled:
            for module in [
                self.c5_semantic_projector,
                self.c5_semantic_reshaper,
                self.c5_semantic_back_projector,
            ]:
                for p in module.parameters():
                    p.requires_grad = False
            self.c5_semantic_gamma.requires_grad = False
        if c4_semantic_reshaper_train_only and self.c4_semantic_reshaper_enabled:
            for p in self.parameters():
                p.requires_grad = False
            for module in [
                self.c4_semantic_projector,
                self.c4_semantic_reshaper,
                self.c4_semantic_adapter,
            ]:
                for p in module.parameters():
                    p.requires_grad = True
            if isinstance(self.c4_semantic_gamma, nn.Parameter):
                self.c4_semantic_gamma.requires_grad = True
        elif c4_semantic_reshaper_freeze and self.c4_semantic_reshaper_enabled:
            for module in [
                self.c4_semantic_projector,
                self.c4_semantic_reshaper,
                self.c4_semantic_adapter,
            ]:
                for p in module.parameters():
                    p.requires_grad = False
            if isinstance(self.c4_semantic_gamma, nn.Parameter):
                self.c4_semantic_gamma.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        # create independent backbone & RPN
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN": 
            # create offline cfg for the pretrained backbone & RPN
            from detectron2.config import get_cfg
            offline_cfg = get_cfg()
            offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
            if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
                offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
                offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
                offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
                offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9
            if cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST:
                offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000

            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg)
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

            if cfg.MODEL.OFFLINE_RPN_TEACHER.ENABLED:
                for p in offline_backbone.parameters(): p.requires_grad = True
                for p in offline_rpn.parameters(): p.requires_grad = True
                offline_backbone.train()
                offline_rpn.train()
            else:
                # convert to evaluation mode
                for p in offline_backbone.parameters(): p.requires_grad = False
                for p in offline_rpn.parameters(): p.requires_grad = False
                offline_backbone.eval()
                offline_rpn.eval()
        # region proposals are ground-truth boxes
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None
        
        backbone = build_backbone(cfg)
        # build language encoder
        if cfg.MODEL.CLIP.GET_CONCEPT_EMB: # extract concept embeddings
            language_encoder = build_clip_language_encoder(cfg)
        else:
            language_encoder = None
        roi_heads = build_roi_heads(cfg, backbone.output_shape())

        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "backbone": backbone,
            "language_encoder": language_encoder, 
            "roi_heads": roi_heads, 
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            "da_pro_enabled": cfg.MODEL.DA_PRO.ENABLED,
            "da_pro_loss_weight": cfg.MODEL.DA_PRO.LOSS_WEIGHT,
            "c3_adapter_enabled": cfg.MODEL.C3_ADAPTER.ENABLED,
            "c3_adapter_hidden_dim": cfg.MODEL.C3_ADAPTER.HIDDEN_DIM,
            "c3_adapter_perturb_scale": cfg.MODEL.C3_ADAPTER.PERTURB_SCALE,
            "c3_adapter_quality_tau": cfg.MODEL.C3_ADAPTER.QUALITY_TAU,
            "c3_adapter_quality_logit_temperature": cfg.MODEL.C3_ADAPTER.QUALITY_LOGIT_TEMPERATURE,
            "c3_adapter_quality_mode": cfg.MODEL.C3_ADAPTER.QUALITY_MODE,
            "c3_adapter_quality_use": cfg.MODEL.C3_ADAPTER.QUALITY_USE,
            "c3_adapter_region_loss_weight_scale": cfg.MODEL.C3_ADAPTER.REGION_LOSS_WEIGHT_SCALE,
            "c3_adapter_scatter_mode": cfg.MODEL.C3_ADAPTER.SCATTER_MODE,
            "c3_adapter_clamp_quality": cfg.MODEL.C3_ADAPTER.CLAMP_QUALITY,
            "c3_adapter_freeze_backbone": cfg.MODEL.C3_ADAPTER.FREEZE_BACKBONE,
            "c3_adapter_observe_period": cfg.MODEL.C3_ADAPTER.OBSERVE_PERIOD,
            "c3_adapter_apply_residual": cfg.MODEL.C3_ADAPTER.APPLY_RESIDUAL,
            "c4_adapter_apply_residual": cfg.MODEL.C3_ADAPTER.C4_APPLY_RESIDUAL,
            "c5_adapter_apply_residual": cfg.MODEL.C3_ADAPTER.C5_APPLY_RESIDUAL,
            "c3_adapter_residual_scale": cfg.MODEL.C3_ADAPTER.RESIDUAL_SCALE,
            "c4_adapter_residual_scale": cfg.MODEL.C3_ADAPTER.C4_RESIDUAL_SCALE,
            "c5_adapter_residual_scale": cfg.MODEL.C3_ADAPTER.C5_RESIDUAL_SCALE,
            "c3_adapter_supervised_proj_loss_weight": cfg.MODEL.C3_ADAPTER.SUPERVISED_PROJ_LOSS_WEIGHT,
            "c3_adapter_supervised_proj_temperature": cfg.MODEL.C3_ADAPTER.SUPERVISED_PROJ_TEMPERATURE,
            "c3_adapter_supervised_proj_use_proposals": cfg.MODEL.C3_ADAPTER.SUPERVISED_PROJ_USE_PROPOSALS,
            "c3_adapter_supervised_proj_feature": cfg.MODEL.C3_ADAPTER.SUPERVISED_PROJ_FEATURE,
            "c3_adapter_supervised_proj_only": cfg.MODEL.C3_ADAPTER.SUPERVISED_PROJ_ONLY,
            "c3_adapter_detach_quality_map": cfg.MODEL.C3_ADAPTER.DETACH_QUALITY_MAP,
            "c3_adapter_freeze_quality_proj": cfg.MODEL.C3_ADAPTER.FREEZE_QUALITY_PROJ,
            "c3_adapter_train_adapter_only": cfg.MODEL.C3_ADAPTER.TRAIN_ADAPTER_ONLY,
            "c3_adapter_domain_decoupled": cfg.MODEL.C3_ADAPTER.DOMAIN_DECOUPLED,
            "c3_adapter_domain_shared_scale": cfg.MODEL.C3_ADAPTER.DOMAIN_SHARED_SCALE,
            "c3_adapter_domain_specific_scale": cfg.MODEL.C3_ADAPTER.DOMAIN_SPECIFIC_SCALE,
            "c3_adapter_orth_loss_weight": cfg.MODEL.C3_ADAPTER.ORTH_LOSS_WEIGHT,
            "c3_adapter_inference_domain": cfg.MODEL.C3_ADAPTER.INFERENCE_DOMAIN,
            "c3_adapter_pooler_resolution": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            "c3_adapter_pooler_sampling_ratio": cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            "c3_adapter_pooler_type": cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
            "c3_adapter_text_emb_dim": cfg.MODEL.CLIP.TEXT_EMB_DIM,
            "region_reshaper_enabled": cfg.MODEL.REGION_RESHAPER.ENABLED,
            "region_reshaper_stage_a_only": cfg.MODEL.REGION_RESHAPER.STAGE_A_ONLY,
            "region_reshaper_train_only": cfg.MODEL.REGION_RESHAPER.TRAIN_RESHAPER_ONLY,
            "region_reshaper_freeze": cfg.MODEL.REGION_RESHAPER.FREEZE,
            "region_reshaper_hidden_dim": cfg.MODEL.REGION_RESHAPER.HIDDEN_DIM,
            "region_reshaper_num_layers": cfg.MODEL.REGION_RESHAPER.NUM_LAYERS,
            "region_reshaper_residual_scale_init": cfg.MODEL.REGION_RESHAPER.RESIDUAL_SCALE_INIT,
            "region_reshaper_source_loss_weight": cfg.MODEL.REGION_RESHAPER.SOURCE_LOSS_WEIGHT,
            "region_reshaper_target_mil_loss_weight": cfg.MODEL.REGION_RESHAPER.TARGET_MIL_LOSS_WEIGHT,
            "region_reshaper_target_topk": cfg.MODEL.REGION_RESHAPER.TARGET_TOPK,
            "c5_semantic_reshaper_enabled": cfg.MODEL.C5_SEMANTIC_RESHAPER.ENABLED,
            "c5_semantic_reshaper_stage_a_only": cfg.MODEL.C5_SEMANTIC_RESHAPER.STAGE_A_ONLY,
            "c5_semantic_reshaper_train_only": cfg.MODEL.C5_SEMANTIC_RESHAPER.TRAIN_ONLY,
            "c5_semantic_reshaper_freeze": cfg.MODEL.C5_SEMANTIC_RESHAPER.FREEZE,
            "c5_semantic_reshaper_hidden_dim": cfg.MODEL.C5_SEMANTIC_RESHAPER.HIDDEN_DIM,
            "c5_semantic_reshaper_num_layers": cfg.MODEL.C5_SEMANTIC_RESHAPER.NUM_LAYERS,
            "c5_semantic_reshaper_residual_scale_init": cfg.MODEL.C5_SEMANTIC_RESHAPER.RESIDUAL_SCALE_INIT,
            "c5_semantic_reshaper_gamma_init": cfg.MODEL.C5_SEMANTIC_RESHAPER.GAMMA_INIT,
            "c5_semantic_reshaper_source_loss_weight": cfg.MODEL.C5_SEMANTIC_RESHAPER.SOURCE_LOSS_WEIGHT,
            "c5_semantic_reshaper_target_mil_loss_weight": cfg.MODEL.C5_SEMANTIC_RESHAPER.TARGET_MIL_LOSS_WEIGHT,
            "c5_semantic_reshaper_target_topk": cfg.MODEL.C5_SEMANTIC_RESHAPER.TARGET_TOPK,
            "c5_semantic_reshaper_adapted_loss_weight": cfg.MODEL.C5_SEMANTIC_RESHAPER.ADAPTED_LOSS_WEIGHT,
            "c4_semantic_reshaper_enabled": cfg.MODEL.C4_SEMANTIC_RESHAPER.ENABLED,
            "c4_semantic_reshaper_stage_a_only": cfg.MODEL.C4_SEMANTIC_RESHAPER.STAGE_A_ONLY,
            "c4_semantic_reshaper_train_only": cfg.MODEL.C4_SEMANTIC_RESHAPER.TRAIN_ONLY,
            "c4_semantic_reshaper_freeze": cfg.MODEL.C4_SEMANTIC_RESHAPER.FREEZE,
            "c4_semantic_reshaper_hidden_dim": cfg.MODEL.C4_SEMANTIC_RESHAPER.HIDDEN_DIM,
            "c4_semantic_reshaper_num_layers": cfg.MODEL.C4_SEMANTIC_RESHAPER.NUM_LAYERS,
            "c4_semantic_reshaper_residual_scale_init": cfg.MODEL.C4_SEMANTIC_RESHAPER.RESIDUAL_SCALE_INIT,
            "c4_semantic_reshaper_gamma_init": cfg.MODEL.C4_SEMANTIC_RESHAPER.GAMMA_INIT,
            "c4_semantic_reshaper_learnable_gamma": cfg.MODEL.C4_SEMANTIC_RESHAPER.LEARNABLE_GAMMA,
            "c4_semantic_reshaper_source_loss_weight": cfg.MODEL.C4_SEMANTIC_RESHAPER.SOURCE_LOSS_WEIGHT,
            "c4_semantic_reshaper_target_mil_loss_weight": cfg.MODEL.C4_SEMANTIC_RESHAPER.TARGET_MIL_LOSS_WEIGHT,
            "c4_semantic_reshaper_target_topk": cfg.MODEL.C4_SEMANTIC_RESHAPER.TARGET_TOPK,
            "c4_semantic_reshaper_semantic_aux_loss_weight": cfg.MODEL.C4_SEMANTIC_RESHAPER.SEMANTIC_AUX_LOSS_WEIGHT,
            "offline_rpn_teacher_enabled": cfg.MODEL.OFFLINE_RPN_TEACHER.ENABLED,
            "offline_rpn_teacher_loss_weight": cfg.MODEL.OFFLINE_RPN_TEACHER.LOSS_WEIGHT,
            "offline_rpn_teacher_obj_threshold": cfg.MODEL.OFFLINE_RPN_TEACHER.OBJ_THRESHOLD,
            "offline_rpn_teacher_semantic_threshold": cfg.MODEL.OFFLINE_RPN_TEACHER.SEMANTIC_THRESHOLD,
            "offline_rpn_teacher_max_pseudo_boxes": cfg.MODEL.OFFLINE_RPN_TEACHER.MAX_PSEUDO_BOXES,
            "offline_rpn_teacher_semantic_filter": cfg.MODEL.OFFLINE_RPN_TEACHER.SEMANTIC_FILTER,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        is_source=False,
        image_level_only=False,
        return_image_level=False,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if image_level_only:
            return self.image_level_predictions(batched_inputs, is_source=is_source)
        if not self.training:
            return self.inference(batched_inputs)
        if self.c4_semantic_reshaper_stage_a_only:
            return self.c4_semantic_reshaper_stage_a_losses(batched_inputs, is_source=is_source)
        if self.c5_semantic_reshaper_stage_a_only:
            return self.c5_semantic_reshaper_stage_a_losses(batched_inputs, is_source=is_source)
        if self.region_reshaper_stage_a_only:
            return self.region_reshaper_stage_a_losses(batched_inputs, is_source=is_source)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        # localization branch: offline modules to get the region proposals
        with torch.no_grad():  
            if self.clip_crop_region_type == "GT":  # from ground-truth
                proposals = []
                for r_i, b_input in enumerate(batched_inputs): 
                    this_gt = copy.deepcopy(b_input["instances"])  # Instance
                    gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                    this_gt._fields = {'proposal_boxes': gt_boxes, 'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)}
                    proposals.append(this_gt)                
            elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
                if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
                    self.offline_backbone.eval() 
                    self.offline_proposal_generator.eval()  
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.recognition_features(images, proposals, is_source=is_source)
        loss_c3_proj = None
        if (
            self.c3_adapter_enabled
            and self.c3_adapter_supervised_proj_loss_weight > 0
            and gt_instances is not None
            and (is_source or not self.da_pro_enabled)
        ):
            loss_c3_proj = (
                self.c3_supervised_projection_loss(
                    features[self.c3_adapter_supervised_proj_feature],
                    gt_instances,
                    proposals,
                    feature_name=self.c3_adapter_supervised_proj_feature,
                )
                * self.c3_adapter_supervised_proj_loss_weight
            )

        if self.da_pro_enabled:
            da_feature_name = "res3" if self.c3_adapter_enabled else "res4"
            da_features = self.c3_adapter_shared_features.get(da_feature_name, features[da_feature_name])
            loss_dis_0, loss_dis_1 = self.Discriminator.loss(da_features)
            if self.C4Discriminator is not None:
                c4_da_features = self.c3_adapter_shared_features.get("res4", features["res4"])
                loss_dis_c4_0, loss_dis_c4_1 = self.C4Discriminator.loss(c4_da_features)


        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                roi_outputs = self.roi_heads(
                    images,
                    features,
                    proposals,
                    gt_instances,
                    res5=self.backbone.layer4,
                    attnpool=self.backbone.attnpool,
                    c5_adapter_fn=self.c5_adapter_roi_features if self.c5_adapter_apply_residual else None,
                    region_loss_quality_fn=self.region_loss_quality if self.use_region_loss_quality() else None,
                    c5_discriminator=self.C5Discriminator,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                    is_source=is_source,
                    return_logits=return_image_level,
                )
            else: # use mean pool
                roi_outputs = self.roi_heads(
                    images,
                    features,
                    proposals,
                    gt_instances,
                    res5=self.backbone.layer4,
                    c5_adapter_fn=self.c5_adapter_roi_features if self.c5_adapter_apply_residual else None,
                    region_loss_quality_fn=self.region_loss_quality if self.use_region_loss_quality() else None,
                    c5_discriminator=self.C5Discriminator,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                    is_source=is_source,
                    return_logits=return_image_level,
                )
        else:  # regular detector setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                roi_outputs = self.roi_heads(
                    images,
                    features,
                    proposals,
                    gt_instances,
                    attnpool=self.backbone.bottom_up.attnpool,
                    region_loss_quality_fn=self.region_loss_quality if self.use_region_loss_quality() else None,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                    is_source=is_source,
                    return_logits=return_image_level,
                )
            else: # use mean pool
                roi_outputs = self.roi_heads(
                    images,
                    features,
                    proposals,
                    gt_instances,
                    region_loss_quality_fn=self.region_loss_quality if self.use_region_loss_quality() else None,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                    is_source=is_source,
                    return_logits=return_image_level,
                )
        if return_image_level:
            _, detector_losses, roi_logits, objectness = roi_outputs
        else:
            _, detector_losses = roi_outputs
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        #visualize_proposals(batched_inputs, proposals, self.input_format)

        losses = {}
        losses.update(detector_losses)
        if loss_c3_proj is not None:
            losses.update({"loss_c3_proj": loss_c3_proj})
        if self.da_pro_enabled:
            losses.update({'loss_dis_0': loss_dis_0})
            losses.update({'loss_dis_1': loss_dis_1})
            if self.C4Discriminator is not None:
                losses.update({'loss_dis_c4_0': loss_dis_c4_0})
                losses.update({'loss_dis_c4_1': loss_dis_c4_1})
        if self.c3_adapter_domain_decoupled and self.c3_adapter_orth_loss_weight > 0:
            for name, loss in self.c3_adapter_orth_losses.items():
                losses[f"loss_orth_{name}"] = loss * self.c3_adapter_orth_loss_weight
        if return_image_level:
            image_probs = [
                self.h2fa_image_level_aggregate(x, o)
                for x, o in zip(roi_logits, objectness)
            ]
            return losses, image_probs
        return losses

    def region_reshaper_zero_loss(self):
        if self.region_reshaper is None:
            return self.pixel_mean.sum() * 0.0
        return sum(p.sum() for p in self.region_reshaper.parameters()) * 0.0

    def c5_semantic_zero_loss(self):
        if not self.c5_semantic_reshaper_enabled:
            return self.pixel_mean.sum() * 0.0
        zero = self.c5_semantic_gamma.sum() * 0.0
        for module in [
            self.c5_semantic_projector,
            self.c5_semantic_reshaper,
            self.c5_semantic_back_projector,
        ]:
            zero = zero + sum(p.sum() for p in module.parameters()) * 0.0
        return zero

    def active_region_reshaper_fn(self):
        if self.c5_semantic_reshaper_enabled and not self.c5_semantic_reshaper_stage_a_only:
            return self.c5_semantic_apply
        if self.region_reshaper_enabled:
            return self.region_reshaper
        return None

    def maybe_apply_active_region_reshaper(self, embeddings: torch.Tensor) -> torch.Tensor:
        fn = self.active_region_reshaper_fn()
        return fn(embeddings) if fn is not None else embeddings

    def c5_semantic_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings.new_zeros((0, self.roi_heads.num_classes))
        text_weight = self.roi_heads.box_predictor.cls_score.weight[: self.roi_heads.num_classes]
        logits = F.normalize(embeddings, dim=1) @ F.normalize(text_weight, dim=1).t()
        temperature = getattr(self.roi_heads.box_predictor, "temperature", 1.0)
        return logits / temperature

    def c5_semantic_certainty(self, semantic_embeddings: torch.Tensor) -> torch.Tensor:
        if semantic_embeddings.numel() == 0:
            return semantic_embeddings.new_zeros((0, 1))
        logits = self.c5_semantic_logits(semantic_embeddings)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
        effective_classes = entropy.exp()
        denom = max(self.roi_heads.num_classes - 1, 1)
        q = (self.roi_heads.num_classes - effective_classes) / denom
        return q.clamp(0.0, 1.0).view(-1, 1)

    def c5_semantic_components(self, embeddings: torch.Tensor):
        semantic = self.c5_semantic_projector(embeddings)
        reshaped_semantic = self.c5_semantic_reshaper(semantic)
        q = self.c5_semantic_certainty(reshaped_semantic).detach()
        residual = self.c5_semantic_back_projector(reshaped_semantic - semantic)
        gamma = self.c5_semantic_gamma.to(dtype=embeddings.dtype)
        adapted = embeddings + gamma * q.to(dtype=embeddings.dtype) * residual
        return semantic, reshaped_semantic, adapted, q

    def c5_semantic_apply(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings
        _, _, adapted, _ = self.c5_semantic_components(embeddings)
        return adapted

    def c5_semantic_target_mil_loss(self, logits, proposals, batched_inputs, prefix):
        labels = self.region_reshaper_image_labels(batched_inputs)
        counts = [len(p) for p in proposals]
        per_image_logits = logits.split(counts, dim=0)
        losses = []
        positive_scores = []
        negative_scores = []
        for img_logits, img_labels in zip(per_image_logits, labels):
            if img_logits.numel() == 0:
                image_logits = img_labels.new_zeros(img_labels.shape)
            else:
                k = min(max(int(self.c5_semantic_reshaper_target_topk), 1), img_logits.shape[0])
                image_logits = img_logits.topk(k, dim=0).values.mean(dim=0)
            losses.append(F.binary_cross_entropy_with_logits(image_logits, img_labels, reduction="mean"))
            if img_labels.sum() > 0:
                positive_scores.append(image_logits[img_labels > 0].detach().sigmoid().mean())
            if (img_labels == 0).any():
                negative_scores.append(image_logits[img_labels == 0].detach().sigmoid().mean())
        storage = get_event_storage()
        storage.put_scalar(
            f"{prefix}/target_labels_per_image",
            float(torch.stack([x.sum() for x in labels]).mean().item()) if labels else 0.0,
            smoothing_hint=False,
        )
        if positive_scores:
            storage.put_scalar(
                f"{prefix}/target_pos_prob",
                float(torch.stack(positive_scores).mean().item()),
                smoothing_hint=False,
            )
        if negative_scores:
            storage.put_scalar(
                f"{prefix}/target_neg_prob",
                float(torch.stack(negative_scores).mean().item()),
                smoothing_hint=False,
            )
        if not losses:
            return self.c5_semantic_zero_loss()
        return torch.stack(losses).mean()

    def c5_semantic_observe(self, reshaped_semantic, adapted, base_embeddings, q, prefix):
        storage = get_event_storage()
        storage.put_scalar(f"{prefix}/q_mean", float(q.mean().item()) if q.numel() else 0.0, smoothing_hint=False)
        storage.put_scalar(f"{prefix}/q_max", float(q.max().item()) if q.numel() else 0.0, smoothing_hint=False)
        storage.put_scalar(f"{prefix}/gamma", float(self.c5_semantic_gamma.detach().item()), smoothing_hint=False)
        if base_embeddings.numel() == 0:
            return
        with torch.no_grad():
            delta = adapted - base_embeddings
            storage.put_scalar(
                f"{prefix}/adapt_delta_norm",
                float(delta.float().norm(dim=1).mean().item()),
                smoothing_hint=False,
            )
            storage.put_scalar(
                f"{prefix}/semantic_norm",
                float(reshaped_semantic.float().norm(dim=1).mean().item()),
                smoothing_hint=False,
            )

    def c5_semantic_reshaper_stage_a_losses(self, batched_inputs, is_source: bool):
        if not self.c5_semantic_reshaper_enabled:
            raise ValueError(
                "C5_SEMANTIC_RESHAPER.STAGE_A_ONLY requires C5_SEMANTIC_RESHAPER.ENABLED=True."
            )
        proposals = self.normal_region_proposals(batched_inputs)
        if is_source:
            if "instances" not in batched_inputs[0]:
                return {"loss_c5_sem_src_ce": self.c5_semantic_zero_loss()}
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            with torch.no_grad():
                proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
            embeddings = self.region_reshaper_embeddings(batched_inputs, proposals, is_source=True)
            _, reshaped_semantic, adapted, q = self.c5_semantic_components(embeddings)
            semantic_logits = self.c5_semantic_logits(reshaped_semantic)
            adapted_logits = self.c5_semantic_logits(adapted)
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if proposals else semantic_logits.new_zeros((0,), dtype=torch.long)
            fg = (gt_classes >= 0) & (gt_classes < self.roi_heads.num_classes)
            self.c5_semantic_observe(reshaped_semantic, adapted, embeddings, q, "c5_semantic_reshaper/source")
            if fg.any():
                src_loss = F.cross_entropy(semantic_logits[fg], gt_classes[fg].long())
                if self.c5_semantic_reshaper_adapted_loss_weight > 0:
                    src_loss = src_loss + self.c5_semantic_reshaper_adapted_loss_weight * F.cross_entropy(
                        adapted_logits[fg],
                        gt_classes[fg].long(),
                    )
                loss = src_loss * self.c5_semantic_reshaper_source_loss_weight
                pred = semantic_logits[fg].argmax(dim=1)
                acc = (pred == gt_classes[fg]).float().mean()
                gt_score = semantic_logits[fg].gather(1, gt_classes[fg].long().view(-1, 1)).squeeze(1)
                masked_logits = semantic_logits[fg].clone()
                masked_logits[torch.arange(masked_logits.shape[0], device=masked_logits.device), gt_classes[fg].long()] = -float("inf")
                margin = (gt_score - masked_logits.max(dim=1).values).mean()
                storage = get_event_storage()
                storage.put_scalar("c5_semantic_reshaper/source_pos", float(fg.sum().item()), smoothing_hint=False)
                storage.put_scalar("c5_semantic_reshaper/source_acc", float(acc.item()), smoothing_hint=False)
                storage.put_scalar("c5_semantic_reshaper/source_margin", float(margin.item()), smoothing_hint=False)
            else:
                loss = self.c5_semantic_zero_loss()
                get_event_storage().put_scalar("c5_semantic_reshaper/source_pos", 0.0, smoothing_hint=False)
            return {"loss_c5_sem_src_ce": loss}

        embeddings = self.region_reshaper_embeddings(batched_inputs, proposals, is_source=False)
        _, reshaped_semantic, adapted, q = self.c5_semantic_components(embeddings)
        semantic_logits = self.c5_semantic_logits(reshaped_semantic)
        adapted_logits = self.c5_semantic_logits(adapted)
        self.c5_semantic_observe(reshaped_semantic, adapted, embeddings, q, "c5_semantic_reshaper/target")
        loss = self.c5_semantic_target_mil_loss(
            semantic_logits,
            proposals,
            batched_inputs,
            "c5_semantic_reshaper",
        )
        if self.c5_semantic_reshaper_adapted_loss_weight > 0:
            loss = loss + self.c5_semantic_reshaper_adapted_loss_weight * self.c5_semantic_target_mil_loss(
                adapted_logits,
                proposals,
                batched_inputs,
                "c5_semantic_reshaper_adapted",
            )
        return {"loss_c5_sem_tgt_mil": loss * self.c5_semantic_reshaper_target_mil_loss_weight}

    def c4_semantic_zero_loss(self):
        if not self.c4_semantic_reshaper_enabled:
            return self.pixel_mean.sum() * 0.0
        zero = self.c4_semantic_gamma.sum() * 0.0
        for module in [
            self.c4_semantic_projector,
            self.c4_semantic_reshaper,
            self.c4_semantic_adapter,
        ]:
            zero = zero + sum(p.sum() for p in module.parameters()) * 0.0
        return zero

    def c4_semantic_feature(self, res4: torch.Tensor, proposals: List[Instances]) -> torch.Tensor:
        if not self.c4_semantic_reshaper_enabled or sum(len(p) for p in proposals) == 0:
            return res4
        boxes = [p.proposal_boxes for p in proposals]
        roi = self.c4_semantic_pooler([res4], boxes)
        roi_vec = roi.mean(dim=[2, 3])
        semantic = self.c4_semantic_projector(roi_vec)
        reshaped = self.c4_semantic_reshaper(semantic)
        q = self.c5_semantic_certainty(reshaped).detach().flatten()
        q_map = self.scatter_region_quality_to_feature(q, proposals, res4.shape[-2:], stride=16.0)
        q_map = q_map.to(dtype=res4.dtype)
        gamma = self.c4_semantic_gamma.to(dtype=res4.dtype)
        adapted = res4 + gamma * q_map * self.c4_semantic_adapter(res4)
        self.c4_semantic_last = {
            "semantic": semantic,
            "reshaped": reshaped,
            "q": q,
            "q_map": q_map,
        }
        return adapted

    def c4_semantic_embeddings(self, batched_inputs, proposals, is_source: bool):
        images = self.preprocess_image(batched_inputs)
        features = self.recognition_features(images, proposals, is_source=is_source)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        if sum(len(x) for x in proposal_boxes) == 0:
            return self.pixel_mean.new_zeros((0, self.roi_heads.box_predictor.cls_score.weight.shape[1]))
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features],
            proposal_boxes,
            self.backbone.layer4,
        )
        if self.use_clip_attpool:
            embeddings = self.backbone.attnpool(box_features)
        else:
            embeddings = box_features.mean(dim=[2, 3])
        return self.maybe_apply_active_region_reshaper(embeddings)

    def c4_semantic_target_mil_loss(self, logits, proposals, batched_inputs, prefix):
        labels = self.region_reshaper_image_labels(batched_inputs)
        counts = [len(p) for p in proposals]
        per_image_logits = logits.split(counts, dim=0)
        losses = []
        positive_scores = []
        negative_scores = []
        for img_logits, img_labels in zip(per_image_logits, labels):
            if img_logits.numel() == 0:
                image_logits = img_labels.new_zeros(img_labels.shape)
            else:
                k = min(max(int(self.c4_semantic_reshaper_target_topk), 1), img_logits.shape[0])
                image_logits = img_logits.topk(k, dim=0).values.mean(dim=0)
            losses.append(F.binary_cross_entropy_with_logits(image_logits, img_labels, reduction="mean"))
            if img_labels.sum() > 0:
                positive_scores.append(image_logits[img_labels > 0].detach().sigmoid().mean())
            if (img_labels == 0).any():
                negative_scores.append(image_logits[img_labels == 0].detach().sigmoid().mean())
        storage = get_event_storage()
        storage.put_scalar(
            f"{prefix}/target_labels_per_image",
            float(torch.stack([x.sum() for x in labels]).mean().item()) if labels else 0.0,
            smoothing_hint=False,
        )
        if positive_scores:
            storage.put_scalar(f"{prefix}/target_pos_prob", float(torch.stack(positive_scores).mean().item()), smoothing_hint=False)
        if negative_scores:
            storage.put_scalar(f"{prefix}/target_neg_prob", float(torch.stack(negative_scores).mean().item()), smoothing_hint=False)
        if not losses:
            return self.c4_semantic_zero_loss()
        return torch.stack(losses).mean()

    def c4_semantic_observe(self, prefix):
        storage = get_event_storage()
        q = self.c4_semantic_last.get("q")
        q_map = self.c4_semantic_last.get("q_map")
        storage.put_scalar(f"{prefix}/gamma", float(self.c4_semantic_gamma.detach().item()), smoothing_hint=False)
        if q is not None and q.numel():
            storage.put_scalar(f"{prefix}/q_mean", float(q.mean().item()), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_max", float(q.max().item()), smoothing_hint=False)
        if q_map is not None and q_map.numel():
            storage.put_scalar(f"{prefix}/q_map_mean", float(q_map.mean().item()), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_map_nonzero", float((q_map > 0).float().mean().item()), smoothing_hint=False)

    def c4_semantic_reshaper_stage_a_losses(self, batched_inputs, is_source: bool):
        if not self.c4_semantic_reshaper_enabled:
            raise ValueError(
                "C4_SEMANTIC_RESHAPER.STAGE_A_ONLY requires C4_SEMANTIC_RESHAPER.ENABLED=True."
            )
        proposals = self.normal_region_proposals(batched_inputs)
        if is_source:
            if "instances" not in batched_inputs[0]:
                return {"loss_c4_sem_src_ce": self.c4_semantic_zero_loss()}
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            with torch.no_grad():
                proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
            embeddings = self.c4_semantic_embeddings(batched_inputs, proposals, is_source=True)
            final_logits = self.c5_semantic_logits(embeddings)
            aux_logits = self.c5_semantic_logits(self.c4_semantic_last["reshaped"])
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if proposals else final_logits.new_zeros((0,), dtype=torch.long)
            fg = (gt_classes >= 0) & (gt_classes < self.roi_heads.num_classes)
            self.c4_semantic_observe("c4_semantic_reshaper/source")
            if fg.any():
                src_loss = F.cross_entropy(final_logits[fg], gt_classes[fg].long())
                src_loss = src_loss + self.c4_semantic_reshaper_semantic_aux_loss_weight * F.cross_entropy(
                    aux_logits[fg],
                    gt_classes[fg].long(),
                )
                loss = src_loss * self.c4_semantic_reshaper_source_loss_weight
                pred = final_logits[fg].argmax(dim=1)
                acc = (pred == gt_classes[fg]).float().mean()
                gt_score = final_logits[fg].gather(1, gt_classes[fg].long().view(-1, 1)).squeeze(1)
                masked_logits = final_logits[fg].clone()
                masked_logits[torch.arange(masked_logits.shape[0], device=masked_logits.device), gt_classes[fg].long()] = -float("inf")
                margin = (gt_score - masked_logits.max(dim=1).values).mean()
                storage = get_event_storage()
                storage.put_scalar("c4_semantic_reshaper/source_pos", float(fg.sum().item()), smoothing_hint=False)
                storage.put_scalar("c4_semantic_reshaper/source_acc", float(acc.item()), smoothing_hint=False)
                storage.put_scalar("c4_semantic_reshaper/source_margin", float(margin.item()), smoothing_hint=False)
            else:
                loss = self.c4_semantic_zero_loss()
                get_event_storage().put_scalar("c4_semantic_reshaper/source_pos", 0.0, smoothing_hint=False)
            return {"loss_c4_sem_src_ce": loss}

        embeddings = self.c4_semantic_embeddings(batched_inputs, proposals, is_source=False)
        final_logits = self.c5_semantic_logits(embeddings)
        aux_logits = self.c5_semantic_logits(self.c4_semantic_last["reshaped"])
        self.c4_semantic_observe("c4_semantic_reshaper/target")
        loss = self.c4_semantic_target_mil_loss(final_logits, proposals, batched_inputs, "c4_semantic_reshaper")
        loss = loss + self.c4_semantic_reshaper_semantic_aux_loss_weight * self.c4_semantic_target_mil_loss(
            aux_logits,
            proposals,
            batched_inputs,
            "c4_semantic_reshaper_aux",
        )
        return {"loss_c4_sem_tgt_mil": loss * self.c4_semantic_reshaper_target_mil_loss_weight}

    def normal_region_proposals(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        with torch.no_grad():
            if self.clip_crop_region_type == "GT":
                proposals = []
                for b_input in batched_inputs:
                    this_gt = copy.deepcopy(b_input["instances"])
                    gt_boxes = this_gt._fields["gt_boxes"].to(self.device)
                    this_gt._fields = {
                        "proposal_boxes": gt_boxes,
                        "objectness_logits": torch.ones(gt_boxes.tensor.size(0), device=self.device),
                    }
                    proposals.append(this_gt)
                return proposals
            if self.clip_crop_region_type != "RPN":
                raise ValueError("Region reshaper Stage A currently expects RPN or GT proposals.")
            if self.offline_backbone.training or self.offline_proposal_generator.training:
                self.offline_backbone.eval()
                self.offline_proposal_generator.eval()
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            proposals, _ = self.offline_proposal_generator(images, features, None)
            return proposals

    def region_reshaper_embeddings(self, batched_inputs, proposals, is_source: bool):
        with torch.no_grad():
            images = self.preprocess_image(batched_inputs)
            features = self.recognition_features(images, proposals, is_source=is_source)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            if sum(len(x) for x in proposal_boxes) == 0:
                return self.pixel_mean.new_zeros((0, self.roi_heads.box_predictor.cls_score.weight.shape[1]))
            if self.use_clip_c4:
                box_features = self.roi_heads._shared_roi_transform(
                    [features[f] for f in self.roi_heads.in_features],
                    proposal_boxes,
                    self.backbone.layer4,
                )
                if self.use_clip_attpool:
                    return self.maybe_apply_active_region_reshaper(self.backbone.attnpool(box_features))
                return self.maybe_apply_active_region_reshaper(box_features.mean(dim=[2, 3]))
            if self.use_clip_attpool:
                box_features = self.roi_heads._shared_roi_transform(
                    [features[f] for f in self.roi_heads.in_features],
                    proposal_boxes,
                    None,
                )
                return self.maybe_apply_active_region_reshaper(self.backbone.bottom_up.attnpool(box_features))
            box_features = self.roi_heads._shared_roi_transform(
                [features[f] for f in self.roi_heads.in_features],
                proposal_boxes,
                None,
            )
            return self.maybe_apply_active_region_reshaper(box_features.mean(dim=[2, 3]))

    def region_reshaper_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings.new_zeros((0, self.roi_heads.num_classes))
        reshaped = self.region_reshaper(embeddings)
        text_weight = self.roi_heads.box_predictor.cls_score.weight[: self.roi_heads.num_classes]
        logits = reshaped @ F.normalize(text_weight, dim=1).t()
        temperature = getattr(self.roi_heads.box_predictor, "temperature", 1.0)
        return logits / temperature

    def region_reshaper_image_labels(self, batched_inputs):
        labels = []
        num_classes = self.roi_heads.num_classes
        for sample in batched_inputs:
            y = self.pixel_mean.new_zeros((num_classes,))
            if "instances" in sample:
                gt_classes = sample["instances"].gt_classes.to(self.device)
                keep = (gt_classes >= 0) & (gt_classes < num_classes)
                if keep.any():
                    y[gt_classes[keep].long()] = 1.0
            labels.append(y)
        return labels

    def region_reshaper_target_mil_loss(self, logits, proposals, batched_inputs):
        labels = self.region_reshaper_image_labels(batched_inputs)
        counts = [len(p) for p in proposals]
        per_image_logits = logits.split(counts, dim=0)
        losses = []
        positive_scores = []
        negative_scores = []
        for img_logits, img_labels in zip(per_image_logits, labels):
            if img_logits.numel() == 0:
                image_logits = img_labels.new_zeros(img_labels.shape)
            else:
                k = min(max(int(self.region_reshaper_target_topk), 1), img_logits.shape[0])
                image_logits = img_logits.topk(k, dim=0).values.mean(dim=0)
            losses.append(F.binary_cross_entropy_with_logits(image_logits, img_labels, reduction="mean"))
            if img_labels.sum() > 0:
                positive_scores.append(image_logits[img_labels > 0].detach().sigmoid().mean())
            if (img_labels == 0).any():
                negative_scores.append(image_logits[img_labels == 0].detach().sigmoid().mean())
        storage = get_event_storage()
        storage.put_scalar(
            "region_reshaper/target_labels_per_image",
            float(torch.stack([x.sum() for x in labels]).mean().item()) if labels else 0.0,
            smoothing_hint=False,
        )
        if positive_scores:
            storage.put_scalar(
                "region_reshaper/target_pos_prob",
                float(torch.stack(positive_scores).mean().item()),
                smoothing_hint=False,
            )
        if negative_scores:
            storage.put_scalar(
                "region_reshaper/target_neg_prob",
                float(torch.stack(negative_scores).mean().item()),
                smoothing_hint=False,
            )
        if not losses:
            return self.region_reshaper_zero_loss()
        return torch.stack(losses).mean()

    def region_reshaper_stage_a_losses(self, batched_inputs, is_source: bool):
        if self.region_reshaper is None:
            raise ValueError("REGION_RESHAPER.STAGE_A_ONLY requires REGION_RESHAPER.ENABLED=True.")
        proposals = self.normal_region_proposals(batched_inputs)
        if is_source:
            if "instances" not in batched_inputs[0]:
                return {"loss_reshaper_src_ce": self.region_reshaper_zero_loss()}
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            with torch.no_grad():
                proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
            embeddings = self.region_reshaper_embeddings(batched_inputs, proposals, is_source=True)
            logits = self.region_reshaper_logits(embeddings)
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0) if proposals else logits.new_zeros((0,), dtype=torch.long)
            fg = (gt_classes >= 0) & (gt_classes < self.roi_heads.num_classes)
            if fg.any():
                loss = F.cross_entropy(logits[fg], gt_classes[fg].long()) * self.region_reshaper_source_loss_weight
                pred = logits[fg].argmax(dim=1)
                acc = (pred == gt_classes[fg]).float().mean()
                gt_score = logits[fg].gather(1, gt_classes[fg].long().view(-1, 1)).squeeze(1)
                masked_logits = logits[fg].clone()
                masked_logits[torch.arange(masked_logits.shape[0], device=masked_logits.device), gt_classes[fg].long()] = -float("inf")
                margin = (gt_score - masked_logits.max(dim=1).values).mean()
                storage = get_event_storage()
                storage.put_scalar("region_reshaper/source_pos", float(fg.sum().item()), smoothing_hint=False)
                storage.put_scalar("region_reshaper/source_acc", float(acc.item()), smoothing_hint=False)
                storage.put_scalar("region_reshaper/source_margin", float(margin.item()), smoothing_hint=False)
            else:
                loss = self.region_reshaper_zero_loss()
                get_event_storage().put_scalar("region_reshaper/source_pos", 0.0, smoothing_hint=False)
            return {"loss_reshaper_src_ce": loss}

        embeddings = self.region_reshaper_embeddings(batched_inputs, proposals, is_source=False)
        logits = self.region_reshaper_logits(embeddings)
        loss = self.region_reshaper_target_mil_loss(logits, proposals, batched_inputs)
        return {"loss_reshaper_tgt_mil": loss * self.region_reshaper_target_mil_loss_weight}

    def offline_rpn_teacher_loss(self, student_inputs, teacher_inputs, teacher_model):
        if (
            not self.offline_rpn_teacher_enabled
            or self.clip_crop_region_type != "RPN"
            or self.offline_backbone is None
            or self.offline_proposal_generator is None
            or teacher_model is None
        ):
            return {}

        with torch.no_grad():
            pseudo_instances = self.offline_rpn_teacher_pseudo_instances(teacher_inputs, teacher_model)

        pseudo_counts = [len(x.gt_boxes) for x in pseudo_instances]
        storage = get_event_storage()
        storage.put_scalar("offline_rpn_teacher/pseudo_boxes", float(sum(pseudo_counts)) / max(len(pseudo_counts), 1))
        if sum(pseudo_counts) == 0:
            zero = next(self.offline_proposal_generator.parameters()).sum() * 0.0
            return {"loss_offline_rpn_pseudo": zero}

        self.offline_backbone.train()
        self.offline_proposal_generator.train()
        student_images = self.offline_preprocess_image(student_inputs)
        student_features = self.offline_backbone(student_images.tensor)
        _, rpn_losses = self.offline_proposal_generator(student_images, student_features, pseudo_instances)
        return {
            "loss_offline_rpn_pseudo": sum(rpn_losses.values()) * self.offline_rpn_teacher_loss_weight
        }

    def offline_rpn_teacher_pseudo_instances(self, batched_inputs, teacher_model):
        teacher_model.eval()
        teacher_images = teacher_model.offline_preprocess_image(batched_inputs)
        teacher_features = teacher_model.offline_backbone(teacher_images.tensor)
        proposals, _ = teacher_model.offline_proposal_generator(teacher_images, teacher_features, None)

        candidate_boxes = []
        candidate_classes = []
        candidate_obj_scores = []
        for proposal in proposals:
            objectness = proposal.objectness_logits.sigmoid()
            keep = objectness >= self.offline_rpn_teacher_obj_threshold
            if keep.sum().item() == 0:
                candidate_boxes.append(Boxes(proposal.proposal_boxes.tensor.new_zeros((0, 4))))
                candidate_classes.append(objectness.new_zeros((0,), dtype=torch.long))
                candidate_obj_scores.append(objectness.new_zeros((0,)))
                continue
            scores = objectness[keep]
            boxes = proposal.proposal_boxes[keep]
            order = torch.argsort(scores, descending=True)[: self.offline_rpn_teacher_max_pseudo_boxes]
            candidate_boxes.append(boxes[order])
            candidate_classes.append(scores.new_zeros((len(order),), dtype=torch.long))
            candidate_obj_scores.append(scores[order])

        semantic_scores, semantic_classes = self.regionclip_semantic_scores_for_boxes(
            batched_inputs,
            candidate_boxes,
        )

        pseudo_instances = []
        kept_counts = []
        semantic_means = []
        objectness_means = []
        for sample, boxes, obj_scores, sem_scores, sem_classes in zip(
            batched_inputs,
            candidate_boxes,
            candidate_obj_scores,
            semantic_scores,
            semantic_classes,
        ):
            if self.offline_rpn_teacher_semantic_filter and len(sem_scores):
                keep = sem_scores >= self.offline_rpn_teacher_semantic_threshold
            else:
                keep = torch.ones((len(boxes),), dtype=torch.bool, device=obj_scores.device)
            if keep.sum().item() > self.offline_rpn_teacher_max_pseudo_boxes:
                combined = obj_scores[keep] + sem_scores[keep]
                selected = torch.argsort(combined, descending=True)[: self.offline_rpn_teacher_max_pseudo_boxes]
                keep_idx = torch.nonzero(keep, as_tuple=False).flatten()[selected]
            else:
                keep_idx = torch.nonzero(keep, as_tuple=False).flatten()

            inst = Instances(sample["instances"].image_size if "instances" in sample else sample["image"].shape[-2:])
            inst.gt_boxes = boxes[keep_idx]
            inst.gt_classes = sem_classes[keep_idx] if len(sem_classes) else obj_scores.new_zeros((0,), dtype=torch.long)
            pseudo_instances.append(inst)
            kept_counts.append(float(len(keep_idx)))
            if len(keep_idx):
                semantic_means.append(float(sem_scores[keep_idx].mean().item()))
                objectness_means.append(float(obj_scores[keep_idx].mean().item()))

        storage = get_event_storage()
        storage.put_scalar("offline_rpn_teacher/after_obj_filter", float(sum(len(x) for x in candidate_boxes)) / max(len(candidate_boxes), 1))
        storage.put_scalar("offline_rpn_teacher/after_sem_filter", sum(kept_counts) / max(len(kept_counts), 1))
        if semantic_means:
            storage.put_scalar("offline_rpn_teacher/semantic_score", sum(semantic_means) / len(semantic_means))
        if objectness_means:
            storage.put_scalar("offline_rpn_teacher/objectness_score", sum(objectness_means) / len(objectness_means))
        return pseudo_instances

    def regionclip_semantic_scores_for_boxes(self, batched_inputs, boxes):
        if sum(len(x) for x in boxes) == 0:
            return (
                [self.pixel_mean.new_zeros((0,)) for _ in boxes],
                [torch.zeros((0,), dtype=torch.long, device=self.device) for _ in boxes],
            )
        images = self.preprocess_image(batched_inputs)
        proposals = []
        for sample, per_image_boxes in zip(batched_inputs, boxes):
            proposal = Instances(sample["instances"].image_size if "instances" in sample else sample["image"].shape[-2:])
            proposal.proposal_boxes = per_image_boxes
            proposal.objectness_logits = per_image_boxes.tensor.new_ones((len(per_image_boxes),))
            proposals.append(proposal)
        features = self.recognition_features(images, proposals, is_source=False)
        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features],
            boxes,
            self.backbone.layer4,
        )
        if self.use_clip_attpool:
            emb = self.backbone.attnpool(box_features)
        else:
            emb = box_features.mean(dim=[2, 3])
        text_weight = self.roi_heads.box_predictor.cls_score.weight[: self.roi_heads.num_classes]
        logits = F.normalize(emb, dim=1) @ F.normalize(text_weight, dim=1).t()
        logits = logits / self.c3_adapter_quality_logit_temperature
        probs = F.softmax(logits, dim=1)
        scores, classes = probs.max(dim=1)

        split_scores = []
        split_classes = []
        start = 0
        for per_image_boxes in boxes:
            end = start + len(per_image_boxes)
            split_scores.append(scores[start:end])
            split_classes.append(classes[start:end])
            start = end
        return split_scores, split_classes

    def image_level_predictions(self, batched_inputs: List[Dict[str, torch.Tensor]], is_source=False):
        with torch.no_grad():
            if self.clip_crop_region_type == "GT":
                proposals = []
                for b_input in batched_inputs:
                    this_gt = copy.deepcopy(b_input["instances"])
                    gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                    this_gt._fields = {
                        'proposal_boxes': gt_boxes,
                        'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device),
                    }
                    proposals.append(this_gt)
            elif self.clip_crop_region_type == "RPN":
                if self.offline_backbone.training or self.offline_proposal_generator.training:
                    self.offline_backbone.eval()
                    self.offline_proposal_generator.eval()
                offline_images = self.offline_preprocess_image(batched_inputs)
                offline_features = self.offline_backbone(offline_images.tensor)
                proposals, _ = self.offline_proposal_generator(offline_images, offline_features, None)

        images = self.preprocess_image(batched_inputs)
        features = self.recognition_features(images, proposals, is_source=is_source)
        if self.use_clip_c4:
            logits = self.roi_heads.image_level_logits(
                features,
                proposals,
                res5=self.backbone.layer4,
                attnpool=self.backbone.attnpool,
                c5_adapter_fn=self.c5_adapter_roi_features if self.c5_adapter_apply_residual else None,
                region_reshaper_fn=self.active_region_reshaper_fn(),
                is_source=is_source,
            )
        elif self.use_clip_attpool:
            logits = self.roi_heads.image_level_logits(
                features,
                proposals,
                attnpool=self.backbone.bottom_up.attnpool,
                region_reshaper_fn=self.active_region_reshaper_fn(),
                is_source=is_source,
            )
        else:
            logits = self.roi_heads.image_level_logits(
                features,
                proposals,
                region_reshaper_fn=self.active_region_reshaper_fn(),
                is_source=is_source,
            )

        objectness = [p.objectness_logits for p in proposals]
        return [self.h2fa_image_level_aggregate(x, o) for x, o in zip(logits, objectness)]

    @staticmethod
    def h2fa_image_level_aggregate(roi_logits: torch.Tensor, objectness_logits: torch.Tensor):
        class_logits = roi_logits[:, :-1]
        if class_logits.numel() == 0:
            return class_logits.new_zeros((class_logits.shape[1],))
        cls_prob = F.softmax(class_logits, dim=1)
        pred_cls = class_logits.argmax(dim=1)
        obj = objectness_logits.to(class_logits.device).float()
        obj_bar = class_logits.new_zeros(class_logits.shape)
        obj_bar[torch.arange(class_logits.shape[0], device=class_logits.device), pred_cls] = obj
        proposal_weights = F.softmax(obj_bar, dim=0)
        return (cls_prob * proposal_weights).sum(dim=0)

    def recognition_features(
        self,
        images: ImageList,
        proposals: List[Instances],
        is_source: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.c3_adapter_shared_features = {}
        self.c3_adapter_orth_losses = {}
        self.c4_semantic_last = {}
        if self.c4_semantic_reshaper_enabled:
            x = images.tensor.type(self.backbone.conv1.weight.dtype)
            x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
            x = self.backbone.relu(self.backbone.bn2(self.backbone.conv2(x)))
            x = self.backbone.relu(self.backbone.bn3(self.backbone.conv3(x)))
            x = self.backbone.avgpool(x)
            x = self.backbone.layer1(x)
            c3 = self.backbone.layer2(x)
            res4 = self.backbone.layer3(c3)
            res4 = self.c4_semantic_feature(res4, proposals)
            return {"res3": c3, "res4": res4}
        if not self.c3_adapter_enabled:
            return self.backbone(images.tensor)

        x = images.tensor.type(self.backbone.conv1.weight.dtype)
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.relu(self.backbone.bn2(self.backbone.conv2(x)))
        x = self.backbone.relu(self.backbone.bn3(self.backbone.conv3(x)))
        x = self.backbone.avgpool(x)
        x = self.backbone.layer1(x)
        c3 = self.backbone.layer2(x)

        if self.c3_adapter_apply_residual:
            quality_map = self.c3_quality_map(c3, proposals, is_source=is_source)
            if self.c3_adapter_detach_quality_map or self.use_region_loss_quality():
                quality_map = quality_map.detach()
            if self.c3_adapter_domain_decoupled:
                c3, c3_shared = self.domain_decoupled_adapter_feature(
                    c3,
                    quality_map,
                    self.c3_shared_adapter,
                    self.domain_specific_adapter(self.c3_source_adapter, self.c3_target_adapter, is_source),
                    self.c3_adapter_residual_scale,
                    "c3",
                )
                self.c3_adapter_shared_features["res3"] = c3_shared
            else:
                c3 = c3 + self.c3_adapter_residual_scale * quality_map * self.c3_adapter(c3)
                self.c3_adapter_shared_features["res3"] = c3
        res4 = self.backbone.layer3(c3)
        if self.c4_adapter_apply_residual:
            quality_map = self.c4_quality_map(res4, proposals, is_source=is_source)
            if self.c3_adapter_detach_quality_map or self.use_region_loss_quality():
                quality_map = quality_map.detach()
            if self.c3_adapter_domain_decoupled:
                res4, res4_shared = self.domain_decoupled_adapter_feature(
                    res4,
                    quality_map,
                    self.c4_shared_adapter,
                    self.domain_specific_adapter(self.c4_source_adapter, self.c4_target_adapter, is_source),
                    self.c4_adapter_residual_scale,
                    "c4",
                )
                self.c3_adapter_shared_features["res4"] = res4_shared
            else:
                res4 = res4 + self.c4_adapter_residual_scale * quality_map * self.c4_adapter(res4)
                self.c3_adapter_shared_features["res4"] = res4
        return {"res3": c3, "res4": res4}

    def use_region_loss_quality(self) -> bool:
        return self.c3_adapter_enabled and self.c3_adapter_quality_use == "region_loss"

    def domain_specific_adapter(self, source_adapter, target_adapter, is_source: bool):
        if self.training:
            return source_adapter if is_source else target_adapter
        return source_adapter if self.c3_adapter_inference_domain == "source" else target_adapter

    def domain_decoupled_adapter_feature(
        self,
        feature: torch.Tensor,
        quality_map: torch.Tensor,
        shared_adapter: nn.Module,
        domain_adapter: nn.Module,
        layer_scale: float,
        prefix: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_residual = shared_adapter(feature)
        domain_residual = domain_adapter(feature)
        quality_map = quality_map.to(dtype=feature.dtype)
        shared_feature = feature + layer_scale * self.c3_adapter_domain_shared_scale * quality_map * shared_residual
        output_feature = shared_feature + layer_scale * self.c3_adapter_domain_specific_scale * quality_map * domain_residual
        self.observe_domain_decoupled_adapter(
            prefix,
            feature,
            quality_map,
            shared_residual,
            domain_residual,
            shared_feature,
            output_feature,
        )
        self.c3_adapter_orth_losses[prefix] = self.orthogonality_loss(
            shared_residual,
            domain_residual,
            quality_map,
            prefix,
        )
        return output_feature, shared_feature

    def orthogonality_loss(
        self,
        shared_residual: torch.Tensor,
        domain_residual: torch.Tensor,
        quality_map: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        if self.c3_adapter_orth_loss_weight <= 0:
            return shared_residual.sum() * 0.0
        shared_norm = F.normalize(shared_residual, dim=1)
        domain_norm = F.normalize(domain_residual, dim=1)
        cos2 = (shared_norm * domain_norm).sum(dim=1, keepdim=True).pow(2)
        weights = quality_map.to(dtype=cos2.dtype)
        while weights.dim() < cos2.dim():
            weights = weights.unsqueeze(-1)
        loss = (cos2 * weights).sum() / weights.sum().clamp_min(1.0)
        if self.training and self.c3_adapter_observe_period > 0:
            storage = get_event_storage()
            if storage.iter % self.c3_adapter_observe_period == 0:
                storage.put_scalar(f"{prefix}_adapter/orth_loss_raw", loss.item(), smoothing_hint=False)
        return loss

    def observe_domain_decoupled_adapter(
        self,
        prefix: str,
        feature: torch.Tensor,
        quality_map: torch.Tensor,
        shared_residual: torch.Tensor,
        domain_residual: torch.Tensor,
        shared_feature: torch.Tensor,
        output_feature: torch.Tensor,
    ):
        if not self.training or self.c3_adapter_observe_period <= 0:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        if storage.iter % self.c3_adapter_observe_period != 0:
            return

        with torch.no_grad():
            eps = 1e-6
            input_vec = feature.detach().float().flatten(1)
            shared_vec = shared_residual.detach().float().flatten(1)
            domain_vec = domain_residual.detach().float().flatten(1)
            shared_delta_vec = (shared_feature - feature).detach().float().flatten(1)
            domain_delta_vec = (output_feature - shared_feature).detach().float().flatten(1)

            input_norm = input_vec.norm(dim=1).mean().clamp_min(eps)
            shared_res_norm = shared_vec.norm(dim=1).mean()
            domain_res_norm = domain_vec.norm(dim=1).mean()
            shared_delta_norm = shared_delta_vec.norm(dim=1).mean()
            domain_delta_norm = domain_delta_vec.norm(dim=1).mean()
            cos = F.cosine_similarity(shared_vec, domain_vec, dim=1, eps=eps)
            q = quality_map.detach().float()

            metric_prefix = f"domain_decoupled/{prefix}"
            storage.put_scalar(f"{metric_prefix}/input_norm", input_norm.item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/shared_res_norm", shared_res_norm.item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/domain_res_norm", domain_res_norm.item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/shared_delta_over_input", (shared_delta_norm / input_norm).item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/domain_delta_over_input", (domain_delta_norm / input_norm).item(), smoothing_hint=False)
            storage.put_scalar(
                f"{metric_prefix}/domain_to_shared_delta_ratio",
                (domain_delta_norm / shared_delta_norm.clamp_min(eps)).item(),
                smoothing_hint=False,
            )
            storage.put_scalar(f"{metric_prefix}/res_cos_mean", cos.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/res_abs_cos_mean", cos.abs().mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/q_mean", q.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{metric_prefix}/q_nonzero_frac", (q > 0).float().mean().item(), smoothing_hint=False)

    def c3_supervised_projection_loss(
        self,
        feature: torch.Tensor,
        gt_instances: List[Instances],
        proposals: Optional[List[Instances]] = None,
        feature_name: str = "res3",
    ) -> torch.Tensor:
        if self.c3_adapter_supervised_proj_use_proposals and proposals is not None:
            with torch.no_grad():
                training_proposals = self.roi_heads.label_and_sample_proposals(
                    copy.deepcopy(proposals),
                    gt_instances,
                )
            boxes = [x.proposal_boxes for x in training_proposals]
            labels = torch.cat([x.gt_classes for x in training_proposals], dim=0)
        else:
            boxes = [x.gt_boxes for x in gt_instances]
            labels = torch.cat([x.gt_classes for x in gt_instances], dim=0)

        num_boxes = sum(len(x) for x in boxes)
        if num_boxes == 0:
            return feature.sum() * 0.0

        valid = (labels >= 0) & (labels < self.roi_heads.num_classes)
        if not valid.any():
            return feature.sum() * 0.0

        if feature_name == "res3":
            pooler = self.c3_quality_pooler
            projector = self.c3_quality_proj
        elif feature_name == "res4":
            pooler = self.c4_quality_pooler
            projector = self.c4_quality_proj
        else:
            raise ValueError("Unknown supervised projection feature: {}".format(feature_name))

        roi = pooler([feature], boxes)
        roi_vec = roi.mean(dim=[2, 3])
        roi_vec = roi_vec[valid]
        labels = labels[valid]

        emb = projector(roi_vec)
        text_weight = self.roi_heads.box_predictor.cls_score.weight[:self.roi_heads.num_classes].detach()
        logits = F.normalize(emb, dim=1) @ F.normalize(text_weight, dim=1).t()
        logits = logits / self.c3_adapter_supervised_proj_temperature
        loss = F.cross_entropy(logits, labels)

        if self.training:
            with torch.no_grad():
                storage = get_event_storage()
                probs = F.softmax(logits, dim=1)
                entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
                prefix = "c3_adapter/{}_sup_proj".format(feature_name)
                storage.put_scalar(prefix + "_acc", (logits.argmax(dim=1) == labels).float().mean().item())
                storage.put_scalar(prefix + "_effective_classes", entropy.exp().mean().item())
                storage.put_scalar(prefix + "_loss_raw", loss.item())
        return loss

    def c3_quality_map(
        self,
        c3: torch.Tensor,
        proposals: List[Instances],
        is_source: bool = False,
    ) -> torch.Tensor:
        if self.use_region_loss_quality():
            return c3.new_ones((c3.shape[0], 1, c3.shape[2], c3.shape[3]))
        if len(proposals) == 0:
            return c3.new_zeros((c3.shape[0], 1, c3.shape[2], c3.shape[3]))
        boxes = [p.proposal_boxes for p in proposals]
        eff = self.c3_effective_classes(c3, boxes)
        aug_boxes = [
            Boxes(self.jitter_boxes(p.proposal_boxes.tensor, p.image_size, self.c3_adapter_perturb_scale))
            for p in proposals
        ]
        eff_aug = self.c3_effective_classes(c3, aug_boxes)
        quality = self.adapter_region_quality(eff, eff_aug)

        quality_maps = self.scatter_region_quality_to_c3(quality, proposals, c3.shape[-2:])
        self.observe_c3_quality(eff, eff_aug, quality, quality_maps, is_source=is_source)
        return quality_maps

    def c4_quality_map(
        self,
        c4: torch.Tensor,
        proposals: List[Instances],
        is_source: bool = False,
    ) -> torch.Tensor:
        if self.use_region_loss_quality():
            return c4.new_ones((c4.shape[0], 1, c4.shape[2], c4.shape[3]))
        if len(proposals) == 0:
            return c4.new_zeros((c4.shape[0], 1, c4.shape[2], c4.shape[3]))
        boxes = [p.proposal_boxes for p in proposals]
        eff = self.c4_effective_classes(c4, boxes)
        aug_boxes = [
            Boxes(self.jitter_boxes(p.proposal_boxes.tensor, p.image_size, self.c3_adapter_perturb_scale))
            for p in proposals
        ]
        eff_aug = self.c4_effective_classes(c4, aug_boxes)
        quality = self.adapter_region_quality(eff, eff_aug)

        quality_maps = self.scatter_region_quality_to_feature(quality, proposals, c4.shape[-2:], stride=16.0)
        self.observe_quality("c4_adapter", eff, eff_aug, quality, quality_maps, is_source=is_source)
        return quality_maps

    def c3_effective_classes(self, c3: torch.Tensor, boxes: List[Boxes]) -> torch.Tensor:
        if sum(len(x) for x in boxes) == 0:
            return c3.new_zeros((0,))
        roi = self.c3_quality_pooler([c3], boxes)
        roi_vec = roi.mean(dim=[2, 3])
        emb = self.c3_quality_proj(roi_vec)
        text_weight = self.roi_heads.box_predictor.cls_score.weight[:self.roi_heads.num_classes]
        logits = F.normalize(emb, dim=1) @ F.normalize(text_weight, dim=1).t()
        logits = logits / self.c3_adapter_quality_logit_temperature
        probs = F.softmax(logits, dim=1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
        return entropy.exp()

    def c4_effective_classes(self, c4: torch.Tensor, boxes: List[Boxes]) -> torch.Tensor:
        if sum(len(x) for x in boxes) == 0:
            return c4.new_zeros((0,))
        roi = self.c4_quality_pooler([c4], boxes)
        roi_vec = roi.mean(dim=[2, 3])
        emb = self.c4_quality_proj(roi_vec)
        text_weight = self.roi_heads.box_predictor.cls_score.weight[:self.roi_heads.num_classes]
        logits = F.normalize(emb, dim=1) @ F.normalize(text_weight, dim=1).t()
        logits = logits / self.c3_adapter_quality_logit_temperature
        probs = F.softmax(logits, dim=1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
        return entropy.exp()

    def c5_adapter_roi_features(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        box_features: torch.Tensor,
        is_source: bool = False,
    ) -> torch.Tensor:
        if box_features.numel() == 0:
            return box_features
        if self.use_region_loss_quality():
            quality = box_features.new_ones((box_features.shape[0], 1, 1, 1))
            if self.c3_adapter_domain_decoupled:
                output_features, shared_features = self.domain_decoupled_adapter_feature(
                    box_features,
                    quality,
                    self.c5_shared_adapter,
                    self.domain_specific_adapter(self.c5_source_adapter, self.c5_target_adapter, is_source),
                    self.c5_adapter_residual_scale,
                    "c5",
                )
                return output_features, shared_features
            return box_features + self.c5_adapter_residual_scale * self.c5_adapter(box_features)
        if self.c3_adapter_detach_quality_map:
            with torch.no_grad():
                quality, eff, eff_aug = self.c5_region_quality(features, proposals, box_features)
            quality = quality.detach()
        else:
            quality, eff, eff_aug = self.c5_region_quality(features, proposals, box_features)
        self.observe_region_quality("c5_adapter", eff, eff_aug, quality, is_source=is_source)
        quality = quality.reshape(-1, 1, 1, 1).to(dtype=box_features.dtype)
        if self.c3_adapter_domain_decoupled:
            output_features, shared_features = self.domain_decoupled_adapter_feature(
                box_features,
                quality,
                self.c5_shared_adapter,
                self.domain_specific_adapter(self.c5_source_adapter, self.c5_target_adapter, is_source),
                self.c5_adapter_residual_scale,
                "c5",
            )
            return output_features, shared_features
        return box_features + self.c5_adapter_residual_scale * quality * self.c5_adapter(box_features)

    def region_loss_quality(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        box_features: torch.Tensor,
        is_source: bool = False,
    ) -> torch.Tensor:
        if "res4" not in features or sum(len(p) for p in proposals) == 0:
            return box_features.new_zeros((0,))
        if self.c3_adapter_detach_quality_map:
            with torch.no_grad():
                quality, eff, eff_aug = self.c4_region_quality_chunked(features["res4"], proposals)
            quality = quality.detach()
        else:
            quality, eff, eff_aug = self.c4_region_quality_chunked(features["res4"], proposals)
        self.observe_region_quality("region_loss_quality", eff, eff_aug, quality, is_source=is_source)
        if self.c3_adapter_region_loss_weight_scale != 1.0:
            quality = quality * self.c3_adapter_region_loss_weight_scale
        if quality.is_cuda:
            torch.cuda.empty_cache()
        return quality

    def c4_region_quality_chunked(
        self,
        c4: torch.Tensor,
        proposals: List[Instances],
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qualities = []
        effs = []
        eff_augs = []
        for image_idx, proposals_per_image in enumerate(proposals):
            boxes_tensor = proposals_per_image.proposal_boxes.tensor
            if boxes_tensor.numel() == 0:
                continue
            aug_tensor = self.jitter_boxes(
                boxes_tensor,
                proposals_per_image.image_size,
                self.c3_adapter_perturb_scale,
            )
            c4_one = c4[image_idx: image_idx + 1]
            for start in range(0, boxes_tensor.shape[0], chunk_size):
                end = min(start + chunk_size, boxes_tensor.shape[0])
                boxes = [Boxes(boxes_tensor[start:end])]
                aug_boxes = [Boxes(aug_tensor[start:end])]
                eff = self.c4_effective_classes(c4_one, boxes)
                eff_aug = self.c4_effective_classes(c4_one, aug_boxes)
                effs.append(eff)
                eff_augs.append(eff_aug)
                qualities.append(self.adapter_region_quality(eff, eff_aug))
        if not qualities:
            empty = c4.new_zeros((0,))
            return empty, empty, empty
        return torch.cat(qualities, dim=0), torch.cat(effs, dim=0), torch.cat(eff_augs, dim=0)

    def c5_region_quality(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        box_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposal_boxes = [x.proposal_boxes for x in proposals]
        if sum(len(x) for x in proposal_boxes) == 0:
            empty = box_features.new_zeros((0,))
            return empty, empty, empty
        eff = self.c5_effective_classes_from_features(box_features)
        aug_boxes = [
            Boxes(
                self.jitter_boxes(
                    proposals_per_image.proposal_boxes.tensor,
                    proposals_per_image.image_size,
                    self.c3_adapter_perturb_scale,
                )
            )
            for proposals_per_image in proposals
        ]
        aug_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features],
            aug_boxes,
            self.backbone.layer4,
        )
        eff_aug = self.c5_effective_classes_from_features(aug_features)
        quality = self.adapter_region_quality(eff, eff_aug)
        return quality, eff, eff_aug

    def adapter_region_quality(self, eff: torch.Tensor, eff_aug: torch.Tensor) -> torch.Tensor:
        num_classes = self.roi_heads.num_classes
        certainty = (float(num_classes) - eff) / max(float(num_classes - 1), 1.0)
        if self.c3_adapter_quality_mode == "certainty_only":
            quality = certainty
        elif self.c3_adapter_quality_mode == "stability_only":
            quality = torch.exp(-(eff_aug - eff).abs() / self.c3_adapter_quality_tau)
        else:
            stability = torch.exp(-(eff_aug - eff).abs() / self.c3_adapter_quality_tau)
            quality = certainty * stability
        if self.c3_adapter_clamp_quality:
            quality = quality.clamp(0.0, 1.0)
        return quality

    def c5_effective_classes_from_features(self, box_features: torch.Tensor) -> torch.Tensor:
        if box_features.numel() == 0:
            return box_features.new_zeros((0,))
        if self.use_clip_attpool:
            emb = self.backbone.attnpool(box_features)
        else:
            emb = box_features.mean(dim=[2, 3])
        text_weight = self.roi_heads.box_predictor.cls_score.weight[:self.roi_heads.num_classes]
        if emb.shape[1] != text_weight.shape[1]:
            raise ValueError(
                "C5 quality requires ROI embedding dim {} to match text classifier dim {}.".format(
                    emb.shape[1], text_weight.shape[1]
                )
            )
        logits = F.normalize(emb, dim=1) @ F.normalize(text_weight, dim=1).t()
        logits = logits / self.c3_adapter_quality_logit_temperature
        probs = F.softmax(logits, dim=1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
        return entropy.exp()

    @staticmethod
    def jitter_boxes(boxes: torch.Tensor, image_size: Tuple[int, int], scale: float) -> torch.Tensor:
        if boxes.numel() == 0 or scale <= 0:
            return boxes.clone()
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        w = (x2 - x1).clamp_min(1.0)
        h = (y2 - y1).clamp_min(1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = (torch.rand_like(cx) * 2.0 - 1.0) * scale * w
        dy = (torch.rand_like(cy) * 2.0 - 1.0) * scale * h
        sw = torch.exp((torch.rand_like(w) * 2.0 - 1.0) * scale)
        sh = torch.exp((torch.rand_like(h) * 2.0 - 1.0) * scale)
        nw = (w * sw).clamp_min(2.0)
        nh = (h * sh).clamp_min(2.0)
        out = torch.stack([cx + dx - 0.5 * nw, cy + dy - 0.5 * nh,
                           cx + dx + 0.5 * nw, cy + dy + 0.5 * nh], dim=1)
        height, width = image_size
        out[:, 0::2].clamp_(0, width)
        out[:, 1::2].clamp_(0, height)
        out[:, 2] = torch.maximum(out[:, 2], out[:, 0] + 1.0)
        out[:, 3] = torch.maximum(out[:, 3], out[:, 1] + 1.0)
        out[:, 0::2].clamp_(0, width)
        out[:, 1::2].clamp_(0, height)
        return out

    def scatter_region_quality_to_c3(
        self,
        quality: torch.Tensor,
        proposals: List[Instances],
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        return self.scatter_region_quality_to_feature(quality, proposals, spatial_size, stride=8.0)

    def scatter_region_quality_to_feature(
        self,
        quality: torch.Tensor,
        proposals: List[Instances],
        spatial_size: Tuple[int, int],
        stride: float,
    ) -> torch.Tensor:
        h, w = spatial_size
        quality_maps = []
        offset = 0
        mode = self.c3_adapter_scatter_mode
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
                raise ValueError("Unknown C3_ADAPTER.SCATTER_MODE: {}".format(mode))
        return torch.stack(quality_maps, dim=0)

    def observe_c3_quality(
        self,
        eff: torch.Tensor,
        eff_aug: torch.Tensor,
        quality: torch.Tensor,
        quality_maps: torch.Tensor,
        is_source: bool = False,
    ):
        self.observe_quality("c3_adapter", eff, eff_aug, quality, quality_maps, is_source=is_source)

    def observe_quality(
        self,
        name: str,
        eff: torch.Tensor,
        eff_aug: torch.Tensor,
        quality: torch.Tensor,
        quality_maps: torch.Tensor,
        is_source: bool = False,
    ):
        if not self.training or self.c3_adapter_observe_period <= 0:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        if storage.iter % self.c3_adapter_observe_period != 0:
            return
        with torch.no_grad():
            prefix = "{}/source".format(name) if is_source else "{}/target".format(name)
            storage.put_scalar(f"{prefix}/effective_classes_mean", eff.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/effective_classes_aug_mean", eff_aug.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/effective_classes_delta_mean", (eff_aug - eff).abs().mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_mean", quality.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_min", quality.min().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_max", quality.max().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_low_frac", (quality < 0.05).float().mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_high_frac", (quality > 0.95).float().mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/Q_mean", quality_maps.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/Q_max", quality_maps.max().item(), smoothing_hint=False)

    def observe_region_quality(
        self,
        name: str,
        eff: torch.Tensor,
        eff_aug: torch.Tensor,
        quality: torch.Tensor,
        is_source: bool = False,
    ):
        if not self.training or self.c3_adapter_observe_period <= 0:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        if storage.iter % self.c3_adapter_observe_period != 0:
            return
        with torch.no_grad():
            prefix = "{}/source".format(name) if is_source else "{}/target".format(name)
            storage.put_scalar(f"{prefix}/effective_classes_mean", eff.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/effective_classes_aug_mean", eff_aug.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/effective_classes_delta_mean", (eff_aug - eff).abs().mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_mean", quality.mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_min", quality.min().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_max", quality.max().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_low_frac", (quality < 0.05).float().mean().item(), smoothing_hint=False)
            storage.put_scalar(f"{prefix}/q_high_frac", (quality > 0.95).float().mean().item(), smoothing_hint=False)

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        
        # localization branch: offline modules to get the region proposals
        if self.clip_crop_region_type == "GT":  # from ground-truth
            proposals = []
            for r_i, b_input in enumerate(batched_inputs): 
                this_gt = copy.deepcopy(b_input["instances"])  # Instance
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes} #, 'objectness_logits': None}
                proposals.append(this_gt)                
        elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            if detected_instances is None:
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     
    
        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.recognition_features(images, proposals)
        #assert not torch.any(torch.isnan(features))


        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                results, _ = self.roi_heads(
                    images,
                    features,
                    proposals,
                    None,
                    res5=self.backbone.layer4,
                    attnpool=self.backbone.attnpool,
                    c5_adapter_fn=self.c5_adapter_roi_features if self.c5_adapter_apply_residual else None,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                )
            else: # use mean pool
                results, _ = self.roi_heads(
                    images,
                    features,
                    proposals,
                    None,
                    res5=self.backbone.layer4,
                    c5_adapter_fn=self.c5_adapter_roi_features if self.c5_adapter_apply_residual else None,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                )
        else:  # regular detector setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                results, _  = self.roi_heads(
                    images,
                    features,
                    proposals,
                    None,
                    attnpool=self.backbone.bottom_up.attnpool,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                )
            else:
                results, _  = self.roi_heads(
                    images,
                    features,
                    proposals,
                    None,
                    region_reshaper_fn=self.active_region_reshaper_fn(),
                )
        
        #visualize
        #from detectron2.utils.visualizer import Visualizer
        #img = batched_inputs[0]["image"]
        #img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        #v_gt = Visualizer(img, None)
        #classname = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        #v_gt_name = ["{} ".format(classname[int(l)]) for l in batched_inputs[0]["instances"].gt_classes.to("cpu")]
        #v_gt = v_gt.overlay_instances(boxes=batched_inputs[0]["instances"].gt_boxes, labels=v_gt_name)
        #anno_img = v_gt.get_image()
        #v_pred = Visualizer(img, None)
        #v_pred = v_pred.draw_instance_predictions(results[0].to("cpu"), 0.8)
        #prop_img = v_pred.get_image()
        #vis_img = np.concatenate((anno_img, prop_img), axis=1)
        #vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
        #f_n = batched_inputs[0]['file_name']
        #to_save = Image.fromarray(np.array(vis_img, np.uint8))
        #to_save.save("output/regions/" + f_n.split("/")[-1].split(".")[0] + ".png")

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPFastRCNN._postprocess(results, batched_inputs)
        else:
            return results

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

@META_ARCH_REGISTRY.register()
class PretrainFastRCNN(nn.Module):
    """
    RegionCLIP: Learning visual region representation via vision-language pretraining from image-text pairs
    1. region-token level matching: learn to match the pseudo region-text pairs, provided by teacher model
    2. image-text level matching: learn to match image-text pairs, obtained from the Internet
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
        roi_heads: nn.Module,
        teacher_backbone: nn.Module,
        teacher_roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_c4: False,
        use_clip_attpool: False,
        offline_input_format: Optional[str] = None,
        offline_pixel_mean: Tuple[float],
        offline_pixel_std: Tuple[float],
        language_encoder: nn.Module,
        matching_temp: None,
        num_regions_per_img: int = 0,
        img_txt_level: None,
        gather_gpus: False,
        concept_emb: None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.offline_backbone = offline_backbone
        self.backbone = backbone
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        
        # image-text level pretraining
        self.img_txt_level = img_txt_level[0]
        self.only_eot = img_txt_level[1]
        if self.img_txt_level:
            self.lang_encoder = language_encoder
            for p in self.lang_encoder.parameters():  # freeze language encoder
                p.requires_grad = False
        self.matching_temp = matching_temp
        self.context_length = 77 # defined in clip_img_txt_pair_tsv class
        self.num_regions_per_img = num_regions_per_img
        self.gather_gpus = gather_gpus

        # region-token level pretraining
        if concept_emb[0]:
            self.register_buffer("concept_emb", torch.load(concept_emb[0]), False) # [#concepts, d]
            self.concept_thres = concept_emb[1]
            self.teacher_backbone = teacher_backbone
            for p in self.teacher_backbone.parameters():  # freeze visual encoder of teacher model
                p.requires_grad = False
            if concept_emb[2] is None: # teacher model uses the same concept embedding as student model
                self.register_buffer("teacher_concept_emb", torch.load(concept_emb[0]), False)
            else: # teacher model uses a seperate concept embedding
                self.register_buffer("teacher_concept_emb", torch.load(concept_emb[2]), False)
            self.teacher_roi_heads = teacher_roi_heads
        else:
            self.concept_emb = None

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN": # create isolated backbone & RPN
            # create offline cfg for the pretrained backbone & RPN
            from detectron2.config import get_cfg
            offline_cfg = get_cfg()
            offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
            if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
                offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
                offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
                offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
                offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
            if cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS:
                offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS 
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH
            
            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg) # build_resnet_fpn_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())
            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE in ["GRID", "RANDOM"]:
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None
        
        # visual encoder and roi_heads of student model
        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        # language encoder of student model
        language_encoder = build_clip_language_encoder(cfg)
        # visual encoder of teacher model
        teacher_cfg = copy.deepcopy(cfg)
        teacher_cfg.defrost()
        teacher_cfg.MODEL.RESNETS.DEPTH = teacher_cfg.MODEL.CLIP.TEACHER_RESNETS_DEPTH
        teacher_backbone = build_backbone(teacher_cfg)
        teacher_cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = teacher_cfg.MODEL.CLIP.TEACHER_POOLER_RESOLUTION
        teacher_roi_heads = build_roi_heads(teacher_cfg, teacher_backbone.output_shape())

        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "backbone": backbone,
            "roi_heads": roi_heads, 
            "teacher_backbone": teacher_backbone,
            "teacher_roi_heads": teacher_roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME == 'PretrainRes5ROIHeads',
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            "language_encoder": language_encoder,
            "matching_temp": cfg.MODEL.CLIP.CLSS_TEMP,
            "num_regions_per_img": cfg.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS,
            "img_txt_level": (cfg.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL, cfg.MODEL.CLIP.PRETRAIN_ONLY_EOT),
            "gather_gpus": cfg.MODEL.CLIP.GATHER_GPUS,
            "concept_emb": (cfg.MODEL.CLIP.CONCEPT_POOL_EMB, cfg.MODEL.CLIP.CONCEPT_THRES, cfg.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        gt_instances = None
        losses = {}
        
        # localization branch: offline modules to get the region proposals
        proposals = self.get_region_proposals(batched_inputs)
        global_proposals = self.create_global_proposals(batched_inputs)

        # recognition branch: get 2D feature maps using the backbone of recognition branch and extract region features
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        region_feats = self.get_region_features(images, features, proposals, gt_instances)
        global_feats = self.get_region_features(images, features, global_proposals, gt_instances)

        # image-text level matching
        if self.img_txt_level:
            self.image_text_matching(batched_inputs, proposals, region_feats, losses, global_feats=global_feats)

        # region-concept level matching
        if self.concept_emb is not None:
            self.region_concept_matching(images, proposals, gt_instances, region_feats, losses)

        return losses

    def region_concept_matching(self, images, proposals, gt_instances, region_feats, losses, use_distill=True, use_contrastive=True):
        # get psuedo concept labels from teacher model
        concept_scores, target_inds, keep_regions, target_embs, label_mtx \
            = self.get_psuedo_concept_labels(images, proposals, gt_instances)

        # prepare region features for the kept regions
        keep_region_feats = region_feats[keep_regions]
        keep_region_feats = keep_region_feats / keep_region_feats.norm(dim=-1, keepdim=True)

        if use_distill:
            # distillation learning: learns from the predictions of teacher model
            concept_emb = self.concept_emb / self.concept_emb.norm(dim=-1, keepdim=True)
            cls_scores = keep_region_feats @ concept_emb.t()  # [#kept_regions, #concepts]
            cls_scores_temp = cls_scores / self.matching_temp
            
            # calculate loss
            cls_loss = F.kl_div(F.softmax(cls_scores_temp, dim=1).log(), concept_scores, reduction='batchmean')  # input is log-probabilities, target is probabilities
            losses.update({"loss_region_distill": cls_loss}) #  * 0.8})

        if use_contrastive:
            # contrastive learning: matching student visual features with target concept embs
            target_embs = target_embs / target_embs.norm(dim=-1, keepdim=True)
            match_scores = keep_region_feats @ target_embs.t()  # [#kept_regions, #kept_regions]
            match_scores_temp = match_scores / self.matching_temp

            # calculate loss given matching scores and label matrix
            contrastive_loss = MILCrossEntropy()(match_scores_temp, label_mtx, weights=None, avg_positives=False)
            losses.update({"loss_concept_contrastive": contrastive_loss})

    def image_text_matching(self, batched_inputs, proposals, region_feats, losses, global_feats):
        # encode text
        num_cap = int(batched_inputs[0][1].size(0) / self.context_length)
        if num_cap == 1:  # one caption per image
            text = [x[1].view(1,-1).to(self.device) for x in batched_inputs]
        else: # multiple caption pers image, then randomly pick one
            rand_ind = [randint(0, num_cap-1) for _ in range(len(batched_inputs))]
            text = [x[1].view(-1,self.context_length)[rand_ind[i]:rand_ind[i]+1].to(self.device) for i, x in enumerate(batched_inputs)]
        text = torch.cat(text, dim=0)
        text_embs = self.lang_encoder.encode_text(text, only_eot=self.only_eot)  # [img_batch, n_ctx, transformer.width] or [img_batch, transformer.width]

        # prepare region features and text embeddings
        region_feats = global_feats
        region_feats = region_feats / region_feats.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        region_feats_full, min_bs = gather_tensors(region_feats) if self.gather_gpus else (region_feats, None)  #  gather across GPUs
        text_embs_full, min_bs = gather_tensors(text_embs) if self.gather_gpus else (text_embs, None)  #  gather across GPUs

        # matching visual features with text embs
        match_scores = region_feats_full @ text_embs_full.view(-1, text_embs_full.size(-1)).t()  # [#regions, img_batch * n_ctx]
        img_b = int(region_feats_full.size(0))
        pooled_score = match_scores

        pooled_score = pooled_score / self.matching_temp
        contrast_target = torch.arange(img_b).to(self.device)
        row_loss = F.cross_entropy(pooled_score, contrast_target)
        col_loss = F.cross_entropy(pooled_score.t(), contrast_target)
        losses.update({"loss_img_txt_level": (row_loss + col_loss) / 2.0}) 

    def get_psuedo_concept_labels(self, images, proposals, gt_instances, s_temp=0.01):
        """ Input images and region proposals, return matching results from teacher model
        """
        with torch.no_grad():
            # extract visual features from teacher model
            features = self.teacher_backbone(images.tensor)
            teacher_region_feats = self.teacher_roi_heads(images, features, proposals, gt_instances, res5=self.teacher_backbone.layer4, attnpool=self.teacher_backbone.attnpool)
            
            # match teacher visual features with teacher concept embs to create pseudo labels
            teacher_region_feats = teacher_region_feats / teacher_region_feats.norm(dim=-1, keepdim=True)
            teacher_concept_emb = self.teacher_concept_emb / self.teacher_concept_emb.norm(dim=-1, keepdim=True)
            concept_scores = teacher_region_feats @ teacher_concept_emb.t()  # [#regions, #concepts]
            concept_scores = F.softmax(concept_scores / s_temp, dim=1)

            max_scores, max_inds = torch.max(concept_scores, dim=1)
            keep_regions = max_scores > self.concept_thres  # only keep the regions that have high matching score with a concept
            if keep_regions.nonzero().size(0) == 0: # if all regions can't match to any concept
                print("all regions can't match to any concept!")
                keep_regions = max_scores > 0.0 
            target_inds = max_inds[keep_regions]
            target_embs = self.concept_emb[target_inds] # the target embedding of student model
            label_mtx = (target_inds.view(-1, 1) == target_inds.view(1, -1)).type_as(teacher_region_feats)
            concept_scores = concept_scores[keep_regions]
                
        return concept_scores, target_inds, keep_regions, target_embs, label_mtx

    def get_region_features(self, images, features, proposals, gt_instances):
        """ Input images and region proposals, return region features
        """
        # Given the proposals, crop region features from 2D image features
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else: # use mean pool
                region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
        else:  # regular detector setting
            region_feats = self.roi_heads(images, features, proposals, gt_instances)
        return region_feats

    def get_region_proposals(self, batched_inputs):
        """ Given image, return object proposals
        """
        with torch.no_grad():  
            if self.clip_crop_region_type == "RANDOM":  # from random proposals
                proposals = self.create_rand_boxes(batched_inputs)         
            elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
                if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
                    self.offline_backbone.eval() 
                    self.offline_proposal_generator.eval()  
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     
            #visualize_proposals(batched_inputs, proposals, self.input_format, vis_pretrain=True)
        
        # randomly select proposals
        if self.training:
            rand_inds = [torch.randperm(len(p))[:self.num_regions_per_img].to(self.device) for p in proposals]
            proposals = [p[rand_inds[i]] for i, p in enumerate(proposals)]
        return proposals

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x[0].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [((x * 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x[0].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [((x * 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def create_rand_boxes(self, batched_inputs, grid_length=8):
        """ create random boxes within an image, output random self.num_regions_per_img boxes
        return a list of Boxes
        """
        images = self.preprocess_image(batched_inputs)
        image_height = images.tensor.size(2)
        image_width = images.tensor.size(3)

        left_top_x = torch.tensor([i*(grid_length) for i in range(image_width // grid_length)])
        left_top_y = torch.tensor([i*(grid_length) for i in range(image_height // grid_length)])
        right_bot_x = torch.tensor([(i+1)*(grid_length) for i in range(image_width // grid_length)])
        right_bot_y = torch.tensor([(i+1)*(grid_length) for i in range(image_height // grid_length)])
        x_inds = torch.randint(0, left_top_x.size(0), (self.num_regions_per_img,))
        y_inds = torch.randint(0, left_top_y.size(0), (self.num_regions_per_img,))

        proposals = []
        for i in range(self.num_regions_per_img):
            rb_x_candidates = right_bot_x[x_inds[i]:]
            rb_x = rb_x_candidates[torch.randperm(rb_x_candidates.size(0))[0]]
            rb_y_candidates = right_bot_y[y_inds[i]:]
            rb_y = rb_y_candidates[torch.randperm(rb_y_candidates.size(0))[0]]
            this_box = torch.cat((left_top_x[x_inds[i]].view(1,1), left_top_y[y_inds[i]].view(1,1), rb_x.view(1,1), rb_y.view(1,1)),dim=-1)
            proposals.append(this_box)
        proposals = torch.cat(proposals).float().to(self.device)
        proposals = [Boxes(proposals) for i in range(len(batched_inputs))] # a list of Boxes
        return proposals

    def create_global_proposals(self, batched_inputs):
        """ create a single global box for an image, so as to extract global image features with RoIAlign on high-resolution images.
        """
        images = self.preprocess_image(batched_inputs)
        image_height = images.tensor.size(2)
        image_width = images.tensor.size(3)

        global_box = torch.tensor([0, 0, image_width, image_height]).view(1,4).float().to(self.device)
        proposals = [Boxes(global_box) for i in range(len(batched_inputs))] # a list of Boxes
        return proposals

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        pass

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(instances, batched_inputs):
            height, width = input_per_image[-1][2] # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


def visualize_proposals(batched_inputs, proposals, input_format, vis_pretrain=False):
    """
    A function used to visualize images and proposals. It shows ground truth
    bounding boxes on the original image and up to 20 top-scoring predicted
    object proposals on the original image. Users can implement different
    visualization functions for different models.

    Args:
        batched_inputs (list): a list that contains input to the model.
        proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
    """
    from detectron2.utils.visualizer import Visualizer

    max_vis_prop = 50
    if vis_pretrain:
        for i, (input, prop) in enumerate(zip(batched_inputs, proposals)):
            img = input[0] * 255.0
            img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = prop_img
            to_save = Image.fromarray(np.array(vis_img, np.uint8))
            to_save.save("output/regions/" + str(i) + ".png")
            #break  # only visualize one image in a batch
    else:
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            #vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            f_n = input['file_name']
            to_save = Image.fromarray(np.array(vis_img, np.uint8))
            to_save.save("output/regions/" + f_n.split("/")[-1].split(".")[0] + ".png")
            #break  # only visualize one image in a batch

import numpy as np

from torch.autograd.function import Function
class GradReverse(Function):
    @classmethod
    def forward(cls, ctx, x):
        #ctx.save_for_backward(result)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #pdb.set_trace()
        #result, = ctx.saved_tensors
        return (grad_output * (-1))

class DAFeatDiscriminator(nn.Module):

    def __init__(self, in_channels, loss_weight=10.0):
        #self.in_channels = in_channels
        self.in_channels = in_channels
        self.loss_weight = loss_weight
        super(DAFeatDiscriminator, self).__init__()
        self._init_layers()
        self.init_weights()
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu2 = nn.LeakyReLU(0.1, inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.cls_domain = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.mse = nn.MSELoss()
        self.gradreverse = GradReverse(1)
        for i, channels in enumerate([[self.in_channels, self.in_channels], 
                                      [self.in_channels, int(self.in_channels/2)], 
                                      [int(self.in_channels/2), 1]]):
            chn_in = channels[0]
            chn_out = channels[1]
            self.cls_domain.append(
                    nn.Conv2d(
                        chn_in,
                        chn_out,
                        1,
                        stride=1,
                        padding=0))
            if i == 2:
                self.norm.append(get_norm('BN', chn_out))
                break
            self.norm.append(get_norm('GN', chn_out))

    def init_weights(self):
        """Initialize weights of the head."""
        def normal_init(module, mean=0, std=1, bias=0):
            nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)
        for m in self.cls_domain:
            normal_init(m, std=0.01)

    def extract_dis_feat(self, x):
        if torch.any(torch.isnan(x)):
            print('0')
        dis_feat = self.gradreverse.apply(x)
        if torch.any(torch.isnan(dis_feat)):
            print('00')
        for idx, (dis_conv, norm) in enumerate(zip(self.cls_domain, self.norm)):
            if idx == 2:
                dis_feat = norm(dis_conv(dis_feat))
                if torch.any(torch.isnan(dis_feat)):
                    print('2')
                break
            dis_feat = self.relu2(norm(dis_conv(dis_feat)))
            if torch.any(torch.isnan(dis_feat)):
                print('1')
        feat_dis_scores = self.sigmoid(dis_feat)
        if torch.any(torch.isnan(dis_feat)):
            print('3')

        return feat_dis_scores

    def loss(self, x):
        # feature domain classification loss
        dis_feat = torch.mean(self.extract_dis_feat(x))
        dis_loss_0 = self.loss_weight*self.mse(dis_feat, torch.tensor(0).cuda().float())
        if torch.isnan(dis_loss_0):
            print('dis_loss_0 is nan!')
            print(torch.any(torch.isnan(dis_feat)))
            print(torch.any(torch.isnan(x)))
            #for name, param in self.cls_domain.named_parameters():
                #print(name)
                #print(param)
            #    print(torch.any(torch.isnan(param)))
            #    print(torch.any(torch.isinf(param)))
        if torch.isinf(dis_loss_0):
            print('dis_loss_0 is inf!')
        dis_loss_1 = self.loss_weight*self.mse(dis_feat, torch.tensor(1).cuda().float())
        if torch.isnan(dis_loss_1):
            print('dis_loss_1 is nan!')
            print(torch.any(torch.isnan(dis_feat)))
            print(torch.any(torch.isnan(x)))
        if torch.isinf(dis_loss_1):
            print('dis_loss_1 is inf!')
        return dis_loss_0, dis_loss_1
