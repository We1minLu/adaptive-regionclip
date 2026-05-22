# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import time
import weakref
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer", "DASimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

class DASimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader_s, data_loader_t, optimizer, is_prompt_tuning, cfg=None, teacher_model=None):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader_s = data_loader_s
        self.data_loader_t = data_loader_t
        self._data_loader_iter_s = iter(data_loader_s)
        self._data_loader_iter_t = iter(data_loader_t)
        self.optimizer = optimizer
        self.is_prompt_tuning = is_prompt_tuning
        self.cfg = cfg
        self.teacher_model = teacher_model
        self.use_image_level_ema_kd = bool(cfg is not None and cfg.MODEL.EMA_TEACHER.ENABLED)
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def sync_teacher_with_student(self):
        if self.teacher_model is None:
            return
        self.teacher_model.load_state_dict(self._unwrap_model().state_dict())
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def update_teacher_ema(self):
        if self.teacher_model is None:
            return
        decay = self.cfg.MODEL.EMA_TEACHER.DECAY
        student = self._unwrap_model()
        student_params = dict(student.named_parameters())
        student_buffers = dict(student.named_buffers())
        with torch.no_grad():
            for name, teacher_param in self.teacher_model.named_parameters():
                if name.startswith("offline_backbone.") or name.startswith("offline_proposal_generator."):
                    continue
                student_param = student_params.get(name)
                if student_param is not None:
                    teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
            for name, teacher_buffer in self.teacher_model.named_buffers():
                if name.startswith("offline_backbone.") or name.startswith("offline_proposal_generator."):
                    continue
                student_buffer = student_buffers.get(name)
                if student_buffer is not None:
                    teacher_buffer.copy_(student_buffer)

    def _strong_augment_batch(self, batched_inputs):
        if not self.use_image_level_ema_kd:
            return batched_inputs
        return [self._strong_augment_input(x) for x in batched_inputs]

    def _strong_augment_input(self, input_dict):
        output = dict(input_dict)
        image = input_dict["image"].clone().float()
        cfg = self.cfg.MODEL.EMA_TEACHER
        if torch.rand(()) < cfg.COLOR_JITTER_PROB:
            strength = cfg.COLOR_JITTER_STRENGTH
            brightness = 1.0 + (torch.rand(()) * 2.0 - 1.0) * strength
            contrast = 1.0 + (torch.rand(()) * 2.0 - 1.0) * strength
            saturation = 1.0 + (torch.rand(()) * 2.0 - 1.0) * strength
            image = image * brightness
            mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - mean) * contrast + mean
            gray = (0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3])
            image = (image - gray) * saturation + gray
        if torch.rand(()) < cfg.GRAYSCALE_PROB:
            gray = (0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3])
            image = gray.expand_as(image)
        if torch.rand(()) < cfg.GAUSSIAN_BLUR_PROB:
            image = self._gaussian_blur(image)
        if torch.rand(()) < cfg.CUTOUT_PROB:
            image = self._cutout(image, cfg.CUTOUT_SCALE)
        output["image"] = image.clone().clamp_(0.0, 255.0).to(input_dict["image"].dtype)
        return output

    def _gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        radius = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.expand(image.shape[0], 1, kernel_size, kernel_size)
        return F.conv2d(image.unsqueeze(0), kernel, padding=radius, groups=image.shape[0]).squeeze(0)

    def _cutout(self, image, scale):
        image = image.clone()
        _, h, w = image.shape
        cut_h = max(1, int(h * scale * float(torch.rand(()))))
        cut_w = max(1, int(w * scale * float(torch.rand(()))))
        y0 = int(torch.randint(0, max(1, h - cut_h + 1), (1,)))
        x0 = int(torch.randint(0, max(1, w - cut_w + 1), (1,)))
        image[:, y0:y0 + cut_h, x0:x0 + cut_w] = 0
        return image

    def _image_level_kd_loss(self, student_probs, teacher_probs):
        losses = []
        for student_prob, teacher_prob in zip(student_probs, teacher_probs):
            student_prob = student_prob.clamp(1e-6, 1.0 - 1e-6)
            teacher_prob = teacher_prob.detach().clamp(0.0, 1.0)
            losses.append(F.binary_cross_entropy(student_prob, teacher_prob, reduction="mean"))
        if not losses:
            return torch.tensor(0.0, device=self._unwrap_model().device)
        return torch.stack(losses).mean()

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        ####################################
        if self.is_prompt_tuning:
            # After pre-trained, freeze the model's parameters and conduct prompt tuning
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
            for name, param in self.model.named_parameters():
                if name == 'roi_heads.box_predictor.DAHead.prompt_learner.ctx_di' or name == 'roi_heads.box_predictor.DAHead.prompt_learner.ctx_ds':
                    param.requires_grad = True
        #print('learning layers:')
        #for name, param in self.model.named_parameters():
        #    if param.requires_grad:
        #        print(name)
            #quit()
        ####################################
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data_s = next(self._data_loader_iter_s)
        data_t = next(self._data_loader_iter_t)
        data_t_student = self._strong_augment_batch(data_t)
        kd_active = (
            self.use_image_level_ema_kd
            and self.iter >= self.cfg.MODEL.EMA_TEACHER.WARMUP_ITERS
        )
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict_s = self.model(data_s, is_source = True)
        import math
        if math.isnan(loss_dict_s['loss_dis_0']) or math.isnan(loss_dict_s['loss_box_reg']):
            print('loss_dis_0 or loss_box_reg is nan!')
            #for name, param in self.model.named_parameters():
            #    if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
            #        print('name:{} param grad:{} '.format(name, param.grad))
            #        if torch.isnan(param).any():
            #            print('aaaaaaaa')
            
        del loss_dict_s['loss_dis_1']
        loss_dict_s.pop('loss_dis_c4_1', None)
        loss_dict_s.pop('loss_dis_c5_1', None)
        teacher_probs = None
        if kd_active:
            with torch.no_grad():
                teacher_probs = self.teacher_model(data_t, is_source=False, image_level_only=True)
        if kd_active:
            loss_dict_t, student_probs = self.model(
                data_t_student,
                is_source=False,
                return_image_level=True,
            )
        else:
            loss_dict_t = self.model(data_t_student, is_source = False)
        del loss_dict_t['loss_dis_0']
        loss_dict_t.pop('loss_dis_c4_0', None)
        loss_dict_t.pop('loss_dis_c5_0', None)
        del loss_dict_t['loss_cls']
        del loss_dict_t['loss_box_reg']
        if kd_active:
            loss_dict_t["loss_img_kd"] = (
                self._image_level_kd_loss(student_probs, teacher_probs)
                * self.cfg.MODEL.EMA_TEACHER.IMG_KD_WEIGHT
            )
        if isinstance(loss_dict_s, torch.Tensor):
            # not used, acturally
            losses_s = loss_dict_s
            losses_t = loss_dict_t
            loss_dict_s = {"total_loss_s": loss_dict_s}
            loss_dict_t = {"total_loss_t": loss_dict_t}
            losses = losses_s + losses_t
        else:
            losses_s = sum(loss_dict_s.values())
            losses_t = sum(loss_dict_t.values())
            losses = losses_s + losses_t

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.requires_grad_(True)
        losses.backward()
        #for name, parms in self.model.named_parameters():
        #    if parms.requires_grad:
        #        print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
        loss_dict = dict(loss_dict_s, **loss_dict_t)

        self._write_metrics(loss_dict, data_time)


        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.is_prompt_tuning:
            self.update_ema_buffer(self.model, 0.99, self.iter)

        self.optimizer.step()
        if self.use_image_level_ema_kd:
            self.update_teacher_ema()
        # EMA update

    # EMA update
    def update_ema_buffer(self, x, alpha, iteration):
        alpha = min(1 - 1 / (iteration + 1), alpha)
        #print(x.roi_heads.box_predictor.DAHead.prompt_learner.ctx_di)
        #print(x.roi_heads.box_predictor.DAHead.prompt_learner.ctx_di_ema)
        x.roi_heads.box_predictor.DAHead.prompt_learner.ctx_di_ema.mul_(alpha).add_(1 - alpha, x.roi_heads.box_predictor.DAHead.prompt_learner.ctx_di)
        x.roi_heads.box_predictor.DAHead.prompt_learner.ctx_ds_ema.mul_(alpha).add_(1 - alpha, x.roi_heads.box_predictor.DAHead.prompt_learner.ctx_ds)

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
