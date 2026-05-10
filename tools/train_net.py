#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, DATrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation.testing import flatten_results_dict

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class BestModelHook(hooks.HookBase):
    def __init__(self, metric_name="bbox/AP50"):
        self._metric_name = metric_name
        self._best_metric = float("-inf")
        self._last_eval_iter = None
        self._logger = logging.getLogger("detectron2.best_model")

    def _maybe_save(self):
        if not comm.is_main_process():
            return
        if not hasattr(self.trainer, "_last_eval_results"):
            return
        if self._last_eval_iter == self.trainer.iter:
            return
        results = self.trainer._last_eval_results
        if not results:
            return
        flattened = flatten_results_dict(results)
        if self._metric_name not in flattened:
            self._logger.warning(
                "Metric %s was not found in eval results: %s",
                self._metric_name,
                sorted(flattened.keys()),
            )
            return
        metric = float(flattened[self._metric_name])
        self._last_eval_iter = self.trainer.iter
        if metric > self._best_metric:
            self._best_metric = metric
            self._logger.info(
                "New best %s=%.4f at iter %d. Saving model_best.pth",
                self._metric_name,
                metric,
                self.trainer.iter,
            )
            self.trainer.checkpointer.save(
                "model_best",
                iteration=self.trainer.iter,
                best_metric=metric,
                best_metric_name=self._metric_name,
            )

    def after_step(self):
        self._maybe_save()

    def after_train(self):
        self._maybe_save()


class TeacherPeriodicCheckpointer(hooks.HookBase):
    def __init__(self, period):
        self._period = period
        self._checkpointer = None

    def _teacher_model(self):
        trainer = getattr(self.trainer, "_trainer", None)
        return getattr(trainer, "teacher_model", None)

    def before_train(self):
        teacher_model = self._teacher_model()
        if teacher_model is not None and comm.is_main_process():
            self._checkpointer = DetectionCheckpointer(
                teacher_model,
                save_dir=self.trainer.cfg.OUTPUT_DIR,
            )

    def _save(self, name):
        if self._checkpointer is None:
            return
        self._checkpointer.save(name, iteration=self.trainer.iter)

    def after_step(self):
        if self._period <= 0:
            return
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            self._save("teacher_model_{:07d}".format(self.trainer.iter))

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._save("teacher_model_final")


class TeacherEvalHook(hooks.HookBase):
    def __init__(self, eval_period, cfg, trainer_cls, metric_name="bbox/AP50"):
        self._period = eval_period
        self._cfg = cfg.clone()
        self._trainer_cls = trainer_cls
        self._metric_name = metric_name
        self._best_metric = float("-inf")
        self._checkpointer = None
        self._logger = logging.getLogger("detectron2.teacher_eval")

    def _teacher_model(self):
        trainer = getattr(self.trainer, "_trainer", None)
        return getattr(trainer, "teacher_model", None)

    def before_train(self):
        teacher_model = self._teacher_model()
        if teacher_model is not None and comm.is_main_process():
            self._checkpointer = DetectionCheckpointer(
                teacher_model,
                save_dir=self._cfg.OUTPUT_DIR,
            )

    def _do_eval(self):
        teacher_model = self._teacher_model()
        if teacher_model is None:
            return
        self._logger.info("Running EMA teacher evaluation ...")
        results = self._trainer_cls.test(self._cfg, teacher_model)
        self.trainer._last_teacher_eval_results = results
        if results:
            flattened = flatten_results_dict(results)
            prefixed = {}
            for k, v in flattened.items():
                prefixed["teacher_" + k] = float(v)
            self.trainer.storage.put_scalars(**prefixed, smoothing_hint=False)
            if self._metric_name in flattened:
                metric = float(flattened[self._metric_name])
                if metric > self._best_metric:
                    self._best_metric = metric
                    self._logger.info(
                        "New best teacher %s=%.4f at iter %d. Saving teacher_model_best.pth",
                        self._metric_name,
                        metric,
                        self.trainer.iter,
                    )
                    if self._checkpointer is not None:
                        self._checkpointer.save(
                            "teacher_model_best",
                            iteration=self.trainer.iter,
                            best_metric=metric,
                            best_metric_name="teacher_" + self._metric_name,
                        )
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def build_hooks(cls):
        ret = super().build_hooks()
        if comm.is_main_process():
            ret.append(BestModelHook("bbox/AP50"))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class DomainTrainer(DATrainer):
    build_evaluator = Trainer.build_evaluator
    test_with_TTA = Trainer.test_with_TTA

    def build_hooks(self):
        ret = super().build_hooks()
        writer = None
        if comm.is_main_process() and ret and isinstance(ret[-1], hooks.PeriodicWriter):
            writer = ret.pop()
        if self.cfg.MODEL.EMA_TEACHER.ENABLED:
            if comm.is_main_process():
                ret.append(TeacherPeriodicCheckpointer(self.cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(TeacherEvalHook(self.cfg.TEST.EVAL_PERIOD, self.cfg, type(self), "bbox/AP50"))
        if comm.is_main_process():
            ret.append(BestModelHook("bbox/AP50"))
        if writer is not None:
            ret.append(writer)
        return ret


def get_trainer_class(cfg):
    return DomainTrainer if len(cfg.DATASETS.TRAIN_S) and len(cfg.DATASETS.TRAIN_T) else Trainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        trainer_cls = get_trainer_class(cfg)
        model = trainer_cls.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
            and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
            and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
                cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
            )
        res = trainer_cls.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(trainer_cls.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = get_trainer_class(cfg)(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
