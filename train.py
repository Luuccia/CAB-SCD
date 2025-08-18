import os
import copy
import itertools
import argparse
from typing import Any, Dict, List, Set

import torch
from torch import nn

from eval import eval
from main_model import ChangeDetection
from data.dataloaders import get_loaders
from configs.base import ReadConfigFile
from util.common import check_dirs, init_seed, gpu_info
from util.loss import SS_CrossEntropyLoss, PossibilityConstraint, ChangeSimilarity
from util.lr_scheduler import build_lr_scheduler
from util.logger import Logger


def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    # import ipdb;
    # ipdb.set_trace()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            hyperparams = copy.copy(defaults)
            if "clip_model" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
            # for deformable detr

            if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")

    return optimizer


def train(args):
    cfg_file = ReadConfigFile(args.config)
    cfg = cfg_file.get_config()

    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SOLVER.CUDA

    output_dir = cfg.OUTPUT_DIR
    save_path, ckp_save_path, dir_info = check_dirs(output_dir)
    best_ckp_file = os.path.join(save_path, "best.pth")
    # copy config file
    cfg_file.save_config(os.path.join(save_path, "config.yaml"))
    # write log
    logger = Logger(filename=os.path.join(save_path, "train.log")).logger
    logger.info("Check Dirs:\n" + dir_info)
    logger.info("Environment Info:\n" + gpu_info())

    model = ChangeDetection(cfg).cuda()
    optimizer = build_optimizer(cfg, model)

    num_classes = cfg.DATASETS.NUM_CLASSES
    train_loader, val_loader, train_len = get_loaders(cfg, logger)

    # 初始化迭代器
    iter_train_loader = iter(train_loader)
    scheduler = build_lr_scheduler(cfg, optimizer)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    bcd_ignore_value = cfg.DATASETS.BCD_IGNORE_VALUE
    bcd_loss_criterion = nn.CrossEntropyLoss(ignore_index=bcd_ignore_value)
    seg_loss_criterion = SS_CrossEntropyLoss(ignore_index=num_classes, weight=cfg.SOLVER.SS_WEIGHT)
    cs_loss_criterion = ChangeSimilarity(ignore_index=bcd_ignore_value)
    pc_loss_criterion = PossibilityConstraint()

    metric_best = {"mIoU": -1, "OA": -1, "F1": -1}
    eval_period = min(train_len, cfg.TEST.EVAL_PERIOD)

    for iteration in range(start_iter, max_iter):
        model.train()

        try:
            data = next(iter_train_loader)
        except StopIteration:
            # 如果迭代器耗尽，则重新初始化迭代器
            iter_train_loader = iter(train_loader)
            data = next(iter_train_loader)
        img_A, img_B, label_BCD, label_SGA, label_SGB, _ = data
        img_A = img_A.cuda()
        img_B = img_B.cuda()
        labels_SGA = label_SGA.cuda().long()
        labels_SGB = label_SGB.cuda().long()
        labels_BCD = label_BCD.cuda().long()

        change_bcd, outputs_A, outputs_B = model(img_A, img_B)

        loss_seg = (seg_loss_criterion(outputs_A, labels_SGA) + seg_loss_criterion(outputs_B, labels_SGB)) * 0.5
        loss_bcd = bcd_loss_criterion(change_bcd, labels_BCD)
        if cfg.SOLVER.USE_CS_LOSS:
            loss_cs = cs_loss_criterion(outputs_A[0], outputs_B[0], labels_BCD)
        else:
            loss_cs = torch.tensor(0)
        if cfg.SOLVER.USE_PC_LOSS:
            loss_pc = pc_loss_criterion(outputs_A[0], outputs_B[0], change_bcd)
        else:
            loss_pc = torch.tensor(0)
        losses = loss_seg + loss_cs + loss_bcd + loss_pc

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        # write logs
        logger.info(
            f"Iteration {iteration} Train Finish. BCD Loss: {round(loss_bcd.item(), 3)}, SEG Loss: {round(loss_seg.item(), 3)}, "
            + f"CS Loss: {round(loss_cs.item(), 3)}, PC Loss: {round(loss_pc.item(), 3)}"
        )

        # validate, at least by epoch
        if (iteration + 1) % eval_period == 0 or (iteration + 1) == max_iter:
            logger.info("\nEval start.")
            eval_results = eval(model, val_loader, num_classes, logger)
            model.train()  # 恢复训练模式

            # save best checkpoint
            refer_metric = eval_results["SCD"]["mIoU"]
            if refer_metric > metric_best["mIoU"]:
                if best_ckp_file is not None and os.path.exists(best_ckp_file):
                    os.remove(best_ckp_file)
                best_ckp_file = os.path.join(
                    ckp_save_path,
                    "_".join(
                        [
                            "best_iter",
                            str(iteration),
                            "IoU",
                            str(round(float(eval_results.get("BCD", {"IoU": 0})["IoU"]), 5)),
                            "SS",
                            str(round(float(eval_results["SS"]["mIoU"]), 5)),
                            "mIoU",
                            str(round(float(eval_results["SCD"]["mIoU"]), 5)),
                        ]
                    )
                    + ".pth",
                )
                checkpoint = {
                    "iteration": iteration,
                    "model": model.state_dict(),
                }
                torch.save(checkpoint, best_ckp_file)
            metric_best = {
                "mIoU": max(metric_best["mIoU"], eval_results["SCD"]["mIoU"]),
                "F1": max(metric_best["F1"], eval_results["SCD"]["F1"]),
                "OA": max(metric_best["OA"], eval_results["SCD"]["OA"]),
            }
            logger.info(
                "Best metrics till now:\n"
                + "Semantic Change Detection: F1: %.2f, mIoU: %.2f, OA: %.2f"
                % (
                    metric_best["F1"] * 100,
                    metric_best["mIoU"] * 100,
                    metric_best["OA"] * 100,
                )
            )
            logger.info("Eval Finish.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Change Detection train")
    parser.add_argument("--config", type=str, default="./configs/hiucd_mini.yaml")

    opt = parser.parse_args()
    train(opt)
