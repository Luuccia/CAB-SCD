import os
import json
import time
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
import torch

from main_model import ChangeDetection
from data.dataloaders import get_eval_loader, index2color
from configs.base import ReadConfigFile
from util.common import check_dirs, init_seed, gpu_info
from util.metrics import SEGEvaluator, BCDEvaluator, SCDEvaluator
from util.logger import Logger


def eval(model, val_loader, num_classes, logger=None, data_semantic=None):
    """
    当训练语义类中不包含unchange
    分别计算 BCD 和 SS 的 mIoU
    """
    evaluator_bcd = BCDEvaluator()
    evaluator_seg = SEGEvaluator(num_classes)
    evaluator_scd = SCDEvaluator(num_classes + 1)

    model.eval()
    start = time.time()
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    with torch.no_grad():
        torch.cuda.empty_cache()
        for _, data in loop:
            img_A, img_B, label_BCD, label_SGA, label_SGB, img_name = data
            img_A = img_A.cuda()
            img_B = img_B.cuda()

            change_bcd, outputs_A, outputs_B = model(img_A, img_B)

            pred_bcd = torch.argmax(change_bcd, dim=1).cpu().detach().numpy().astype("int")
            label_bcd = label_BCD.cpu().numpy().astype("int")
            evaluator_bcd.add_batch(label_bcd, pred_bcd)

            pred_seg_A = torch.argmax(outputs_A, dim=1).cpu().detach().numpy().astype("int")
            pred_seg_B = torch.argmax(outputs_B, dim=1).cpu().detach().numpy().astype("int")
            pred_seg = np.concatenate([pred_seg_A, pred_seg_B], axis=0)
            label_seg = torch.cat([label_SGA, label_SGB], dim=0).cpu().numpy().astype("int")
            evaluator_seg.add_batch(label_seg, pred_seg)

            unchange_mask = label_bcd == 0
            label_scd_A = label_SGA.cpu().numpy().astype("int") + 1
            label_scd_B = label_SGB.cpu().numpy().astype("int") + 1
            label_scd_A[unchange_mask] = 0
            label_scd_B[unchange_mask] = 0
            label_scd = np.concatenate([label_scd_A, label_scd_B], axis=0)

            pred_unchange = pred_bcd == 0
            pred_scd_A = np.copy(pred_seg_A) + 1
            pred_scd_A[pred_unchange] = 0
            pred_scd_B = np.copy(pred_seg_B) + 1
            pred_scd_B[pred_unchange] = 0
            pred_scd = np.concatenate([pred_scd_A, pred_scd_B], axis=0)
            evaluator_scd.add_batch(label_scd, pred_scd)

    curr_time = time.time() - start
    res_dict = {
        "BCD": evaluator_bcd.get_scores(),
        "SS": evaluator_seg.get_scores(),
        "SCD": evaluator_scd.get_scores(),
    }
    class_iou = {}
    for idx, iou in enumerate(res_dict["SS"]["class_IoUs"]):
        if data_semantic is not None:
            class_iou[data_semantic[idx]] = {"IoU": iou, "F1": res_dict["SS"]["class_F1s"][idx]}
        else:
            class_iou[f"Class {idx}"] = {"IoU": iou, "F1": res_dict["SS"]["class_F1s"][idx]}
    res_dict["SS"]["class"] = class_iou
    res_dict["SS"].pop("class_F1s")
    res_dict["SS"].pop("class_IoUs")

    if logger is not None:
        logger.info(f"eval for {curr_time:.1f}s:")
        logger.info(json.dumps(res_dict, indent=4))
    else:
        print(f"eval for {curr_time:.1f}s:")
        print(json.dumps(res_dict, indent=4))

    return res_dict


def load_model(model, checkpoint_path, logger=None):
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} doesn't exist."
    checkpoint = torch.load(checkpoint_path)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    ckp_iter = checkpoint["iteration"]
    if logger is not None:
        logger.info("Loaded checkpoint from '{}' (iteration {})".format(checkpoint_path, ckp_iter))
        if missing_keys:
            logger.info(f"The model is missing the following keys (unloaded): {missing_keys}")
        if unexpected_keys:
            logger.info(f"The checkpoint contains redundant keys (unused): {unexpected_keys}")
    del checkpoint
    return model


def test(args):
    cfg_file = ReadConfigFile(args.config)
    cfg = cfg_file.get_config()

    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SOLVER.CUDA

    output_dir = cfg.OUTPUT_DIR
    save_path, results_save_path, dir_info = check_dirs(output_dir, phase="eval")
    ckp_file = args.checkpoint
    # write log
    logger = Logger(filename=os.path.join(save_path, "eval.log")).logger
    logger.info("Check Dirs:\n" + dir_info)
    logger.info("Environment Info:\n" + gpu_info())

    model = ChangeDetection(cfg).cuda()
    model = load_model(model, ckp_file, logger)
    model.eval()

    num_classes = cfg.DATASETS.NUM_CLASSES
    test_loader, data_config = get_eval_loader(cfg, logger)
    eval(model, test_loader, num_classes, logger, data_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Change Detection eval")
    parser.add_argument("--config", type=str, default="configs/landsat.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )

    opt = parser.parse_args()
    test(opt)
