import os
import shutil
import time
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
import torch

from main_model import ChangeDetection
from data.dataloaders import index2color
from configs.base import ReadConfigFile
from util.common import init_seed

# Landsat
OUTPUT_DIR = "/home/daiqiulu@corp.sse.tongji.edu.cn/workspace/experiments/landsat"
root = "/archive/hot1/dql/Landsat-split/test"
imgs = ["From1990To2001_02.png", "From1990To2015_01.png"]
splits = ["im1", "im2"]

# HRSCD
# root = "/archive/hot0/dql/CD-data/HRSCD/test"
# splits = ["im1", "im2"]
# OUTPUT_DIR = "/home/daiqiulu@corp.sse.tongji.edu.cn/workspace/experiments/hrscd"
# imgs = [
#     "0470-6905-LA93_8_2.tif",
#     "0330-6770-LA93_2_8.tif",
#     "0330-6780-LA93_9_8.tif",
#     "0460-6915-LA93_9_16.tif",
#     "0335-6770-LA93_5_15.tif",
#     "0335-6780-LA93_14_10.tif",
#     "0350-6800-LA93_12_14.tif",
#     "0355-6775-LA93_3_18.tif",
#     "0370-6790-LA93_13_3.tif",
#     "0425-6900-LA93_17_0.tif",
#     "0335-6795-LA93_6_8.tif",
#     "0340-6780-LA93_7_6.tif",
#     "0340-6795-LA93_0_0.tif",
# ]
# OUTPUT_DIR = "/archive/hot0/dql/CD-data/HRSCD/test/visual"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# imgs = os.listdir("/archive/hot0/dql/CD-data/HRSCD/test/im1")


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


def predict(args):
    cfg_file = ReadConfigFile(args.config)
    cfg = cfg_file.get_config()

    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SOLVER.CUDA
    data_type = cfg.DATASETS.TYPE
    num_classes = cfg.DATASETS.NUM_CLASSES

    ckp_file = args.checkpoint
    model = ChangeDetection(cfg).cuda()
    model = load_model(model, ckp_file)
    model.eval()

    begin_time = time.time()
    for img in tqdm(imgs):
        img1_path = os.path.join(root, splits[0], img)
        img2_path = os.path.join(root, splits[1], img)

        img_name = img.split(".")[0]
        suffix = img.split(".")[-1]
        save_path = os.path.join(OUTPUT_DIR, img_name)
        if not os.path.exists(save_path):
            os.makedirs(os.path.join(save_path, "predict1"))
            os.makedirs(os.path.join(save_path, "predict2"))
            shutil.copyfile(img1_path, os.path.join(save_path, "t1." + suffix))
            shutil.copyfile(img2_path, os.path.join(save_path, "t2." + suffix))
            shutil.copyfile(os.path.join(root, "label1", img_name + ".png"), os.path.join(save_path, "label1.png"))
            shutil.copyfile(os.path.join(root, "label2", img_name + ".png"), os.path.join(save_path, "label2.png"))

        img_A = np.array(Image.open(img1_path)).astype(np.float32).transpose((2, 0, 1))
        img_A = torch.from_numpy(img_A).type(torch.FloatTensor).unsqueeze(0).cuda()
        img_B = np.array(Image.open(img2_path)).astype(np.float32).transpose((2, 0, 1))
        img_B = torch.from_numpy(img_B).type(torch.FloatTensor).unsqueeze(0).cuda()

        with torch.no_grad():
            change_bcd, outputs_A, outputs_B = model(img_A, img_B)

            pred_bcd = torch.argmax(change_bcd, dim=1).cpu().detach().squeeze().numpy().astype("int")
            pred_seg_A = torch.argmax(outputs_A, dim=1).cpu().detach().squeeze().numpy().astype("int")
            pred_seg_B = torch.argmax(outputs_B, dim=1).cpu().detach().squeeze().numpy().astype("int")

            pred_unchange = pred_bcd == 0
            pred_scd_A = np.copy(pred_seg_A)
            pred_scd_A[pred_unchange] = num_classes
            pred_scd_B = np.copy(pred_seg_B)
            pred_scd_B[pred_unchange] = num_classes

            pred_A = index2color(pred_scd_A, data_type)
            palette_image1 = Image.fromarray(pred_A)
            palette_image1.save(os.path.join(save_path, "predict1", "CAB-SCD.png"))

            pred_B = index2color(pred_scd_B, data_type)
            palette_image2 = Image.fromarray(pred_B)
            palette_image2.save(os.path.join(save_path, "predict2", "CAB-SCD.png"))

    time_use = time.time() - begin_time
    print("Total time: %.2fs" % time_use)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Change Detection predict")
    parser.add_argument("--config", type=str, default="weights/landsat/20250809_1841_train/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/landsat/eval_v3_64.92/best_iter_29987_IoU_0.79714_SS_0.90966_mIoU_0.87357.pth",
    )

    opt = parser.parse_args()
    predict(opt)
