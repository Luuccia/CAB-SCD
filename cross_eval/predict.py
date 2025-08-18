import os
import torch
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
from PIL import Image

from main_model import ChangeDetection
from configs.base import ReadConfigFile

# Dataset config
DATASETS = {
    "whucd": {
        "root": "/archive/hot0/dql/CD-data/whucd-256",
        "folders": ["A", "B"],
        "process": "building",
        "config": "eval_ov/binary.yaml",
    },
    "levir": {
        "root": "/archive/hot0/dql/CD-data/levir-256",
        "folders": ["A", "B"],
        "process": "building",
        "config": "eval_ov/binary.yaml",
    },
}
MIDDLE_DIR = "exp/middle_scd"


def predict(dataset, checkpoint_path, cfg):
    root = DATASETS[dataset]["root"]
    folders = DATASETS[dataset]["folders"]

    output_dir = os.path.join(root, MIDDLE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    for dname in ["im1", "im2", "binary"]:
        dname = os.path.join(root, MIDDLE_DIR, dname)
        os.makedirs(dname, exist_ok=True)

    model = ChangeDetection(cfg).cuda()
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} doesn't exist."
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=False)
    ckp_iter = checkpoint["iteration"]
    print("Loaded checkpoint from '{}' (iteration {})".format(checkpoint_path, ckp_iter))
    del checkpoint
    model.eval()

    with torch.no_grad():
        for file_name in tqdm(os.listdir(os.path.join(root, folders[0])), desc="Processing", unit="iteration"):
            # Construct file paths for the images
            img1_path = os.path.join(root, folders[0], file_name)
            img2_path = os.path.join(root, folders[1], file_name)

            # Read the input images
            # img_A = torch.tensor(imread(img1_path).transpose(2, 0, 1)).unsqueeze(0).cuda()
            # img_B = torch.tensor(imread(img2_path).transpose(2, 0, 1)).unsqueeze(0).cuda()

            img_A = np.array(Image.open(img1_path)).astype(np.float32).transpose((2, 0, 1))
            img_A = torch.from_numpy(img_A).type(torch.FloatTensor).unsqueeze(0).cuda()
            img_B = np.array(Image.open(img2_path)).astype(np.float32).transpose((2, 0, 1))
            img_B = torch.from_numpy(img_B).type(torch.FloatTensor).unsqueeze(0).cuda()

            change_bcd, outputs_A, outputs_B = model(img_A, img_B)
            pred_bcd = torch.argmax(change_bcd, dim=1).squeeze().cpu().detach().numpy().astype(np.uint8)
            pred_seg_A = torch.argmax(outputs_A, dim=1).squeeze().cpu().detach().numpy().astype(np.uint8)
            pred_seg_B = torch.argmax(outputs_B, dim=1).squeeze().cpu().detach().numpy().astype(np.uint8)
            imsave(os.path.join(output_dir, "binary", file_name), pred_bcd)
            imsave(os.path.join(output_dir, "im1", file_name), pred_seg_A)
            imsave(os.path.join(output_dir, "im2", file_name), pred_seg_B)


def building_post_process(dataset):
    root = DATASETS[dataset]["root"]
    folders = [MIDDLE_DIR + "/im1", MIDDLE_DIR + "/im2", MIDDLE_DIR + "/binary"]
    targets = ["bcd", "ss", "and", "or"]

    for dname in targets:
        output_dir = os.path.join(root, "exp", "test_" + dname)
        os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(os.path.join(root, folders[0])), desc="Processing", unit="iteration"):
        img1_path = os.path.join(root, folders[0], file_name)
        img2_path = os.path.join(root, folders[1], file_name)
        binary_path = os.path.join(root, folders[2], file_name)

        img1 = imread(img1_path)
        img2 = imread(img2_path)
        bcd = imread(binary_path)
        shape = img1.shape[:2]

        ss_change = ((img1 == 0) & (img1 != img2)) | ((img2 == 0) & (img1 != img2))
        bcd_change = bcd > 0
        ori_img = np.zeros((shape[0], shape[1]), dtype=np.uint8)

        bcd_result = np.copy(ori_img)
        bcd_result[bcd_change] = 255
        imsave(os.path.join(root, "exp", "test_" + targets[0], file_name), bcd_result)

        ss_result = np.copy(ori_img)
        ss_result[ss_change] = 255
        imsave(os.path.join(root, "exp", "test_" + targets[1], file_name), ss_result)

        and_result = np.copy(ori_img)
        and_result[bcd_change & ss_change] = 255
        imsave(os.path.join(root, "exp", "test_" + targets[2], file_name), and_result)

        or_result = np.copy(ori_img)
        or_result[bcd_change | ss_change] = 255
        imsave(os.path.join(root, "exp", "test_" + targets[3], file_name), or_result)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    ckp_path = "weights/hiucd_mini/eval_37.48_v3/best_iter_41399_IoU_0.5443_SS_0.55218_mIoU_0.7414.pth"

    for dataset in DATASETS.keys():
        cfg_path = DATASETS[dataset]["config"]
        cfg_file = ReadConfigFile(cfg_path)
        cfg = cfg_file.get_config()
        predict(dataset, ckp_path, cfg)

        process = DATASETS[dataset]["process"]
        if process == "building":
            building_post_process(dataset)
