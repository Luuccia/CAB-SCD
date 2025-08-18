import os
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.dataloaders import index2color


# step 1: 将 10000*10000 的图像分成 512*512 的不重叠图像
# step 2: 统计整个数据集中的变化像素占比 平均值 7.7%
def count_change_ratio(root, size=512):
    """
    统计数据集中所有 change=1 的像素占比(发现语义图里似乎没有不变类)
    Args:
        root (str): 存放标签图像的目录
    Returns:
        list: 要绘制直方图的变化像素占比
    """
    total_change = 0
    total_pixels = 0
    data = []
    pixels = size * size
    for fname in tqdm(os.listdir(root)):
        img_path = os.path.join(root, fname)
        img = Image.open(img_path)
        img = np.asarray(img)
        label = img[:, :, -1]
        total_change += np.sum(label == 1)
        total_pixels += pixels
        ratio = np.sum(label == 1) / pixels
        data.append(ratio)

    avg_ratio = total_change / total_pixels
    print(avg_ratio)
    return data


# step 3: 选择变化像素占比大于 5% 的数据 按 3:1:1 的比例划分
def split_dataset(root, size=512):
    pixels = size * size
    val_ratio = 1 / 5.0
    test_ratio = 1 / 5.0

    label_dir = os.path.join(root, "label")
    img_dir1 = os.path.join(root, "im1")
    img_dir2 = os.path.join(root, "im2")
    for mode in ["train", "test", "val"]:
        os.makedirs(os.path.join(root, mode, "im1"))
        os.makedirs(os.path.join(root, mode, "im2"))
        os.makedirs(os.path.join(root, mode, "label"))
    valid_list = []

    for fname in tqdm(os.listdir(label_dir)):
        img_path = os.path.join(label_dir, fname)
        img = Image.open(img_path)
        img = np.asarray(img)
        label = img[:, :, -1]
        ratio = np.sum(label == 1) / pixels
        if ratio > 0.005:
            valid_list.append(fname)

    num_pics = len(valid_list)
    num_val = int(num_pics * val_ratio)
    num_test = int(num_pics * test_ratio)
    num_train = num_pics - num_val - num_test
    print("num_pics: %d, num_train: %d, num_val:%d, num_test:%d." % (num_pics, num_train, num_val, num_test))

    random.shuffle(valid_list)
    for idx, it in enumerate(valid_list):
        if idx < num_train:
            mode = "train"
        elif idx < (num_train + num_val):
            mode = "val"
        else:
            mode = "test"
        shutil.copy(os.path.join(label_dir, it), os.path.join(root, mode, "label"))
        shutil.copy(os.path.join(img_dir1, it), os.path.join(root, mode, "im1"))
        shutil.copy(os.path.join(img_dir2, it), os.path.join(root, mode, "im2"))

    print(len(os.listdir(os.path.join(root, "train", "label"))))


# 从 处理后的三通道汇总标签 中提取 有颜色的语义变化图
def generate_semantic_maps(root):
    label1_path = os.path.join(root, "label1")
    label2_path = os.path.join(root, "label2")
    if not os.path.exists(label1_path):
        os.makedirs(label1_path)
        os.makedirs(label2_path)
    else:
        print(f"The folder have been generated.")
        return

    ori_path = os.path.join(root, "label")
    img_names = os.listdir(ori_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(ori_path, img_name)
        img = np.asarray(Image.open(img_path))
        new_name = img_name.replace("tif", "png")
        change_map = img[:, :, 2]

        label1 = img[:, :, 0].copy()
        label1[change_map == 0] = 5
        label1 = index2color(label1, "HRSCD")
        palette_image1 = Image.fromarray(label1)
        palette_image1.save(os.path.join(label1_path, new_name))

        label2 = img[:, :, 1].copy()
        label2[change_map == 0] = 5
        label2 = index2color(label2, "HRSCD")
        palette_image2 = Image.fromarray(label2)
        palette_image2.save(os.path.join(label2_path, new_name))


def has_pixel(image_path, target_pixel):
    img_rgb = np.asarray(Image.open(image_path))
    return np.any(np.all(img_rgb == target_pixel, axis=-1))


if __name__ == "__main__":
    # root = "/archive/hot0/dql/CD-data/HRSCD/test"
    # generate_semantic_maps(root)
    # img_path = "/archive/hot0/dql/CD-data/HRSCD/test/label/0350-6780-LA93_12_0.tif"
    # img = np.asarray(Image.open(img_path))
    # unique, counts = np.unique(img[:, :, 2], return_counts=True)
    # print(dict(zip(unique, counts)))
    tmp = ["0460-6915-LA93_9_16", "0350-6815-LA93_17_15", "0325-6775-LA93_12_6"]
    for path in tmp:
        shutil.copyfile(f"/archive/hot0/dql/CD-data/HRSCD/test/label1/{path}.png", f"./label/{path}_label1.png")
        shutil.copyfile(f"/archive/hot0/dql/CD-data/HRSCD/test/label2/{path}.png", f"./label/{path}_label2.png")
