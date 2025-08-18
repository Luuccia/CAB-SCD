import numpy as np
import os
import random
from PIL import Image
from tqdm import tqdm

AUG_NAMES = ["rotate", "Crop", "ZheDang", "Zhedang"]
# Landsat-SCD original:
# 0: No change, 1: Farmland-desert, 2: Farmland-building, 3: Desert-farmland, 4: Desert-building,
# 5: Desert-water, 6: Building-farmland, 7: Building-desert, 8: water-farmland, 9: water-desert
#
# after processed:
# 0: No change, 1: farmland, 2: desert, 3: building, 4: water
MAP_A = [0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
MAP_B = [0, 2, 3, 1, 3, 4, 1, 2, 1, 2]
SCD_COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]


def is_aug(it_name):
    for aug_name in AUG_NAMES:
        if aug_name in it_name:
            return True
    return False


def remove_aug_samples(root, folders=["A", "B", "label"]):
    """
    去除数据集中与原图像重复、进行了数据增强的图片
    """
    for folder in folders:
        working_dir = os.path.join(root, folder)
        data_list = os.listdir(working_dir)

        for it in data_list:
            if is_aug(it):
                img_path1 = os.path.join(working_dir, it)
                os.remove(img_path1)


def MCD2SCD(label):
    """
    数据集中的标签数值表示From-To的变化类型，需要处理为单独的语义类型
    """
    label = np.asarray(label)
    mapA = np.asarray(MAP_A)
    mapB = np.asarray(MAP_B)
    labelA = mapA[label]
    labelB = mapB[label]
    return labelA.astype(np.uint8), labelB.astype(np.uint8)


def Index2Color(mask):
    """
    将标签索引转换为RGB颜色
    """
    colormap = np.asarray(SCD_COLORMAP, dtype="uint8")
    x = np.asarray(mask, dtype="int32")
    return colormap[x, :]


def mask_invalid(root, name, labelA_rgb, labelB_rgb):
    """
    如果某像素在 imgA 和 imgB 中均为纯白色，则视为无效区域，需要在RGB标签中进行更改
    """
    img_dir1 = os.path.join(root, "A")
    img_dir2 = os.path.join(root, "B")

    img_path1 = os.path.join(img_dir1, name)
    img_path2 = os.path.join(img_dir2, name)
    imgA = np.array(Image.open(img_path1))
    imgB = np.array(Image.open(img_path2))

    invalid_maskA = np.all(imgA == 255, axis=2)
    invalid_maskB = np.all(imgB == 255, axis=2)
    invalid_mask = invalid_maskA * invalid_maskB

    labelA_rgb[invalid_mask] = 255
    labelB_rgb[invalid_mask] = 255
    return labelA_rgb, labelB_rgb


def process_landsat(root):
    remove_aug_samples(root)
    label_path = os.path.join(root, "label")
    data_list = os.listdir(label_path)
    img_nums1 = len(os.listdir(os.path.join(root, "A")))
    img_nums2 = len(os.listdir(os.path.join(root, "B")))
    assert (
        len(data_list) == img_nums1 and len(data_list) == img_nums2
    ), f"After removal, there are {len(data_list)} labels left. However, image numbers are {img_nums1} and {img_nums2}"
    print(f"Dataset Number: {img_nums1}")

    target_dir1 = os.path.join(root, "label1")
    target_dir2 = os.path.join(root, "label2")
    if not os.path.exists(target_dir1):
        os.mkdir(target_dir1)
        os.mkdir(target_dir2)
    for img_name in tqdm(data_list):
        if img_name[-4:] == ".png":
            src_path = os.path.join(label_path, img_name)
            dst_pathA = os.path.join(target_dir1, img_name)
            dst_pathB = os.path.join(target_dir2, img_name)
            label = Image.open(src_path)
            labelA, labelB = MCD2SCD(label)
            labelA = Index2Color(labelA)
            labelB = Index2Color(labelB)
            labelA, labelB = mask_invalid(root, img_name, labelA, labelB)
            Image.fromarray(labelA).save(dst_pathA)
            Image.fromarray(labelB).save(dst_pathB)
    print("Done.")


def split_dataset(root_dir):
    val_ratio = 1 / 5.0
    test_ratio = 1 / 5.0
    img_dir = os.path.join(root_dir, "A")
    train_info = open(os.path.join(root_dir, "train.txt"), "w")
    val_info = open(os.path.join(root_dir, "val.txt"), "w")
    test_info = open(os.path.join(root_dir, "test.txt"), "w")

    img_list = os.listdir(img_dir)
    valid_list = []
    train_list = []
    val_list = []
    test_list = []

    # 校验后缀格式
    for it in img_list:
        it_ext = it[-4:]
        if it_ext == ".png":
            valid_list.append(it)
    num_pics = len(valid_list)
    num_val = int(num_pics * val_ratio)
    num_test = int(num_pics * test_ratio)
    num_train = num_pics - num_val - num_test
    print("num_pics: %d, num_train: %d, num_val:%d, num_test:%d." % (num_pics, num_train, num_val, num_test))

    random.shuffle(valid_list)
    for idx, it in enumerate(valid_list):
        if idx < num_train:
            train_list.append(it)
        elif idx < (num_train + num_val):
            val_list.append(it)
        else:
            test_list.append(it)
    for it in train_list:
        train_info.write(it + "\n")
    for it in val_list:
        val_info.write(it + "\n")
    for it in test_list:
        test_info.write(it + "\n")


if __name__ == "__main__":
    root = "/archive/hot0/dql/CD-data/Landsat-SCD_dataset"
    # process_landsat(root)
    split_dataset(root)
