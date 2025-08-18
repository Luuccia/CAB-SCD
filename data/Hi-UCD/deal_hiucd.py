import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

from data.dataloaders import index2color


def process_unlabel(mask_floder, save_floder):
    img_names = os.listdir(mask_floder)

    for img_name in tqdm(img_names):
        if img_name.endswith("png"):
            img_path = os.path.join(mask_floder, img_name)

            img = np.asarray(Image.open(img_path))
            img_new = img.copy()
            img_new[:, :, 0:2] = img[:, :, 0:2] - 1
            img_new[:, :, 0:2][img[:, :, 0:2] == 0] = 9  # semantic: unlabeled equal num

            img_new[:, :, -1] = img[:, :, -1] - 1
            img_new[:, :, -1][img[:, :, -1] == 0] = 255  # binary: unlabeled equal 255

            img_path = os.path.join(save_floder, img_name)
            Image.fromarray(img_new).save(img_path)


def process_hiucd_mini(
    data_root, splits=["train", "val", "test"], img_folder="image", mask_folder="mask_merge", divide=False
):
    """test: 136*4=544, train: 300*4=1200, val: 59*4=236
    1. Hi-UCD 与 Hi-UCD-mini 的文件路径不完全相同
    2. Hi-UCD-mini 的图片大小为 1024 -> 如果 divide=True 则裁剪为 512
    3. Hi-UCD-mini 的测试集中数据量不一致，仅保留有相同名称的
    """
    for split in splits:
        if not os.path.exists(os.path.join(data_root, split)):
            continue
        dir_path = os.path.join(data_root, split)

        # 处理标签
        if not os.path.exists(os.path.join(dir_path, mask_folder)):
            print(f"The {split} folder may be processed.")
            continue
        tmp_path = os.listdir(os.path.join(dir_path, mask_folder))[0]
        mask_path = os.path.join(dir_path, mask_folder, tmp_path, "9")
        save_mask = os.path.join(dir_path, "label")
        print(f"Process label in {mask_path}, save results to {save_mask}.")
        os.makedirs(save_mask)
        process_unlabel(mask_path, save_mask)
        shutil.rmtree(os.path.join(dir_path, mask_folder))

        img_names = os.listdir(os.path.join(dir_path, img_folder))
        img_names.sort()
        img_path1 = os.path.join(dir_path, img_folder, img_names[0], "9")
        img_path2 = os.path.join(dir_path, img_folder, img_names[1], "9")
        # 更改图片文件夹的名称 并移动位置
        if not os.path.exists(img_path1):
            print("The folder may be processed.")
            continue
        shutil.move(img_path1, os.path.join(dir_path, "im1"))
        shutil.move(img_path2, os.path.join(dir_path, "im2"))
        shutil.rmtree(os.path.join(dir_path, img_folder))

        if divide:
            divide_images(dir_path)


def divide_images(split_path, folders=["im1", "im2", "label"]):
    final_names = dict()
    for folder_name in folders:
        path = os.path.join(split_path, folder_name)
        print(f"Dividing {path}:")
        filenames = os.listdir(path)
        for fname in tqdm(filenames):
            prefix = fname.split(".")[0]
            image = Image.open(os.path.join(path, fname))
            w, h = image.size

            img1 = image.crop((0, 0, w / 2, h / 2))
            img2 = image.crop((w / 2, 0, w, h / 2))
            img3 = image.crop((0, h / 2, w / 2, h))
            img4 = image.crop((w / 2, h / 2, w, h))

            img1.save(os.path.join(path, f"{prefix}_1.png"))
            img2.save(os.path.join(path, f"{prefix}_2.png"))
            img3.save(os.path.join(path, f"{prefix}_3.png"))
            img4.save(os.path.join(path, f"{prefix}_4.png"))

            os.remove(os.path.join(path, fname))
        final_names[folder_name] = os.listdir(path)

    same_names = set(final_names[folders[0]]) & set(final_names[folders[1]]) & set(final_names[folders[2]])
    if len(same_names) == len(final_names[folders[0]]):
        print(f"Finish. Length: {len(same_names)}")
    else:
        for folder_name in folders:
            delete_names = set(final_names[folder_name]) - same_names
            for fname in delete_names:
                os.remove(os.path.join(split_path, folder_name, fname))
        print(
            f"The data volume is inconsistent. Only the image pairs with the same name are retained. Length: {len(same_names)}"
        )


# 从 处理后的三通道汇总标签 中提取 有颜色的语义图
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
        if img_name.endswith("png"):
            img_path = os.path.join(ori_path, img_name)
            img = np.asarray(Image.open(img_path))
            change_map = img[:, :, 2]

            label1 = img[:, :, 0].copy()
            label1[change_map == 0] = 9
            label1 = index2color(label1, "Hi-UCD")
            palette_image1 = Image.fromarray(label1)
            palette_image1.save(os.path.join(label1_path, img_name))

            label2 = img[:, :, 1].copy()
            label2[change_map == 0] = 9
            label2 = index2color(label2, "Hi-UCD")
            palette_image2 = Image.fromarray(label2)
            palette_image2.save(os.path.join(label2_path, img_name))


if __name__ == "__main__":
    root = "/archive/hot0/dql/CD-data/Hi-UCD-mini-512/test"
    generate_semantic_maps(root)
