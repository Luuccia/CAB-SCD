import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from configs.base import ReadConfigFile
from data.datatransforms import test_transforms, get_train_transforms

# "classes": semantics that can help understanding
_DATASET_CONFIG = {
    "SECOND-CD": {
        "dual_label": True,
        "color_map": np.array(
            [(0, 0, 255), (128, 128, 128), (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0), (255, 255, 255)]
        ),
        "classes": [
            "water",
            "ground",
            "low vegetation",
            "tree",
            "building",
            "sports field",
            "unchanged",
        ],
    },  # 6
    "Landsat-SCD": {
        "dual_label": True,
        "color_map": np.array([(0, 155, 0), (255, 165, 0), (230, 30, 100), (0, 170, 240), (255, 255, 255)]),
        "classes": [
            "farmland",
            "desert",
            "building",
            "water",
            "No change",
        ],
    },
    "Hi-UCD": {
        "dual_label": False,
        "color_map": np.array(
            [
                [0, 153, 255],
                [202, 255, 122],
                [255, 0, 0],
                [230, 0, 255],
                [255, 230, 0],
                [255, 181, 197],
                [0, 255, 230],
                [175, 122, 255],
                [0, 255, 0],
                [0, 0, 0],
            ]
        ),
        "classes": [
            "water",
            "grass",
            "building",
            "green house",
            "road",
            "bridge",
            "others",
            "bare Land",
            "woodland",
            "unlabeled",
        ],
    },
    "HRSCD": {
        "dual_label": False,
        "color_map": np.array([(128, 0, 0), (128, 128, 128), (0, 128, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]),
        "classes": [
            "artificial surfaces",
            "farmland, agricultural areas",
            "forest",
            "wetland",
            "water",
            "No change",
        ],
    },
}


def index2color(pred, data_type):
    """将索引值转换为对应的颜色值

    Args:
        pred (numpy.ndarray): (H, W) 模型预测结果，表示每个像素的类别索引
        data_type (string): 指定数据集的类型

    Returns:
        numpy.ndarray: (H, W, 3) 表示每个像素的 RGB 值
    """
    assert data_type in _DATASET_CONFIG.keys()
    colormap = _DATASET_CONFIG[data_type]["color_map"].astype("uint8")
    x = np.asarray(pred, dtype="int32")
    return colormap[x, :]


def get_loaders(opt, logger=None):
    """
    载入训练集和验证集
    """
    train_dataset = CDDataset(opt, phase="train")
    val_dataset = CDDataset(opt, phase="val")

    num_workers = opt.DATALOADER.NUM_WORKERS
    batch_size = opt.DATALOADER.BATCH_SIZE

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    if logger is not None:
        logger.info("Making batched data loader with batch_size=%d", batch_size)
        logger.info(f"Train Number: {len(train_loader) * batch_size}, Val Number: {len(val_loader) * batch_size}\n")
    length = len(train_loader)
    return train_loader, val_loader, length


def get_eval_loader(opt, logger=None):
    """
    载入有标签的测试集
    """
    dataset = CDDataset(opt, phase="test")

    num_workers = opt.DATALOADER.NUM_WORKERS
    batch_size = 1
    dataset_type = opt.DATASETS.TYPE

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    if logger is not None:
        logger.info("Making batched data loader with batch_size=%d", batch_size)
        logger.info(f"Test Number: {len(dataset)}.\n")
    return loader, _DATASET_CONFIG[dataset_type]["classes"]


class CDDataset(Dataset):
    """Custom dataset for change detection. An example of file structure is as
    followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── img_dir2
        │   │   │   ├── xxx{img_suffix2}
        │   │   │   ├── yyy{img_suffix2}
        │   │   │   ├── zzz{img_suffix2}
        │   │   ├── ann_dir
        │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   ├── zzz{seg_map_suffix}
        │   │   ├── ann_dir2
        │   │   │   ├── xxx{seg_map_suffix2}
        │   │   │   ├── yyy{seg_map_suffix2}
        │   │   │   ├── zzz{seg_map_suffix2}

    The image names in img_dir and img_dir2 should be consistent.
    The annotation names in ann_dir and ann_dir2 should also be consistent.
    The img/gt_semantic_seg pair should be of the same except suffix.
    A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.


    Args: Config has following parameter
        ann_file (str): Annotation file path. Defaults to ''.
        type (str): Dataset name which must appear in _DATASET_CONFIG
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img_path=None, img_path2=None, seg_map_path=None, seg_map_path2=None).
        img_suffix (str): Suffix of images. Default: '.png'
        img_suffix2 (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        seg_map_suffix2 (str): Suffix of segmentation maps. Default: '.png'
    """

    def __init__(self, cfg, phase):
        self.img_suffix = cfg.DATASETS.SUFFIX.IMG_SUFFIX
        self.img_suffix2 = cfg.DATASETS.SUFFIX.IMG_SUFFIX2
        self.seg_map_suffix = cfg.DATASETS.SUFFIX.SEG_MAP_SUFFIX
        self.seg_map_suffix2 = cfg.DATASETS.SUFFIX.SEG_MAP_SUFFIX2

        self.data_root = cfg.DATASETS.DATA_ROOT
        self.dataset_type = cfg.DATASETS.TYPE
        assert phase in ["train", "test", "val"], "Param phase should be ['train', 'test', 'val']."
        if phase == "train":
            self.split_file = os.path.join(self.data_root, cfg.DATASETS.ANN_FILE.TRAIN)
            self.transform = get_train_transforms()
        elif phase == "test":
            self.split_file = os.path.join(self.data_root, cfg.DATASETS.ANN_FILE.TEST)
            self.transform = test_transforms
        elif phase == "val":
            self.split_file = os.path.join(self.data_root, cfg.DATASETS.ANN_FILE.VAL)
            self.transform = test_transforms

        assert (
            self.dataset_type in _DATASET_CONFIG.keys()
        ), f"Invalid Dataset Name with {self.dataset_type}. It should be included in {_DATASET_CONFIG.keys()}"
        self.dual_label = _DATASET_CONFIG[self.dataset_type]["dual_label"]  # 决定标签载入逻辑
        if self.dual_label:
            self.color_map = _DATASET_CONFIG[self.dataset_type]["color_map"]
        else:
            self.color_map = None

        if os.path.exists(os.path.join(self.data_root, phase)):
            data_prefix = os.path.join(self.data_root, phase)
        else:
            data_prefix = self.data_root
        self.img_dir = os.path.join(data_prefix, cfg.DATASETS.DATA_PREFIX.IMG_PATH)
        self.img_dir2 = os.path.join(data_prefix, cfg.DATASETS.DATA_PREFIX.IMG_PATH2)
        self.ann_dir = os.path.join(data_prefix, cfg.DATASETS.DATA_PREFIX.SEG_MAP_PATH)
        self.ann_dir2 = os.path.join(data_prefix, cfg.DATASETS.DATA_PREFIX.SEG_MAP_PATH2)

        self.data_list = self.load_data_list()

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if os.path.exists(self.split_file) and os.path.isfile(self.split_file):
            # 从数据集划分文件中读入部分文件
            with open(self.split_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                img_name = line.strip()
                if "." in os.path.basename(img_name):
                    img_name, img_ext = os.path.splitext(img_name)
                    self.img_suffix = img_ext
                    self.img_suffix2 = img_ext
                data_info = dict(
                    img_path=os.path.abspath(os.path.join(self.img_dir, img_name + self.img_suffix)),
                    img_path2=os.path.abspath(os.path.join(self.img_dir2, img_name + self.img_suffix2)),
                )

                if self.dual_label:
                    seg_map = img_name + self.seg_map_suffix
                    data_info["seg_map_path"] = os.path.abspath(os.path.join(self.ann_dir, seg_map))
                    seg_map2 = img_name + self.seg_map_suffix2
                    data_info["seg_map_path2"] = os.path.abspath(os.path.join(self.ann_dir2, seg_map2))
                else:
                    seg_map = img_name + self.seg_map_suffix
                    data_info["seg_map_path"] = os.path.abspath(os.path.join(self.ann_dir, seg_map))

                data_list.append(data_info)
        else:
            # 按照名称对应关系读入文件夹下所有文件
            for img in os.listdir(self.img_dir):
                if "." in os.path.basename(img):
                    img, img_ext = os.path.splitext(img)
                    self.img_suffix = img_ext
                    self.img_suffix2 = img_ext
                data_info = dict(
                    img_path=os.path.abspath(os.path.join(self.img_dir, img + self.img_suffix)),
                    img_path2=os.path.abspath(os.path.join(self.img_dir2, img + self.img_suffix2)),
                )

                if self.dual_label:
                    seg_map = img + self.seg_map_suffix
                    data_info["seg_map_path"] = os.path.abspath(os.path.join(self.ann_dir, seg_map))
                    seg_map2 = img + self.seg_map_suffix2
                    data_info["seg_map_path2"] = os.path.abspath(os.path.join(self.ann_dir2, seg_map2))
                else:
                    seg_map = img + self.seg_map_suffix
                    data_info["seg_map_path"] = os.path.abspath(os.path.join(self.ann_dir, seg_map))

                data_list.append(data_info)

            data_list = sorted(data_list, key=lambda x: x["img_path"])
        return data_list

    def rgb2label(self, rgb_label, color_map):
        """将每个像素的 RGB 转化为索引"""
        num_classes = color_map.shape[0]
        gray_label = np.zeros(shape=(rgb_label.shape[0], rgb_label.shape[1]), dtype=np.int16)
        for i in range(num_classes):
            index = np.where(np.all(rgb_label == color_map[i], axis=-1))
            gray_label[index] = i
            gray_label = gray_label * (gray_label < num_classes)
        return gray_label

    def gen_changelabel(self, label1, label2):
        """generate BCD map. unchanged=0, changed=1"""
        unchange_type = np.max(label1)
        binary_label = np.ones_like(label1, dtype=np.uint8)
        binary_label[(label1 == unchange_type) | (label2 == unchange_type)] = 0
        return binary_label

    def __getitem__(self, index):
        data_info = self.data_list[index]

        img_name = data_info["img_path"].split("/")[-1].split(".")[0]
        img_A = Image.open(data_info["img_path"])
        img_B = Image.open(data_info["img_path2"])

        if self.dataset_type in ["SECOND-CD", "Landsat-SCD"]:
            label_A = np.asarray(Image.open(data_info["seg_map_path"]), dtype=np.int16)
            label_B = np.asarray(Image.open(data_info["seg_map_path2"]), dtype=np.int16)
            label_A = self.rgb2label(label_A, self.color_map)
            label_B = self.rgb2label(label_B, self.color_map)
            label_BCD = self.gen_changelabel(label_A, label_B)
        elif self.dataset_type in ["Hi-UCD", "HRSCD"]:
            label = np.asarray(Image.open(data_info["seg_map_path"]))
            label_A = label[:, :, 0]
            label_B = label[:, :, 1]
            label_BCD = label[:, :, 2]

        sample = {
            "img_A": img_A,
            "img_B": img_B,
            "label_BCD": Image.fromarray(label_BCD),
            "label_SGA": Image.fromarray(label_A),
            "label_SGB": Image.fromarray(label_B),
        }
        sample = self.transform(sample)
        return sample["img_A"], sample["img_B"], sample["label_BCD"], sample["label_SGA"], sample["label_SGB"], img_name

    def __len__(self):
        return len(self.data_list)


def test_dataset(cfg):
    # test dataset read by annotations
    train_dataset = CDDataset(cfg, "train")
    assert len(train_dataset) == 2

    # test data get
    train_data = train_dataset[0]
    print(train_data[0].shape)  # torch.Size([3, 512, 512])

    # test dataset read by dir
    test_dataset = CDDataset(cfg, "test")
    assert len(test_dataset) == 4

    # test data get
    test_data = test_dataset[0]
    print(test_data[0].shape)  # torch.Size([3, 512, 512])


def test_loader(cfg):
    train_loader, _, _ = get_loaders(cfg)
    print(len(train_loader))
    for idx, batch_inputs in enumerate(train_loader):
        for img in batch_inputs[:5]:
            print(img.shape)
        print(torch.unique(batch_inputs[2], return_counts=True))
        print(torch.unique(batch_inputs[3], return_counts=True))
        print(torch.unique(batch_inputs[4], return_counts=True))
        print(batch_inputs[5])
        break


if __name__ == "__main__":
    path = "/home/daiqiulu@corp.sse.tongji.edu.cn/workspace/SCD/configs/hiucd_mini.yaml"
    cfg = ReadConfigFile(path).get_config()

    # test_dataset(cfg)
    test_loader(cfg)
