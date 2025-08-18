import os
import time
import torch
import random
import shutil
import numpy as np


def init_seed(seed=123):
    """
    初始化随机种子以实现实验的可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True


def check_dirs(dir_name, phase="train"):
    execute_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    save_path = os.path.join(dir_name, execute_time + f"_{phase}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    info = "checkpoints & results are saved at: {}".format(os.path.abspath(save_path))

    if phase == "train":
        ckp_save_path = os.path.join(save_path, "every_ckp")
        os.mkdir(ckp_save_path)
    else:
        ckp_save_path = os.path.join(save_path, "results")  # 推理的可视化结果
        label_path1 = os.path.join(ckp_save_path, "label1")
        label_path2 = os.path.join(ckp_save_path, "label2")
        os.mkdir(ckp_save_path)
        os.mkdir(label_path1)
        os.mkdir(label_path2)

    return save_path, ckp_save_path, info


# 打印GPU相关信息
def gpu_info():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


if __name__ == "__main__":
    pass
