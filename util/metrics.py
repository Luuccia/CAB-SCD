import numpy as np
import math
from scipy import stats

np.seterr(divide="ignore", invalid="ignore")


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class BCDEvaluator(object):
    """only see change class"""

    def __init__(self, num_class=2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # row is True col is pred
        self.pre_cal = False

    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP[1] + self.FN[1] + self.FP[1]
        IoU = np.where(IoU == 0, 0, self.TP[1] / IoU)
        IoU = np.round(np.nanmean(IoU), 5)
        return IoU

    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        F1 = np.round(F1, 5)
        return F1[-1]

    def _generate_matrix(self, gt_image, pre_image):
        """
        对于一个 num_class*num_class 的混淆矩阵:
            行: 真实标签的类别 gt_image
            列: 预测标签的类别 pre_image

        矩阵中每个元素 M_{ij} = 真实类别为 i 但预测为 j 的像素数量
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True

    def get_scores(self):
        return {
            "OA": self.Overall_Accuracy(),
            "F1": self.F1_score(),
            "IoU": self.Intersection_over_Union(),
        }


class SEGEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # row is True col is pred
        self.pre_cal = False

    def Overall_Accuracy(self):
        self._pre_cal() if not self.pre_cal else 0
        OA = np.round(np.sum(self.TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def Precision(self):  # precision
        self._pre_cal() if not self.pre_cal else 0
        pre = self.TP + self.FP
        pre = np.where(pre == 0, 0, self.TP / pre)
        pre = np.round(pre, 5)
        return pre

    def Recall(self):  # recall
        self._pre_cal() if not self.pre_cal else 0
        rec = self.TP + self.FN
        rec = np.where(rec == 0, 0, self.TP / rec)
        rec = np.round(rec, 5)
        return rec

    def Mean_Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        # 计算每个类的IoU
        IoU_per_class = self.TP + self.FN + self.FP
        IoU_per_class = np.where(IoU_per_class == 0, 0, self.TP / IoU_per_class)
        # 计算平均IoU
        MIoU = np.round(np.nanmean(IoU_per_class), 5)
        return MIoU, np.round(IoU_per_class, 5)

    def Intersection_over_Union(self):
        self._pre_cal() if not self.pre_cal else 0
        IoU = self.TP + self.FN + self.FP
        IoU = np.where(IoU == 0, 0, self.TP / IoU)
        return IoU

    def F1_score(self):
        self._pre_cal() if not self.pre_cal else 0
        F1_per_class = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        F1 = np.nanmean(F1_per_class)
        F1 = np.round(F1, 5)
        return F1, np.round(F1_per_class, 5)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _pre_cal(self):
        self.TP = np.diag(self.confusion_matrix)
        self.FP = np.sum(self.confusion_matrix, 0) - self.TP
        self.FN = np.sum(self.confusion_matrix, 1) - self.TP
        self.pre_cal = True

    def get_scores(self):
        mIoU, class_IoUs = self.Mean_Intersection_over_Union()
        F1, class_F1s = self.F1_score()
        metrics = {
            "OA": self.Overall_Accuracy(),
            "F1": F1,
            "mIoU": mIoU,
            "class_IoUs": class_IoUs.tolist(),
            "class_F1s": class_F1s.tolist(),
        }
        return metrics


class SCDEvaluator(object):
    """
    data_loader的载入逻辑:
    n个语义类型=[1, n], 不变类型=0
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Overall_Accuracy(self):
        TP = np.diag(self.confusion_matrix)
        OA = np.round(np.sum(TP) / np.sum(self.confusion_matrix), 5)
        return OA

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = gt_image[mask].astype("int") + self.num_class * pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def get_scores(self):
        hist = self.confusion_matrix
        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]  # TN
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]  # FP. hist.sum(1): pred_hist
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]  # FN. hist.sum(0): label_hist
        c2hist[1][1] = hist_fg.sum()  # TP
        print("bn_hist: TP %d, FN %d, FP %d, TN %d" % (c2hist[1][1], c2hist[1][0], c2hist[0][1], c2hist[0][0]))

        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_c = iu[1]
        mIoU = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_c)) / math.e
        Score = 0.3 * mIoU + 0.7 * Sek

        pixel_sum = hist.sum()
        change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - hist.sum(0)[0].sum()
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        SC_Recall = SC_TP / change_label_sum
        F1 = stats.hmean([SC_Precision, SC_Recall])

        metrics = {
            "OA": self.Overall_Accuracy(),
            "F1": np.round(F1, 5),
            "mIoU": np.round(mIoU, 5),
            "IoU_c": np.round(IoU_c, 5),
            "Sek": np.round(Sek, 5),
            "Total Score": np.round(Score, 5),
        }
        return metrics
