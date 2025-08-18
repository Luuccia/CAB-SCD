import torch
import torch.nn as nn
import torch.nn.functional as F


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction="mean")

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class SS_CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, weight=[1, 0.05, 0.1, 0.15]):
        super(SS_CrossEntropyLoss, self).__init__()
        self.ce = CrossEntropyLoss2d(ignore_index=ignore_index)
        self.weight = weight

    def forward(self, inputs, target):
        losses = 0
        # 考虑辅助分支
        for output_, w in zip(inputs, self.weight):
            loss = self.ce(output_, target)
            losses += loss * w
        return losses


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ChangeSimilarity(nn.Module):
    """
    计算余弦相似度，使：
    未变化区域 两个时相的预测概率分布相似
    变化区域   两个时相的预测概率分布不相似

    input: x1, x2 multi-class predictions, c = class_num
    label_change: binary change map (0=未变化, 1=变化, 255=忽略)
    """

    def __init__(self, reduction="mean", ignore_index=255):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0.0, reduction="none")
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()

        # 计算预测的概率分布
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        # 创建有效区域掩码
        valid_mask = label_change != self.ignore_index
        label_unchange = ~label_change.bool() & valid_mask  # 未变化区域掩码
        target = label_unchange.float()
        target = target - (label_change.float() * valid_mask.float())  # 1=未变化，-1=变化，忽略区域为0
        target = torch.reshape(target, [b * h * w])

        # 计算有效区域的余弦相似度损失
        loss = self.loss_f(x1, x2, target)
        valid_mask = torch.reshape(valid_mask, [b * h * w])
        loss = loss[valid_mask]

        # 返回损失
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss.view(b, h, w)[valid_mask.view(b, h, w)]


class PossibilityConstraint(nn.Module):
    """
    若两个时相预测的类别相同，则令其概率与非变化概率相近
    若两个时相预测的类别不同，则令其概率与变化概率相近
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, out_A, out_B, out_change):
        logit_A = torch.softmax(out_A, dim=1)
        logit_B = torch.softmax(out_B, dim=1)
        logit_Change = torch.softmax(out_change, dim=1)

        pred_A = torch.argmax(logit_A, dim=1)
        pred_B = torch.argmax(logit_B, dim=1)

        unchange = pred_A == pred_B
        change = ~unchange

        CD_logit = logit_Change[:, 0, :, :] * unchange + logit_Change[:, 1, :, :] * change

        max_A, _ = torch.max(logit_A, dim=1)
        max_B, _ = torch.max(logit_B, dim=1)
        SCD_logit = 0.5 * (max_A + max_B)

        loss = self.mse(SCD_logit, CD_logit.detach())
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        ignore_index: Label index to ignore (default: None)
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, ignore_index=None, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1

        # Create mask for valid pixels (where target != ignore_index)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()  # [N, D], 0 or 1
        else:
            mask = torch.ones_like(target)

        # Apply mask to both predict and target
        masked_predict = predict * mask  # [N, D]
        masked_target = target * mask  # [N, D]

        # Calculate numerator (intersection) and denominator (union)
        intersection = torch.sum(masked_predict * masked_target, dim=1)  # [N]
        union = torch.sum(masked_predict.pow(self.p) + masked_target.pow(self.p), dim=1)  # [N]

        # Dice coefficient and loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice  # [N]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class BCEWithIgnoreLoss(nn.Module):
    def __init__(self, ignore_index=255, OHEM=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.OHEM = OHEM

    def forward(self, logits, target):
        if len(logits.shape) != len(target.shape) and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        target = target.float()
        valid_mask = target != self.ignore_index
        loss = self.bce(logits, target)

        # OHEM
        if self.OHEM:
            loss_, _ = loss.contiguous().view(-1).sort()
            min_value = loss_[int(0.5 * loss.numel())]

            loss = loss[valid_mask]
            loss = loss[loss >= min_value]
        else:
            loss = loss[valid_mask]

        return loss.mean()


class BCDLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(BCDLoss, self).__init__()
        self.dice = BinaryDiceLoss(ignore_index=ignore_index)
        self.bce = BCEWithIgnoreLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        logits = F.logsigmoid(logits).exp()
        losses = self.dice(logits, target) + self.bce(logits, target)
        return losses
