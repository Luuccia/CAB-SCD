import torch
import torch.nn as nn
from torch.nn import functional as F

from configs.base import ReadConfigFile
from models.block.CSR import CSR
from models.Encoder.clip import CustomCLIP
from models.Decoder.aggregator import Aggregator
from models.Decoder.text_guided_fpn import FPNNeck


class ChangeDetection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_CSR = cfg.MODEL.CSR.ENABLE
        if self.use_CSR:
            clip_feature_dim = cfg.MODEL.ENCODER.CLIP_DIM
            self.CSR = CSR(in_dim=clip_feature_dim)

        self.encoder = CustomCLIP(cfg)
        self.SCD = Aggregator(cfg)
        self.BCD = FPNNeck(fuse=cfg.MODEL.BCD_HEAD.FUSE, rotate=cfg.MODEL.USE_ROTATE)

    def forward(self, xa, xb):
        _, _, height, width = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        clip_vis_a, fusion_features_a, text_sem, text_embed = self.encoder(xa)
        clip_vis_b, fusion_features_b, text_sem, text_embed = self.encoder(xb)

        if self.use_CSR:
            clip_vis_a, clip_vis_b = self.CSR(clip_vis_a, clip_vis_b)

        # BCD: torch.Size([B, 2, H/4, W/4])
        change_bcd = self.BCD(clip_vis_a, fusion_features_a, clip_vis_b, fusion_features_b, text_embed)

        # Semantic Segmentation: torch.Size([B, seg_classes, H/4, W/4])
        change_s1 = self.SCD(clip_vis_a, text_sem, fusion_features_a)
        change_s2 = self.SCD(clip_vis_b, text_sem, fusion_features_b)

        change_bcd = F.interpolate(change_bcd, size=(height, width), mode="bilinear", align_corners=False)
        if self.training:
            for i, output_ in enumerate(change_s1):
                output_ = F.interpolate(output_, size=(height, width), mode="bilinear", align_corners=False)
                change_s1[i] = output_
            for i, output_ in enumerate(change_s2):
                output_ = F.interpolate(output_, size=(height, width), mode="bilinear", align_corners=False)
                change_s2[i] = output_
        else:
            change_s1 = F.interpolate(change_s1[0], size=(height, width), mode="bilinear", align_corners=False)
            change_s2 = F.interpolate(change_s2[0], size=(height, width), mode="bilinear", align_corners=False)

        return change_bcd, change_s1, change_s2


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = ReadConfigFile().get_config()

    net = ChangeDetection(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    net.eval()

    batch_size = 1
    img1 = torch.rand(batch_size, 3, 512, 512).to(device)
    img2 = torch.rand(batch_size, 3, 512, 512).to(device)
    res = net(img1, img2)

    for feat in res:
        print(feat.shape)
        print(feat.dtype)
