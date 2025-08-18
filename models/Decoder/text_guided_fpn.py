import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.base import Conv3Relu, DropBlock
from models.block.se_layer import SELayer
from models.Decoder.context_decoder import ContextDecoder


class FPNNeck(nn.Module):
    def __init__(self, in_channels=[128, 256, 512, 642], out_channels=128, text_dim=640, fuse=True, rotate=True):
        super().__init__()
        self.fuse = fuse
        self.rotate = rotate
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.context_decoder = ContextDecoder()

        # 拼接 img1 和 img2 的特征后还原其通道数
        self.concat_conv = nn.Sequential(
            Conv3Relu(in_channels[0] * 2, in_channels[0]),
            Conv3Relu(in_channels[1] * 2, in_channels[1]),
            Conv3Relu(in_channels[2] * 2, in_channels[2]),
            Conv3Relu(in_channels[3] * 2, in_channels[3]),
        )
        # 拼接差分特征后降维
        self.channel_att = nn.Sequential(
            SELayer(in_channels[0] * 3, in_channels[0]),
            SELayer(in_channels[1] * 3, in_channels[1]),
            SELayer(in_channels[2] * 3, in_channels[2]),
            SELayer(in_channels[3] * 3, in_channels[3]),
        )

        self.stage4_Conv_after_up = Conv3Relu(in_channels[3], in_channels[2])
        self.stage3_Conv_after_up = Conv3Relu(in_channels[2], in_channels[1])
        self.stage2_Conv_after_up = Conv3Relu(in_channels[1], in_channels[0])

        self.stage3_Conv2 = Conv3Relu(in_channels[2] * 2, in_channels[2])
        self.stage2_Conv2 = Conv3Relu(in_channels[1] * 2, in_channels[1])
        self.stage1_Conv2 = Conv3Relu(in_channels[0] * 2, in_channels[0])

        if self.fuse:
            self.stage4_Conv3 = Conv3Relu(in_channels[3], out_channels)  # 降维
            self.stage3_Conv3 = Conv3Relu(in_channels[2], out_channels)
            self.stage2_Conv3 = Conv3Relu(in_channels[1], out_channels)
            if in_channels[0] == out_channels:
                self.stage1_Conv3 = Conv3Relu(in_channels[0], out_channels)
            else:
                self.stage1_Conv3 = None
            self.final_Conv = Conv3Relu(out_channels * 4, out_channels)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.drop = DropBlock(rate=0.15, size=7, step=30)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(out_channels, 2, 3, 1, 1))  # 使用0.1的dropout

    def get_score_map(self, img_feats, text_embeddings):
        if self.rotate:
            feats0 = img_feats[0]
            feats1 = torch.rot90(img_feats[1], k=3, dims=(2, 3))
            feats2 = torch.rot90(img_feats[2], k=2, dims=(2, 3))
            feats3 = torch.rot90(img_feats[3], k=1, dims=(2, 3))
            vis_dense = (feats0 + feats1 + feats2 + feats3) / 4
        else:
            vis_dense = img_feats[0]

        B, C, H, W = vis_dense.shape  # (B, 640, 16, 16)
        global_feat = F.adaptive_avg_pool2d(vis_dense, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 640, 1)
        visual_context = torch.cat([global_feat.reshape(B, C, 1), vis_dense.reshape(B, C, H * W)], dim=2)
        visual_context = visual_context.permute(0, 2, 1)  # (B, 257, 640)

        # update text_embeddings by visual_context
        text_diff = self.context_decoder(text_embeddings, visual_context)  # (B, 1, C)
        text_embeddings = text_embeddings + self.gamma * text_diff  # (B, K, C)

        # compute score map
        vis_norm = F.normalize(vis_dense, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum("bchw,bkc->bkhw", vis_norm, text)

        return vis_dense, score_map

    def forward(self, dense_a, fusion_a, dense_b, fusion_b, text):
        fa1, fa2, fa3 = (fusion_a["res2"], fusion_a["res3"], fusion_a["res4"])
        fb1, fb2, fb3 = (fusion_b["res2"], fusion_b["res3"], fusion_b["res4"])
        H, W = fa1.size(2), fa1.size(3)

        dense_a, score_map_a = self.get_score_map(dense_a, text)
        dense_b, score_map_b = self.get_score_map(dense_b, text)
        fa4 = torch.cat([dense_a, score_map_a], dim=1)
        fb4 = torch.cat([dense_b, score_map_b], dim=1)
        fa = [fa1, fa2, fa3, fa4]
        fb = [fb1, fb2, fb3, fb4]

        x = [self.concat_conv[i](torch.cat([fa[i], fb[i]], dim=1)) for i in range(len(fa))]
        x_minus = [torch.abs(fa[i] - fb[i]) for i in range(len(fa))]
        x_diff = [torch.sigmoid(1 - torch.cosine_similarity(fa[i], fb[i], dim=1)).unsqueeze(1) for i in range(len(fa))]
        x = [torch.cat([x[i] * x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        change1, change2, change3, change4 = [self.channel_att[i](x[i]) for i in range(len(x))]

        # fpn (Feature Pyramid Network) 多尺度特征融合
        change3_2 = self.stage4_Conv_after_up(self.up(change4))
        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1))

        change2_2 = self.stage3_Conv_after_up(self.up(change3))
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1))

        change1_2 = self.stage2_Conv_after_up(self.up(change2))
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1))

        if self.fuse:
            change4 = self.stage4_Conv3(F.interpolate(change4, size=(H, W), mode="bilinear", align_corners=True))
            change3 = self.stage3_Conv3(F.interpolate(change3, size=(H, W), mode="bilinear", align_corners=True))
            change2 = self.stage2_Conv3(F.interpolate(change2, size=(H, W), mode="bilinear", align_corners=True))
            if self.stage1_Conv3 is not None:
                change1 = self.stage1_Conv3(change1)

            [change1, change2, change3, change4] = self.drop([change1, change2, change3, change4])  # dropblock
            change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))
        else:
            change = change1

        change = self.classifier(change)
        return change


if __name__ == "__main__":
    model = FPNNeck(fuse=False)

    data = {
        "res2": torch.rand(2, 128, 128, 128),
        "res3": torch.rand(2, 256, 64, 64),
        "res4": torch.rand(2, 512, 32, 32),
    }
    text_embeddings = torch.randn(2, 2, 640)
    clip_vis_dense = torch.randn(2, 640, 16, 16)
    feats = [clip_vis_dense, clip_vis_dense, clip_vis_dense, clip_vis_dense]

    feature = model(feats, data, feats, data, text_embeddings)
    print(feature.shape)
