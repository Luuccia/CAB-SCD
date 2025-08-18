from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base import configurable, ReadConfigFile
from models.Decoder.text_up import Up
from models import imagenet_templates


class Aggregator(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        rotate=False,
        decoder_dims=(64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_corr_guidance_dims=[80, 32, 16],
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4,
        hidden_dim=128,
        pooling_size=(6, 6),
        attention_type="linear",
        prompt_channel=80,
        cnext_type="V1",
        kernel_size=(7, 7, 7),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if rotate:  # 使用旋转特征时，特征维度会变化
            ratio = 4
        else:
            ratio = 1
        self.conv1 = nn.Conv2d(prompt_channel * ratio, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.text_guidance_projection = None

        self.decoder_guidance_projection = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    )
                    for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
                ]
            )
            if decoder_guidance_dims[0] > 0
            else None
        )

        decoder_corr_guidance_dims[0] = prompt_channel * ratio
        decoder_corr_guidance_dims = tuple(decoder_corr_guidance_dims)
        self.decoder_corr_guidance_projection = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConvTranspose2d(d, dp, kernel_size=2, stride=2),
                        nn.ReLU(),
                    )
                    for d, dp in zip(decoder_corr_guidance_dims, decoder_guidance_proj_dims)
                ]
            )
            if decoder_corr_guidance_dims[0] > 0
            else None
        )

        text_guidance_channels = [0, 0, 0]
        self.decoder1 = Up(
            hidden_dim,
            decoder_dims[0],
            decoder_guidance_proj_dims[0],
            text_guidance_channels[0],
            nheads,
            attention_type,
            pooling_size,
            cnext_type,
            kernel_size[0],
        )
        self.decoder2 = Up(
            decoder_dims[0],
            decoder_dims[1],
            decoder_guidance_proj_dims[1],
            text_guidance_channels[1],
            nheads,
            attention_type,
            pooling_size,
            cnext_type,
            kernel_size[1],
        )
        self.decoder3 = Up(
            decoder_dims[1],
            decoder_dims[2],
            decoder_guidance_proj_dims[2],
            text_guidance_channels[2],
            nheads,
            attention_type,
            pooling_size,
            cnext_type,
            kernel_size[2],
        )

        # 监督辅助分支 仅用于训练
        self.sem_head0 = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        self.sem_head1 = nn.Conv2d(decoder_dims[0], 1, kernel_size=3, stride=1, padding=1)
        self.sem_head2 = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.sem_head3 = nn.Conv2d(decoder_dims[2], 1, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_config(cls, cfg):
        ret = {}

        ret["rotate"] = cfg.MODEL.USE_ROTATE
        prompt_ensemble_type = cfg.MODEL.ENCODER.PROMPT_ENSEMBLE_TYPE
        if prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif prompt_ensemble_type == "single":
            prompt_templates = [
                "A photo of a {} in the scene",
            ]
        else:
            raise NotImplementedError
        ret["prompt_channel"] = len(prompt_templates)

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["nheads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dim"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_size"] = tuple(cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES)
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE
        ret["cnext_type"] = cfg.MODEL.SEM_SEG_HEAD.CNEXT_TYPE
        ret["kernel_size"] = cfg.MODEL.SEM_SEG_HEAD.KERNEL_SIZE

        return ret

    def correlation(self, img_feats, text_feats):
        """
        计算图像特征和文本特征之间的相关性
        """
        if len(img_feats) == 1:
            img_feats = F.normalize(img_feats[0], dim=1)  # B C H W
            text_feats = F.normalize(text_feats, dim=-1)  # B T P C
            corr = torch.einsum("bchw, btpc -> bpthw", img_feats, text_feats)
        else:
            img_feats0 = F.normalize(img_feats[0], dim=1)  # B C H W
            img_feats1 = F.normalize(img_feats[1], dim=1)  # B C H W
            img_feats2 = F.normalize(img_feats[2], dim=1)  # B C H W
            img_feats3 = F.normalize(img_feats[3], dim=1)  # B C H W
            img_feats1 = torch.rot90(img_feats1, k=3, dims=(2, 3))
            img_feats2 = torch.rot90(img_feats2, k=2, dims=(2, 3))
            img_feats3 = torch.rot90(img_feats3, k=1, dims=(2, 3))
            # B × N × C × H × W, N=4
            img_feats = torch.cat(
                (
                    img_feats0.unsqueeze(dim=1),
                    img_feats1.unsqueeze(dim=1),
                    img_feats2.unsqueeze(dim=1),
                    img_feats3.unsqueeze(dim=1),
                ),
                dim=1,
            )

            text_feats = F.normalize(text_feats, dim=-1)  # B T P C
            corr = torch.einsum("bnchw, btpc -> bnpthw", img_feats, text_feats)
            corr = rearrange(corr, "B N P T H W -> B (N P) T H W")
        return corr

    def corr_embed(self, x):
        # 将相关性特征通过卷积转换为嵌入表示
        B = x.shape[0]
        corr_embed = rearrange(x, "B P T H W -> (B T) P H W")
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, "(B T) C H W -> B C T H W", B=B)
        return corr_embed

    def conv_decoder(self, x, text_guidance, guidance, corr_guidance):
        """
        x:             torch.Size([B, 128, num_classes, 24, 24])
        text_guidance: list
        guidance:      list
        corr_guidance: list

        0:
            corr_embed:  torch.Size([B*num_classes, 24, 24])
            mask_aux:  torch.Size([B, num_classes, 24, 24])
        1:
            corr_embed:  torch.Size([B*num_classes, 64, 48, 48])
            mask_aux0:  torch.Size([B, num_classes, 48, 48])
        2:
            corr_embed:  torch.Size([B*num_classes, 32, 96, 96])
            mask_aux1:  torch.Size([B, num_classes, 96, 96])
        """
        B = x.shape[0]
        # 通过 detach 阻止梯度反向传播
        # 以确保辅助分支仅用于提供额外的监督信号，而不会干扰主分支的训练过程
        corr_embed = rearrange(x, "B C T H W -> (B T) C H W")
        mask_aux = self.sem_head0(corr_embed.detach())
        mask_aux = rearrange(mask_aux, "(B T) () H W -> B T H W", B=B)

        corr_embed = self.decoder1(corr_embed, text_guidance[0], guidance[0], corr_guidance[0])
        mask_aux0 = self.sem_head1(corr_embed.detach())
        mask_aux0 = rearrange(mask_aux0, "(B T) () H W -> B T H W", B=B)

        corr_embed = self.decoder2(corr_embed, text_guidance[1], guidance[1], corr_guidance[1])
        mask_aux1 = self.sem_head2(corr_embed.detach())
        mask_aux1 = rearrange(mask_aux1, "(B T) () H W -> B T H W", B=B)

        corr_embed = self.decoder3(corr_embed, text_guidance[2], guidance[2], corr_guidance[2])
        corr_embed = self.sem_head3(corr_embed)
        corr_embed = rearrange(corr_embed, "(B T) () H W -> B T H W", B=B)

        return [corr_embed, mask_aux, mask_aux0, mask_aux1]

    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: list of (B, C, H, W)
            text_feats: (B, T, P, C)
            apperance_guidance: tuple of (B, C, H, W)
        """
        # 1. 计算视觉特征与文本特征的相关性
        corr = self.correlation(img_feats, text_feats)
        # 2. 生成相关性嵌入
        corr_embed = self.corr_embed(corr)

        projected_text_guidance, projected_decoder_guidance = [None, None, None], [None, None, None]
        projected_corr_decoder_guidance = [None, None, None]

        # 3. 反转中间层视觉特征的顺序 并投影
        vis = [appearance_guidance[k] for k in appearance_guidance.keys()][::-1]
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, vis)]

        # 4. 相关性特征投影
        if self.decoder_corr_guidance_projection is not None:
            corr = rearrange(corr, "B P T H W -> (B T) P H W")
            for i, proj in enumerate(self.decoder_corr_guidance_projection):
                corr = proj(corr)
                projected_corr_decoder_guidance[i] = corr

        seg_logits = self.conv_decoder(
            corr_embed, projected_text_guidance, projected_decoder_guidance, projected_corr_decoder_guidance
        )

        return seg_logits


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ReadConfigFile().get_config()
    model = Aggregator(cfg)
    model.to(device)
    model.eval()

    img = torch.randn(2, 640, 16, 16).to(device)
    text = torch.randn(2, 6, 80, 640).to(device)
    guidance = {
        "res2": torch.randn(2, 128, 128, 128).to(device),
        "res3": torch.randn(2, 256, 64, 64).to(device),
        "res4": torch.randn(2, 512, 32, 32).to(device),
        # "res5": torch.randn(2, 1024, 16, 16).to(device),
    }
    if cfg.MODEL.USE_ROTATE:
        logits = model([img, img, img, img], text, guidance)
    else:
        logits = model([img], text, guidance)
    for feat in logits:
        print(feat.shape)
