# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.convnext import ConvNextBlock, ConvNextV2Block


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float("-inf"))

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type="linear"):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == "linear":
            self.attention = LinearAttention()
        elif attention_type == "full":
            self.attention = FullAttention()
        else:
            raise NotImplementedError

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.nheads)
        k = rearrange(k, "B S (H D) -> B S H D", H=self.nheads)
        v = rearrange(v, "B S (H D) -> B S H D", H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, "B L H D -> B L (H D)")
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type="linear", pooling_size=(4, 4)) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size)
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.pool(x)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, _, _, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()

        x_pool = rearrange(x_pool, "B C T H W -> (B H W) T C")
        if guidance is not None:
            guidance = repeat(guidance, "B T C -> (B H W) T C", H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance)  # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool))  # MLP

        x_pool = rearrange(x_pool, "(B H W) T C -> (B T) C H W", H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode="bilinear", align_corners=True)
        x_pool = rearrange(x_pool, "(B T) C H W -> B C T H W", B=B)

        x = x + x_pool  # Residual
        return x


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        guidance_channels,
        text_guidance_channels,
        nheads,
        attention_type,
        pooling_size,
        cnext_type="V1",
        kernel_size=7,
    ):
        super().__init__()
        corr_guidance_channels = guidance_channels

        # 上采样层
        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        # 双卷积层
        self.conv = DoubleConv(in_channels + corr_guidance_channels, out_channels)
        if cnext_type == "V1":
            self.cnext = ConvNextBlock(in_channels, kernel_size, 0.0, 1.0)
        elif cnext_type == "V2":
            self.cnext = ConvNextV2Block(in_channels, kernel_size, 0.0)
        else:
            raise NotImplementedError
        # 注意力层
        self.attention = ClassTransformerLayer(
            in_channels, text_guidance_channels, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size
        )

    def forward(self, x, text_guidance, guidance=None, corr_guidance=None):
        B, _, _, _ = guidance.size()
        x = self.cnext(x)  # 1. Spatial-level特征增强
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        x = self.attention(x, text_guidance)  # 2. Class-level特征整合
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.up(x)  # 3. 特征上采样
        # 4. 多尺度特征融合
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        if corr_guidance is not None:
            x = torch.cat([x, corr_guidance], dim=1)
        return self.conv(x)
