import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from models import imagenet_templates
from configs.base import configurable, ReadConfigFile


class CustomCLIP(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        # clip params
        clip_name: str,
        clip_pretrained: str,
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        clip_resolution: Tuple[float],
        clip_finetune: str,
        out_features: list,
        rotate: bool,
        # text params
        class_json: str,
        prompt_ensemble_type: str,
        context_length: int,
        token_embed_dim: int,
    ):
        super().__init__()
        self.clip_finetune = clip_finetune
        self.clip_resolution = clip_resolution
        self.out_features = out_features
        self.rotate = rotate
        self.clip_finetune = clip_finetune

        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        with open(class_json, "r") as f_in:
            self.class_texts = json.load(f_in)
        assert self.class_texts != None
        device = "cuda" if torch.cuda.is_available() else "cpu"

        name, pretrain = (clip_name, clip_pretrained)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device=device,
        )

        self.tokenizer = open_clip.get_tokenizer(name)
        self.context_length = context_length
        self.prompt_ensemble_type = prompt_ensemble_type
        context_length = 77 - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = [
                "A photo of a {} in the scene.",
            ]
        else:
            raise NotImplementedError

        self.text_features = (
            self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        )
        self.cls_text = self.get_cls_text(self.class_texts)  # torch.Size([1, 2, context_length])

        self.clip_model = clip_model.float()
        for name, params in self.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "conv":
                    params.requires_grad = True if "conv" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                elif clip_finetune == "mlp":
                    params.requires_grad = True if "mlp" in name or "position" in name else False
                elif clip_finetune == "full_res5":
                    if "stages.3" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = {}

        ret["clip_name"] = cfg.MODEL.ENCODER.CLIP_NAME
        ret["clip_pretrained"] = cfg.MODEL.ENCODER.CLIP_PRETRAINED
        ret["clip_pixel_mean"] = cfg.MODEL.ENCODER.CLIP_PIXEL_MEAN
        ret["clip_pixel_std"] = cfg.MODEL.ENCODER.CLIP_PIXEL_STD
        ret["clip_finetune"] = cfg.MODEL.ENCODER.CLIP_FINETUNE
        ret["clip_resolution"] = tuple(cfg.MODEL.ENCODER.CLIP_RESOLUTION)

        ret["out_features"] = cfg.MODEL.ENCODER.OUT_FEATURES
        ret["rotate"] = cfg.MODEL.USE_ROTATE

        ret["class_json"] = cfg.DATASETS.CLASS_JSON
        ret["prompt_ensemble_type"] = cfg.MODEL.ENCODER.PROMPT_ENSEMBLE_TYPE
        ret["context_length"] = cfg.MODEL.ENCODER.CONTEXT_LENGTH
        ret["token_embed_dim"] = cfg.MODEL.ENCODER.CLIP_DIM

        return ret

    @torch.no_grad()
    def get_cls_text(self, classnames):
        fore = ", ".join(["remote sensing image foreground objects"] + classnames)
        back = ", ".join(["remote sensing image background objects"])
        return (
            torch.cat([self.tokenizer(c, context_length=self.context_length) for c in [back, fore]]).cuda().unsqueeze(0)
        )

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ", " in classname:
                classname_splits = classname.split(", ")
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class

            texts = self.tokenizer(texts).cuda()

            class_embeddings = clip_model.encode_text(texts)
            # 多子类
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def encode_context(self, text, context):
        """
        Args:
            text: 固定语义类别
            context: 可学习文本参数 两张长度之和为 77

        Returns: CLIP 编码的文本嵌入
        """
        x_text = self.clip_model.token_embedding(text)  # torch.Size([2, 2, 38, context_length])
        N, K, N1, C = x_text.shape
        N, B, N2, C = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(N, K).unsqueeze(1).reshape(-1)

        context = context.expand(N, K, N2, C)

        x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2).reshape(N * K, N1 + N2, C)

        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.clip_model.text_projection
        x = x.reshape(N, K, C)

        return x

    def forward(self, batched_inputs, dense=True):
        B = batched_inputs.shape[0]

        # torch.Size([B, 3, H, W])
        clip_images = (batched_inputs - self.clip_pixel_mean) / self.clip_pixel_std
        # torch.Size([B, 3, 512, 512]) <-- CLIP_RESOLUTION
        clip_images = F.interpolate(
            clip_images,
            size=self.clip_resolution,
            mode="bilinear",
            align_corners=False,
        )

        clip_features = self.clip_model.encode_image(clip_images, dense=dense)
        """
        stem           torch.Size([B, 128, H/4, W/4])      4倍下采样
        res2           torch.Size([B, 128, H/4, W/4])
        res3           torch.Size([B, 256, H/8, W/8])      2倍下采样
        res4           torch.Size([B, 512, H/16, W/16])    2倍下采样
        res5           torch.Size([B, 1024, H/32, W/32])   2倍下采样
        clip_vis_dense torch.Size([B, 640, H/32, W/32])
        """
        clip_vis_dense = clip_features["clip_vis_dense"]
        # 通过 clone().detach() 使 fusion_features 中的特征不会参与梯度计算，也不会将梯度传递回图像编码器
        fusion_features = {k: v.clone().detach() for k, v in clip_features.items() if k in self.out_features}

        if self.rotate:
            # 旋转90度
            clip_images_resized_90 = torch.rot90(clip_images, k=1, dims=(2, 3))
            # 旋转180度
            clip_images_resized_180 = torch.rot90(clip_images, k=2, dims=(2, 3))
            # 旋转270度
            clip_images_resized_270 = torch.rot90(clip_images, k=3, dims=(2, 3))
            clip_vis_dense1 = self.clip_model.encode_image(clip_images_resized_90, dense=True)["clip_vis_dense"]
            clip_vis_dense2 = self.clip_model.encode_image(clip_images_resized_180, dense=True)["clip_vis_dense"]
            clip_vis_dense3 = self.clip_model.encode_image(clip_images_resized_270, dense=True)["clip_vis_dense"]

            clip_features = [clip_vis_dense, clip_vis_dense1, clip_vis_dense2, clip_vis_dense3]
        else:
            clip_features = [clip_vis_dense]

        text_semantic = self.text_features.repeat(batched_inputs.shape[0], 1, 1, 1)
        cls_text_ = torch.cat([self.cls_text] * int(B), dim=0)
        contexts_ = torch.cat([self.contexts] * int(B), dim=0)
        text_embeddings = self.encode_context(cls_text_, contexts_)

        return clip_features, fusion_features, text_semantic, text_embeddings


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ReadConfigFile("/home/daiqiulu@corp.sse.tongji.edu.cn/workspace/SCD/configs/hiucd_mini.yaml").get_config()
    model = CustomCLIP(cfg)
    model.to(device)
    model.eval()

    img = torch.randn(2, 3, 512, 512)
    clip_vis_dense, fusion_features, text_semantic, text_embeddings = model(img.to(device))
    print(clip_vis_dense[0].shape)
    for k, v in fusion_features.items():
        print(k, v.shape)
    print(text_semantic.shape)  # torch.Size([2, 6, 80, 640])  json文件中的数量一致
    print(text_embeddings.shape)  # torch.Size([2, 2, 640])
