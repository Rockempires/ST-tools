"""
图像尺寸节点（ST_ImageMaskLatentSize）

职责：接收图像/遮罩/latent（全部 optional，至少接一个），输出对齐 vae_unit 后的图像、遮罩、latent 和最终宽高。

尺寸确定优先级：
    自定义尺寸（自定义尺寸=true） > 预设比例（预设比例≠关闭）
    > 图像输入尺寸 > latent 输入尺寸 > 兜底 1024×1024
确定后统一对齐到 vae_unit 倍数。

输入处理优先级：
    图像输入 > latent 输入。图像路径内部根据缩放方式不同：
        中心裁剪 → 按目标宽高比裁剪中心 → resize 到目标尺寸
        等比缩放 → 以原始图像尺寸 × 缩放倍数（忽略预设/自定义尺寸）
    latent 路径同理：
        中心裁剪 → 按目标比例裁剪 latent 网格 → bilinear 到目标 latent 尺寸
        等比缩放 → latent 原始空间尺寸 × 倍数

vae_unit 来源优先级：latent["downscale_ratio_spacial"] > VAE.downscale_ratio > 兜底 8
latent 通道数来源：latent["samples"].shape[1] > 兜底 4

遮罩处理：
    有遮罩输入 → 与图像做完全相同的 resize/crop
    无遮罩输入 → 创建全白遮罩

latent 路径（无图像输入）：
    对齐后用 torch.zeros 创建空 latent，shape = [B, channels, H/vae_unit, W/vae_unit]
"""

import torch
import numpy as np
import math
from PIL import Image
from ..config.presets import PRESETS, get_size_from_preset


class ST_ImageMaskLatentSize:
    """图像尺寸节点：统一处理图像/遮罩/latent 尺寸，按 VAE 下采样倍率对齐。"""
    DISPLAY_NAME = "图像尺寸"
    
    @classmethod
    def INPUT_TYPES(cls):
        ratio_options = [name for name, size in PRESETS]
        
        return {
            "required": {
                "自定义尺寸": ("BOOLEAN", {"default": False}),
                "预设比例": (ratio_options, {"default": ratio_options[0] if ratio_options else "无可用比例"}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "缩放方式": (["中心裁剪", "等比缩放"], {"default": "中心裁剪"}),
                "缩放倍数": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT")
    RETURN_NAMES = ("图像", "遮罩", "latent", "宽度", "高度")
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = '处理优先级：\n 1. 尺寸优先级：自定义尺寸 > 预设尺寸 > 图像输出尺寸 > Latent 输入尺寸\n 2. 输入处理优先级：图像/遮罩输入 > Latent 输入\n处理方法：\n 中心裁剪：按目标比例裁剪中心调整大小\n 等比缩放：按目标比例乘以倍数等比缩放\n Latent：按目标比例调整大小（按VAE下采样倍率对齐）\n 遮罩不存在时：创建全白遮罩'

    @staticmethod
    def _resolve_vae_unit_and_channels(kwargs):
        """
        解析 vae_unit 和 latent_channels。
        优先级：latent["downscale_ratio_spacial"] > VAE.downscale_ratio > 兜底 8/4
        """
        vae_unit = 8
        latent_channels = 4
        
        latent_in = kwargs.get("latent")
        vae = kwargs.get("vae")
        
        # latent 自带最准的信息
        if latent_in is not None:
            vae_unit = int(latent_in.get("downscale_ratio_spacial", vae_unit))
            if "samples" in latent_in and latent_in["samples"].dim() == 4:
                latent_channels = int(latent_in["samples"].shape[1])
        
        # VAE 可补充（latent 没带时）
        if vae is not None:
            vae_unit = int(getattr(vae, "downscale_ratio", vae_unit))
        
        # 防御
        if not isinstance(vae_unit, int) or vae_unit <= 0:
            vae_unit = 8
        
        return vae_unit, latent_channels

    @staticmethod
    def _align_to_vae_unit(size, vae_unit):
        return math.ceil(size / vae_unit) * vae_unit

    def run(self, **kwargs):
        """主入口：确定目标尺寸 → 按输入类型分发处理。"""
        try:
            vae_unit, latent_channels = self._resolve_vae_unit_and_channels(kwargs)
            
            use_custom = kwargs["自定义尺寸"]
            ratio_name = kwargs["预设比例"]
            width = kwargs["宽度"]
            height = kwargs["高度"]
            缩放方式 = kwargs["缩放方式"]
            缩放倍数 = kwargs["缩放倍数"]
            vae = kwargs.get("vae")
            
            has_image = kwargs.get("图像") is not None
            has_latent = kwargs.get("latent") is not None

            # 1. 确定目标尺寸（统一对齐 vae_unit）
            target_width, target_height = self._determine_target_size(
                kwargs, use_custom, ratio_name, width, height, has_image, has_latent, vae_unit
            )

            # 2. 按输入类型分发
            if has_image:
                if vae is not None:
                    return self._process_image_with_vae(
                        kwargs, target_width, target_height, 缩放方式, vae, vae_unit, latent_channels
                    )
                return self._process_image(
                    kwargs, target_width, target_height, 缩放方式, vae_unit, latent_channels
                )
            if has_latent:
                return self._process_latent(
                    kwargs, target_width, target_height, 缩放方式, 缩放倍数, vae_unit
                )
            
            # 无任何输入：仅返回尺寸对齐后的空 latent
            latent = {
                "samples": torch.zeros((1, latent_channels,
                    target_height // vae_unit, target_width // vae_unit)),
                "downscale_ratio_spacial": vae_unit
            }
            return (None, None, latent, target_width, target_height)
        
        except Exception as e:
            print(f"处理尺寸时出错: {e}")
            latent = {
                "samples": torch.zeros((1, 4, 1024 // 8, 1024 // 8)),
                "downscale_ratio_spacial": 8
            }
            return (None, None, latent, 1024, 1024)

    def _determine_target_size(self, kwargs, use_custom, ratio_name, width, height, has_image, has_latent, vae_unit):
        """确定目标尺寸，优先级：自定义 > 预设 > 图像 > latent > 兜底。最后统一对齐 vae_unit。"""
        if use_custom:
            target_width, target_height = width, height
        elif ratio_name != "关闭":
            target_width, target_height = get_size_from_preset(ratio_name)
        elif has_image:
            _, img_h, img_w, _ = kwargs["图像"].shape
            target_width, target_height = img_w, img_h
        elif has_latent:
            samples = kwargs["latent"]["samples"]
            target_width = samples.shape[-1] * vae_unit
            target_height = samples.shape[-2] * vae_unit
        else:
            target_width = target_height = 1024
        
        # 限幅 + 对齐
        target_width  = self._align_to_vae_unit(max(64, min(8192, target_width)), vae_unit)
        target_height = self._align_to_vae_unit(max(64, min(8192, target_height)), vae_unit)
        
        return target_width, target_height

    def _process_image(self, kwargs, target_width, target_height, 缩放方式, vae_unit, latent_channels):
        """图像 + 遮罩 resize/crop（无 VAE 编码路径），最后创建空 latent。"""
        image = kwargs["图像"]
        mask = kwargs.get("遮罩")
        缩放倍数 = kwargs.get("缩放倍数", 1.0)

        batch_size, height1, width1, _ = image.shape
        
        # 确定处理后尺寸
        if 缩放方式 == "等比缩放":
            # 以原始图像尺寸为基准乘倍数（忽略预设/自定义尺寸）
            scaled_width = int(width1 * 缩放倍数)
            scaled_height = int(height1 * 缩放倍数)
        else:
            scaled_width, scaled_height = target_width, target_height

        new_images, new_masks = [], []
        for i in range(batch_size):
            # 图像 → PIL → resize/crop → tensor
            img = Image.fromarray(np.clip(255. * image[i].cpu().numpy(), 0, 255).astype(np.uint8))
            if 缩放方式 == "中心裁剪":
                img = self._center_crop(img, scaled_width, scaled_height, width1, height1)
            img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
            new_images.append(np.array(img).astype(np.float32) / 255.0)

            # 遮罩（有遮罩 → 做相同处理；无遮罩 → 全白）
            if mask is not None:
                m = Image.fromarray(np.clip(255. * mask[i].cpu().numpy(), 0, 255).astype(np.uint8))
                if 缩放方式 == "中心裁剪":
                    m = self._center_crop(m, scaled_width, scaled_height, width1, height1)
                m = m.resize((scaled_width, scaled_height), Image.LANCZOS)
                new_masks.append(np.array(m).astype(np.float32) / 255.0)
            else:
                new_masks.append(np.ones((scaled_height, scaled_width), dtype=np.float32))

        new_image = torch.tensor(np.stack(new_images, axis=0))
        new_mask = torch.tensor(np.stack(new_masks, axis=0))
        
        # 最后对齐一次 vae_unit（PIL resize 结果可能没对齐）
        out_w = self._align_to_vae_unit(scaled_width, vae_unit)
        out_h = self._align_to_vae_unit(scaled_height, vae_unit)
        if out_w != scaled_width or out_h != scaled_height:
            new_image = self._resize_image_tensor(new_image, out_w, out_h)
            new_mask = self._resize_mask_tensor(new_mask, out_w, out_h)
            scaled_width, scaled_height = out_w, out_h
        
        latent = {
            "samples": torch.zeros((new_image.shape[0], latent_channels,
                scaled_height // vae_unit, scaled_width // vae_unit), device=new_image.device),
            "downscale_ratio_spacial": vae_unit
        }
        
        return (new_image, new_mask, latent, scaled_width, scaled_height)

    def _process_image_with_vae(self, kwargs, target_width, target_height, 缩放方式, vae, vae_unit, latent_channels):
        """先走 _process_image（图像+遮罩），再用 VAE encode 生成原生 latent。"""
        new_image, new_mask, _, scaled_width, scaled_height = self._process_image(
            kwargs, target_width, target_height, 缩放方式, vae_unit, latent_channels
        )
        
        latent = {
            "samples": vae.encode(new_image),
            "downscale_ratio_spacial": vae_unit
        }
        
        return (new_image, new_mask, latent, scaled_width, scaled_height)

    def _process_latent(self, kwargs, target_width, target_height, 缩放方式, 缩放倍数, vae_unit):
        """latent 输入路径：等比缩放或中心裁剪 latent 网格。"""
        latent = kwargs["latent"]
        batch_size, channels, latent_h, latent_w = latent["samples"].shape
        
        if 缩放方式 == "等比缩放":
            # 以 latent 原始空间尺寸 × 倍数（忽略预设/自定义尺寸）
            orig_w = latent_w * vae_unit
            orig_h = latent_h * vae_unit
            scaled_w = self._align_to_vae_unit(int(orig_w * 缩放倍数), vae_unit)
            scaled_h = self._align_to_vae_unit(int(orig_h * 缩放倍数), vae_unit)
            
            new_samples = torch.nn.functional.interpolate(
                latent["samples"],
                size=(scaled_h // vae_unit, scaled_w // vae_unit),
                mode="bilinear", align_corners=False
            )
            return (None, None, {"samples": new_samples, "downscale_ratio_spacial": vae_unit}, scaled_w, scaled_h)
        
        # 中心裁剪：先按目标比例裁 latent 网格，再 bilinear 到目标尺寸
        orig_w = latent_w * vae_unit
        orig_h = latent_h * vae_unit
        target_ratio = target_width / target_height
        orig_ratio = orig_w / orig_h
        
        if orig_ratio > target_ratio:
            new_w = int(orig_h * target_ratio)
            lw = new_w // vae_unit
            sx = (latent_w - lw) // 2
            cropped = latent["samples"][:, :, :, sx:sx + lw]
        else:
            new_h = int(orig_w / target_ratio)
            lh = new_h // vae_unit
            sy = (latent_h - lh) // 2
            cropped = latent["samples"][:, :, sy:sy + lh, :]
        
        new_samples = torch.nn.functional.interpolate(
            cropped,
            size=(target_height // vae_unit, target_width // vae_unit),
            mode="bilinear", align_corners=False
        )
        return (None, None, {"samples": new_samples, "downscale_ratio_spacial": vae_unit}, target_width, target_height)

    @staticmethod
    def _center_crop(img, target_width, target_height, original_width, original_height):
        """PIL 图像的中心裁剪。按目标宽高比裁掉长边多余部分。"""
        aspect_ratio = target_width / target_height
        img_ratio = original_width / original_height
        
        if img_ratio > aspect_ratio:
            # 宽度过大 → 裁宽度
            new_width = int(original_height * aspect_ratio)
            left = (original_width - new_width) // 2
            return img.crop((left, 0, left + new_width, original_height))
        
        # 高度过大 → 裁高度
        new_height = int(original_width / aspect_ratio)
        top = (original_height - new_height) // 2
        return img.crop((0, top, original_width, top + new_height))

    @staticmethod
    def _resize_image_tensor(tensor, width, height):
        """[B,H,W,C] → bilinear → [B,H,W,C]"""
        t = tensor.movedim(-1, 1)
        resized = torch.nn.functional.interpolate(
            t, size=(height, width), mode="bilinear", align_corners=False
        )
        return resized.movedim(1, -1)

    @staticmethod
    def _resize_mask_tensor(mask, width, height):
        """[B,H,W] → unsqueeze → bilinear → squeeze → [B,H,W]"""
        t = mask.unsqueeze(1)
        resized = torch.nn.functional.interpolate(
            t, size=(height, width), mode="bilinear", align_corners=False
        )
        return resized.squeeze(1)


NODE_CLASS_MAPPINGS = {
    "ST_ImageMaskLatentSize": ST_ImageMaskLatentSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_ImageMaskLatentSize": "图像尺寸"
}
