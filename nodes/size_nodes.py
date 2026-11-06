
from .. import st_log
"""
图像尺寸节点（ST_ImageMaskLatentSize）

职责：接收图像/遮罩/latent（全部 optional，至少接一个），输出对齐 vae_unit 后的图像、遮罩、latent 和最终宽高。

尺寸确定优先级：
    自定义尺寸（自定义尺寸开启） > 预设比例（预设比例≠关闭）
    > 图像输入尺寸 > latent 输入尺寸 > 兜底 1024×1024
确定后统一对齐到 vae_unit 倍数。

输入处理（所有已接入的输入都会处理并输出，未接入的输出为 None）：
    图像 → 中心裁剪按目标比例裁剪中心再 resize；等比缩放按原图 × 缩放倍数
    latent → 与图像同规则作用于 latent 网格，兼容 4D[B,C,H,W] 与 5D[B,C,F,H,W]
    遮罩 → 跟随图像同尺寸处理；无图像时按画布比例独立处理
    VAE + 图像 → 直接调用系统 vae.encode() 编码处理后的图像；无 VAE 但有 latent 输入则缩放该 latent

vae_unit 来源优先级：latent["downscale_ratio_spacial"] > VAE.downscale_ratio > 兜底 8
latent 通道数来源：latent["samples"].shape[1]（4D/5D 均在 dim1） > 兜底 4

遮罩处理：
    随图像接入 → 与图像做完全相同的 resize/crop；未接遮罩 → 全白遮罩兜底

无图像无 latent 时：用 torch.zeros 创建空 latent，shape = [B, channels, H/vae_unit, W/vae_unit]
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
                "自定义尺寸": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后用填写的宽高作为目标尺寸。已有图像或 Latent 且选择等比缩放时，改用原尺寸乘倍数。"}),
                "预设比例": (ratio_options, {"default": ratio_options[0] if ratio_options else "无可用比例", "tooltip": "关闭自定义尺寸时可选择预设画布；选「关闭」则从图像或 Latent 推导目标尺寸。"}),
                "画面方向": (["横向", "竖向"], {"default": "横向", "tooltip": "决定预设画布的横向或竖向，不改变已填写的自定义宽高。"}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "自定义画布宽度（像素）。处理后会按 VAE 的下采样倍率向上对齐。"}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "自定义画布高度（像素）。处理后会按 VAE 的下采样倍率向上对齐。"}),
                "缩放方式": (["中心裁剪", "等比缩放"], {"default": "中心裁剪", "tooltip": "中心裁剪按目标比例裁掉边缘再缩放；等比缩放按原尺寸乘倍数，适用于图像或 Latent。"}),
                "缩放倍数": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1, "tooltip": "仅在图像或 Latent 走等比缩放时使用，1.0 保持原始比例与大小，再做 VAE 对齐。"}),
            },
            "optional": {
                "图像": ("IMAGE", {"tooltip": "可选。接入后输出缩放图像；未接遮罩时同时输出全白遮罩。"}),
                "遮罩": ("MASK", {"tooltip": "可选。随图像一起处理，或独立调整到输出画布。"}),
                "latent": ("LATENT", {"tooltip": "可选。缩放已有潜变量；如果同时接入图像和 VAE，则改由图像编码生成 Latent。"}),
                "vae": ("VAE", {"tooltip": "可选。提供空间下采样倍率，并在接入图像时对处理后的图像编码。"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT")
    RETURN_NAMES = ("图像", "遮罩", "latent", "宽度", "高度")
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = """接线与输出
接入图像、遮罩或 Latent。输出处理后的数据和实际宽高（像素）。实际宽高向上对齐到 VAE 空间下采样倍率的整数倍。

中心裁剪
先按目标宽高比裁掉长边的两侧，再缩放到目标尺寸。
处理图像或 Latent 时，以下目标尺寸优先级仅在中心裁剪模式生效：
1. 开启自定义尺寸：使用填写的宽度和高度。
2. 自定义尺寸关闭且选了预设比例：使用预设尺寸，横竖方向也只作用于预设。
3. 两者都关闭：取图像尺寸，其次取 Latent 对应的像素尺寸，最后使用 1024×1024。

等比缩放
有图像时按原图宽高乘「缩放倍数」；只有 Latent 时按 Latent 对应的像素宽高乘该倍数。1.0 表示不主动放大或缩小，之后仍会做 VAE 对齐。
仅接遮罩，或图像与 Latent 都未接入时，按目标尺寸处理。

遮罩与 Latent
有图像时，遮罩随图像处理；未接遮罩则输出全白遮罩。只有 Latent 时，可把外接遮罩调整到同一画布。
同时接入图像和 VAE 时，Latent 由处理后的图像编码得到。否则使用已接入的 Latent 并调整尺寸；未接 Latent 时生成空 Latent。"""

    @staticmethod
    def _resolve_vae_unit_and_channels(kwargs):
        """
        解析 vae_unit / latent_channels / vae_is_3d。
        优先级：latent 自带标记与通道 > VAE 属性 > 兜底 8/4/2D。
        注意时序 VAE（Qwen/Wan/Hunyuan/Minimax）的 downscale_ratio 是
        元组 (时间函数, 空间H倍率, 空间W倍率)，不能直接 int()。
        """
        vae_unit = 8
        latent_channels = 4
        vae_is_3d = False

        latent_in = kwargs.get("latent")
        vae = kwargs.get("vae")

        # latent 自带最准的信息
        if latent_in is not None:
            if "downscale_ratio_spacial" in latent_in:
                vae_unit = int(latent_in["downscale_ratio_spacial"])
            # 4D:[B,C,H,W] 5D:[B,C,F,H,W]，通道都在 dim1
            if "samples" in latent_in and latent_in["samples"].dim() in (4, 5):
                latent_channels = int(latent_in["samples"].shape[1])
                vae_is_3d = latent_in["samples"].dim() == 5

        # VAE 补充（latent 没带时），不能覆盖 latent 自带值
        latent_has_unit = latent_in is not None and "downscale_ratio_spacial" in latent_in
        if vae is not None and not latent_has_unit:
            vae_unit = ST_ImageMaskLatentSize._parse_vae_spatial_ratio(vae)
        if vae is not None:
            vae_ch = getattr(vae, "latent_channels", None)
            if isinstance(vae_ch, int) and vae_ch > 0:
                latent_channels = vae_ch
            vae_is_3d = vae_is_3d or getattr(vae, "latent_dim", 2) == 3

        # 防御
        if not isinstance(vae_unit, int) or vae_unit <= 0:
            vae_unit = 8

        return vae_unit, latent_channels, vae_is_3d

    @staticmethod
    def _parse_vae_spatial_ratio(vae):
        """取 VAE 空间压缩比。int 直接用；时序 VAE 元组 (时间函数,H,W) 取空间值 W(=H)。"""
        raw = getattr(vae, "downscale_ratio", 8)
        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            raw = raw[-1]  # (temporal_fn, spatial_h, spatial_w)
        try:
            unit = int(raw)
            return unit if unit > 0 else 8
        except (TypeError, ValueError):
            return 8

    @staticmethod
    def _empty_latent(batch, width, height, vae_unit, latent_channels, vae_is_3d, device=None):
        """按 VAE 规格创建空 latent；3D VAE 为 5D [B,C,1,H,W]，2D 为 4D。"""
        shape = (batch, latent_channels, height // vae_unit, width // vae_unit)
        if vae_is_3d:
            shape = (batch, latent_channels, 1, height // vae_unit, width // vae_unit)
        return {
            "samples": torch.zeros(shape, device=device),
            "downscale_ratio_spacial": vae_unit
        }

    @staticmethod
    def _align_to_vae_unit(size, vae_unit):
        return math.ceil(size / vae_unit) * vae_unit

    def run(self, **kwargs):
        """主入口：确定目标尺寸 → 所有已接入的输入各自处理并返回，缺省的输出为 None。"""
        try:
            vae_unit, latent_channels, vae_is_3d = self._resolve_vae_unit_and_channels(kwargs)
            use_custom = kwargs["自定义尺寸"]
            ratio_name = kwargs["预设比例"]
            画面方向 = kwargs["画面方向"]
            width = kwargs["宽度"]
            height = kwargs["高度"]
            缩放方式 = kwargs["缩放方式"]
            缩放倍数 = kwargs["缩放倍数"]
            vae = kwargs.get("vae")

            image_in = kwargs.get("图像")
            mask_in = kwargs.get("遮罩")
            latent_in = kwargs.get("latent")
            has_image = image_in is not None
            has_latent = latent_in is not None

            # 1. 确定目标尺寸（统一对齐 vae_unit）
            target_width, target_height = self._determine_target_size(
                kwargs, use_custom, ratio_name, 画面方向, width, height, has_image, has_latent, vae_unit
            )

            # 2. 图像路径：图像/遮罩一起处理；latent 优先 VAE 编码，其次缩放输入 latent，最后空 latent
            if has_image:
                new_image, new_mask, _, canvas_w, canvas_h = self._process_image(
                    kwargs, target_width, target_height, 缩放方式, vae_unit, latent_channels
                )
                if vae is not None:
                    # 直接调用系统 VAE 编码（同内置 VAE Encode 节点）
                    out_latent = {
                        "samples": vae.encode(new_image),
                        "downscale_ratio_spacial": vae_unit
                    }
                elif has_latent:
                    out_latent = self._latent_to_canvas(
                        latent_in, canvas_w, canvas_h, 缩放方式, vae_unit
                    )
                else:
                    out_latent = self._empty_latent(
                        new_image.shape[0], canvas_w, canvas_h, vae_unit,
                        latent_channels, vae_is_3d, device=new_image.device
                    )
                return (new_image, new_mask, out_latent, canvas_w, canvas_h)

            # 3. 无图像：处理 latent（遮罩若有则单独处理到同一画布）
            if has_latent:
                samples = latent_in["samples"]
                latent_h, latent_w = samples.shape[-2], samples.shape[-1]
                if 缩放方式 == "等比缩放":
                    canvas_w = self._align_to_vae_unit(int(latent_w * vae_unit * 缩放倍数), vae_unit)
                    canvas_h = self._align_to_vae_unit(int(latent_h * vae_unit * 缩放倍数), vae_unit)
                else:
                    canvas_w, canvas_h = target_width, target_height

                out_latent = self._latent_to_canvas(
                    latent_in, canvas_w, canvas_h, 缩放方式, vae_unit
                )
                new_mask = None
                if mask_in is not None:
                    new_mask = self._process_mask_to_canvas(
                        mask_in, canvas_w, canvas_h, 缩放方式
                    )
                return (None, new_mask, out_latent, canvas_w, canvas_h)

            # 4. 仅遮罩：遮罩处理到目标画布
            if mask_in is not None:
                new_mask = self._process_mask_to_canvas(
                    mask_in, target_width, target_height, 缩放方式
                )
                latent = self._empty_latent(
                    1, target_width, target_height, vae_unit, latent_channels, vae_is_3d
                )
                return (None, new_mask, latent, target_width, target_height)

            # 5. 无任何输入：仅返回尺寸对齐后的空 latent
            latent = self._empty_latent(
                1, target_width, target_height, vae_unit, latent_channels, vae_is_3d
            )
            return (None, None, latent, target_width, target_height)

        except Exception as e:
            # 出错也必须原样透传真实输入，绝不能返回写死的全零 latent（VAE 解码会变绿屏）
            image_in = kwargs.get("图像")
            mask_in = kwargs.get("遮罩")
            latent_in = kwargs.get("latent")

            if image_in is not None:
                passthrough_w, passthrough_h = image_in.shape[2], image_in.shape[1]
            elif latent_in is not None and "samples" in latent_in:
                try:
                    unit = int(latent_in.get("downscale_ratio_spacial", 8))
                except (TypeError, ValueError):
                    unit = 8
                passthrough_w = latent_in["samples"].shape[-1] * unit
                passthrough_h = latent_in["samples"].shape[-2] * unit
            elif mask_in is not None:
                passthrough_w, passthrough_h = mask_in.shape[-1], mask_in.shape[-2]
            else:
                passthrough_w = passthrough_h = 1024

            if latent_in is None:
                fb_vae = kwargs.get("vae")
                fb_unit = self._parse_vae_spatial_ratio(fb_vae) if fb_vae is not None else 8
                fb_ch = int(getattr(fb_vae, "latent_channels", 4)) if fb_vae is not None else 4
                fb_3d = fb_vae is not None and getattr(fb_vae, "latent_dim", 2) == 3
                latent_in = self._empty_latent(
                    1, passthrough_w, passthrough_h, fb_unit, fb_ch, fb_3d
                )
            return (image_in, mask_in, latent_in, passthrough_w, passthrough_h)

    def _determine_target_size(self, kwargs, use_custom, ratio_name, orientation, width, height, has_image, has_latent, vae_unit):
        """确定目标尺寸，优先级：自定义 > 预设 > 图像 > latent > 兜底。最后统一对齐 vae_unit。"""
        if use_custom:
            target_width, target_height = width, height
        elif ratio_name != "关闭":
            target_width, target_height = get_size_from_preset(ratio_name, orientation)
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

    def _latent_to_canvas(self, latent_in, canvas_w, canvas_h, 缩放方式, vae_unit):
        """把输入 latent 裁/缩放到指定像素画布，兼容 4D[B,C,H,W] 与 5D[B,C,F,H,W]。"""
        samples = latent_in["samples"]
        latent_h, latent_w = samples.shape[-2], samples.shape[-1]

        # 裁剪区在 latent 网格中的位置（等比缩放不裁剪）
        top, left, crop_h, crop_w = 0, 0, latent_h, latent_w

        if 缩放方式 == "中心裁剪":
            target_ratio = canvas_w / canvas_h
            orig_ratio = latent_w / latent_h
            if orig_ratio > target_ratio:
                crop_w = max(1, int(latent_h * target_ratio))
                left = (latent_w - crop_w) // 2
            else:
                crop_h = max(1, int(latent_w / target_ratio))
                top = (latent_h - crop_h) // 2

        cropped = self._spatial_crop(samples, top, left, crop_h, crop_w)
        new_grid_h, new_grid_w = canvas_h // vae_unit, canvas_w // vae_unit
        new_samples = self._spatial_resize(cropped, new_grid_h, new_grid_w)

        # 复制原 dict，保留 noise_mask 等附加键（直接重建会丢遮罩）
        out_latent = dict(latent_in)
        out_latent["samples"] = new_samples
        out_latent["downscale_ratio_spacial"] = vae_unit
        if out_latent.get("noise_mask") is not None:
            out_latent["noise_mask"] = self._transform_noise_mask(
                out_latent["noise_mask"], top, left, crop_h, crop_w, new_grid_h, new_grid_w
            )
        return out_latent

    def _process_mask_to_canvas(self, mask, canvas_w, canvas_h, 缩放方式):
        """独立遮罩输入：与图像同规则（中心裁剪按画布比例）缩放到画布。"""
        if 缩放方式 == "中心裁剪":
            mask = self._center_crop_mask_tensor(mask, canvas_w / canvas_h)
        return self._resize_mask_tensor(mask, canvas_w, canvas_h)

    @staticmethod
    def _center_crop_mask_tensor(mask, aspect_ratio):
        """[B,H,W] 遮罩按目标宽高比中心裁剪。"""
        mask_h, mask_w = mask.shape[-2], mask.shape[-1]
        if mask_w / mask_h > aspect_ratio:
            new_w = int(mask_h * aspect_ratio)
            left = (mask_w - new_w) // 2
            return mask[:, :, left:left + new_w]
        new_h = int(mask_w / aspect_ratio)
        top = (mask_h - new_h) // 2
        return mask[:, top:top + new_h, :]

    @staticmethod
    def _spatial_crop(samples, top, left, crop_h, crop_w):
        """仅在宽高维中心裁剪，兼容 4D[B,C,H,W] 与 5D[B,C,F,H,W]，时间维不动。"""
        if samples.dim() == 5:
            return samples[:, :, :, top:top + crop_h, left:left + crop_w]
        return samples[:, :, top:top + crop_h, left:left + crop_w]

    @staticmethod
    def _spatial_resize(samples, new_h, new_w):
        """仅缩放宽高（bilinear），5D 时把时间维折进 batch 逐帧插值后还原。"""
        if samples.dim() == 5:
            B, C, F, H, W = samples.shape
            frames = samples.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            frames = torch.nn.functional.interpolate(
                frames, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            return frames.reshape(B, F, C, new_h, new_w).permute(0, 2, 1, 3, 4)
        return torch.nn.functional.interpolate(
            samples, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

    @staticmethod
    def _transform_noise_mask(mask, top, left, crop_h, crop_w, new_h, new_w):
        """noise_mask 跟随 latent 网格做相同裁剪+缩放，nearest 保持 {0,1} 二值。"""
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        mask = mask[:, top:top + crop_h, left:left + crop_w]
        return torch.nn.functional.interpolate(
            mask.unsqueeze(1).float(), size=(new_h, new_w), mode="nearest"
        ).squeeze(1) == 1

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

