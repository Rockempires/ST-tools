"""
编辑图像节点 + 编辑对齐节点

ST_ImageEditor：
    图像编码 / 纯文本编码一体化节点。
    - 无图像输入时直接当作文本编码器使用（VAE 也可不接）
    - 三种模式：
        flux2klein   — 左上放置（右下补边），无视觉编码，reference_latents 仅写正面
        qwenedit-krea2 — 左上放置（右下补边），视觉编码（带 llama_template），
                       reference_latents 仅写正面
        qwenimage21  — 居中放置（四周补黑），keep_vision 协议，reference_latents 正负都写

    flux2klein / qwenedit-krea2 共用 _prepare_reference_latent：
        fit-within 等比缩放 → 黑色画布左上放置 → VAE encode

    qwenimage21 使用 _prepare_reference_latent(placement="center")：
        fit-within 等比缩放 → 黑色画布居中放置 → VAE encode → 视觉塔

    补边策略：
        画布（采样器目标）按 ref_width/ref_height 对齐 vae_unit；
        图像 fit-within 保持宽高比不变形；
        pad_info 记录补边几何 + target_w/h，供 ST_ImageSizeAligner 裁剪还原。

    遮罩（可选输入 mask）：
        白色区域重绘、黑色区域保留；写入输出 latent["noise_mask"]。

ST_ImageSizeAligner：
    接收 pad_info + KSampler 输出图像，强制 resize 到 target_w/target_h（编辑节点输入尺寸）。
"""

import torch
import math
import node_helpers
import comfy.utils
import comfy.model_management
from comfy_api.latest import io


_MODE_SPECS = {
    "flux2klein":  (128, 16, 4),
    "qwenedit-krea2": (16, 8, 5),    # Wan 3D VAE：单图 encode 为 5D [B,C,F,H,W]
    "qwenimage21": (64, 16, 4),
}

_EMPTY_PAD_INFO = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0, "target_w": 0, "target_h": 0}


def _debug_mask(label, noise_mask, samples_out, ref_latents, mode):
    """打印遮罩链路关键值。"""
    print(f"[ST-tools] [遮罩调试] === {mode} ({label}) ===")
    print(f"  采样输出 shape={tuple(samples_out.shape)} dtype={samples_out.dtype} "
          f"范围=[{samples_out.min():.3f}, {samples_out.max():.3f}] 非零数={torch.count_nonzero(samples_out).item()}")
    if noise_mask is not None:
        print(f"  遮罩 latent  shape={tuple(noise_mask.shape)} dtype={noise_mask.dtype} "
              f"范围=[{noise_mask.min():.3f}, {noise_mask.max():.3f}] "
              f"白色像素={torch.count_nonzero(noise_mask).item()}/{noise_mask.numel()}")
    else:
        print(f"  遮罩 latent  = 无")
    if ref_latents:
        r = ref_latents[0]
        print(f"  参考 latent[0] shape={tuple(r.shape)} dtype={r.dtype} "
              f"范围=[{r.min():.3f}, {r.max():.3f}] 非零数={torch.count_nonzero(r).item()}")

def _extract_downscale_ratio(vae):
    """从 VAE 解析空间压缩比。3D VAE 取最后一个元素（空间倍率），2D VAE 直接取值。"""
    raw = getattr(vae, "downscale_ratio", None)
    if raw is None:
        raw = getattr(vae, "downscale", None)
    if raw is None:
        return 8
    x = raw
    if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
        if len(x) == 0:
            return 8
        x = x[-1]
    try:
        return int(x)
    except (TypeError, ValueError):
        return 8


def _align_canvas(ref_width, ref_height, vae_unit):
    """目标宽高 ceil 对齐 vae_unit，返回画布像素尺寸。"""
    return (math.ceil(ref_width / vae_unit) * vae_unit,
            math.ceil(ref_height / vae_unit) * vae_unit)


def _empty_latent(channels, canvas_width, canvas_height, vae_unit, latent_dim):
    """按规格构造零 latent：4D→[1,C,lh,lw]，5D→[1,C,1,lh,lw]。"""
    lw = max(1, canvas_width // vae_unit)
    lh = max(1, canvas_height // vae_unit)
    shape = (1, channels, 1, lh, lw) if latent_dim == 5 else (1, channels, lh, lw)
    return torch.zeros(shape, device=comfy.model_management.intermediate_device())


def _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae=None, placement="top_left", zero_padding=False):
    """
    参考图像 → latent 处理流程。

    placement:
        "top_left" — fit-within 等比缩放后左上放置 → 右下补黑
        "center"   — fit-within 等比缩放后居中放置 → 四周补黑
    zero_padding:
        True  — VAE encode 后将补边区 latent 置零
        False — 保留完整 encode

    流程：
        1. fit-within 等比缩放，保持宽高比
        2. canvas 按 ref 对齐 vae_unit
        3. 按 placement 放置图像 → 黑边区填 0,0,0
        4. VAE 已接时 encode → zero_padding=True 则补边区置零

    返回：
        (encoded_latent, pad_info_dict, scaled_w, scaled_h, canvas_w, canvas_h, scale_by, canvas_image)
    """
    original_w = samples.shape[3]
    original_h = samples.shape[2]

    # fit-within：取宽高比中较小者，保证图像完整落在目标外框内
    scale_by_w = ref_width / original_w
    scale_by_h = ref_height / original_h
    scale_by = min(scale_by_w, scale_by_h)
    scaled_width = int(round(original_w * scale_by))
    scaled_height = int(round(original_h * scale_by))

    # canvas 按 ref 对齐 vae_unit → latent shape 可预测
    canvas_width = math.ceil(ref_width / vae_unit) * vae_unit
    canvas_height = math.ceil(ref_height / vae_unit) * vae_unit

    # 按 placement 计算图像在画布上的偏移
    if placement == "center":
        pad_left = (canvas_width - scaled_width) // 2
        pad_top = (canvas_height - scaled_height) // 2
    else:
        pad_left, pad_top = 0, 0

    # fit-within 等比缩放
    resized = comfy.utils.common_upscale(
        samples, scaled_width, scaled_height, "lanczos", "center"
    )

    # 黑色画布 + 按 placement 放置
    canvas = torch.zeros(
        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
        dtype=samples.dtype, device=samples.device
    )
    canvas[:, :, pad_top:pad_top + scaled_height, pad_left:pad_left + scaled_width] = resized

    encoded_latent = None
    if vae is not None:
        vae_img = canvas.movedim(1, -1)[:, :, :, :3]
        encoded_latent = vae.encode(vae_img)

        if zero_padding:
            # 将补边区 latent 置零 → 采样起点从噪声开始（扩图必需）
            if encoded_latent.dim() == 5:
                latent_h, latent_w = encoded_latent.shape[3], encoded_latent.shape[4]
            else:
                latent_h, latent_w = encoded_latent.shape[2], encoded_latent.shape[3]

            img_l = max(0, pad_left // vae_unit)
            img_t = max(0, pad_top // vae_unit)
            img_r = min(latent_w, (pad_left + scaled_width) // vae_unit)
            img_b = min(latent_h, (pad_top + scaled_height) // vae_unit)

            # 构造布尔 mask：True=补边区（置零），False=原图区（保留）
            rows = torch.arange(latent_h, device=encoded_latent.device)
            cols = torch.arange(latent_w, device=encoded_latent.device)
            img_area = (rows[:, None] >= img_t) & (rows[:, None] < img_b) & \
                       (cols[None, :] >= img_l) & (cols[None, :] < img_r)
            padding = ~img_area  # True 表示补边区

            if encoded_latent.dim() == 5:
                encoded_latent = encoded_latent.masked_fill(padding[None, None, None, :, :], 0.0)
            else:
                encoded_latent = encoded_latent.masked_fill(padding[None, None, :, :], 0.0)

    # pad_info：x/y 是图像左上偏移，width/height 是画布总补边量
    pad_info_dict = {
        "x": pad_left,
        "y": pad_top,
        "width": canvas_width - scaled_width,
        "height": canvas_height - scaled_height,
        "scale_by": round(scale_by, 3),
        "placement": placement,
        "target_w": ref_width,
        "target_h": ref_height,
    }

    return (encoded_latent, pad_info_dict,
            scaled_width, scaled_height, canvas_width, canvas_height, scale_by, canvas)


def _build_pixel_mask(mask, canvas_width, canvas_height,
                      scaled_width=None, scaled_height=None, placement="top_left",
                      fill_unscaled=1.0):
    """
    像素空间 mask：和 canvas_image 同尺寸、同 fit-within 比例、同 placement。

    fill_unscaled: 补边区（scaled 外）填什么值。
        1.0 — 扩图模式：补边区自由生成
        0.0 — 纯局部重绘：补边区锁定为"非重绘"
    返回 shape [1, H, W]，值 {0,1}。
    """
    if scaled_width is None:
        scaled_width, scaled_height = canvas_width, canvas_height

    if placement == "center":
        ox = (canvas_width - scaled_width) // 2
        oy = (canvas_height - scaled_height) // 2
    else:
        ox, oy = 0, 0

    if mask is None:
        # 无用户 mask：scaled 区全 1（全图自由采样），补边区填 fill_unscaled
        canvas_m = torch.full(
            (canvas_height, canvas_width), fill_unscaled, dtype=torch.float32
        )
        canvas_m[oy:oy + scaled_height, ox:ox + scaled_width] = 1.0
    else:
        m = mask[0] if mask.dim() == 3 else mask  # [H, W]
        m = torch.nn.functional.interpolate(
            m[None, None].float(), size=(scaled_height, scaled_width), mode="bilinear"
        )[0, 0].round()  # [scaled_h, scaled_w]
        canvas_m = torch.full(
            (canvas_height, canvas_width), fill_unscaled, dtype=m.dtype, device=m.device
        )
        canvas_m[oy:oy + scaled_height, ox:ox + scaled_width] = m

    return canvas_m[None]  # [1, canvas_h, canvas_w]


def _build_noise_mask(pixel_mask, vae_unit, latent_dim, grow=0):
    """
    像素空间 pixel_mask → latent 空间 noise_mask。

    pixel_mask: [B, H, W]  {0,1}，和 canvas_image 同尺寸。
    grow:       像素空间边缘膨胀（dilation），消除 latent 下采样硬边，默认 10。
    返回:        noise_mask tensor，对齐 latent_dim（4D→[B,1,lh,lw]，5D→[B,1,1,lh,lw]）。
    """
    # 统一转 4D [B, 1, H, W] 处理
    m = pixel_mask
    if m.dim() == 3:
        m = m[:, None]  # [B, H, W] → [B, 1, H, W]

    # grow 膨胀：全 1 卷积核 → 边缘外扩
    if grow > 0:
        kernel = torch.ones((1, 1, grow, grow), device=m.device)
        padding = math.ceil((grow - 1) / 2)
        m = torch.clamp(
            torch.nn.functional.conv2d(m.round(), kernel, padding=padding), 0, 1
        )

    # nearest 下采样到 latent 尺寸（精确对齐 ref_latents 的空间网格）
    _, _, ph, pw = m.shape
    lh = ph // vae_unit
    lw = pw // vae_unit
    grid = torch.nn.functional.interpolate(
        m, size=(lh, lw), mode="nearest"
    ).round()

    # 4D→[B,1,lh,lw]，5D→[B,1,1,lh,lw]
    if latent_dim == 5:
        grid = grid[:, None]  # [B,1,lh,lw] → [B,1,1,lh,lw]
    return grid


def _build_concat_latent_image(canvas_pixels, pixel_mask, vae):
    """
    mask=1（重绘区）填 0.5 灰度后 VAE encode，给模型层明确信号。

    canvas_pixels: [B, C, H, W]（像素空间，和 latent 同 canvas 尺寸）
    pixel_mask:    [B, H, W]  {0,1}（0=保持原图，1=重绘）
    vae:           VAE 实例
    返回:           VAE encode 后的 latent tensor
    """
    # canvas_pixels 是 [B, C, H, W]，取前3通道 → [B, 3, H, W]
    pixels = canvas_pixels[:, :3].clone()
    # pixel_mask: [B, H, W] → [B, 1, H, W]
    m = (1.0 - pixel_mask.round()).unsqueeze(1)
    # 中心化 → 乘 mask → 加回去
    pixels -= 0.5
    pixels *= m
    pixels += 0.5
    # VAE encode 期望 [B, H, W, C]
    return vae.encode(pixels.movedim(1, -1))


class ST_ImageEditor(io.ComfyNode):
    """图像编辑 / 文本编码节点：把输入图像编码为 CLIP conditioning + reference_latents + pad_info。"""

    @classmethod
    def define_schema(cls):
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("图片"),
            names=[f"图片{i}" for i in range(1, 11)],
            min=0,
        )

        return io.Schema(
            node_id="ST_ImageEditor",
            display_name="编辑图像",
            category="🎯 石头工具/图像编辑",
            description="图像编码 / 文本编码一体化节点，无图像输入时可直接当文本编码器使用（VAE 也可不接）。\n\n三种模式：\n- flux2klein：左上放置，右下补边，纯文本编码\n- qwenedit-krea2：左上放置，右下补边，视觉编码+官方编辑模板\n- qwenimage21：居中放置，四周补黑，视觉编码（keep_vision）\n\n等比缩放、兰佐斯插值、按VAE下采样倍率对齐；可选遮罩白色重绘黑色保留。\n\n输出：正面/负面提示词编码、latent、补边信息（供编辑对齐节点裁剪）。",
            inputs=[
                io.Clip.Input("clip", display_name="CLIP模型"),
                io.Vae.Input("vae", display_name="VAE模型", optional=True,
                             tooltip="可选。不接时不产出参考 latent；纯文生图无需连接。"),
                io.Autogrow.Input("图片", template=image_template,
                                  tooltip="可选。不接图片时本节点作为纯文本编码器使用。"),
                io.Mask.Input("mask", display_name="遮罩", optional=True,
                              tooltip="可选。白色区域重绘、黑色区域保留原图；输出 latent 自带 noise_mask（需把 latent 接入采样器）。"),
                io.String.Input("正面提示词", multiline=True, dynamic_prompts=True),
                io.String.Input("负面提示词", multiline=True, dynamic_prompts=True),
                io.Combo.Input("对齐模式",
                               options=["flux2klein", "qwenedit-krea2", "qwenimage21"],
                               default="flux2klein"),
                io.Int.Input("生成图像宽度", default=1024, min=1, max=4096, step=8,
                             tooltip="画布宽度，直接作为输出尺寸。"),
                io.Int.Input("生成图像高度", default=1024, min=1, max=4096, step=8,
                             tooltip="画布高度，直接作为输出尺寸。"),
            ],
            outputs=[
                io.Conditioning.Output("正面输出"),
                io.Conditioning.Output("负面输出"),
                io.Latent.Output("latent"),
                io.AnyType.Output("补边信息"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae=None, 图片=None, mask=None, 正面提示词="", 负面提示词="",
                对齐模式="flux2klein", 生成图像宽度=1024, 生成图像高度=1024) -> io.NodeOutput:
        input_images = [im for im in (图片 or {}).values() if im is not None]
        print(f"[ST-tools] [遮罩入口] 模式={对齐模式} 图像数={len(input_images)} "
              f"遮罩={'无' if mask is None else f'形状={tuple(mask.shape)} 范围=[{mask.min():.3f},{mask.max():.3f}]'}")

        if 对齐模式 == "flux2klein":
            result = cls._process_flux2klein(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )
        elif 对齐模式 == "qwenedit-krea2":
            result = cls._process_qwen(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )
        elif 对齐模式 == "qwenimage21":
            result = cls._process_qwen21(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )

        return io.NodeOutput(*result)

    @classmethod
    def _process_qwen(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """qwenedit-krea2 模式：左上放置（右下补边），视觉编码（带 llama_template）+ reference_latents 仅写正面。"""
        channels, def_unit, latent_dim = _MODE_SPECS["qwenedit-krea2"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        pad_info = dict(_EMPTY_PAD_INFO)
        ref_latents = []
        vl_images = []
        noise_mask = None
        image_prompt = ""

        for i, image in enumerate(input_images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            vl_target_size = 384
            total = int(vl_target_size * vl_target_size)
            scale_vl = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            vl_w = round(samples.shape[3] * scale_vl)
            vl_h = round(samples.shape[2] * scale_vl)
            s = comfy.utils.common_upscale(samples, vl_w, vl_h, "lanczos", "center")
            vl_images.append(s.movedim(1, -1)[:, :, :, :3])
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by, canvas_img = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == 0:
                pad_info = pi
                print(f"[ST-tools] [遮罩构建] qwenedit-krea2: 遮罩={'无' if mask is None else f'形状={tuple(mask.shape)}'}, "
                      f"画布={canvas_w}x{canvas_h}, 缩放后={scaled_w}x{scaled_h}, "
                      f"VAE倍率={vae_unit}, 图像数={len(input_images)}")
                pixel_mask = _build_pixel_mask(
                    mask, canvas_w, canvas_h, scaled_w, scaled_h,
                    placement="top_left", fill_unscaled=1.0  # 补边区自由生成（扩图）
                )
                noise_mask = _build_noise_mask(pixel_mask, vae_unit, latent_dim, grow=10)

        if len(vl_images) > 0:
            full_prompt = image_prompt + positive_prompt
            llama_template = (
                "<|im_start|>system\n"
                "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                "then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
                "<|im_start|>user\n{}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            positive_out = clip.encode_from_tokens_scheduled(
                clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
            )
        else:
            positive_out = clip.encode_from_tokens_scheduled(clip.tokenize(positive_prompt))
        negative_out = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        # Qwen 特有：reference_latents 仅写正面
        if len(ref_latents) > 0:
            positive_out = node_helpers.conditioning_set_values(
                positive_out, {"reference_latents": ref_latents}, append=True
            )
            samples_out = ref_latents[0]
        else:
            canvas_w, canvas_h = _align_canvas(ref_width, ref_height, vae_unit)
            samples_out = _empty_latent(channels, canvas_w, canvas_h, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
            # 遮罩区填灰后 VAE encode，作为 concat_latent_image
            if len(ref_latents) > 0 and vae is not None:
                concat_latent = _build_concat_latent_image(canvas_img, pixel_mask, vae)
                cond_updates = {"concat_mask": noise_mask, "concat_latent_image": concat_latent}
                positive_out = node_helpers.conditioning_set_values(positive_out, cond_updates, append=True)
                negative_out = node_helpers.conditioning_set_values(negative_out, cond_updates, append=True)
        _debug_mask("qwenedit-krea2", noise_mask, samples_out, ref_latents, "qwenedit-krea2")
        return (positive_out, negative_out, latent_out, pad_info)

    @classmethod
    def _process_flux2klein(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """flux2klein 模式：左上放置（右下补边），无视觉编码，reference_latents 仅写正面。"""
        channels, def_unit, latent_dim = _MODE_SPECS["flux2klein"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        pad_info = dict(_EMPTY_PAD_INFO)
        ref_latents = []
        noise_mask = None

        for i, image in enumerate(input_images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by, canvas_img = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == 0:
                pad_info = pi
                print(f"[ST-tools] [遮罩构建] flux2_klein: 遮罩={'无' if mask is None else f'形状={tuple(mask.shape)}'}, "
                      f"画布={canvas_w}x{canvas_h}, 缩放后={scaled_w}x{scaled_h}, "
                      f"VAE倍率={vae_unit}, 图像数={len(input_images)}")
                pixel_mask = _build_pixel_mask(
                    mask, canvas_w, canvas_h, scaled_w, scaled_h,
                    placement="top_left", fill_unscaled=1.0  # 补边区自由生成（扩图）
                )
                noise_mask = _build_noise_mask(pixel_mask, vae_unit, latent_dim, grow=10)
        positive_out = clip.encode_from_tokens_scheduled(clip.tokenize(positive_prompt))
        negative_out = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if len(ref_latents) > 0:
            positive_out = node_helpers.conditioning_set_values(
                positive_out, {"reference_latents": ref_latents}, append=True
            )
            samples_out = ref_latents[0]
        else:
            canvas_w, canvas_h = _align_canvas(ref_width, ref_height, vae_unit)
            samples_out = _empty_latent(channels, canvas_w, canvas_h, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
            # 遮罩区填灰后 VAE encode，作为 concat_latent_image
            if len(ref_latents) > 0 and vae is not None:
                concat_latent = _build_concat_latent_image(canvas_img, pixel_mask, vae)
                cond_updates = {"concat_mask": noise_mask, "concat_latent_image": concat_latent}
                positive_out = node_helpers.conditioning_set_values(positive_out, cond_updates, append=True)
                negative_out = node_helpers.conditioning_set_values(negative_out, cond_updates, append=True)
        _debug_mask("flux2_klein", noise_mask, samples_out, ref_latents, "flux2_klein")
        return (positive_out, negative_out, latent_out, pad_info)

    @classmethod
    def _process_qwen21(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """
        qwenimage21 模式：居中放置 + fit-within 等比缩放 + 四周纯黑补边。

        keep_vision 协议，reference_latents 正负都写（CFG 对称）。
        pad_info 记录居中偏移 + target_w/target_h，供 aligner 强制 resize 还原。
        提示词直接透传用户原始输入。
        """
        channels, def_unit, latent_dim = _MODE_SPECS["qwenimage21"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        pad_info = dict(_EMPTY_PAD_INFO)
        ref_latents = []
        vl_images = []
        noise_mask = None

        for i, image in enumerate(input_images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            result = _prepare_reference_latent(
                samples, ref_width, ref_height, vae_unit, vae, placement="center"
            )
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by, canvas = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)

            cw_32 = max(32, round(canvas_w / 32) * 32)
            ch_32 = max(32, round(canvas_h / 32) * 32)
            if (cw_32, ch_32) != (canvas_w, canvas_h):
                canvas_vl = comfy.utils.common_upscale(canvas, cw_32, ch_32, "lanczos", "disabled")
            else:
                canvas_vl = canvas
            vl_images.append(canvas_vl.movedim(1, -1)[:, :, :, :3])

            if i == 0:
                pad_info = pi
                print(f"[ST-tools] [遮罩构建] qwenimage21: 遮罩={'无' if mask is None else f'形状={tuple(mask.shape)}'}, "
                      f"画布={canvas_w}x{canvas_h}, 缩放后={scaled_w}x{scaled_h}, "
                      f"VAE倍率={vae_unit}, 图像数={len(input_images)}")
                pixel_mask = _build_pixel_mask(
                    mask, canvas_w, canvas_h, scaled_w, scaled_h,
                    placement="center", fill_unscaled=1.0  # 补边区自由生成（扩图）
                )
                noise_mask = _build_noise_mask(pixel_mask, vae_unit, latent_dim, grow=10)

        # 文本编码，与官方 TextEncodeQwenImage21 行为一致
        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(positive_prompt, images=vl_images, keep_vision=keep_vision, prevent_empty_text=True)
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt, images=vl_images, keep_vision=keep_vision, prevent_empty_text=True)
        )

        # 正负面都写 ref：CFG 两侧序列等长，差异只隔离文本方向
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )

        # samples：有 mask 且有 ref_latents → 用 VAE encode canvas（noise_mask=0 区保持原图）
        # 否则零 latent（参考图走 keep_vision context 拼接，不作采样起点）
        if noise_mask is not None and len(ref_latents) > 0:
            samples_out = ref_latents[0]
        else:
            canvas_w, canvas_h = _align_canvas(ref_width, ref_height, vae_unit)
            samples_out = _empty_latent(channels, canvas_w, canvas_h, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        _debug_mask("qwenimage21", noise_mask, samples_out, ref_latents, "qwenimage21")
        return (positive, negative, latent_out, pad_info)


class ST_ImageSizeAligner:
    """编辑对齐节点：根据补边信息裁剪 KSampler 输出，还原 fit-within 尺寸。"""

    DISPLAY_NAME = "编辑对齐"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "补边信息": ("ANY", ),
                "生成图像": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("对齐后图像",)
    FUNCTION = "align"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = "接收编辑图像节点的补边信息，对K采样器生成的图像进行反向裁剪/对齐，让生成图像与原图尺寸完全一致。"

    def align(self, 补边信息, 生成图像):
        target_w = 补边信息.get("target_w")
        target_h = 补边信息.get("target_h")

        img = 生成图像.movedim(-1, 1)
        cur_h, cur_w = img.shape[2], img.shape[3]

        # 有 target 尺寸：以 target 为唯一准绳
        if target_w and target_h:
            if cur_w == target_w and cur_h == target_h:
                # canvas 对齐 vae_unit 后恰好等于 target → 直接返回
                return (img.movedim(1, -1),)
            # canvas 对齐后 ≠ target（如 1080→1088）→ resize 还原
            img = comfy.utils.common_upscale(img, target_w, target_h, "lanczos", "disabled")
            return (img.movedim(1, -1),)

        # 无 target（旧 pad_info 兼容）→ 回退到裁剪补边
        x = 补边信息.get("x", 0)
        y = 补边信息.get("y", 0)
        width_padding = 补边信息.get("width", 0)
        height_padding = 补边信息.get("height", 0)
        placement = 补边信息.get("placement", "top_left")

        if placement == "center":
            cropped_img = img[
                :, :,
                y:cur_h - (height_padding - y),
                x:cur_w - (width_padding - x)
            ]
        else:
            cropped_img = img[
                :, :,
                y:cur_h - height_padding,
                x:cur_w - width_padding
            ]

        return (cropped_img.movedim(1, -1),)


NODE_CLASS_MAPPINGS = {
    "ST_ImageEditor": ST_ImageEditor,
    "ST_ImageSizeAligner": ST_ImageSizeAligner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_ImageEditor": "编辑图像",
    "ST_ImageSizeAligner": "编辑对齐",
}
