
from .. import _reg_set, _reg_get
"""
编辑图像节点 + 编辑对齐节点

ST_ImageEditor：
    图像编码 / 纯文本编码一体化节点。
    - 无图像输入时直接当作文本编码器使用（VAE 也可不接）
    - 三种模式：
        flux2klein     — 左上放置（右下补边），无视觉编码，reference_latents 仅写正面
        qwenedit-krea2 — 左上放置（右下补边），视觉编码（带 llama_template），
                         reference_latents 仅写正面
        qwenimage21    — 入口检测尺寸后分两套独立处理：
            · 同尺寸（宽/高为 0 或与图像1一致）→ 纯编辑：官方 TextEncodeQwenImage21
              按短边 vis_res 编码，采样后编辑对齐节点放大回图像1原尺寸
            · 不同尺寸（宽/高均>0 且与图像1不同）→ 扩图：每张参考图 fit-within
              居中贴纯黑画布，视觉塔+VAE 整张画布编码，输出即输入画布尺寸

    flux2klein / qwenedit-krea2 共用 _prepare_reference_latent：
        fit-within 等比缩放 → 黑色画布左上放置 → VAE encode

    补边策略：
        画布（采样器目标）按 ref_width/ref_height 对齐 vae_unit；
        图像 fit-within 保持宽高比不变形；
        pad_info 记录补边几何 + target_w/h 或 orig_w/h，供 ST_ImageSizeAligner 裁剪/放大还原。

    遮罩（可选输入 mask）：
        白色区域重绘、黑色区域保留；写入输出 latent["noise_mask"]。

ST_ImageSizeAligner：
    接收 pad_info + KSampler 输出图像还原最终尺寸：
    优先按 orig_w/h（qwenimage21 纯编辑放大回图像1原尺寸）或 target_w/h（扩图/其他模式的画布尺寸）
    lanczos resize；二者都缺失时按补边几何（x/y/width/height/placement）裁掉补边区。
"""

import torch
import math
import node_helpers
import comfy.utils
import comfy.model_management
from comfy_api.latest import io
from comfy_extras.nodes_qwen import TextEncodeQwenImage21


_MODE_SPECS = {
    "flux2klein":  (128, 16, 4),
    "qwenedit-krea2": (16, 8, 5),    # Wan 3D VAE：单图 encode 为 5D [B,C,F,H,W]
    "qwenimage21": (64, 16, 4),
}

_EMPTY_PAD_INFO = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0, "target_w": 0, "target_h": 0}


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


def _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae=None, placement="top_left"):
    """
    参考图像 → latent 处理流程。

    placement:
        "top_left" — fit-within 等比缩放后左上放置 → 右下补黑
        "center"   — fit-within 等比缩放后居中放置 → 四周补黑

    流程：
        1. fit-within 等比缩放，保持宽高比
        2. canvas 按 ref ceil 对齐 vae_unit
        3. 按 placement 放置图像 → 黑边区填 0,0,0
        4. VAE 已接时对整张画布 encode

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

    # canvas 按 ref ceil 对齐 vae_unit → latent shape 可预测
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
    像素空间 mask：与画布同尺寸，用户遮罩缩放到 scaled 子区并按 placement 放入。

    fill_unscaled: 补边区（scaled 子区外）的取值，当前各模式统一传 1.0（补边区自由生成）。
        用户未接遮罩时整个画布恒为 1（全图自由采样）。
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
    grow:       像素空间边缘膨胀（dilation）像素数，0 表示不膨胀；调用方统一传 10。
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
            description="""接线与参考图
接入与模式匹配的 CLIP；用参考图编辑时接入匹配的 VAE 以生成参考 Latent。可接多张参考图。无参考图时可用于文生图，VAE 输入可省略。

Flux 2 Klein 与 Qwen Image Krea2
flux2klein 使用文本编码，qwenedit-krea2 使用视觉编码和编辑模板；两者都将参考图放在画布左上角并在右下补边，目标宽高请填至少 32。

QwenImage 2.1
qwenimage21 用首张已连接参考图判断编辑或扩图。宽高任一为 0，或目标尺寸与参考图各自取整到 32 的倍数后相同，则走普通编辑。否则将参考图居中补边至目标画布。无参考图时，宽或高填 0 的对应边按 1024 处理。

遮罩与输出
可选遮罩：白色区域参与重绘，黑色区域限制去噪。
将正面、负面和 Latent 输出接采样器；解码后把图像与本节点的补边信息一起接「编辑对齐」，恢复原图或目标画布尺寸。""",
            inputs=[
                io.Clip.Input("clip", display_name="CLIP模型"),
                io.Vae.Input("vae", display_name="VAE模型", optional=True,
                             tooltip="可选。图像编辑时接入匹配的 VAE 以生成参考 Latent；纯文生图可省略。"),
                io.Autogrow.Input("图片", template=image_template,
                                  tooltip="可选。不接图片时本节点作为纯文本编码器使用。"),
                io.Mask.Input("mask", display_name="遮罩", optional=True,
                              tooltip="可选。白色区域参与重绘，黑色区域限制去噪。将带噪声遮罩的 Latent 接入采样器。"),
                io.String.Input("正面提示词", multiline=True, dynamic_prompts=True),
                io.String.Input("负面提示词", multiline=True, dynamic_prompts=True),
                io.Combo.Input("对齐模式",
                               options=["flux2klein", "qwenedit-krea2", "qwenimage21"],
                               default="flux2klein",
                               tooltip="按模型选择 Flux 2 Klein、Qwen Image Krea2 或 QwenImage 2.1 的处理模式。"),
                io.Int.Input("生成图像宽度", default=1024, min=0, max=16384, step=8,
                             tooltip="qwenimage21：有参考图时宽高任一填 0，或按 32 对齐后与首张参考图尺寸相同，即普通编辑；否则用填写的宽高作为扩图目标。无图时此项填 0 按 1024 处理。其他模式请填至少 32。"),
                io.Int.Input("生成图像高度", default=1024, min=0, max=16384, step=8,
                             tooltip="qwenimage21：有参考图时宽高任一填 0，或按 32 对齐后与首张参考图尺寸相同，即普通编辑；否则用填写的宽高作为扩图目标。无图时此项填 0 按 1024 处理。其他模式请填至少 32。"),
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
        scene = "编辑" if input_images else "文生图"

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
            # 入口检测同尺寸（纯编辑）/不同尺寸（扩图），传原始 图片 dict 保留编号槽位
            result = cls._process_qwen21(
                clip, vae, 图片, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )

        # 把元信息登记到 registry（用 id(positive_out) 做 key，因为 conditioning 是 Python list 不能打属性）
        positive_out = result[0]
        clip_name = _reg_get(clip, 'filename') if clip is not None else None
        _reg_set(positive_out, meta={
            "mode": 对齐模式,
            "scene": scene,
            "mask": "有" if mask is not None else "无",
            "clip_name": clip_name,
            "pos_len": len(正面提示词),
            "neg_len": len(负面提示词),
        })

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
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, _scale_by, canvas_img = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == 0:
                pad_info = pi
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
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, _scale_by, canvas_img = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == 0:
                pad_info = pi
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
            if len(ref_latents) > 0 and vae is not None:
                concat_latent = _build_concat_latent_image(canvas_img, pixel_mask, vae)
                cond_updates = {"concat_mask": noise_mask, "concat_latent_image": concat_latent}
                positive_out = node_helpers.conditioning_set_values(positive_out, cond_updates, append=True)
                negative_out = node_helpers.conditioning_set_values(negative_out, cond_updates, append=True)
        return (positive_out, negative_out, latent_out, pad_info)

    @classmethod
    def _process_qwen21(cls, clip, vae, images_map, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """qwenimage21 入口：先检测同尺寸 / 不同尺寸，再分发到两套完全独立的处理。

        同尺寸（宽高输入任一为 0，或输入尺寸与图像1一致）→ _process_qwen21_edit
        不同尺寸（输入宽高均 >0 且与图像1尺寸不同）      → _process_qwen21_outpaint
        """
        images_map = images_map or {}
        ordered_keys = sorted(
            images_map.keys(),
            key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0),
        )
        imgs = [images_map[k] for k in ordered_keys if images_map.get(k) is not None]

        if not imgs:
            return cls._process_qwen21_edit(
                clip, vae, images_map, positive_prompt, negative_prompt,
                ref_width, ref_height, mask)

        first_img = imgs[0]
        orig_h, orig_w = int(first_img.shape[1]), int(first_img.shape[2])
        ow = max(32, round(orig_w / 32) * 32)
        oh = max(32, round(orig_h / 32) * 32)

        if ref_width and ref_width > 0 and ref_height and ref_height > 0:
            cw = max(32, round(ref_width / 32) * 32)
            ch = max(32, round(ref_height / 32) * 32)
        else:
            cw, ch = ow, oh

        if (cw, ch) == (ow, oh):
            return cls._process_qwen21_edit(
                clip, vae, images_map, positive_prompt, negative_prompt,
                ref_width, ref_height, mask)

        return cls._process_qwen21_outpaint(
            clip, vae, images_map, positive_prompt, negative_prompt,
            ref_width, ref_height, mask)

    @classmethod
    def _process_qwen21_edit(cls, clip, vae, images_map, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """qwenimage21 纯编辑模式（独立）。

        参考图按短边 vis_res（1024~2560）等比缩放后交官方 TextEncodeQwenImage21 编码，
        采样在缩后尺寸进行，补边信息记录图像1原尺寸，由编辑对齐节点 lanczos 放大回原尺寸。
        遮罩 nearest 铺满整个生成画布。宽/高输入框仅在无图纯文本时用于零 latent。
        """
        images_map = images_map or {}
        ordered_keys = sorted(
            images_map.keys(),
            key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0),
        )
        official_images = {f"image_{i + 1}": images_map.get(k)
                           for i, k in enumerate(ordered_keys)}
        channels, vae_unit, _ = _MODE_SPECS["qwenimage21"]  # 64 / 16 / 4

        # 无图纯文本：官方编码，零 latent 取宽高输入框
        if not any(images_map.get(k) is not None for k in ordered_keys):
            out = TextEncodeQwenImage21.execute(
                clip, positive_prompt, negative_prompt,
                vae=vae, resolution=0, images=official_images)
            positive, negative, latent_out = out[0], out[1], out[2]
            cw = max(32, round((ref_width or 1024) / 32) * 32)
            ch = max(32, round((ref_height or 1024) / 32) * 32)
            latent_out["samples"] = torch.zeros(
                [1, channels, ch // vae_unit, cw // vae_unit],
                device=comfy.model_management.intermediate_device())
            if mask is not None:
                gh, gw = latent_out["samples"].shape[-2], latent_out["samples"].shape[-1]
                m = mask[0] if mask.dim() == 3 else mask
                m = torch.nn.functional.interpolate(
                    m[None, None].float(), size=(gh * vae_unit, gw * vae_unit), mode="nearest")[0, 0]
                grid = torch.nn.functional.interpolate(
                    m[None, None], size=(gh, gw), mode="nearest")[0, 0].round()
                latent_out["noise_mask"] = grid[None, None]
            return (positive, negative, latent_out, dict(_EMPTY_PAD_INFO))

        first_img = next(images_map[k] for k in ordered_keys if images_map.get(k) is not None)
        orig_h, orig_w = int(first_img.shape[1]), int(first_img.shape[2])
        vis_res = max(1024, min(orig_w, orig_h, 2560))

        out = TextEncodeQwenImage21.execute(
            clip, positive_prompt, negative_prompt,
            vae=vae, resolution=vis_res, images=official_images)
        positive, negative, latent_out = out[0], out[1], out[2]
        zero = latent_out["samples"]
        scaled_w, scaled_h = zero.shape[-1] * vae_unit, zero.shape[-2] * vae_unit

        if mask is not None:
            gh, gw = zero.shape[-2], zero.shape[-1]
            m = mask[0] if mask.dim() == 3 else mask
            m = torch.nn.functional.interpolate(
                m[None, None].float(), size=(gh * vae_unit, gw * vae_unit), mode="nearest")[0, 0]
            grid = torch.nn.functional.interpolate(
                m[None, None], size=(gh, gw), mode="nearest")[0, 0].round()
            latent_out["noise_mask"] = grid[None, None]

        pad_info = {
            "x": 0, "y": 0, "width": 0, "height": 0,
            "scale_by": orig_w / scaled_w if scaled_w > 0 else 1.0,
            "orig_w": orig_w, "orig_h": orig_h,
        }
        return (positive, negative, latent_out, pad_info)

    @classmethod
    def _process_qwen21_outpaint(cls, clip, vae, images_map, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """qwenimage21 不同尺寸扩图处理（独立，与纯编辑互不共用）。

        每张参考图各自 fit-within 等比缩放后居中贴入同一尺寸纯黑画布（四周补黑），
        视觉塔取 32 对齐画布，VAE 直接 encode 整张画布作 reference_latents（正/负都写）。
        接了 VAE 时采样起点取整张画布 encode（未接遮罩即等价全图重绘），未接 VAE 时用零 latent。
        pad_info 记录居中偏移 + target_w/h，供编辑对齐节点还原到输入画布尺寸。
        """
        channels, def_unit, latent_dim = _MODE_SPECS["qwenimage21"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        ordered_keys = sorted(
            (images_map or {}).keys(),
            key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0),
        )
        input_images = [images_map[k] for k in ordered_keys if images_map.get(k) is not None]

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
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, _scale_by, canvas = result
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
                pixel_mask = _build_pixel_mask(
                    mask, canvas_w, canvas_h, scaled_w, scaled_h,
                    placement="center", fill_unscaled=1.0  # 补边区自由生成（扩图）
                )
                noise_mask = _build_noise_mask(pixel_mask, vae_unit, latent_dim, grow=10)

        # 视觉塔编码，keep_vision 协议；reference_latents 正/负都写（CFG 两侧序列等长）
        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(positive_prompt, images=vl_images, keep_vision=keep_vision, prevent_empty_text=True)
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt, images=vl_images, keep_vision=keep_vision, prevent_empty_text=True)
        )

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )

        # 有 noise_mask 且有 ref_latents → 采样起点用画布 encode；否则零 latent
        if noise_mask is not None and len(ref_latents) > 0:
            samples_out = ref_latents[0]
        else:
            cw, ch = _align_canvas(ref_width, ref_height, vae_unit)
            samples_out = _empty_latent(channels, cw, ch, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        return (positive, negative, latent_out, pad_info)


class ST_ImageSizeAligner:
    """编辑对齐节点：按补边信息把 KSampler 输出 resize 回目标尺寸，或裁掉补边区。"""

    DISPLAY_NAME = "编辑对齐"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "补边信息": ("ANY",  {"tooltip": "接同一次「编辑图像」的补边信息，用来恢复输出尺寸。"}),
                "生成图像": ("IMAGE",  {"tooltip": "接采样后经 VAE 解码的图像。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("对齐后图像",)
    FUNCTION = "align"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = """接线
将采样结果经 VAE 解码后的图像接入本节点，并把同一次「编辑图像」执行的「补边信息」接过来。

尺寸恢复与输出
普通编辑按记录恢复参考图原尺寸；扩图按记录恢复目标画布尺寸。其他带补边的编辑按记录裁掉补边再还原。输出是处理后的图像，可接预览或保存。"""

    def align(self, 补边信息, 生成图像):
        # 目标尺寸优先级：orig_w/h（纯编辑放大回图像1原尺寸）> target_w/h（画布输入尺寸）
        target_w = 补边信息.get("orig_w") or 补边信息.get("target_w")
        target_h = 补边信息.get("orig_h") or 补边信息.get("target_h")

        img = 生成图像.movedim(-1, 1)
        cur_h, cur_w = img.shape[2], img.shape[3]

        if target_w and target_h:
            if cur_w == target_w and cur_h == target_h:
                return (img.movedim(1, -1),)
            img = comfy.utils.common_upscale(img, target_w, target_h, "lanczos", "disabled")
            return (img.movedim(1, -1),)

        # 无目标尺寸（无图纯文本的空补边信息）→ 按补边几何裁掉补边区
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

