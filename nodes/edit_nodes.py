"""
编辑图像节点 + 编辑对齐节点

ST_ImageEditor：
    图像编码 / 纯文本编码一体化节点。
    - 无图像输入时直接当作文本编码器使用（VAE 也可不接）
    - 五种模式：
        flux2klein  — 无视觉编码，reference_latents 仅写正面
        qwenedit    — 有视觉编码（带 llama_template），reference_latents 仅写正面
        boogu       — 有视觉编码，reference_latents 同时写正负 → CFG 下抵消原图结构
        qwenimage21— QwenImage 2.1 协议：keep_vision 双模式 + image_slots，ref 写正负，零 latent
        mingimage  — MingImage 协议：视觉塔吃原图，3D VAE ref_frames，direct_context 自动携带
    
    旧三模式参考图处理（_prepare_reference_latent）：
        fit-within 等比缩放 → 黑色画布左上放置 → VAE encode
        latent shape = ceil(ref/vae_unit) * vae_unit → 与采样器目标尺寸精确匹配
        VAE 未接时只走视觉塔文本条件，不产出 reference_latents。
    
    补边策略：
        画布（采样器目标）按 ref 对齐 vae_unit；
        图像 fit-within 保持宽高比不变形，右下留白 → 黑色补边；
        pad_info 记录右下补边量，供 ST_ImageSizeAligner 裁剪还原。
    qwenimage21 / mingimage 无补边（ref 独立画布或拉伸首图画布），pad_info 补边量为 0。

    遮罩（可选输入 mask）：
        旧三模式与主图共享 fit-within + 补边几何，nearest 缩放保 {0,1} 二值；
        新模式按目标画布整体 nearest；
        白色区域重绘、黑色区域保留；写入输出 latent["noise_mask"]。

ST_ImageSizeAligner：
    接收 pad_info + KSampler 输出图像，裁剪掉右下补边区域，还原 fit-within 尺寸。
"""

import torch
import math
import node_helpers
import comfy.utils
import comfy.model_management
from comfy_api.latest import io


# 无图 / 无 VAE 时的兜底规格：(latent 通道, 空间压缩比, latent 维度)，以 latent_formats.py 为准
_MODE_SPECS = {
    "flux2klein":  (128, 16, 4),
    "qwenedit":    (16, 8, 5),    # Wan 3D VAE：单图 encode 为 5D [B,C,F,H,W]
    "boogu":       (16, 8, 4),    # Flux 格式
    "qwenimage21": (64, 16, 4),
    "mingimage":   (16, 8, 5),    # 3D VAE
}

_EMPTY_PAD_INFO = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0}


def _extract_downscale_ratio(vae):
    """从 VAE 解析空间压缩比。兼容 int/float/tensor/tuple/list/None，失败兜底 8。"""
    raw = getattr(vae, "downscale_ratio", None)
    if raw is None:
        raw = getattr(vae, "downscale", None)
    if raw is None:
        return 8
    x = raw
    while hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
        if len(x) == 0:
            return 8
        x = x[0]
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


def _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae=None):
    """
    旧三模式统一的参考图像 → latent 处理流程

    流程：
        1. fit-within 等比缩放，保持宽高比（scaled 不对齐 vae_unit，不变形）
        2. canvas（采样器目标）按 ref 外框对齐 vae_unit
        3. 黑色画布左上放置 → 右下补边区为纯黑
        4. VAE 已接时 encode → latent shape = canvas/vae_unit；未接时返回 None（只走视觉塔）

    参数：
        samples: [B, C, H, W] 原图（已 movedim 到 CHW）
        ref_width, ref_height: 生成目标宽高（像素）
        vae_unit: VAE 空间压缩比
        vae: VAE 模型（可选）

    返回：
        (encoded_latent, pad_info_dict, scaled_w, scaled_h, canvas_w, canvas_h, scale_by)
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

    # resize 到 fit-within 尺寸（等比，不变形）
    resized = comfy.utils.common_upscale(
        samples, scaled_width, scaled_height, "lanczos", "center"
    )

    # 黑色画布 + 左上放置（右下补边区填黑，让采样器自由生成而非锁定 VAE 边界伪影）
    canvas = torch.zeros(
        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
        dtype=samples.dtype, device=samples.device
    )
    canvas[:, :, :scaled_height, :scaled_width] = resized

    encoded_latent = None
    if vae is not None:
        # canvas 尺寸 → latent shape = canvas / vae_unit
        vae_img = canvas.movedim(1, -1)[:, :, :, :3]
        encoded_latent = vae.encode(vae_img)

    # pad_info：像素级，记录右下补边总量，供 aligner 裁剪回 scaled 尺寸
    pad_info_dict = {
        "x": 0,
        "y": 0,
        "width": canvas_width - scaled_width,
        "height": canvas_height - scaled_height,
        "scale_by": round(scale_by, 3),
    }

    return (encoded_latent, pad_info_dict,
            scaled_width, scaled_height, canvas_width, canvas_height, scale_by)


def _build_noise_mask(mask, canvas_width, canvas_height, vae_unit, latent_dim,
                      scaled_width=None, scaled_height=None):
    """
    输入遮罩 → 采样器 noise_mask。

    scaled_width/height 给定时（旧三模式）：先缩到 fit-within 子区再左上放入画布，
        右下补边区保持 0；不给定时（新模式）：遮罩铺满整个目标画布。
    nearest 全程保 {0,1} 不产生灰边，round 二值化。
    返回形状对齐 latent：4D→[1,1,lh,lw]，5D→[1,1,1,lh,lw]（核心 reshape_mask 约定）。
    """
    if scaled_width is None:
        scaled_width, scaled_height = canvas_width, canvas_height

    m = mask[0] if mask.dim() == 3 else mask  # 取主图遮罩 [H,W]
    m = torch.nn.functional.interpolate(
        m[None, None].float(), size=(scaled_height, scaled_width), mode="nearest"
    )[0, 0]

    canvas_m = torch.zeros((canvas_height, canvas_width), dtype=m.dtype, device=m.device)
    canvas_m[:scaled_height, :scaled_width] = m

    grid = torch.nn.functional.interpolate(
        canvas_m[None, None],
        size=(canvas_height // vae_unit, canvas_width // vae_unit),
        mode="nearest"
    )[0, 0].round()

    return grid[None, None, None] if latent_dim == 5 else grid[None, None]


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
            description="图像编码 / 文本编码一体化节点，无图像输入时可直接当文本编码器使用（VAE 也可不接）。\n\n五种模式：\n- flux2klein：Flux2模型模式（仅参考潜空间）\n- qwenedit：旧版Qwen图像编辑（视觉编码+参考潜空间）\n- boogu：Boogu图像编辑（视觉编码+参考潜空间写正负）\n- qwenimage21：QwenImage 2.1（视觉token双模式，参考图按槽位拼入序列）\n- mingimage：MingImage（视觉塔直接吃原图，3D VAE参考帧）\n\n等比缩放、兰佐斯插值、按VAE下采样倍率对齐；可选遮罩白色重绘黑色保留。\n\n输出：正面/负面提示词编码、latent、补边信息（供编辑对齐节点裁剪）。",
            inputs=[
                io.Clip.Input("clip", display_name="CLIP模型"),
                io.Vae.Input("vae", display_name="VAE模型", optional=True,
                             tooltip="可选。不接 VAE 时图像只经文本编码器视觉塔条件化，不产出参考 latent；纯文生图无需连接。"),
                io.Autogrow.Input("图片", template=image_template,
                                  tooltip="可选。不接图片时本节点作为纯文本编码器使用。"),
                io.Mask.Input("mask", display_name="遮罩", optional=True,
                              tooltip="可选。白色区域重绘、黑色区域保留原图；输出 latent 自带 noise_mask（需把 latent 接入采样器）。"),
                io.String.Input("正面提示词", multiline=True, dynamic_prompts=True),
                io.String.Input("负面提示词", multiline=True, dynamic_prompts=True),
                io.Combo.Input("对齐模式",
                               options=["flux2klein", "qwenedit", "boogu", "qwenimage21", "mingimage"],
                               default="flux2klein"),
                io.Int.Input("生成图像宽度", default=1024, min=16, max=4096, step=8),
                io.Int.Input("生成图像高度", default=1024, min=16, max=4096, step=8),
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

        if 对齐模式 == "flux2klein":
            result = cls._process_flux2klein(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )
        elif 对齐模式 == "qwenedit":
            result = cls._process_qwen(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )
        elif 对齐模式 == "boogu":
            result = cls._process_boogu(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )
        elif 对齐模式 == "qwenimage21":
            result = cls._process_qwen21(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )
        else:
            result = cls._process_ming(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度, mask
            )

        return io.NodeOutput(*result)

    @classmethod
    def _process_boogu(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """Boogu 模式：视觉编码 + reference_latents 同时写正负 → CFG 下抵消原图结构。"""
        channels, def_unit, latent_dim = _MODE_SPECS["boogu"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        pad_info = dict(_EMPTY_PAD_INFO)
        ref_latents = []
        images_vl = []
        noise_mask = None

        # 第一张非空图为主图（pad_info 取自主图的缩放/补边数据）
        main_image_index = -1
        for i, image in enumerate(input_images):
            if image is not None:
                main_image_index = i
                break

        for i, image in enumerate(input_images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            # 视觉编码路径：缩放到 384×384（面积不变）
            vl_target_size = 384
            total = int(vl_target_size * vl_target_size)
            scale_vl = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            vl_w = round(samples.shape[3] * scale_vl)
            vl_h = round(samples.shape[2] * scale_vl)
            s = comfy.utils.common_upscale(samples, vl_w, vl_h, "lanczos", "center")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])

            # 参考 latent 路径（VAE 未接时跳过）
            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == main_image_index:
                pad_info = pi
                if mask is not None:
                    noise_mask = _build_noise_mask(
                        mask, canvas_w, canvas_h, vae_unit, latent_dim, scaled_w, scaled_h
                    )

        # 正面：带视觉图像；负面：纯文本
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(positive_prompt, images=images_vl)
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt)
        )

        # Boogu 特有：reference_latents 同时写正负 → CFG 下相互抵消，保留原图结构
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )
            samples_out = ref_latents[0]
        else:
            canvas_w, canvas_h = _align_canvas(ref_width, ref_height, vae_unit)
            samples_out = _empty_latent(channels, canvas_w, canvas_h, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        return (positive, negative, latent_out, pad_info)

    @classmethod
    def _process_qwen(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """Qwen 模式：视觉编码（带 llama_template）+ reference_latents 仅写正面。"""
        channels, def_unit, latent_dim = _MODE_SPECS["qwenedit"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        pad_info = dict(_EMPTY_PAD_INFO)
        ref_latents = []
        vl_images = []
        noise_mask = None
        image_prompt = ""

        main_image_index = -1
        for i, image in enumerate(input_images):
            if image is not None:
                main_image_index = i
                break

        for i, image in enumerate(input_images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            # 视觉编码路径：缩放到 384×384 + 构建图像占位符 prompt
            vl_target_size = 384
            total = int(vl_target_size * vl_target_size)
            scale_vl = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            vl_w = round(samples.shape[3] * scale_vl)
            vl_h = round(samples.shape[2] * scale_vl)
            s = comfy.utils.common_upscale(samples, vl_w, vl_h, "lanczos", "center")
            vl_images.append(s.movedim(1, -1)[:, :, :, :3])
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

            # 参考 latent 路径（VAE 未接时跳过）
            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == main_image_index:
                pad_info = pi
                if mask is not None:
                    noise_mask = _build_noise_mask(
                        mask, canvas_w, canvas_h, vae_unit, latent_dim, scaled_w, scaled_h
                    )

        # 有图：image_prompt + positive_prompt + 编辑专用 llama_template；无图：纯文本编码
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
        return (positive_out, negative_out, latent_out, pad_info)

    @classmethod
    def _process_flux2klein(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """Flux2Klein 模式：无视觉编码，reference_latents 仅写正面。"""
        channels, def_unit, latent_dim = _MODE_SPECS["flux2klein"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        pad_info = dict(_EMPTY_PAD_INFO)
        ref_latents = []
        noise_mask = None

        main_image_index = -1
        for i, image in enumerate(input_images):
            if image is not None:
                main_image_index = i
                break

        for i, image in enumerate(input_images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            # 参考 latent 路径（VAE 未接时跳过）
            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by = result
            if encoded_latent is not None:
                ref_latents.append(encoded_latent)
            if i == main_image_index:
                pad_info = pi
                if mask is not None:
                    noise_mask = _build_noise_mask(
                        mask, canvas_w, canvas_h, vae_unit, latent_dim, scaled_w, scaled_h
                    )

        # 无视觉编码，纯文本 tokenize
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
        return (positive_out, negative_out, latent_out, pad_info)

    @classmethod
    def _process_qwen21(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """QwenImage 2.1 模式：复刻官方 TextEncodeQwenImage21 协议。"""
        channels, def_unit, latent_dim = _MODE_SPECS["qwenimage21"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        ref_latents = []
        images_vl = []
        # 参考图按生成画布面积等比缩放（官方数学：面积基准 + 32 倍数对齐，VL/VAE 共享同一 resize）
        target_area = ref_width * ref_height

        for image in input_images:
            samples = image[:1].movedim(-1, 1)
            ratio = samples.shape[3] / samples.shape[2]
            width = max(32, round(math.sqrt(target_area * ratio) / 32) * 32)
            height = max(32, round(math.sqrt(target_area / ratio) / 32) * 32)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                # 视觉塔看 alpha 合成白底，VAE 保留全部通道
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
            images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(s))

        # keep_vision：无 VAE latent 时视觉 token 留在文本序列；有 latent 时 TE 剥离并产出 image_slots 供 DiT 拼接
        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(positive_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
        )

        # 正负面都写 ref：CFG 两侧序列等长，差异只隔离文本方向
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )

        # 目标 latent 为零 latent（参考图走 context 槽位拼接，不作采样起点）
        canvas_w, canvas_h = _align_canvas(ref_width, ref_height, vae_unit)
        samples_out = _empty_latent(channels, canvas_w, canvas_h, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if mask is not None:
            latent_out["noise_mask"] = _build_noise_mask(
                mask, canvas_w, canvas_h, vae_unit, latent_dim
            )
        return (positive, negative, latent_out, dict(_EMPTY_PAD_INFO))

    @classmethod
    def _process_ming(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """MingImage 模式：复刻官方 TextEncodeMingImageEdit 协议。"""
        channels, def_unit, latent_dim = _MODE_SPECS["mingimage"]
        vae_unit = _extract_downscale_ratio(vae) if vae is not None else def_unit

        # 视觉塔直接吃原图：TE 内部固定 451584 像素智能切分，预缩只会被内部再放大、损失质量
        images_vl = [image[:, :, :, :3] for image in input_images]
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(positive_prompt, images=images_vl)
        )
        # 官方节点单 conditioning，负面给纯文本供可选 CFG；direct_context 由 TE 自动携带
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        # 首图画布尺寸（有图时无论 VAE 是否连接都要用）
        first_h = input_images[0].shape[1] if len(input_images) > 0 else None
        first_w = input_images[0].shape[2] if len(input_images) > 0 else None

        ref_latents = []
        if vae is not None and len(images_vl) > 0:
            # 后续图全部 bilinear 拉到第一张画布（vendor 设计），3D VAE encode 出 5D
            h, w = first_h, first_w
            for image in input_images:
                s = comfy.utils.common_upscale(
                    image.movedim(-1, 1), w, h, "bilinear", "disabled"
                ).movedim(1, -1)
                ref_latents.append(vae.encode(s))
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )

        # 目标画布：有图按首图原尺寸，无图按生成宽高
        if len(images_vl) > 0:
            canvas_w = math.ceil(first_w / vae_unit) * vae_unit
            canvas_h = math.ceil(first_h / vae_unit) * vae_unit
        else:
            canvas_w, canvas_h = _align_canvas(ref_width, ref_height, vae_unit)
        samples_out = _empty_latent(channels, canvas_w, canvas_h, vae_unit, latent_dim)

        latent_out = {"samples": samples_out}
        if mask is not None:
            latent_out["noise_mask"] = _build_noise_mask(
                mask, canvas_w, canvas_h, vae_unit, latent_dim
            )
        return (positive, negative, latent_out, dict(_EMPTY_PAD_INFO))


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
        # pad_info 全部描述右下补边（x=0, y=0, width=右补, height=下补）
        x = 补边信息.get("x", 0)
        y = 补边信息.get("y", 0)
        width_padding = 补边信息.get("width", 0)
        height_padding = 补边信息.get("height", 0)

        img = 生成图像.movedim(-1, 1)

        # 裁掉右下补边区域
        cropped_img = img[
            :, :,
            y:img.shape[2] - height_padding,
            x:img.shape[3] - width_padding
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
