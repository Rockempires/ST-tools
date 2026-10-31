"""
编辑图像节点 + 编辑对齐节点

ST_ImageEditor：
    图像编码 / 纯文本编码一体化节点。
    - 无图像输入时直接当作文本编码器使用（VAE 也可不接）
    - 五种模式：
        flux2klein  — 无视觉编码，reference_latents 仅写正面
        qwenedit    — 有视觉编码（带 llama_template），reference_latents 仅写正面
        boogu       — 有视觉编码，reference_latents 同时写正负 → CFG 下抵消原图结构
        qwenimage21— 直接调用官方 TextEncodeQwenImage21.execute(resolution=0)：条件/参考latent/零latent
                     与官方节点逐字节一致；每张参考图保持自身尺寸（仅 32 对齐），零 latent 画布=第一张
                     参考图原尺寸（//16，64ch），大图 1:1 编辑；外层仅追加可选遮罩与补边信息输出
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
from comfy_extras.nodes_qwen import TextEncodeQwenImage21


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
                io.Int.Input("生成图像宽度", default=1024, min=16, max=4096, step=8,
                             tooltip="qwenimage21 模式：画布严格保持第一张参考图原尺寸（仅 32 对齐），2560x1440 等大图 1:1 编辑不缩放不裁切；此处宽/高仅在未接图片作纯文本编码时用于构造零 latent。"),
                io.Int.Input("生成图像高度", default=1024, min=16, max=4096, step=8,
                             tooltip="qwenimage21 模式：画布严格保持第一张参考图原尺寸（仅 32 对齐），2560x1440 等大图 1:1 编辑不缩放不裁切；此处宽/高仅在未接图片作纯文本编码时用于构造零 latent。"),
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
            # 2.1 直接走官方节点实现：传原始图片 dict（保留编号槽位），而非拍平后的 list
            result = cls._process_qwen21(
                clip, vae, 图片, 正面提示词, 负面提示词,
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
    def _process_qwen21(cls, clip, vae, images_map, positive_prompt, negative_prompt, ref_width, ref_height, mask=None):
        """QwenImage 2.1 — 直接复刻官方 TextEncodeQwenImage21：

        核心约束（来自官方 nodes_qwen.py:153 注释）：
          "same resize for the text encoder and the VAE, so every vision slot covers 2x2 latents"
          视觉塔 ViT merger (32px/token) 和 VAE latent (16px/token) 之间有固定的
          1 merger = 2×2 latent 覆盖关系。三者（视觉塔、ref_latents、零 latent）
          必须用同一张缩后图！之前拆成不同尺寸导致 image_slots 和 ref_latents 对不齐。

        硬约束：qwen3vl.py:20 num_position_embeddings=2304（merger token 上限）。
        resolution=0 时 2560×1440 原图产出 3600 merger tokens 超限 → 官方节点自己也崩。
        所以 resolution 用安全值 vis_res = min(短边, 1536)（merger ≤ 2304）。"""
        import os as _os, time as _time
        _LOG = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "qwen21_debug.log")
        def _stat(t):
            try:
                return "shape=%s dtype=%s min=%.4f max=%.4f mean=%.5f" % (
                    tuple(t.shape), t.dtype, t.min().item(), t.max().item(), t.float().mean().item())
            except Exception as e:
                return "<stat fail %s>" % e
        dbg = ["=" * 70, "[%s] ST_ImageEditor qwenimage21 RUN (official-clone)" % _time.strftime("%Y-%m-%d %H:%M:%S")]

        images_map = images_map or {}
        ordered_keys = sorted(
            images_map.keys(),
            key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0),
        )
        has_image = any(images_map.get(k) is not None for k in ordered_keys)

        # 映射到官方 image_1..image_N 格式
        official_images = {}
        for idx, key in enumerate(ordered_keys):
            official_images[f"image_{idx + 1}"] = images_map.get(key)

        if not has_image:
            # 无图：调官方 resolution=0 拿 conditioning（纯文本），
            # 但零 latent 用 ST 宽高输入框自己构造（官方固定 1024）
            out = TextEncodeQwenImage21.execute(
                clip, positive_prompt, negative_prompt,
                vae=vae, resolution=0, images=official_images,
            )
            positive, negative, _ = out[0], out[1], out[2]
            channels, vae_unit, _ = _MODE_SPECS["qwenimage21"]
            cw = max(32, round(ref_width / 32) * 32)
            ch = max(32, round(ref_height / 32) * 32)
            zero = torch.zeros(
                [1, channels, ch // vae_unit, cw // vae_unit],
                device=comfy.model_management.intermediate_device(),
            )
            latent_out = {"samples": zero}
            dbg.append(f"NO_IMAGE: canvas={cw}x{ch} (from ref_width={ref_width} ref_height={ref_height})")
            if mask is not None:
                gh, gw = zero.shape[-2], zero.shape[-1]
                latent_out["noise_mask"] = _build_noise_mask(mask, gw * 16, gh * 16, 16, 4)
            with open(_LOG, "a", encoding="utf-8") as _f:
                _f.write("\n".join(dbg) + "\n")
            return (positive, negative, latent_out, dict(_EMPTY_PAD_INFO))

        # 有图：ref_latents + 零 latent 同尺寸（官方缩后）→ 编辑正确
        # pad_info 带原图尺寸 → size aligner 放大回原图尺寸输出
        # 为什么？官方 ref_latents 和零 latent 始终同尺寸，不同尺寸会崩（DiT build_sequence 位置编码依赖同尺寸奇偶对齐）
        channels, vae_unit, _ = _MODE_SPECS["qwenimage21"]
        first_img = next(images_map[k] for k in ordered_keys if images_map.get(k) is not None)
        orig_h, orig_w = int(first_img.shape[1]), int(first_img.shape[2])
        # vis_res 上限放大到 2560（用户实测 resolution=2560 正确，fast_pos_embed_interpolate 插值适配任意网格）
        vis_res = max(1024, min(orig_w, orig_h, 2560))
        dbg.append(f"REF_IMG={orig_w}x{orig_h} vis_res={vis_res}")

        # Step 1: 调官方节点 → ref_latents 和零 latent 同尺寸（缩后）
        out = TextEncodeQwenImage21.execute(
            clip, positive_prompt, negative_prompt,
            vae=vae, resolution=vis_res, images=official_images,
        )
        positive, negative, latent_out = out[0], out[1], out[2]
        zero = latent_out["samples"]
        dbg.append(f"Official zero latent: {_stat(zero)}")

        # 缩后像素尺寸
        scaled_h = zero.shape[-2] * vae_unit
        scaled_w = zero.shape[-1] * vae_unit
        dbg.append(f"Scaled pixel size: {scaled_w}x{scaled_h} → need upscale to {orig_w}x{orig_h}")

        # Step 2: pad_info 带原图尺寸 → size aligner 放大回原图
        pad_info = {
            "x": 0, "y": 0, "width": 0, "height": 0,
            "scale_by": orig_w / scaled_w if scaled_w > 0 else 1.0,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }

        # mask
        if mask is not None:
            gh, gw = zero.shape[-2], zero.shape[-1]
            latent_out["noise_mask"] = _build_noise_mask(mask, gw * 16, gh * 16, 16, 4)

        dbg.append(f"FINAL: zero={zero.shape} scaled={scaled_w}x{scaled_h} orig={orig_w}x{orig_h}")
        with open(_LOG, "a", encoding="utf-8") as _f:
            _f.write("\n".join(dbg) + "\n")

        return (positive, negative, latent_out, pad_info)

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
        x = 补边信息.get("x", 0)
        y = 补边信息.get("y", 0)
        width_padding = 补边信息.get("width", 0)
        height_padding = 补边信息.get("height", 0)
        orig_w = 补边信息.get("orig_w")
        orig_h = 补边信息.get("orig_h")

        img = 生成图像.movedim(-1, 1)

        # 1. 裁掉补边区（旧三模式用，qwenimage21 模式补边为 0）
        if width_padding or height_padding:
            img = img[:, :, y:img.shape[2] - height_padding, x:img.shape[3] - width_padding]

        # 2. 如果指定了 orig_w/orig_h，缩放回原图尺寸
        # qwenimage21 模式：官方 resolution=vis_res 输出缩后（或放大）尺寸，
        #    这里放回到输入图原尺寸
        cur_h, cur_w = img.shape[2], img.shape[3]
        if orig_w and orig_h and (cur_w != orig_w or cur_h != orig_h):
            img = comfy.utils.common_upscale(img, orig_w, orig_h, "lanczos", "disabled")

        return (img.movedim(1, -1),)


NODE_CLASS_MAPPINGS = {
    "ST_ImageEditor": ST_ImageEditor,
    "ST_ImageSizeAligner": ST_ImageSizeAligner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_ImageEditor": "编辑图像",
    "ST_ImageSizeAligner": "编辑对齐",
}
