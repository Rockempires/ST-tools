"""
编辑图像节点 + 编辑对齐节点

ST_ImageEditor：
    接收 1-10 张图片，输出 CLIP conditioning + reference_latents + pad_info。
    三种模式差异在 reference_latents 写入位置：
        flux2klein  — 无视觉编码，reference_latents 仅写正面
        qwenedit    — 有视觉编码（带 llama_template），reference_latents 仅写正面
        boogu       — 有视觉编码，reference_latents 同时写正负 → CFG 下抵消原图结构
    
    参考图像处理流程（三种模式共享 _prepare_reference_latent）：
        fit-within 等比缩放 → 黑色画布左上放置 → VAE encode
        latent shape = ceil(ref/vae_unit) * vae_unit → 与采样器目标尺寸精确匹配
    
    补边策略：
        画布（采样器目标）按 ref 外框对齐 vae_unit；
        图像 fit-within 保持宽高比不变形，右下留白 → 黑色补边；
        pad_info 记录右下补边量，供 ST_ImageSizeAligner 裁剪还原。

ST_ImageSizeAligner：
    接收 pad_info + KSampler 输出图像，裁剪掉右下补边区域，还原 fit-within 尺寸。
    注意：裁剪到 fit-within 尺寸而非原图尺寸——因为等比缩放后本身就会变小，
    原始尺寸信息在这个节点链路中不可用。
"""

import torch
import numpy as np
import math
import node_helpers
import comfy.utils
from comfy_api.latest import io


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


def _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae):
    """
    统一的参考图像 → latent 处理流程（三种模式共享）
    
    流程：
        1. fit-within 等比缩放，保持宽高比（scaled 不对齐 vae_unit，不变形）
        2. canvas（采样器目标）按 ref 外框对齐 vae_unit
        3. 黑色画布左上放置 → 右下补边区为纯黑
        4. VAE encode → latent shape = canvas/vae_unit = 采样器目标 ✅
    
    参数：
        samples: [B, C, H, W] 原图（已 movedim 到 CHW）
        ref_width, ref_height: 生成目标宽高（像素）
        vae_unit: VAE 空间压缩比
        vae: VAE 模型
    
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
    
    # VAE encode（canvas 尺寸 → latent shape = canvas / vae_unit）
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


class ST_ImageEditor(io.ComfyNode):
    """图像编辑节点：把输入图像编码为 CLIP conditioning + reference_latents + pad_info。"""
    
    @classmethod
    def define_schema(cls):
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("图片"),
            names=[f"图片{i}" for i in range(1, 11)],
        )
        
        return io.Schema(
            node_id="ST_ImageEditor",
            display_name="编辑图像",
            category="🎯 石头工具/图像编辑",
            description="图像编辑节点，用于将输入图像编码为条件向量和潜空间表示，支持多种对齐模式。\n\n功能特性：\n- 动态输入：支持1-10张图片，接入后自动显示新输入点\n- 三种对齐模式：\n  - flux2klein：Flux2模型标准模式（仅参考潜空间）\n  - qwenedit：Qwen模型编辑模式（视觉编码+参考潜空间）\n  - boogu：Boogu图像编辑模式（视觉编码+参考潜空间）\n- 等比缩放：等比适配目标宽高，兰佐斯插值\n- 自动补边：填充至VAE对齐尺寸（按VAE下采样倍率对齐）\n\n输出：\n- 正面输出：正面提示词编码\n- 负面输出：负面提示词编码\n- latent：参考图像的潜空间表示（尺寸即采样器目标尺寸）\n- 补边信息：记录补边区域，供编辑对齐节点裁剪",
            inputs=[
                io.Clip.Input("clip", display_name="CLIP模型"),
                io.Vae.Input("vae", display_name="VAE模型"),
                io.Autogrow.Input("图片", template=image_template),
                io.String.Input("正面提示词", multiline=True, dynamic_prompts=True),
                io.String.Input("负面提示词", multiline=True, dynamic_prompts=True),
                io.Combo.Input("对齐模式", options=["flux2klein", "qwenedit", "boogu"], default="flux2klein"),
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
    def execute(cls, clip, vae, 图片, 正面提示词, 负面提示词, 对齐模式, 生成图像宽度, 生成图像高度) -> io.NodeOutput:
        input_images = list(图片.values())
        
        if 对齐模式 == "flux2klein":
            result = cls._process_flux2klein(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度
            )
        elif 对齐模式 == "qwenedit":
            result = cls._process_qwen(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度
            )
        else:
            result = cls._process_boogu(
                clip, vae, input_images, 正面提示词, 负面提示词,
                生成图像宽度, 生成图像高度
            )
        
        return io.NodeOutput(*result)

    @classmethod
    def _process_boogu(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height):
        """Boogu 模式：视觉编码 + reference_latents 同时写正负 → CFG 下抵消原图结构。"""
        vae_unit = _extract_downscale_ratio(vae)
        
        pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0}
        ref_latents = []
        images_vl = []
        
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
            
            # 参考 latent 路径
            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by = result
            ref_latents.append(encoded_latent)
            if i == main_image_index:
                pad_info = pi
        
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
            samples_out = torch.zeros(1, 4, 128, 128)
        
        return (positive, negative, {"samples": samples_out}, pad_info)

    @classmethod
    def _process_qwen(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height):
        """Qwen 模式：视觉编码（带 llama_template）+ reference_latents 仅写正面。"""
        vae_unit = _extract_downscale_ratio(vae)
        
        pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0}
        ref_latents = []
        vl_images = []
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
            
            # 参考 latent 路径
            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by = result
            ref_latents.append(encoded_latent)
            if i == main_image_index:
                pad_info = pi
        
        # Qwen 特有：image_prompt + positive_prompt + llama_template
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
        negative_out = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt)
        )
        
        # Qwen 特有：reference_latents 仅写正面
        if len(ref_latents) > 0:
            positive_out = node_helpers.conditioning_set_values(
                positive_out, {"reference_latents": ref_latents}, append=True
            )
            samples_out = ref_latents[0]
        else:
            samples_out = torch.zeros(1, 4, 128, 128)
        
        return (positive_out, negative_out, {"samples": samples_out}, pad_info)

    @classmethod
    def _process_flux2klein(cls, clip, vae, input_images, positive_prompt, negative_prompt, ref_width, ref_height):
        """Flux2Klein 模式：无视觉编码，reference_latents 仅写正面。"""
        vae_unit = _extract_downscale_ratio(vae)
        
        pad_info = {"x": 0, "y": 0, "width": 0, "height": 0, "scale_by": 1.0}
        ref_latents = []
        
        main_image_index = -1
        for i, image in enumerate(input_images):
            if image is not None:
                main_image_index = i
                break
        
        for i, image in enumerate(input_images):
            if image is None:
                continue
            
            samples = image.movedim(-1, 1)
            
            # 参考 latent 路径
            result = _prepare_reference_latent(samples, ref_width, ref_height, vae_unit, vae)
            encoded_latent, pi, scaled_w, scaled_h, canvas_w, canvas_h, scale_by = result
            ref_latents.append(encoded_latent)
            if i == main_image_index:
                pad_info = pi
        
        # 无视觉编码，纯文本 tokenize
        positive_out = clip.encode_from_tokens_scheduled(clip.tokenize(positive_prompt))
        negative_out = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        
        if len(ref_latents) > 0:
            positive_out = node_helpers.conditioning_set_values(
                positive_out, {"reference_latents": ref_latents}, append=True
            )
            samples_out = ref_latents[0]
        else:
            samples_out = torch.zeros(1, 16, 128, 128)
        
        return (positive_out, negative_out, {"samples": samples_out}, pad_info)


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
