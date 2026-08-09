import torch
import numpy as np
import math
import node_helpers
import comfy.utils
from comfy_api.latest import io


class ST_ImageEditor(io.ComfyNode):
    """
    图像编辑节点
    将输入图像编码为条件向量和潜空间表示，支持三种对齐模式（flux2klein、qwenedit、boogu）
    """
    
    @classmethod
    def define_schema(cls):
        """
        定义节点架构
        包括输入参数、输出参数、显示名称、分类等
        """
        
        # 定义动态图像输入模板，支持图片1到图片10
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("图片"),
            names=[f"图片{i}" for i in range(1, 11)],
        )
        
        return io.Schema(
            node_id="ST_ImageEditor",
            display_name="编辑图像",
            category="🎯 石头工具",
            description="图像编辑节点，用于将输入图像编码为条件向量和潜空间表示，支持多种对齐模式。\n\n功能特性：\n- 动态输入：支持1-10张图片，接入后自动显示新输入点\n- 三种对齐模式：\n  - flux2klein: Flux2模型标准模式（仅参考潜空间）\n  - qwenedit: Qwen模型编辑模式（视觉编码+参考潜空间）\n  - boogu: Boogu图像编辑模式（视觉编码+参考潜空间）\n- 自动缩放：按最长边比例缩放，兰佐斯插值\n- 自动补边：填充至VAE对齐尺寸（8/16的倍数）\n- 遮罩支持：可选遮罩输入，生成噪声掩码\n\n输出：\n- 正面输出：正面提示词编码\n- 负面输出：负面提示词编码\n- latent：参考图像的潜空间表示\n- 补边信息：记录填充区域尺寸\n- 主图像：处理后的主图像（经过缩放和补边）\n- 遮罩：噪声掩码（如有）",
            inputs=[
                io.Clip.Input("clip", display_name="CLIP模型"),
                io.Vae.Input("vae", display_name="VAE模型"),
                io.Autogrow.Input("图片", template=image_template),
                io.String.Input("正面提示词", multiline=True, dynamic_prompts=True),
                io.String.Input("负面提示词", multiline=True, dynamic_prompts=True),
                io.Combo.Input("对齐模式", options=["flux2klein", "qwenedit", "boogu"], default="flux2klein"),
                io.Int.Input("生成图像最长边", default=1024, min=16, max=4096, step=1),
                io.Mask.Input("遮罩", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("正面输出"),
                io.Conditioning.Output("负面输出"),
                io.Latent.Output("latent"),
                io.AnyType.Output("补边信息"),
                io.Image.Output("主图像"),
                io.Mask.Output("遮罩"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, 图片, 正面提示词, 负面提示词, 对齐模式, 生成图像最长边, 遮罩=None) -> io.NodeOutput:
        """
        节点执行方法
        根据选择的对齐模式，调用对应的处理方法
        
        参数：
            clip: CLIP模型，用于文本和图像编码
            vae: VAE模型，用于图像转潜空间
            图片: 动态输入的图片字典 {图片id: 图像张量}
            正面提示词: 用户输入的正面提示词
            负面提示词: 用户输入的负面提示词
            对齐模式: 选择的处理模式（flux2klein/qwenedit/boogu）
            生成图像最长边: 目标图像的最长边尺寸
            遮罩: 可选的遮罩输入
        
        返回：
            NodeOutput: 包含正面输出、负面输出、latent、补边信息、主图像、遮罩
        """
        
        # 将字典转换为列表，按顺序获取所有输入图片
        input_images = list(图片.values())
        
        # 固定处理参数
        resize_mode = "longest_edge"  # 按最长边缩放
        crop_method = "pad"           # 补边模式
        upscale_method = "lanczos"    # 兰佐斯插值
        
        # 根据对齐模式调用对应的处理方法
        if 对齐模式 == "flux2klein":
            result = cls._process_flux2klein(clip, vae, input_images, 遮罩, 正面提示词, 负面提示词, 生成图像最长边, resize_mode, crop_method, upscale_method)
        elif 对齐模式 == "qwenedit":
            result = cls._process_qwen(clip, vae, input_images, 遮罩, 正面提示词, 负面提示词, 生成图像最长边, resize_mode, crop_method, upscale_method)
        else:
            result = cls._process_boogu(clip, vae, input_images, 遮罩, 正面提示词, 负面提示词, 生成图像最长边, resize_mode, crop_method, upscale_method)
        
        return io.NodeOutput(*result)

    @classmethod
    def _process_boogu(cls, clip, vae, input_images, mask, positive_prompt, negative_prompt, ref_longest_edge, resize_mode, crop_method, upscale_method):
        """
        Boogu图像编辑模式处理方法
        特点：所有图片同时通过视觉编码器和VAE编码，正面和负面条件都设置参考潜空间
        
        参数：
            clip: CLIP模型
            vae: VAE模型
            input_images: 输入图片列表
            mask: 遮罩（此模式不使用）
            positive_prompt: 正面提示词
            negative_prompt: 负面提示词
            ref_longest_edge: 目标最长边尺寸
            resize_mode: 缩放模式
            crop_method: 裁剪方式
            upscale_method: 缩放方法
        
        返回：
            (positive, negative, latent_out, pad_info, main_image, None)
        """
        
        # VAE对齐单位为8（Boogu模型要求）
        vae_unit = 8
        
        # 初始化补边信息，记录填充区域尺寸和缩放比例
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0,
        }
        
        # 存储处理后的参考潜空间、视觉编码器图像、VAE图像
        ref_latents = []
        images_vl = []
        vae_images = []
        
        # 遍历所有输入图片
        for i, image in enumerate(input_images):
            # 跳过空图片
            if image is None:
                continue
            
            # 配置参数：同时用于参考和视觉编码，第一张图作为主图
            to_ref = True
            to_vl = True
            ref_main_image = (i == 0)
            vl_target_size = 384      # 视觉编码器目标尺寸
            vl_crop = "center"        # 居中裁剪
            vl_upscale = "lanczos"    # 兰佐斯插值
            
            # 如果既不用于参考也不用于视觉编码，则跳过
            if not to_ref and not to_vl:
                continue
            
            # 将图像维度从 [B, H, W, C] 转换为 [B, C, H, W]
            samples = image.movedim(-1, 1)
            
            # 处理视觉编码器输入：缩放到384x384
            if to_vl:
                # 计算缩放比例，保持面积不变
                total = int(vl_target_size * vl_target_size)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                
                # 执行缩放并转换回 [B, H, W, C] 格式，取前3通道（去除Alpha）
                s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                images_vl.append(s.movedim(1, -1)[:, :, :, :3])
            
            # 处理参考图像：按最长边缩放并补边
            if to_ref:
                # 根据缩放模式计算缩放比例
                if resize_mode == "area":
                    # 面积模式：保持总面积不变
                    total = int(ref_longest_edge * ref_longest_edge)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    scaled_width = int(round(samples.shape[3] * scale_by))
                    scaled_height = int(round(samples.shape[2] * scale_by))
                else:
                    # 最长边模式：按最长边比例缩放
                    ori_longest_edge = max(samples.shape[2], samples.shape[3])
                    scale_by = ori_longest_edge / ref_longest_edge
                    scaled_height = int(round(samples.shape[2] / scale_by))
                    scaled_width = int(round(samples.shape[3] / scale_by))
                
                # 补边模式：创建画布并居中放置图像
                if crop_method == "pad":
                    crop = "center"
                    # 计算画布尺寸，向上取整到VAE单位的倍数
                    canvas_width = math.ceil(scaled_width / vae_unit) * vae_unit
                    canvas_height = math.ceil(scaled_height / vae_unit) * vae_unit
                    
                    # 创建黑色画布
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    # 缩放图像到目标尺寸
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    
                    # 将缩放后的图像放置在画布左上角
                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    
                    # 如果是主图，记录补边信息
                    if ref_main_image:
                        current_total = (samples.shape[3] * samples.shape[2])
                        total_px = int(resized_width * resized_height)
                        scale_by_val = math.sqrt(total_px / current_total)
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": round(1 / scale_by_val, 3)
                        }
                    
                    s = canvas
                else:
                    # 非补边模式：直接裁剪到目标尺寸
                    crop = crop_method
                    width = round(scaled_width / vae_unit) * vae_unit
                    height = round(scaled_height / vae_unit) * vae_unit
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                
                # 将图像转换回 [B, H, W, C] 格式，取前3通道
                vae_img = s.movedim(1, -1)[:, :, :, :3]
                vae_images.append(vae_img)
                
                # 使用VAE编码为潜空间表示
                ref_latents.append(vae.encode(vae_img))
        
        # 使用CLIP编码正面提示词（包含视觉编码器图像）
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(positive_prompt, images=images_vl))
        
        # 使用CLIP编码负面提示词（不包含图像）
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        
        # 如果有参考潜空间，同时设置到正面和负面条件中（Boogu模式特点）
        # 这样在CFG下会相互抵消，保持原图结构
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        
        # 输出第一张图的潜空间作为latent
        samples_out = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples_out}
        
        # 获取主图像（第一张经过处理的图像）
        main_image = vae_images[0] if len(vae_images) > 0 else None
        
        # Boogu模式不支持遮罩输出
        return (positive, negative, latent_out, pad_info, main_image, None)

    @classmethod
    def _process_qwen(cls, clip, vae, input_images, mask, positive_prompt, negative_prompt, ref_longest_edge, resize_mode, crop_method, upscale_method):
        """
        Qwen模型编辑模式处理方法
        特点：所有图片通过视觉编码器和VAE编码，参考潜空间仅设置在正面条件中
        
        参数：
            clip: CLIP模型
            vae: VAE模型
            input_images: 输入图片列表
            mask: 遮罩
            positive_prompt: 正面提示词
            negative_prompt: 负面提示词
            ref_longest_edge: 目标最长边尺寸
            resize_mode: 缩放模式
            crop_method: 裁剪方式
            upscale_method: 缩放方法
        
        返回：
            (positive_out, negative_out, latent_out, pad_info, main_image, noise_mask)
        """
        
        # VAE对齐单位为8（Qwen模型要求）
        vae_unit = 8
        
        # 初始化补边信息
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0,
        }
        
        # 存储处理后的参考潜空间、VAE图像、视觉编码器图像
        ref_latents = []
        vae_images = []
        vl_images = []
        noise_mask = None
        image_prompt = ""
        
        # 找到第一张非空图片作为主图
        main_image_index = -1
        for i, image in enumerate(input_images):
            if image is None:
                continue
            if main_image_index == -1:
                main_image_index = i
        
        # 遍历所有输入图片
        for i, image in enumerate(input_images):
            # 跳过空图片
            if image is None:
                continue
            
            # 配置参数：同时用于参考和视觉编码
            to_ref = True
            ref_main_image = (i == main_image_index)
            to_vl = True
            vl_resize = True
            vl_target_size = 384      # 视觉编码器目标尺寸
            vl_crop = "center"        # 居中裁剪
            vl_upscale = "bicubic"    # 双三次插值
            
            # 如果既不用于参考也不用于视觉编码，则跳过
            if not to_ref and not to_vl:
                continue
            
            # 将图像维度从 [B, H, W, C] 转换为 [B, C, H, W]
            samples = image.movedim(-1, 1)
            
            # 如果有遮罩，扩展遮罩维度以匹配图像通道数
            if mask is not None:
                _, c, _, _ = samples.shape
                sample_masks = mask.unsqueeze(1).repeat(1, c, 1, 1)
            
            # 处理参考图像：按最长边缩放并补边
            if to_ref:
                ori_longest_edge = max(samples.shape[2], samples.shape[3])
                
                # 根据缩放模式计算缩放比例
                if resize_mode == "area":
                    total = int(ref_longest_edge * ref_longest_edge)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                else:
                    scale_by = ori_longest_edge / ref_longest_edge
                
                scaled_height = int(round(samples.shape[2] / scale_by))
                scaled_width = int(round(samples.shape[3] / scale_by))
                
                # 补边模式：创建画布并居中放置图像
                if crop_method == "pad":
                    crop = "center"
                    canvas_width = math.ceil(scaled_width / vae_unit) * vae_unit
                    canvas_height = math.ceil(scaled_height / vae_unit) * vae_unit
                    
                    # 创建黑色画布
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    # 缩放图像到目标尺寸
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    
                    # 将缩放后的图像放置在画布左上角
                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    
                    # 如果是主图，记录补边信息
                    if ref_main_image:
                        current_total = (samples.shape[3] * samples.shape[2])
                        total = int(resized_width * resized_height)
                        scale_by_val = math.sqrt(total / current_total)
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": round(1 / scale_by_val, 3)
                        }
                    
                    s = canvas
                    
                    # 如果有遮罩且是主图，处理遮罩并生成噪声掩码
                    if mask is not None and ref_main_image:
                        mask_canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_sample_masks = comfy.utils.common_upscale(sample_masks, scaled_width, scaled_height, upscale_method, crop)
                        mask_canvas[:, :, :resized_height, :resized_width] = resized_sample_masks
                        noise_mask = mask_canvas[:, :1, :, :].squeeze(1)
                else:
                    # 非补边模式：直接裁剪到目标尺寸
                    crop = crop_method
                    width = round(scaled_width / vae_unit) * vae_unit
                    height = round(scaled_height / vae_unit) * vae_unit
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    
                    # 如果有遮罩且是主图，缩放遮罩
                    if mask is not None and ref_main_image:
                        m = comfy.utils.common_upscale(sample_masks, width, height, upscale_method, crop)
                        noise_mask = m[:, :1, :, :].squeeze(1)
                
                # 将图像转换回 [B, H, W, C] 格式
                image = s.movedim(1, -1)
                
                # 使用VAE编码为潜空间表示
                ref_latents.append(vae.encode(image[:, :, :, :3]))
                vae_images.append(image)
            
            # 处理视觉编码器输入：缩放到目标尺寸
            if to_vl:
                # 根据是否调整尺寸计算目标面积
                if vl_resize:
                    total = int(vl_target_size * vl_target_size)
                else:
                    total = int(samples.shape[3] * samples.shape[2])
                    # 限制最大面积为2048x2048
                    if total > 2048 * 2048:
                        total = 2048 * 2048
                
                # 计算缩放比例
                current_total = (samples.shape[3] * samples.shape[2])
                scale_by = math.sqrt(total / current_total)
            
                # 计算缩放后的尺寸
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                
                # 执行缩放
                s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                
                # 转换回 [B, H, W, C] 格式
                image = s.movedim(1, -1)
                
                # 添加图片占位符到提示词中
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                vl_images.append(image)
        
        # 组合图片占位符和正面提示词
        full_prompt = image_prompt + positive_prompt
        
        # Llama模板：指导模型理解图像并执行编辑
        llama_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # 使用CLIP编码正面提示词（包含视觉编码器图像和Llama模板）
        positive_tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        positive_out = clip.encode_from_tokens_scheduled(positive_tokens)
        
        # 使用CLIP编码负面提示词
        negative_tokens = clip.tokenize(negative_prompt)
        negative_out = clip.encode_from_tokens_scheduled(negative_tokens)
        
        # 如果有参考潜空间，仅设置到正面条件中（Qwen模式特点）
        if len(ref_latents) > 0:
            positive_out = node_helpers.conditioning_set_values(positive_out, {"reference_latents": ref_latents}, append=True)
            samples = ref_latents[main_image_index]
        else:
            samples = torch.zeros(1, 4, 128, 128)
        
        # 构造latent输出
        latent_out = {"samples": samples}
        
        # 如果有噪声掩码，添加到latent中
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        
        # 获取主图像（主图对应的处理后图像）
        main_image = vae_images[main_image_index] if len(vae_images) > 0 else None
        
        return (positive_out, negative_out, latent_out, pad_info, main_image, noise_mask)

    @classmethod
    def _process_flux2klein(cls, clip, vae, input_images, mask, positive_prompt, negative_prompt, ref_longest_edge, resize_mode, crop_method, upscale_method):
        """
        Flux2Klein模式处理方法
        特点：图片仅通过VAE编码为参考潜空间，不使用视觉编码器
        
        参数：
            clip: CLIP模型
            vae: VAE模型
            input_images: 输入图片列表
            mask: 遮罩
            positive_prompt: 正面提示词
            negative_prompt: 负面提示词
            ref_longest_edge: 目标最长边尺寸
            resize_mode: 缩放模式
            crop_method: 裁剪方式
            upscale_method: 缩放方法
        
        返回：
            (positive_out, negative_out, latent_out, pad_info, main_image, noise_mask)
        """
        
        # VAE对齐单位为16（Flux2模型要求）
        vae_unit = 16
        
        # 初始化补边信息
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0,
        }
        
        # 存储处理后的参考潜空间和VAE图像
        ref_latents = []
        vae_images = []
        noise_mask = None
        
        # 找到第一张非空图片作为主图
        main_image_index = -1
        for i, image in enumerate(input_images):
            if image is None:
                continue
            if main_image_index == -1:
                main_image_index = i
        
        # 遍历所有输入图片
        for i, image in enumerate(input_images):
            # 跳过空图片
            if image is None:
                continue
            
            # 配置参数：仅用于参考（不使用视觉编码器）
            to_ref = True
            ref_main_image = (i == main_image_index)
            
            # 如果不用于参考，则跳过
            if not to_ref:
                continue
            
            # 将图像维度从 [B, H, W, C] 转换为 [B, C, H, W]
            samples = image.movedim(-1, 1)
            
            # 如果有遮罩，扩展遮罩维度以匹配图像通道数
            if mask is not None:
                _, c, _, _ = samples.shape
                sample_masks = mask.unsqueeze(1).repeat(1, c, 1, 1)
            
            # 计算原图最长边
            ori_longest_edge = max(samples.shape[2], samples.shape[3])
            
            # 根据缩放模式计算缩放比例
            if resize_mode == "area":
                total = int(ref_longest_edge * ref_longest_edge)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            else:
                scale_by = ori_longest_edge / ref_longest_edge
            
            scaled_height = int(round(samples.shape[2] / scale_by))
            scaled_width = int(round(samples.shape[3] / scale_by))
            
            # 补边模式：创建画布并居中放置图像
            if crop_method == "pad":
                crop = "center"
                canvas_width = math.ceil(scaled_width / vae_unit) * vae_unit
                canvas_height = math.ceil(scaled_height / vae_unit) * vae_unit
                
                # 创建黑色画布
                canvas = torch.zeros(
                    (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                    dtype=samples.dtype,
                    device=samples.device
                )
                
                # 缩放图像到目标尺寸
                resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                resized_width = resized_samples.shape[3]
                resized_height = resized_samples.shape[2]
                
                # 将缩放后的图像放置在画布左上角
                canvas[:, :, :resized_height, :resized_width] = resized_samples
                
                # 如果是主图，记录补边信息
                if ref_main_image:
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(resized_width * resized_height)
                    scale_by_val = math.sqrt(total / current_total)
                    pad_info = {
                        "x": 0,
                        "y": 0,
                        "width": canvas_width - resized_width,
                        "height": canvas_height - resized_height,
                        "scale_by": round(1 / scale_by_val, 3)
                    }
                
                s = canvas
                
                # 如果有遮罩且是主图，处理遮罩并生成噪声掩码
                if mask is not None and ref_main_image:
                    mask_canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    resized_sample_masks = comfy.utils.common_upscale(sample_masks, scaled_width, scaled_height, upscale_method, crop)
                    mask_canvas[:, :, :resized_height, :resized_width] = resized_sample_masks
                    noise_mask = mask_canvas[:, :1, :, :].squeeze(1)
            else:
                # 非补边模式：直接裁剪到目标尺寸
                crop = crop_method
                width = round(scaled_width / vae_unit) * vae_unit
                height = round(scaled_height / vae_unit) * vae_unit
                s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                
                # 如果有遮罩且是主图，缩放遮罩
                if mask is not None and ref_main_image:
                    m = comfy.utils.common_upscale(sample_masks, width, height, upscale_method, crop)
                    noise_mask = m[:, :1, :, :].squeeze(1)
            
            # 将图像转换回 [B, H, W, C] 格式
            image = s.movedim(1, -1)
            
            # 使用VAE编码为潜空间表示
            ref_latents.append(vae.encode(image[:, :, :, :3]))
            vae_images.append(image)
        
        # 使用CLIP编码正面提示词（Flux2模式不使用视觉编码器）
        positive_tokens = clip.tokenize(positive_prompt)
        positive_out = clip.encode_from_tokens_scheduled(positive_tokens)
        
        # 使用CLIP编码负面提示词
        negative_tokens = clip.tokenize(negative_prompt)
        negative_out = clip.encode_from_tokens_scheduled(negative_tokens)
        
        # 如果有参考潜空间，仅设置到正面条件中（Flux2模式特点）
        if len(ref_latents) > 0:
            positive_out = node_helpers.conditioning_set_values(positive_out, {"reference_latents": ref_latents}, append=True)
            samples = ref_latents[main_image_index]
        else:
            samples = torch.zeros(1, 4, 128, 128)
        
        # 构造latent输出
        latent_out = {"samples": samples}
        
        # 如果有噪声掩码，添加到latent中
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        
        # 获取主图像（主图对应的处理后图像）
        main_image = vae_images[main_image_index] if len(vae_images) > 0 else None
        
        return (positive_out, negative_out, latent_out, pad_info, main_image, noise_mask)


class ST_ImageSizeAligner:
    """
    编辑对齐节点
    接收编辑图像节点的补边信息，对K采样器生成的图像进行反向裁剪/对齐，
    去除补边区域，让生成图像与原图尺寸完全一致
    """
    
    DISPLAY_NAME = "编辑对齐"
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数类型
        """
        return {
            "required": {
                "补边信息": ("ANY", ),
                "生成图像": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("对齐后图像",)
    FUNCTION = "align"
    CATEGORY = "🎯 石头工具"
    DESCRIPTION = "说明：编辑对齐节点\n\n接收编辑图像节点的补边信息，对K采样器生成的图像进行反向裁剪/对齐，让生成图像与原图尺寸完全一致。"

    def align(self, 补边信息, 生成图像):
        """
        对齐方法
        根据补边信息，反向裁剪生成图像，去除补边区域
        
        参数：
            补边信息: 编辑图像节点输出的补边信息字典
            生成图像: K采样器生成的图像张量
        
        返回：
            对齐后图像: 去除补边区域后的图像
        """
        
        # 从补边信息中提取填充参数
        x = 补边信息.get("x", 0)
        y = 补边信息.get("y", 0)
        width_padding = 补边信息.get("width", 0)
        height_padding = 补边信息.get("height", 0)
        
        # 将图像维度从 [B, H, W, C] 转换为 [B, C, H, W]
        img = 生成图像.movedim(-1, 1)
        
        # 计算原始内容区域的尺寸
        original_content_width = img.shape[3] - width_padding
        original_content_height = img.shape[2] - height_padding
        
        # 裁剪图像，去除补边区域
        cropped_img = img[:, :, y:original_content_height, x:original_content_width]
        
        # 将图像维度转换回 [B, H, W, C]
        aligned_image = cropped_img.movedim(1, -1)
        
        return (aligned_image,)
