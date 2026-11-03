import torch
import folder_paths
import random
import os
import comfy.samplers
from PIL import Image
import numpy as np

from .. import st_log, st_flush, _reg_get, _reg_get_by_role

class ST_KSamplerWithVAE:
    """解码采样器节点 - 集成K采样器和VAE解码功能，一步生成图像"""
    DISPLAY_NAME = "解码采样器"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # 定义节点输入参数
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "用于去噪输入潜在空间的模型。"}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "用于创建噪声的随机种子。"}),
                "步数": ("INT", {"default": 8, "min": 1, "max": 10000, "tooltip": "去噪过程中使用的步数。"}),
                "CFG": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "分类器自由引导尺度。"}),
                "采样器名称": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "采样时使用的算法。"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "噪声调度器。"}),
                "正面条件": ("CONDITIONING", {"tooltip": "正面提示词的条件。"}),
                "负面条件": ("CONDITIONING", {"tooltip": "负面提示词的条件。"}),
                "Latent": ("LATENT", {"tooltip": "要去噪的潜在图像。"}),
                "降噪": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "应用的降噪量。"}),
                "VAE": ("VAE", {"tooltip": "用于解码潜在空间的VAE模型。"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/基础流程"
    DESCRIPTION = "集成K采样器和VAE解码功能，一步生成图像。"

    def run(self, 模型, 种子, 步数, CFG, 采样器名称, 调度器, 正面条件, 负面条件, Latent, 降噪, VAE, unique_id, save_preview=True, return_ui=True):
        # 开头先清 buffer 残留（防止上一次 KSampler 异常退出导致日志累加）
        st_flush()

        # 所有 st_log 集中在 KSampler 开头一次性写入（KSampler 每次必执行，不受 ComfyUI 缓存影响）
        # 通过 registry 查各对象的登记信息（用 id(obj) 做 key，对任何对象类型都生效）
        # Fallback：UNET model 可能被 ComfyUI 内部多次 clone（LoRA / common_ksampler），精确 id 查不到时扫 registry
        unet_name = _reg_get(模型, 'filename') or _reg_get_by_role('unet', 'filename')
        vae_name = _reg_get(VAE, 'filename') or _reg_get_by_role('vae', 'filename')
        lora_list = _reg_get(模型, 'lora_list') or _reg_get_by_role('unet', 'lora_list')
        meta = _reg_get(正面条件, 'meta')

        if unet_name:
            st_log(f"UNET={unet_name}")
        # CLIP 优先从 meta 拿（edit_nodes 登记的），拿不到就按 role 搜 registry
        clip_name = (meta.get('clip_name') if meta else None) or _reg_get_by_role('clip', 'filename')
        if clip_name:
            st_log(f"CLIP={clip_name}")
        if vae_name:
            st_log(f"VAE={vae_name}")
        if lora_list:
            for ln, ls in lora_list:
                st_log(f"LoRA={ln} 强度={ls}")
        if meta:
            st_log(f"模式={meta['mode']} {meta['scene']} 遮罩={meta['mask']}")
            st_log(f"CLIP编码 正={meta['pos_len']}字 负={meta['neg_len']}字")

        # 执行K采样
        from nodes import common_ksampler
        latent = common_ksampler(
            model=模型,
            seed=种子,
            steps=步数,
            cfg=CFG,
            sampler_name=采样器名称,
            scheduler=调度器,
            positive=正面条件,
            negative=负面条件,
            latent=Latent,
            denoise=降噪
        )[0]
        
        # 执行VAE解码
        latent_samples = latent["samples"]
        if latent_samples.is_nested:
            latent_samples = latent_samples.unbind()[0]

        images = VAE.decode(latent_samples)
        if len(images.shape) == 5:  # 合并批次
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        image = images
        
        # 处理返回值
        st_flush()  # KSampler 一般是工作流最后一个 st-tools 节点，这里统一打出本轮所有日志
        if not save_preview and not return_ui:
            return (latent, image)

        results = []
        if save_preview and isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[0] > 0:
            temp_dir = folder_paths.get_temp_directory()
            t = image[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"sampler_vae_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        if return_ui:
            return {"ui": {"images": results}, "result": (latent, image)}
        return (latent, image)
