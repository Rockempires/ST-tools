import folder_paths
import torch
from nodes import UNETLoader, CLIPLoader, VAELoader

class ST_ModelsLoader:
    """模型加载节点 - 同时加载UNET、CLIP和VAE模型"""
    DISPLAY_NAME = "模型加载"
    
    @classmethod
    def INPUT_TYPES(cls):
        unet_files = ["无"] + folder_paths.get_filename_list("diffusion_models")
        clip_files = ["无"] + folder_paths.get_filename_list("text_encoders")
        vae_files = ["无"] + VAELoader.vae_list(VAELoader)

        weight_dtype_opts = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
        
        clip_loader_input_types = CLIPLoader.INPUT_TYPES()
        clip_type_opts = clip_loader_input_types["required"]["type"][0]
        
        device_opts = ["default", "cpu"]

        return {
            "required": {
                "UNET模型": (unet_files, {"default": "无", "tooltip": "扩散模型 (UNET) 的名称。"}),
                "权重类型": (weight_dtype_opts, {"advanced": True, "tooltip": "模型权重的数据类型。"}),
                "CLIP模型": (clip_files, {"default": "无", "tooltip": "CLIP模型的名称。"}),
                "CLIP类型": (clip_type_opts, {"default": "flux2", "tooltip": "CLIP模型的类型。"}),
                "VAE模型": (vae_files, {"default": "无", "tooltip": "VAE模型的名称。"}),
            },
            "optional": {
                "设备": (device_opts, {"advanced": True, "tooltip": "CLIP模型的加载设备。"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("模型", "CLIP", "VAE")
    FUNCTION = "load_models"
    CATEGORY = "🎯 石头工具/基础流程"
    DESCRIPTION = "同时加载UNET、CLIP和VAE模型。"

    def load_models(self, **kwargs):
        print(f"\n[ST] === 开始加载模型 ===")
        
        # 获取参数值
        unet_name = kwargs.get("UNET模型")
        weight_dtype = kwargs.get("权重类型")
        clip_name = kwargs.get("CLIP模型")
        type = kwargs.get("CLIP类型", "stable_diffusion")
        vae_name = kwargs.get("VAE模型")
        device = kwargs.get("设备", "default")
        
        # 加载UNET模型
        if unet_name != "无":
            unet_loader = UNETLoader()
            model = unet_loader.load_unet(unet_name=unet_name, weight_dtype=weight_dtype)[0]
            print(f"[ST] UNET: {unet_name}")
        else:
            model = None
            print(f"[ST] UNET: 未选择")

        # 加载CLIP模型
        if clip_name != "无":
            clip_loader = CLIPLoader()
            clip = clip_loader.load_clip(clip_name=clip_name, type=type, device=device)[0]
            print(f"[ST] CLIP: {clip_name}")
        else:
            clip = None
            print(f"[ST] CLIP: 未选择")

        # 加载VAE模型
        if vae_name != "无":
            vae_loader = VAELoader()
            vae = vae_loader.load_vae(vae_name=vae_name)[0]
            print(f"[ST] VAE: {vae_name}")
        else:
            vae = None
            print(f"[ST] VAE: 未选择")

        print(f"[ST] === 模型加载完成 ===\n")
        return (model, clip, vae)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ST_ModelsLoader": ST_ModelsLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_ModelsLoader": "模型加载"
}
