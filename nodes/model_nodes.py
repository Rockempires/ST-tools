import folder_paths
import torch
from nodes import UNETLoader, CLIPLoader, VAELoader

from .. import _reg_set

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
        unet_name = kwargs.get("UNET模型")
        weight_dtype = kwargs.get("权重类型")
        clip_name = kwargs.get("CLIP模型")
        clip_type = kwargs.get("CLIP类型", "stable_diffusion")
        vae_name = kwargs.get("VAE模型")
        device = kwargs.get("设备", "default")

        model = clip = vae = None

        if unet_name != "无":
            model = UNETLoader().load_unet(unet_name=unet_name, weight_dtype=weight_dtype)[0]
            _reg_set(model, filename=unet_name, role='unet')
        if clip_name != "无":
            clip = CLIPLoader().load_clip(clip_name=clip_name, type=clip_type, device=device)[0]
            _reg_set(clip, filename=clip_name, role='clip')
        if vae_name != "无":
            vae = VAELoader().load_vae(vae_name=vae_name)[0]
            _reg_set(vae, filename=vae_name, role='vae')

        return (model, clip, vae)

