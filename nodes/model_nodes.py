import folder_paths
import torch
from nodes import UNETLoader, CLIPLoader, VAELoader

from .. import _reg_set

class ST_ModelsLoader:
    """模型加载节点 - 同时加载UNET、CLIP和两个独立的VAE模型"""
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
                "VAE模型": (vae_files, {"default": "无", "display_name": "VAE模型1", "tooltip": "第一路VAE，双VAE视频工作流中可选择视频VAE。只加载一个VAE时，任一槽位均可，两个输出都会提供该VAE。"}),
            },
            "optional": {
                "VAE模型2": (vae_files, {"default": "无", "tooltip": "第二路VAE，双VAE视频工作流中可选择音频VAE。两路均选择时分别从VAE1、VAE2输出；只选择本槽位时两个输出均可使用。"}),
                "设备": (device_opts, {"advanced": True, "tooltip": "CLIP模型的加载设备。"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("模型", "CLIP", "VAE1", "VAE2")
    FUNCTION = "load_models"
    CATEGORY = "🎯 石头工具/基础流程"
    DESCRIPTION = """接线
分别加载扩散模型（UNET）、CLIP 和 VAE。将模型接采样器或 LoRA 加载，将 CLIP 接提示词编码器，将 VAE 接编码或解码节点。

加载选项
UNET、CLIP 或 VAE 选择「无」可跳过该组件。CLIP 类型要匹配工作流的模型架构；「设备」只控制 CLIP 的加载设备，「权重类型」只用于 UNET。

VAE 输出
两个 VAE 都选时分别从 VAE1、VAE2 输出，可按工作流需要连接视频和音频 VAE。只选一路时，两个输出都提供同一个 VAE；两路选同一文件时只加载一次。"""

    def load_models(self, **kwargs):
        unet_name = kwargs.get("UNET模型")
        weight_dtype = kwargs.get("权重类型")
        clip_name = kwargs.get("CLIP模型")
        clip_type = kwargs.get("CLIP类型", "stable_diffusion")
        vae_name = kwargs.get("VAE模型", "无")
        vae_name2 = kwargs.get("VAE模型2", "无")
        device = kwargs.get("设备", "default")

        model = clip = vae = vae2 = None

        if unet_name != "无":
            model = UNETLoader().load_unet(unet_name=unet_name, weight_dtype=weight_dtype)[0]
            _reg_set(model, filename=unet_name, role='unet')
        if clip_name != "无":
            clip = CLIPLoader().load_clip(clip_name=clip_name, type=clip_type, device=device)[0]
            _reg_set(clip, filename=clip_name, role='clip')
        if vae_name and vae_name != "无":
            vae = VAELoader().load_vae(vae_name=vae_name)[0]
            _reg_set(vae, filename=vae_name, role='vae')

        if vae_name2 and vae_name2 != "无":
            # 两个槽位选择同一文件时复用本次加载结果，避免重复读取权重。
            if vae_name2 == vae_name:
                vae2 = vae
            else:
                vae2 = VAELoader().load_vae(vae_name=vae_name2)[0]
                _reg_set(vae2, filename=vae_name2, role='vae')

        # 保留原有输出位置；只加载一路时，任一槽位均能供原工作流使用。
        return (model, clip, vae if vae is not None else vae2, vae2 if vae2 is not None else vae)

