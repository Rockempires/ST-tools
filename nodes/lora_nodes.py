import folder_paths
import comfy.sd
import comfy.utils

from .. import _reg_set, _reg_get

class ST_LoraStack:
    """Lora加载节点 - 同时加载多个Lora模型到模型和CLIP中"""
    
    # 节点显示名称
    DISPLAY_NAME = "Lora加载"
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取所有可用的Lora文件列表，添加"无"选项
        lora_files = ["无"] + folder_paths.get_filename_list("loras")
        
        # 定义必填输入参数
        required = {
            "模型": ("MODEL", {"tooltip": "扩散模型，Lora将应用到该模型上。"}),
            # 第一个Lora
            "lora加载": (lora_files, {"default": "无", "tooltip": "Lora模型的名称。"}),
            "强度": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "修改模型的强度。"}),
            # 第二个Lora
            "lora加载 ": (lora_files, {"default": "无", "tooltip": "Lora模型的名称。"}),
            "强度 ": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "修改模型的强度。"}),
            # 第三个Lora
            "lora加载  ": (lora_files, {"default": "无", "tooltip": "Lora模型的名称。"}),
            "强度  ": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "修改模型的强度。"}),
            # 第四个Lora
            "lora加载   ": (lora_files, {"default": "无", "tooltip": "Lora模型的名称。"}),
            "强度   ": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "修改模型的强度。"}),
        }
        
        # 定义可选输入参数
        optional = {
            "CLIP": ("CLIP", {"tooltip": "CLIP模型，Lora将应用到该模型上。"}),
        }
        
        return {"required": required, "optional": optional}

    # 定义节点的返回类型
    RETURN_TYPES = ("MODEL", "CLIP")
    # 定义返回类型的名称
    RETURN_NAMES = ("模型", "CLIP")
    # 定义节点的主要函数
    FUNCTION = "load_loras"
    # 定义节点的分类
    CATEGORY = "🎯 石头工具/基础流程"
    # 定义节点的描述
    DESCRIPTION = "同时加载多个Lora模型到模型和CLIP中。"

    def load_loras(self, 模型, **kwargs):
        # 加载Lora模型到输入的模型和CLIP中
        
        # 获取CLIP（可选）
        CLIP = kwargs.get("CLIP")
        
        # 应用 LoRA 前先拍 registry 快照（load_lora_for_models 返回的是 model.clone()，id 会变）
        old_model_info = _reg_get(模型)
        old_clip_info = _reg_get(CLIP) if CLIP is not None else {}
        
        lora_list = []
        
        # 处理4个Lora输入
        lora_keys = ["lora加载", "lora加载 ", "lora加载  ", "lora加载   "]
        strength_keys = ["强度", "强度 ", "强度  ", "强度   "]
        
        for i, (lora_key, strength_key) in enumerate(zip(lora_keys, strength_keys), 1):
            lora_name = kwargs.get(lora_key)
            lora_strength = kwargs.get(strength_key, 1.0)
            
            if lora_name == "无" or lora_strength == 0:
                continue
            
            try:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = self.loaded_loras.get(lora_path)
                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_loras[lora_path] = lora
                
                if CLIP is not None:
                    模型, CLIP = comfy.sd.load_lora_for_models(模型, CLIP, lora, lora_strength, lora_strength)
                else:
                    模型, _ = comfy.sd.load_lora_for_models(模型, None, lora, lora_strength, lora_strength)
                lora_list.append((lora_name, lora_strength))
            except Exception as e:
                pass
        
        # load_lora_for_models 返回的是 model.clone()（新对象 id 变了），
        # 把旧 registry 快照搬到新 clone 上，但显式跳过 lora_list——
        # lora_list 是本次 run 全新生成的，不能从旧对象搬（会跨 run 累加）
        if old_model_info:
            for k, v in old_model_info.items():
                if k != 'lora_list':
                    _reg_set(模型, **{k: v})
        if old_clip_info and CLIP is not None:
            for k, v in old_clip_info.items():
                _reg_set(CLIP, **{k: v})
        # lora_list 直接 set，不 existing + new——每次 run 都是全新的 4 个 slot 收集结果
        if lora_list:
            _reg_set(模型, lora_list=lora_list)
        
        return (模型, CLIP)

