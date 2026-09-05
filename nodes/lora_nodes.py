import folder_paths
import comfy.sd
import comfy.utils

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
        
        # 统计加载的Lora数量
        lora_count = 0
        
        # 处理4个Lora输入
        lora_keys = ["lora加载", "lora加载 ", "lora加载  ", "lora加载   "]
        strength_keys = ["强度", "强度 ", "强度  ", "强度   "]
        
        for i, (lora_key, strength_key) in enumerate(zip(lora_keys, strength_keys), 1):
            # 获取当前Lora的名称和强度值
            lora_name = kwargs.get(lora_key)
            lora_strength = kwargs.get(strength_key, 1.0)
            
            # 检查是否选择了"无"或强度值为0
            if lora_name == "无" or lora_strength == 0:
                continue
            
            # 增加Lora计数
            lora_count += 1
            
            try:
                # 获取Lora文件路径
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                
                # 检查是否已加载过该Lora
                lora = None
                if lora_path in self.loaded_loras:
                    lora = self.loaded_loras[lora_path]
                else:
                    # 加载Lora文件
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_loras[lora_path] = lora
                
                # 应用Lora到模型和CLIP（如果CLIP存在）
                if CLIP is not None:
                    模型, CLIP = comfy.sd.load_lora_for_models(模型, CLIP, lora, lora_strength, lora_strength)
                else:
                    # 只应用Lora到模型
                    模型, _ = comfy.sd.load_lora_for_models(模型, None, lora, lora_strength, lora_strength)
                print(f"[ST] 应用Lora: {lora_name} | 强度={lora_strength}")
            except Exception as e:
                print(f"[ST] ✗ 错误: {e}")
        
        # 返回加载了Lora的模型和CLIP（如果CLIP存在）
        return (模型, CLIP)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ST_LoraStack": ST_LoraStack
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_LoraStack": "Lora加载"
}
