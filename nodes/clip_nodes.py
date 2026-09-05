import torch
from nodes import CLIPTextEncode, ConditioningZeroOut

class ST_CLIPTextEncoder:
    """CLIP编码器节点 - 合并正向和负向提示词输入"""
    DISPLAY_NAME = "CLIP编码器"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # 定义节点输入参数
        return {
            "required": {
                "CLIP": ("CLIP", {"tooltip": "用于编码文本的CLIP模型。"}),
                "正向提示词": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip": "正向提示词，描述你想要在图像中包含的内容。", "rows": 6}),
                "负向提示词": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip": "负向提示词，描述你想要在图像中排除的内容。", "rows": 3}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("正面条件", "负面条件", "负面条件零化")
    FUNCTION = "encode"
    CATEGORY = "🎯 石头工具/基础流程"
    DESCRIPTION = "合并CLIP文本编码功能，同时处理正向和负向提示词。"

    def encode(self, CLIP, 正向提示词, 负向提示词):
        # 执行CLIP文本编码操作
        try:
            # 确保提示词是字符串类型
            if not isinstance(正向提示词, str):
                正向提示词 = str(正向提示词)
            if not isinstance(负向提示词, str):
                负向提示词 = str(负向提示词)
            
            # 直接使用CLIP模型编码文本
            # 编码正向提示词
            positive_tokens = CLIP.tokenize(正向提示词)
            positive_conditioning = CLIP.encode_from_tokens_scheduled(positive_tokens)
            
            # 编码负向提示词
            negative_tokens = CLIP.tokenize(负向提示词)
            negative_conditioning = CLIP.encode_from_tokens_scheduled(negative_tokens)
            
            # 零化负向提示词
            zero_conditioning = []
            for t in negative_conditioning:
                d = t[1].copy()
                pooled_output = d.get("pooled_output", None)
                if pooled_output is not None:
                    d["pooled_output"] = torch.zeros_like(pooled_output)
                conditioning_lyrics = d.get("conditioning_lyrics", None)
                if conditioning_lyrics is not None:
                    d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
                n = [torch.zeros_like(t[0]), d]
                zero_conditioning.append(n)
            
            # 返回三个条件
            return (positive_conditioning, negative_conditioning, zero_conditioning)
        except Exception as e:
            # 错误处理
            print(f"[ST_tools] CLIP文本编码器执行失败: {str(e)}")
            # 直接使用CLIP模型创建空条件
            empty_tokens = CLIP.tokenize("")
            empty_conditioning = CLIP.encode_from_tokens_scheduled(empty_tokens)
            # 返回空条件
            return (empty_conditioning, empty_conditioning, empty_conditioning)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ST_CLIPTextEncoder": ST_CLIPTextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_CLIPTextEncoder": "CLIP编码器"
}