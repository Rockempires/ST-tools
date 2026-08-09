# __init__.py
import os
import logging
import sys
import subprocess

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 在导入节点前尝试确保 requirements.txt 中的依赖已安装
def _ensure_requirements():
    try:
        req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        if not os.path.exists(req_path):
            return

        # 快速检查可能会被导入的关键包，避免在导入节点时崩溃
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
            import scipy  # noqa: F401
            import argostranslate  # noqa: F401
            return
        except Exception:
            logger.info("ST_tools: 检测到缺失依赖，尝试自动安装 requirements.txt 中的依赖...")

        # 调用 pip 安装依赖（使用当前 python 解释器）
        try:
            res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_path], capture_output=True, text=True, timeout=900)
            if res.returncode == 0:
                logger.info("ST_tools: 依赖安装成功")
            else:
                logger.warning(f"ST_tools: 依赖安装失败，pip 返回码 {res.returncode}, stderr: {res.stderr}")
        except Exception as e:
            logger.warning(f"ST_tools: 自动安装依赖时发生异常: {e}")
    except Exception:
        # 不要让自动安装导致导入失败
        pass

# 立即尝试保证依赖（在导入节点前运行）
_ensure_requirements()

from .nodes import ST_ImageMaskLatentSize, ST_OfflineTranslator, ST_ImagePostProcessing, ST_FilterShader, ST_ImageEditor, ST_ImageSizeAligner, ST_ModelsLoader, ST_LoraStack, ST_KSamplerWithVAE, ST_CLIPTextEncoder

# 启动日志
logger.info("--------------------------------------------------------------------------------")
logger.info("🎯石头版ComfyUI封装包 Q群：33320584")
logger.info(r"🎯石头工具 节点已加载，用法详阅节点说明")
logger.info("--------------------------------------------------------------------------------")

NODE_CLASS_MAPPINGS = {
    "ST_ModelsLoader": ST_ModelsLoader,
    "ST_LoraStack": ST_LoraStack,
    "ST_CLIPTextEncoder": ST_CLIPTextEncoder,
    "ST_ImageMaskLatentSize": ST_ImageMaskLatentSize,
    "ST_KSamplerWithVAE": ST_KSamplerWithVAE,
    "ST_ImagePostProcessing": ST_ImagePostProcessing,
    "ST_FilterShader": ST_FilterShader,
    "ST_ImageEditor": ST_ImageEditor,
    "ST_ImageSizeAligner": ST_ImageSizeAligner,
    "ST_OfflineTranslator": ST_OfflineTranslator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_ModelsLoader": "模型加载",
    "ST_LoraStack": "Lora加载",
    "ST_CLIPTextEncoder": "CLIP编码器",
    "ST_ImageMaskLatentSize": "图像尺寸",
    "ST_KSamplerWithVAE": "解码采样器",
    "ST_ImagePostProcessing": "图像调色",
    "ST_FilterShader": "图像滤镜",
    "ST_ImageEditor": "编辑图像",
    "ST_ImageSizeAligner": "编辑对齐",
    "ST_OfflineTranslator": "离线翻译",
}

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from comfy_api.latest import ComfyExtension
from comfy_api.latest import io

class STToolsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ST_ImageEditor,
        ]

async def comfy_entrypoint() -> STToolsExtension:
    return STToolsExtension()