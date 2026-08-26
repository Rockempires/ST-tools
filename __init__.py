# __init__.py
import os
import logging
import re
import sys
import subprocess
import importlib.metadata
import platform

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

from .nodes import ST_ImageMaskLatentSize, ST_OfflineTranslator, ST_ImagePostProcessing, ST_FilterShader, ST_ImageEditor, ST_ImageSizeAligner, ST_ModelsLoader, ST_LoraStack, ST_KSamplerWithVAE, ST_CLIPTextEncoder, ST_DLSSNRImage, ST_DLSSNRVideo, ST_DLSSNRRuntimeInfo

# 启动日志
logger.info("--------------------------------------------------------------------------------")
logger.info("🎯石头版ComfyUI-v12-20260817  +Q群获取更新：33320584")
logger.info(r"🎯石头工具 节点已加载，用法详阅节点说明")
logger.info("--------------------------------------------------------------------------------")

def _log_hardware_info():
    def _ver(pkg):
        try:
            return importlib.metadata.version(pkg)
        except Exception:
            return "未安装"

    def _ver_full(pkg):
        try:
            dist = importlib.metadata.distribution(pkg)
            return dist.version
        except Exception:
            return "未安装"

    def _fmt_cudnn(v):
        try:
            v = int(v)
            if v >= 10000:
                major = v // 10000
                minor = (v % 10000) // 1000
                patch = v % 1000
            else:
                major = v // 1000
                minor = (v % 1000) // 100
                patch = v % 100
            return f"{major}.{minor}.{patch}"
        except Exception:
            return str(v)

    def _get_cpu_model():
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            m = re.search(r'(i[3579]-\d+[KXF]+|Ryzen\s+\d+\s+\w+)', cpu_name, re.IGNORECASE)
            if m:
                return m.group(1)
            return cpu_name.strip()
        except Exception:
            return platform.processor() or "未知"

    def _get_motherboard():
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\BIOS")
            product, _ = winreg.QueryValueEx(key, "BaseBoardProduct")
            winreg.CloseKey(key)
            m = re.search(r'([A-Z]+\d+[A-Z]*(?:-[A-Z0-9]+)?)', product)
            if m:
                chipset = m.group(1)
                rest = product[m.end():].strip()
                stop = ['GAMING', 'WIFI', 'BLUETOOTH', 'ROG', 'STRIX', 'TUF',
                        'MAG', 'MPG', 'PRO', 'PLUS', 'ULTRA', 'EDITION']
                words = rest.split()
                model_words = [w for w in words if w.upper() not in stop and w.upper() != chipset.upper()]
                if model_words:
                    return chipset + ' ' + ' '.join(model_words)
                return chipset
            return product.strip()
        except Exception:
            return "未知"

    def _get_os_name():
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            display_version, _ = winreg.QueryValueEx(key, "DisplayVersion")
            build, _ = winreg.QueryValueEx(key, "CurrentBuild")
            edition_id, _ = winreg.QueryValueEx(key, "EditionID")
            winreg.CloseKey(key)
            build_num = int(build)
            win_ver = "Windows 11" if build_num >= 22000 else "Windows 10"
            edition_map = {
                "Professional": "专业版",
                "ProfessionalWorkstation": "工作站专业版",
                "Enterprise": "企业版",
                "Education": "教育版",
                "Home": "家庭版",
            }
            edition_cn = edition_map.get(edition_id, edition_id)
            if display_version:
                return f"{win_ver} {display_version} {edition_cn}"
            return f"{win_ver} {edition_cn}"
        except Exception:
            return platform.platform()

    def _get_disk_info():
        try:
            ps_path = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            if not os.path.exists(ps_path):
                return ""
            cmd = (
                "Get-PhysicalDisk | "
                "ForEach-Object { "
                "$sizeTB = [math]::Ceiling($_.Size/1TB); "
                "$sizeTB.ToString() + '|' + $_.BusType "
                "}"
            )
            result = subprocess.run(
                [ps_path, "-Command", cmd],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0 or not result.stdout.strip():
                return ""
            disks = []
            bus_types = set()
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) >= 2:
                    disks.append(f"{parts[0]}T")
                    bus_types.add(parts[1])
            if not disks:
                return ""
            type_map = {"NVMe": "M.2", "SATA": "SATA", "SAS": "SAS"}
            type_names = [type_map.get(t, t) for t in bus_types]
            return "+".join(disks) + " " + "+".join(sorted(set(type_names)))
        except Exception:
            return ""

    lines = []
    lines.append("")
    lines.append("🖥️ 本计算机硬件信息：")

    try:
        lines.append(f"  主板: {_get_motherboard()}")
    except Exception:
        pass

    try:
        lines.append(f"  CPU: {_get_cpu_model()}")
    except Exception:
        lines.append(f"  CPU: {platform.processor() or '未知'}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        lines.append(f"  内存: {round(mem.total / 1024**3, 1)} GB")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            m = re.search(r'(RTX\s+\d+\s*(?:Ti|Super)?|GTX\s+\d+\s*(?:Ti)?|T\d+)', gpu_name, re.IGNORECASE)
            if m:
                gpu_name = m.group(1)
            lines.append(f"  显卡: {gpu_name}")
            lines.append(f"  CUDA版本: {torch.version.cuda}")
            try:
                lines.append(f"  cudNN版本: {_fmt_cudnn(torch.backends.cudnn.version())}")
            except Exception:
                lines.append("  cudNN版本: 未检测到")
        else:
            lines.append("  显卡: CUDA 不可用")
    except Exception:
        lines.append("  显卡: torch 未加载")

    disk_info = _get_disk_info()
    if disk_info:
        lines.append(f"  硬盘: {disk_info}")

    try:
        lines.append(f"  操作系统: {_get_os_name()}")
    except Exception:
        pass

    lines.append("")
    lines.append("🧩 ComfyUI核心环境依赖版本:")
    lines.append(f"  Python: {sys.version.split()[0]}")
    lines.append(f"  PyTorch: {_ver_full('torch')}")
    lines.append(f"  torchvision: {_ver_full('torchvision')}")
    lines.append(f"  torchaudio: {_ver_full('torchaudio')}")
    lines.append(f"  SageAttention: {_ver_full('sageattention')}")
    lines.append(f"  triton-windows: {_ver_full('triton-windows')}")
    lines.append(f"  Flash-Attention: {_ver_full('flash-attn')}")
    lines.append(f"  llama-cpp-python: {_ver_full('llama-cpp-python')}")
    lines.append(f"  numpy: {_ver('numpy')}")
    lines.append(f"  transformers: {_ver('transformers')}")
    lines.append(f"  diffusers: {_ver('diffusers')}")
    lines.append(f"  accelerate: {_ver('accelerate')}")
    lines.append(f"  opencv-contrib-python-headless: {_ver('opencv-contrib-python-headless')}")
    lines.append(f"  Pillow: {_ver('Pillow')}")
    lines.append(f"  fastapi: {_ver('fastapi')}")
    lines.append(f"  einops: {_ver('einops')}")
    lines.append(f"  safetensors: {_ver('safetensors')}")

    for line in lines:
        logger.info(line)

    logger.info("--------------------------------------------------------------------------------")

_log_hardware_info()

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
    "ST_DLSSNRImage": ST_DLSSNRImage,
    "ST_DLSSNRVideo": ST_DLSSNRVideo,
    "ST_DLSSNRRuntimeInfo": ST_DLSSNRRuntimeInfo,
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
    "ST_DLSSNRImage": "渲染图像(DLSS)",
    "ST_DLSSNRVideo": "渲染视频(DLSS)",
    "ST_DLSSNRRuntimeInfo": "渲染检测(DLSS)",
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