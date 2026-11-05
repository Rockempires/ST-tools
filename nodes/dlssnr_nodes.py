# DLSS 超分辨率 + 神经渲染节点
# 通过调用 dlssnr.exe 实现 NVIDIA DLSS SR + Neural Rendering
# 两种处理模式：
#   图像模式：批次内每张图独立处理（--nr-run）
#   视频模式：整段视频经 --nr-video 光流运动矢量处理，时序更稳定；
#             帧以原始 RGBA 经管道直传 exe，不走 ffmpeg、不产生临时文件
# exe 查找顺序：1. 环境变量 DLSSNR_EXE（指向 exe 完整路径）；2. ST-tools/dlssnr/bin/dlssnr.exe

import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from fractions import Fraction

import numpy as np
import torch
from aiohttp import web
from server import PromptServer

from .. import st_log

# ST-tools 根目录（本文件位于 nodes/ 下）
_TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 下拉选项中文显示 -> CLI 数值
STYLES = {"默认": 0, "自然": 1, "电影": 2}
# 三个自定义档位自动保存控件参数，切换时恢复；图像和视频分别存储。
# 保留各档位原有的后端预设编号，渲染参数仍随当前节点输入传递。
PRESETS = {"默认预设（推荐）": 0, "预设配置一": 1, "预设配置二": 2, "预设配置三": 3}
ENGINES = {"自动": "auto", "NVOF光流": "nvof", "LK光流": "lk"}

# 旧版工作流兼容映射：下拉项改名后，历史工作流 JSON 中仍保存旧键名
_PRESET_ALIASES = {
    "默认": "默认预设（推荐）", "Default": "默认预设（推荐）",
    "默认模型(推荐)": "默认预设（推荐）", "默认模型（推荐）": "默认预设（推荐）",
    "默认预设模型(推荐)": "默认预设（推荐）", "默认预设模型（推荐）": "默认预设（推荐）",
    "默认预设(推荐)": "默认预设（推荐）",
    "预设1": "预设配置一", "Preset 1": "预设配置一",
    "预设2": "预设配置二", "Preset 2": "预设配置二",
    "预设3": "预设配置三", "Preset 3": "预设配置三",
}


def _norm_preset(v):
    """渲染预设归一化：旧工作流值/英文原值/CLI 数值统一映射为当前下拉键"""
    if v in PRESETS:
        return v
    if v in _PRESET_ALIASES:
        return _PRESET_ALIASES[v]
    try:  # CLI 数值 0-3 兜底
        for k, idx in PRESETS.items():
            if idx == int(v):
                return k
    except (TypeError, ValueError):
        pass
    return v


def _norm_engine(v):
    """运动引擎归一化：兼容旧英文值 auto/nvof/lk"""
    if v in ENGINES:
        return v
    if v in ("auto", "nvof", "lk"):
        return v
    return "auto"


def _validate_nr_inputs(kwargs):
    """下拉类参数校验：旧工作流键名自动放行（运行时归一化），未知值才报错"""
    if "渲染风格" in kwargs and kwargs["渲染风格"] not in STYLES:
        return f"渲染风格的值无效：{kwargs['渲染风格']!r}，请重新选择"
    if "渲染预设" in kwargs:
        v = _norm_preset(kwargs["渲染预设"])
        if v not in PRESETS:
            return f"渲染预设的值无效：{kwargs['渲染预设']!r}，请重新选择"
    if "运动引擎" in kwargs and kwargs["运动引擎"] not in ENGINES \
            and kwargs["运动引擎"] not in ("auto", "nvof", "lk"):
        return f"运动引擎的值无效：{kwargs['运动引擎']!r}，请重新选择"
    return True


# ----------------------------------------------------------------------------- 工具查找

def find_exe():
    """查找 dlssnr.exe，找不到则报错提示放置路径"""
    env = os.environ.get("DLSSNR_EXE", "").strip().strip('"')
    cands = [env] if env else []
    cands.append(os.path.join(_TOOL_ROOT, "dlssnr", "bin", "dlssnr.exe"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise RuntimeError(
        "未找到 dlssnr.exe。请设置环境变量 DLSSNR_EXE 指向该程序，"
        "或将发布包的 bin 文件夹复制到："
        + os.path.join(_TOOL_ROOT, "dlssnr", "bin"))


def nr_args(风格, 预设, 渲染强度, 局部结构, 局部色调, 肤色, 全局色调, 细节融合,
            色彩融合, UI校正, 自动遮罩, HDR模式):
    """组装 NR 模型与合成参数，参数名与 CLI 的 --nr-* 一一对应"""
    预设 = _norm_preset(预设)
    a = ["--nr-style", str(STYLES[风格]), "--nr-preset", str(PRESETS[预设]),
         "--nr-intensity", f"{渲染强度}", "--nr-local-structure", f"{局部结构}",
         "--nr-local-tone", f"{局部色调}", "--nr-skin", f"{肤色}",
         "--nr-global-tone", f"{全局色调}", "--nr-detail", f"{细节融合}",
         "--nr-color", f"{色彩融合}", "--nr-ui-correction", "1" if UI校正 else "0"]
    if 自动遮罩:
        a.append("--nr-auto-mask")
    if HDR模式:
        a.append("--nr-hdr")
    return a


def out_dims(in_w, in_h, width, scale):
    """输出尺寸计算：指定宽度时按宽高比推高度，否则按倍率；尺寸取偶数"""
    if width > 0:
        out_w, out_h = int(width), round(in_h * width / in_w)
    else:
        out_w, out_h = round(in_w * scale), round(in_h * scale)
    return out_w - out_w % 2, out_h - out_h % 2


# ----------------------------------------------------------------------------- numpy 处理核心

def run_image_np(img_u8, width, scale, nr, exe, adapter=0):
    """单张 uint8 RGB [H,W,3] 图像走 --nr-run，返回 uint8 RGB [H',W',3]"""
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="dlssnr_")
    try:
        src = os.path.join(tmp, "in.png")
        out_dir = os.path.join(tmp, "out")
        Image.fromarray(np.ascontiguousarray(img_u8), "RGB").save(src)
        cmd = [exe, "--nr-run", "--in", src, "--out", out_dir, "--adapter", str(adapter)] + nr
        if width > 0:
            cmd += ["--nr-width", str(int(width))]
        elif abs(scale - 1.0) > 1e-6:
            cmd += ["--nr-scale", f"{scale}"]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        hits = glob.glob(os.path.join(out_dir, "*_nr.png"))
        if p.returncode != 0 or not hits:
            raise RuntimeError(f"dlssnr 处理失败 (退出码 {p.returncode})\n"
                               + ((p.stdout or "") + (p.stderr or ""))[-3000:])
        return np.array(Image.open(hits[0]).convert("RGB"), dtype=np.uint8)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_video_np(frames_u8, width, scale, nr, motion, engine, motion_vis, exe, adapter=0,
                 progress=None, log=None):
    """uint8 RGB [B,H,W,3] 帧序列经管道流式送入 --nr-video，返回 uint8 RGB [B,H',W',3]

    progress(n)：工具每报告完成帧数时回调；log 为列表时收集非进度 stderr 行
    （使用的后端、最终统计、警告等）。
    """
    b, h, w, _ = frames_u8.shape
    out_w, out_h = out_dims(w, h, width, scale)
    cmd = [exe, "--nr-video", "--nr-in", f"{w}x{h}", "--adapter", str(adapter)] + nr + [
        "--nr-motion", "1" if motion else "0", "--nr-motion-engine", engine]
    if motion_vis:
        cmd.append("--nr-motion-vis")
    if (out_w, out_h) != (w, h):  # 显式传偶数尺寸，保证工具输出与读取端一致
        cmd += ["--nr-width", str(out_w), "--nr-height", str(out_h)]

    # 补全 alpha 通道拼成 RGBA 原始帧
    alpha = np.full((b, h, w, 1), 255, np.uint8)
    rgba = np.concatenate([frames_u8, alpha], axis=3)

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    err = []

    def feed():
        try:
            for i in range(b):
                p.stdin.write(rgba[i].tobytes())
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def drain():
        prog = re.compile(rb"^NRPROG (\d+) ")
        for raw in iter(p.stderr.readline, b""):
            m = prog.match(raw)
            if m:
                if progress:
                    progress(int(m.group(1)))
            else:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    err.append(line)

    tf = threading.Thread(target=feed, daemon=True)
    td = threading.Thread(target=drain, daemon=True)
    tf.start()
    td.start()

    # 按固定字节数读回 RGBA 结果
    need = b * out_w * out_h * 4
    buf = bytearray(need)
    view = memoryview(buf)
    got = 0
    while got < need:
        n = p.stdout.readinto(view[got:])
        if not n:
            break
        got += n
    rc = p.wait()
    tf.join(timeout=5)
    td.join(timeout=5)
    if log is not None:
        log.extend(err)
    if rc != 0 or got != need:
        raise RuntimeError(f"dlssnr 处理失败 (退出码 {rc}，收到 {got}/{need} 字节)\n"
                           + "\n".join(err)[-3000:])
    return np.frombuffer(buf, np.uint8).reshape(b, out_h, out_w, 4)[..., :3]


# ----------------------------------------------------------------------------- tensor 转换

def to_u8(t):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> uint8 RGB [B,H,W,3]"""
    x = t.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return np.ascontiguousarray(x[..., :3])


def to_tensor(u8):
    return torch.from_numpy(np.ascontiguousarray(u8).astype(np.float32) / 255.0)


def make_video(images, audio, frame_rate):
    """把帧序列、音频、帧率封装为 ComfyUI VIDEO 类型

    旧版 ComfyUI 没有 VIDEO 类型时返回 None，IMAGE 输出仍可正常使用。
    """
    try:
        from comfy_api.input_impl import VideoFromComponents
        from comfy_api.util import VideoComponents
    except Exception:
        return None
    return VideoFromComponents(VideoComponents(images=images, audio=audio, frame_rate=frame_rate))


def _nr_inputs():
    """两个节点共用的 NR 参数定义"""
    f = lambda d, lo, hi: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": 0.05})
    return {
        "渲染风格": (list(STYLES), {"default": "电影"}),
        "渲染预设": (list(PRESETS), {"default": "默认预设（推荐）",
                                     "tooltip": "预设配置一/二/三为用户自定义档位：修改参数后自动保存，切换时恢复，重启后可复用。"
                                                "图像与视频分别保存；默认预设恢复初始参数，不覆盖自定义档位。"
                                                "各档位仍使用原有后端预设编号0/1/2/3。外接输入参数不参与控件档位切换。"}),
        "渲染强度": f(1.0, 0.0, 2.0),
        "局部结构": f(1.0, 0.0, 2.0),
        "局部色调": f(1.0, 0.0, 2.0),
        "肤色增强": f(-1.0, -1.0, 2.0),       # -1 = 模型默认
        "全局色调": f(-1.0, -1.0, 2.0),       # <0 = 模型默认
        "细节融合": f(1.0, 0.0, 2.0),         # 合成比例：0 = 原片，1 = 完整 NR
        "色彩融合": f(1.0, 0.0, 1.0),         # 0 = 保留原色相，1 = NR 色彩
        "UI校正": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭"}),
        "自动遮罩": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭"}),
        "HDR模式": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭"}),
        "放大倍数": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.05}),
        "输出宽度": ("INT", {"default": 0, "min": 0, "max": 7680, "step": 2,
                            "tooltip": "输出宽度（像素），高度按宽高比自动计算；填 0 则按放大倍数缩放"}),
        "显卡序号": ("INT", {"default": 0, "min": 0, "max": 15,
                            "tooltip": "DXGI 显卡适配器序号，多显卡时选择使用哪张 GPU"}),
    }


# ----------------------------------------------------------------------------- 节点

class ST_DLSSNRImage:
    """渲染图像(DLSS) - 批次内每张图像独立进行 DLSS 超分 + 神经渲染"""
    DISPLAY_NAME = "渲染图像(DLSS)"

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        return {"required": {"图像": ("IMAGE",), **_nr_inputs()}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/DLSS渲染"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 存在该方法后核心跳过内置下拉校验，旧工作流键名在此统一兼容
        return _validate_nr_inputs(kwargs)

    DESCRIPTION = ("DLSS 超分辨率 + 神经渲染（图像模式），批次内每张图独立处理。\n"
                   "输出宽度：指定输出宽度（高度按比例），填0则按放大倍数缩放。\n\n"
                   "参数说明：\n"
                   "渲染风格：默认/自然/电影\n"
                   "渲染预设：配置一/二/三自动保存用户调节参数，切换时恢复，重启后可复用。\n"
                   "默认预设恢复初始参数；图像和视频档位分别保存。各档位保留原后端预设编号。\n"
                   "参数修改后即保存，无需执行渲染；读取工作流时保留工作流自身参数。\n"
                   "渲染强度：NR 整体效果强度\n"
                   "局部结构/局部色调：局部结构与明暗增强\n"
                   "肤色增强：肤色优化，-1为模型默认\n"
                   "全局色调：全局色调映射，<0为模型默认\n"
                   "细节融合：0=原片细节，1=完整NR细节\n"
                   "色彩融合：0=保留原色相，1=NR色彩\n"
                   "UI校正：校正UI类画面元素\n"
                   "自动遮罩：自动识别需要处理的区域\n"
                   "HDR模式：请求后端启用HDR处理；节点仍以普通RGB图像格式返回结果\n"
                   "放大倍数：1-3倍超分倍率\n"
                   "显卡序号：多显卡时选择 GPU")

    def run(self, 图像, 渲染风格, 渲染预设, 渲染强度, 局部结构, 局部色调, 肤色增强, 全局色调,
            细节融合, 色彩融合, UI校正, 自动遮罩, HDR模式, 放大倍数, 输出宽度, 显卡序号):
        """逐张执行神经渲染"""
        exe = find_exe()
        nr = nr_args(渲染风格, 渲染预设, 渲染强度, 局部结构, 局部色调, 肤色增强, 全局色调,
                     细节融合, 色彩融合, UI校正, 自动遮罩, HDR模式)
        src = to_u8(图像)
        outs = [run_image_np(src[i], 输出宽度, 放大倍数, nr, exe, 显卡序号)
                for i in range(src.shape[0])]
        return (to_tensor(np.stack(outs)),)


class ST_DLSSNRVideo:
    """渲染视频(DLSS) - 整段视频经光流运动矢量处理，时序稳定

    可接入核心 Load Video 节点的 VIDEO 输出（自动继承音频与帧率），
    也可接入普通 IMAGE 帧批次（如 VideoHelperSuite），同时返回帧序列与封装好的 VIDEO。
    """
    DISPLAY_NAME = "渲染视频(DLSS)"

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        return {
            "required": {
                **_nr_inputs(),
                "运动矢量": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭",
                                       "tooltip": "启用光流运动矢量，保证帧间时序稳定"}),
                "运动引擎": (list(ENGINES), {"default": "自动",
                                            "tooltip": "光流后端：自动/NVOF英伟达光流/LK光流"}),
                "运动可视化": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭",
                                           "tooltip": "调试用：输出光流场可视化而非NR结果"}),
            },
            "optional": {
                "视频": ("VIDEO", {"tooltip": "接核心 Load Video 节点，音频和帧率自动继承到视频输出"}),
                "图像序列": ("IMAGE", {"tooltip": "帧批次（如 VideoHelperSuite 加载视频），未接视频时使用"}),
                "帧率": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1,
                                 "tooltip": "仅 IMAGE 输入时生效：给 VIDEO 输出标记帧率，"
                                            "不会增减帧数；接了 VIDEO 时忽略"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("图像序列", "视频")
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/DLSS渲染"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 存在该方法后核心跳过内置下拉校验，旧工作流键名在此统一兼容
        return _validate_nr_inputs(kwargs)

    DESCRIPTION = ("DLSS 超分辨率 + 神经渲染（视频模式），可启用光流运动矢量改善时序稳定性。\n"
                   "接入核心 Load Video（VIDEO）或 IMAGE 帧批次均可，"
                   "输出处理后的帧序列和封装好的 VIDEO（保留音频与帧率）。\n"
                   "运动矢量：用于改善帧间时序稳定性；运动引擎：自动/NVOF/LK。\n"
                   "渲染预设：配置一/二/三自动保存用户参数，切换时恢复，重启后可复用。\n"
                   "视频档位包含运动参数和帧率，与图像档位分别保存；默认预设恢复初始参数。\n"
                   "各档位保留原后端预设编号。参数修改后即保存，无需执行；读取工作流时保留工作流自身参数。")

    def run(self, 渲染风格, 渲染预设, 渲染强度, 局部结构, 局部色调, 肤色增强, 全局色调,
            细节融合, 色彩融合, UI校正, 自动遮罩, HDR模式, 放大倍数, 输出宽度, 显卡序号,
            运动矢量, 运动引擎, 运动可视化, 视频=None, 图像序列=None, 帧率=24):
        """流式处理整段视频"""
        audio, frame_rate = None, None
        if 视频 is not None:
            comps = 视频.get_components()
            图像序列, audio, frame_rate = comps.images, comps.audio, comps.frame_rate
        if 图像序列 is None:
            raise RuntimeError("请接入 VIDEO（核心 Load Video）或 IMAGE 帧批次（如 VHS 加载视频）。")
        if frame_rate is None:  # IMAGE 批次不带帧率，此处仅用于 VIDEO 输出标记
            frame_rate = Fraction(帧率).limit_denominator(1000)

        exe = find_exe()
        nr = nr_args(渲染风格, 渲染预设, 渲染强度, 局部结构, 局部色调, 肤色增强, 全局色调,
                     细节融合, 色彩融合, UI校正, 自动遮罩, HDR模式)
        src = to_u8(图像序列)
        total = src.shape[0]
        pbar = None
        try:
            from comfy.utils import ProgressBar  # 仅 ComfyUI 内存在
            pbar = ProgressBar(total)
        except Exception:
            pass

        def progress(done):
            if pbar is not None:
                pbar.update_absolute(min(done, total))

        out = run_video_np(src, 输出宽度, 放大倍数, nr, 运动矢量, ENGINES[_norm_engine(运动引擎)],
                           运动可视化, exe, 显卡序号, progress)
        out_t = to_tensor(out)
        return (out_t, make_video(out_t, audio, frame_rate))


def _zh_log_line(line):
    """工具 stderr 日志英译中：匹配常见固定格式行，未识别的原样保留（保留诊断价值）"""
    s = line.strip()
    style_names = {0: "默认", 1: "自然", 2: "电影"}

    # video: 1280x720 -> 1280x720, style 0, intensity 1.00, detail 1.00 (pipelined)
    m = re.match(r"^video:\s*(\d+)[x×](\d+)\s*(?:->|→)\s*(\d+)[x×](\d+),\s*"
                 r"style\s*(\d+),\s*intensity\s*([\d.]+),\s*detail\s*([\d.]+)"
                 r"(?:\s*\(([^)]*)\))?", s)
    if m:
        iw, ih, ow, oh, st, inten, det, tag = m.groups()
        tag_zh = "，流水线模式" if tag and "pipeline" in tag else ""
        return (f"视频: {iw}×{ih} → {ow}×{oh}，风格 {st}({style_names.get(int(st), st)})，"
                f"强度 {inten}，细节 {det}{tag_zh}")

    if s.startswith("feature ready"):
        return "模型特性就绪；流式处理中（读取+解包 | GPU推理 | 打包+写入）..."

    # NVOFA output grid: 1x1 (supported: 1,2,4)
    m = re.match(r"^NVOFA output grid:\s*(\d+)[x×](\d+)\s*\(supported:\s*([^)]*)\)", s)
    if m:
        gx, gy, sup = m.groups()
        return f"NVOFA 输出网格: {gx}×{gy}（支持: {sup}）"

    # optical flow: NVIDIA NVOFA (hardware)
    m = re.match(r"^optical flow:\s*(.*)", s)
    if m:
        engine = (m.group(1)
                  .replace("NVIDIA NVOFA (hardware)", "NVIDIA NVOFA（硬件光流）")
                  .replace("(hardware)", "（硬件）")
                  .replace("(software)", "（软件）"))
        return f"光流引擎: {engine}"

    # done: 2 frames in 0.1 s (32.5 fps)
    m = re.match(r"^done:\s*(\d+)\s*frames?\s*in\s*([\d.]+)\s*s(?:\s*\(([\d.]+)\s*fps\))?", s)
    if m:
        n, sec, fps = m.groups()
        return f"完成: {n} 帧，耗时 {sec} 秒" + (f"，{fps} fps" if fps else "")

    return s  # 未识别的行（警告/错误等）保留原文


class ST_DLSSNRRuntimeInfo:
    """渲染检测(DLSS) - 推送两帧测试数据走真实管线，报告本机 NR 是否可用"""
    DISPLAY_NAME = "渲染检测(DLSS)"

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        return {"required": {"显卡序号": ("INT", {"default": 0, "min": 0, "max": 15})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("检测信息",)
    FUNCTION = "info"
    CATEGORY = "🎯 石头工具/DLSS渲染"
    OUTPUT_NODE = True
    DESCRIPTION = "跑一个微型测试任务，报告本机神经渲染是否可用及当前使用的光流后端。"

    def info(self, 显卡序号):
        """执行 2 帧 1280x720 原生分辨率测试并收集日志"""
        lines = []
        try:
            exe = find_exe()
        except RuntimeError as e:
            return (str(e),)
        lines.append(f"程序路径: {exe}")
        try:
            if torch.cuda.is_available():
                lines.append(f"显卡设备: {torch.cuda.get_device_name(0)}")
            else:
                lines.append("显卡设备: CUDA 不可用")
        except Exception:
            pass

        # 真实 2 帧任务：1280x720、原生分辨率（仅NR）、开运动矢量
        h, w = 720, 1280
        yy, xx = np.mgrid[0:h, 0:w]
        f = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) % 256], -1).astype(np.uint8)
        frames = np.stack([f, f])
        nr = nr_args("默认", "默认预设（推荐）", 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, False, False, False)
        log = []
        t0 = time.time()
        try:
            out = run_video_np(frames, 0, 1.0, nr, True, "auto", False, exe, 显卡序号, log=log)
            lines.append(f"神经渲染: 正常 ✓（{w}×{h}，2帧，耗时 {time.time() - t0:.1f} 秒，"
                         f"输出 {out.shape[2]}×{out.shape[1]}）")
        except RuntimeError as e:
            lines.append("神经渲染: 失败 ✗")
            lines.append(str(e))
        if log:
            lines.append("")
            lines.append("【工具运行日志】")
            lines.extend(_zh_log_line(x) for x in log)
        return ("\n".join(lines)[-4000:],)


# ----------------------------------------------------------------------------- 用户预设存储与原生控件联动

_PROFILE_FILE = os.path.join(_TOOL_ROOT, "config", "dlssnr_presets.json")
_PROFILE_LOCK = threading.Lock()
_PROFILE_SCRIPT_PATH = "/st-tools/dlssnr-presets.js"
_PROFILE_SCRIPT_VERSION = "20261105-2"


def _profile_specs(kind):
    """只允许保存本节点的普通控件，不保存图像、视频和预设选择本身。"""
    classes = {"image": ST_DLSSNRImage, "video": ST_DLSSNRVideo}
    if kind not in classes:
        raise ValueError("预设节点类型无效")
    specs = {}
    for group in classes[kind].INPUT_TYPES().values():
        for name, spec in group.items():
            if name != "渲染预设" and len(spec) > 1 and "default" in spec[1]:
                specs[name] = spec
    return specs


def _profile_values(kind, values):
    """按节点输入定义检查并补全配置，拒绝无效的类型和数值。"""
    if not isinstance(values, dict):
        raise ValueError("预设参数必须为对象")
    result = {}
    for name, (input_type, options) in _profile_specs(kind).items():
        value = values.get(name, options["default"])
        if isinstance(input_type, list):
            valid = value in input_type
        elif input_type == "BOOLEAN":
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if valid and input_type == "INT":
                valid = int(value) == value
            if valid:
                valid = options.get("min", value) <= value <= options.get("max", value)
        if not valid:
            raise ValueError(f"预设参数无效：{name}")
        result[name] = value
    return result


def _read_profiles():
    """调用者持有文件锁；损坏的文件报错，不静默覆盖用户配置。"""
    if not os.path.isfile(_PROFILE_FILE):
        return {"image": {}, "video": {}}
    with open(_PROFILE_FILE, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or any(not isinstance(data.get(kind, {}), dict) for kind in ("image", "video")):
        raise ValueError("预设配置文件格式无效")
    return data


@PromptServer.instance.routes.get("/st-tools/dlssnr-presets")
async def _get_profiles(request):
    try:
        with _PROFILE_LOCK:
            data = _read_profiles()
        defaults = {kind: _profile_values(kind, {}) for kind in ("image", "video")}
        profiles = {
            kind: {name: _profile_values(kind, data.get(kind, {}).get(name, {}))
                   for name in PRESETS if PRESETS[name] != 0}
            for kind in defaults
        }
        return web.json_response({"defaults": defaults, "profiles": profiles}, headers={"Cache-Control": "no-store"})
    except (OSError, ValueError) as error:
        return web.json_response({"error": f"读取渲染预设失败：{error}"}, status=500)


@PromptServer.instance.routes.post("/st-tools/dlssnr-presets")
async def _save_profile(request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("预设请求格式无效")
        kind = payload.get("kind")
        preset = _norm_preset(payload.get("preset"))
        if preset not in PRESETS or PRESETS[preset] == 0:
            raise ValueError("只能保存预设配置一、二、三")
        values = _profile_values(kind, payload.get("values"))
    except (ValueError, TypeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    try:
        with _PROFILE_LOCK:
            data = _read_profiles()
            data.setdefault(kind, {})[preset] = values
            directory = os.path.dirname(_PROFILE_FILE)
            os.makedirs(directory, exist_ok=True)
            # 同目录写入后原子替换，避免中途退出留下半份配置。
            descriptor, temporary = tempfile.mkstemp(prefix="dlssnr_presets_", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, _PROFILE_FILE)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        return web.json_response({"ok": True})
    except (OSError, ValueError) as error:
        return web.json_response({"error": f"保存渲染预设失败：{error}"}, status=500)


_PROFILE_SCRIPT = r'''
import { app } from "../scripts/app.js";
import { api } from "../scripts/api.js";

const kinds = {ST_DLSSNRImage: "image", ST_DLSSNRVideo: "video"};
const defaultPreset = "默认预设（推荐）";
const customPresets = ["预设配置一", "预设配置二", "预设配置三"];
const aliases = {
    "默认": defaultPreset, "Default": defaultPreset,
    "默认模型(推荐)": defaultPreset, "默认模型（推荐）": defaultPreset,
    "默认预设模型(推荐)": defaultPreset, "默认预设模型（推荐）": defaultPreset,
    "默认预设(推荐)": defaultPreset,
    "预设1": customPresets[0], "Preset 1": customPresets[0],
    "预设2": customPresets[1], "Preset 2": customPresets[1],
    "预设3": customPresets[2], "Preset 3": customPresets[2],
};
const normalize = value => aliases[value] || ([0, 1, 2, 3].includes(Number(value)) ? [defaultPreset, ...customPresets][Number(value)] : value);
let storage;
let writes = Promise.resolve();
let ready;

function report(error) {
    console.error("石头工具：渲染预设保存或读取失败", error);
    app.ui?.dialog?.show(`渲染预设操作失败：${error.message}`);
}

async function request(path, options) {
    const response = await api.fetchApi(path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "预设接口请求失败");
    return result;
}

function loadStorage() {
    ready ??= request("/st-tools/dlssnr-presets", {cache: "no-store"}).then(data => {storage = data;});
    return ready;
}

function connected(node, widget) {
    return node.inputs?.some(input => (input.widget?.name === widget.name || input.name === widget.name) && input.link != null);
}

function capture(node) {
    const values = {};
    for (const widget of node.widgets || []) {
        if (widget.name in storage.defaults[node._stDLSSKind] && !connected(node, widget)) values[widget.name] = widget.value;
    }
    return values;
}

function save(node) {
    clearTimeout(node._stDLSSTimer);
    node._stDLSSTimer = null;
    const preset = node._stDLSSActive;
    if (!customPresets.includes(preset)) return;
    const kind = node._stDLSSKind;
    const values = {...storage.defaults[kind], ...capture(node)};
    storage.profiles[kind][preset] = values;
    // 逐次提交，确保快速调节或连续切换时，较早请求不会覆盖较新的值。
    writes = writes.then(() => request("/st-tools/dlssnr-presets", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({kind, preset, values}), keepalive: true,
    })).catch(report);
}

function apply(node, values) {
    node._stDLSSApplying = true;
    try {
        for (const widget of node.widgets || []) {
            if (widget.name in values && !connected(node, widget)) {
                widget.value = values[widget.name];
                widget.callback?.(widget.value);
            }
        }
    } finally {
        node._stDLSSApplying = false;
    }
    node.setDirtyCanvas(true, true);
    node.graph?.change();
}

function changed(node, name, value) {
    if (!storage || node._stDLSSApplying || node._stDLSSConfiguring) return;
    if (name === "渲染预设") {
        const selected = normalize(value);
        if (selected === node._stDLSSActive) return;
        save(node);
        node._stDLSSActive = selected;
        const kind = node._stDLSSKind;
        const values = selected === defaultPreset ? storage.defaults[kind] : storage.profiles[kind][selected];
        if (values) apply(node, values);
    } else if (name in storage.defaults[node._stDLSSKind]) {
        // 数值拖动做短暂合并，切换档位时立即提交，离开页面时也会补交。
        clearTimeout(node._stDLSSTimer);
        if (customPresets.includes(node._stDLSSActive)) node._stDLSSTimer = setTimeout(() => save(node), 200);
    }
}

function install(node) {
    if (node._stDLSSInstalled) return;
    const kind = kinds[node.comfyClass];
    const preset = node.widgets?.find(widget => widget.name === "渲染预设");
    if (!kind || !preset) return;
    node._stDLSSInstalled = true;
    node._stDLSSKind = kind;
    node._stDLSSActive = normalize(preset.value);
    for (const widget of node.widgets) {
        if (widget.type === "toggle") Object.assign(widget.options, {on: "开启", off: "关闭"});
        if (widget.name !== "渲染预设" && !(widget.name in storage.defaults[kind])) continue;
        // 监听原生控件数值赋值，兼容箭头、直接输入及新版前端的更新路径。
        let owner = widget;
        let descriptor;
        while (owner && !descriptor) {
            descriptor = Object.getOwnPropertyDescriptor(owner, "value");
            owner = Object.getPrototypeOf(owner);
        }
        let current = widget.value;
        Object.defineProperty(widget, "value", {
            configurable: true, enumerable: true,
            get() {return descriptor?.get ? descriptor.get.call(this) : current;},
            set(value) {
                const previous = this.value;
                if (descriptor?.set) descriptor.set.call(this, value);
                else current = value;
                if (previous !== value) changed(node, widget.name, value);
            },
        });
        const callback = widget.callback;
        widget.callback = function(value, ...args) {
            callback?.call(this, value, ...args);
            changed(node, widget.name, value);
        };
    }
}

app.registerExtension({
    name: "ST_tools.DLSSPresets",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!kinds[nodeData.name]) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            clearTimeout(this._stDLSSTimer);
            this._stDLSSConfiguring = true;
            try {return configure.call(this, info);}
            finally {
                this._stDLSSConfiguring = false;
                this._stDLSSActive = normalize(this.widgets?.find(w => w.name === "渲染预设")?.value);
            }
        };
        const widgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(name, value) {
            const result = widgetChanged?.apply(this, arguments);
            if (this._stDLSSInstalled) changed(this, name, value);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            if (this._stDLSSInstalled && this._stDLSSTimer) save(this);
            return removed?.apply(this, arguments);
        };
    },
    async nodeCreated(node) {
        if (!kinds[node.comfyClass]) return;
        try {await loadStorage(); install(node);}
        catch (error) {report(error);}
    },
    async setup() {
        try {await loadStorage();}
        catch (error) {report(error);}
        window.addEventListener("pagehide", () => {
            for (const node of app.graph?._nodes || []) {
                if (node._stDLSSInstalled && node._stDLSSTimer) save(node);
            }
        });
    },
});
'''


@PromptServer.instance.routes.get(_PROFILE_SCRIPT_PATH)
async def _profile_script(request):
    return web.Response(text=_PROFILE_SCRIPT, content_type="application/javascript", headers={"Cache-Control": "no-store"})


@web.middleware
async def _profile_extension(request, handler):
    response = await handler(request)
    if request.path.rstrip("/") in ("/extensions", "/api/extensions") and response.status == 200:
        extensions = json.loads(response.body)
        script_url = f"{_PROFILE_SCRIPT_PATH}?v={_PROFILE_SCRIPT_VERSION}"
        if script_url not in extensions:
            extensions.append(script_url)
        return web.json_response(extensions)
    return response


PromptServer.instance.app.middlewares.append(_profile_extension)
