import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.interpolate import CubicSpline
import cv2
import folder_paths
import random
import os

class ST_ImagePostProcessing:
    """图像调色节点 - 调整图像的亮度、对比度、饱和度、Gamma、色彩平衡、模糊/锐化、HDR效果和自适应画质增强"""
    DISPLAY_NAME = "图像调色"
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        return {
            "required": {
                "图像": ("IMAGE",),
                "亮度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "对比度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "饱和度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Gamma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "红色_青色": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "绿色_洋红": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "蓝色_黄色": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "模糊_锐化": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "HDR强度": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
                "自适应画质增强": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = "说明：所有参数范围为0.0-2.0，1.0为默认值（不改画面），步数为0.01\n\n图像：输入的图像\n亮度：调整图像明暗程度\n对比度：调整图像明暗对比\n饱和度：调整图像色彩鲜艳程度\nGamma：调整图像亮度曲线\n红色_青色：调整红色和青色平衡\n绿色_洋红：调整绿色和洋红平衡\n蓝色_黄色：调整蓝色和黄色平衡\n模糊_锐化：值<1为模糊，值>1为锐化\nHDR强度：增强局部对比度，保留更多细节（1.0保持原图，1-2增加HDR强度\n自适应画质增强：增强图像边缘和纹理细节"

    def run(self, 图像, 亮度, 对比度, 饱和度, Gamma, 红色_青色, 绿色_洋红, 蓝色_黄色, 模糊_锐化, HDR强度, 自适应画质增强, unique_id, save_preview=True, return_ui=True):
        """执行图像处理操作"""
        try:
            # 处理批量图像
            if len(图像) > 1:
                tensors = []
                for img in 图像:
                    processed_img = self._process_single_image(
                        img, 亮度, 对比度, 饱和度, Gamma, 
                        红色_青色, 绿色_洋红, 蓝色_黄色, 
                        模糊_锐化, HDR强度, 自适应画质增强
                    )
                    tensors.append(processed_img)
                tensors = torch.cat(tensors, dim=0)
            else:
                # 处理单个图像
                img = 图像
                tensors = self._process_single_image(
                    img, 亮度, 对比度, 饱和度, Gamma, 
                    红色_青色, 绿色_洋红, 蓝色_黄色, 
                    模糊_锐化, HDR强度, 自适应画质增强
                )

            # 处理返回值
            if not save_preview and not return_ui:
                return tensors

            preview_tensor = tensors
            if isinstance(preview_tensor, torch.Tensor) and preview_tensor.dim() == 3:
                preview_tensor = preview_tensor.unsqueeze(0)

            results = []
            if save_preview and isinstance(preview_tensor, torch.Tensor) and preview_tensor.dim() == 4 and preview_tensor.shape[0] > 0:
                temp_dir = folder_paths.get_temp_directory()
                t = preview_tensor[0]
                img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                filename = f"hsl_preview_{random.randint(1, 1000000)}.png"
                img_pil.save(os.path.join(temp_dir, filename))
                results.append({"filename": filename, "subfolder": "", "type": "temp"})

            if return_ui:
                return {"ui": {"bg_image": results}, "result": (tensors,)}
            return tensors
        except Exception as e:
            # 错误处理
            print(f"处理图像时出错: {e}")
            # 返回原始图像
            return 图像

    def _process_single_image(self, img, 亮度, 对比度, 饱和度, Gamma, 红色_青色, 绿色_洋红, 蓝色_黄色, 模糊_锐化, HDR强度, 自适应画质增强):
        """处理单个图像"""
        pil_image = None
        
        # 亮度调整
        if 亮度 != 1.0:
            img = np.clip(img * 亮度, 0.0, 1.0)
        
        # 对比度调整
        if 对比度 != 1.0:
            midpoint = 0.5
            img = np.clip((img - midpoint) * 对比度 + midpoint, 0.0, 1.0)
        
        # Gamma调整
        if Gamma != 1.0:
            pil_image = self.tensor2pil(img)
            pil_image = self.gamma_trans(pil_image, Gamma)
        
        # 色彩平衡调整
        if 红色_青色 != 1.0 or 绿色_洋红 != 1.0 or 蓝色_黄色 != 1.0:
            pil_image = pil_image if pil_image else self.tensor2pil(img)
            cr = 红色_青色 - 1.0
            mg = 绿色_洋红 - 1.0
            yb = 蓝色_黄色 - 1.0
            pil_image = self.color_balance(pil_image, cr, mg, yb)
        
        # 饱和度调整
        if 饱和度 != 1.0:
            pil_image = pil_image if pil_image else self.tensor2pil(img)
            pil_image = ImageEnhance.Color(pil_image).enhance(饱和度)
        
        # 模糊/锐化调整
        if 模糊_锐化 != 1.0:
            pil_image = pil_image if pil_image else self.tensor2pil(img)
            if 模糊_锐化 > 1.0:
                # 锐化处理
                sharpness_factor = 1.0 + (模糊_锐化 - 1.0) * 3
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness_factor)
            else:
                # 模糊处理
                blur_radius = (1.0 - 模糊_锐化) * 10
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # HDR效果
        if HDR强度 > 1.0:
            pil_image = pil_image if pil_image else self.tensor2pil(img)
            pil_image = self._apply_hdr_effect(pil_image, HDR强度)
        
        # 自适应画质增强
        if 自适应画质增强:
            pil_image = pil_image if pil_image else self.tensor2pil(img)
            pil_image = pil_image.filter(ImageFilter.DETAIL)
        
        # 转换回张量
        out_image = (self.pil2tensor(pil_image) if pil_image else img)
        return out_image

    def _apply_hdr_effect(self, pil_image, hdr_strength):
        """应用HDR效果"""
        from scipy.ndimage import gaussian_filter
        
        img_array = np.array(pil_image) / 255.0
        local_mean = gaussian_filter(img_array, sigma=3)
        
        # 增强局部对比度
        hdr_factor = (hdr_strength - 1.0) * 0.5
        enhanced = local_mean + (img_array - local_mean) * (1.0 + hdr_factor * 1.0)
        
        # 轻微降低饱和度，避免色彩过于鲜艳
        saturation_factor = 1.0 - hdr_factor * 0.1
        gray = np.mean(enhanced, axis=2, keepdims=True)
        enhanced = enhanced * saturation_factor + gray * (1 - saturation_factor)
        
        # 转换回PIL图像
        enhanced = np.clip(enhanced, 0.0, 1.0) * 255.0
        enhanced = enhanced.astype(np.uint8)
        return Image.fromarray(enhanced)

    def pil2tensor(self, image):
        """将PIL图像转换为张量"""
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:
            np_image = np_image[None, None, ...]
        elif np_image.ndim == 3:
            np_image = np_image[None, ...]
        return torch.from_numpy(np_image)

    def tensor2pil(self, image):
        """将张量转换为PIL图像"""
        try:
            # 使用更安全的转换方式
            img_array = image.cpu().numpy().squeeze()
            # 确保值在有效范围内
            img_array = np.clip(img_array, 0, 1)
            img_array = (255. * img_array).astype(np.uint8)
            return Image.fromarray(img_array)
        except Exception as e:
            print(f"[ST_tools] 图像转换警告：在将张量转换为PIL图像时遇到问题 - {str(e)}")
            print("[ST_tools] 解决建议：检查输入图像格式及节点是否被链接或启用。")
            # 尝试使用更安全的方式转换
            try:
                img_array = image.cpu().numpy().squeeze()
                # 确保值在有效范围内
                img_array = np.clip(img_array, 0, 1)
                img_array = (255. * img_array).astype(np.uint8)
                return Image.fromarray(img_array)
            except Exception as e2:
                print(f"[ST_tools] 图像转换失败：{str(e2)}")
                print("[ST_tools] 解决建议：检查输入图像格式及节点是否被链接或启用。")
                # 返回一个空图像作为 fallback
                return Image.new('RGB', (100, 100), color='black')

    def adjust_saturation(self, img_array, factor):
        """调整图像饱和度"""
        # 将RGB转换为HSL
        # 计算亮度
        gray = np.mean(img_array, axis=2, keepdims=True)
        
        # 应用饱和度调整
        adjusted = gray + factor * (img_array - gray)
        
        # 确保值在有效范围内
        adjusted = np.clip(adjusted, 0.0, 1.0)
        
        return adjusted

    def gamma_trans(self, image, gamma):
        """进行Gamma校正"""
        cv2_image = self.pil2cv2(image)
        # 标准Gamma校正公式：输出 = (输入/255)^(1/gamma) * 255
        # 当gamma < 1时，图像变亮；当gamma > 1时，图像变暗
        inv_gamma = 1.0 / gamma
        gamma_table = [np.power(x/255.0, inv_gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        _corrected = cv2.LUT(cv2_image, gamma_table)
        return self.cv22pil(_corrected)

    def pil2cv2(self, pil_image):
        """将PIL图像转换为OpenCV格式"""
        np_img_array = np.asarray(pil_image)
        return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

    def cv22pil(self, cv2_img):
        """将OpenCV图像转换为PIL格式"""
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_img)

    def color_balance(self, image, cyan_red, magenta_green, yellow_blue):
        """调整色彩平衡"""
        # 将PIL图像转换为tensor
        img = self.pil2tensor(image)
        img_copy = img.clone()
        
        # 计算原始亮度（如果需要保持亮度）
        original_luminance = 0.2126 * img_copy[0, ..., 0] + 0.7152 * img_copy[0, ..., 1] + 0.0722 * img_copy[0, ..., 2]
        
        # 定义调整曲线
        def adjust(x, center, value, max_adjustment):
            # 缩放调整值
            value = value * max_adjustment
            
            # 定义控制点
            points = torch.tensor([[0, 0], [center, center + value], [1, 1]])
            
            # 创建三次样条曲线
            cs = CubicSpline(points[:, 0], points[:, 1])
            
            # 应用三次样条曲线到颜色通道
            return torch.clamp(torch.from_numpy(cs(x)), 0, 1)
        
        # 应用调整到每个颜色通道
        # 青色/红色调整红色通道，洋红/绿色调整绿色通道，黄色/蓝色调整蓝色通道
        adjustments = [-cyan_red, -magenta_green, -yellow_blue]
        for i, adjustment in enumerate(adjustments):
            img_copy[0, ..., i] = adjust(img_copy[0, ..., i], 0.15, adjustment, 0.1)  # 阴影
            img_copy[0, ..., i] = adjust(img_copy[0, ..., i], 0.5, adjustment, 1.0)   # 中间调
            img_copy[0, ..., i] = adjust(img_copy[0, ..., i], 0.8, adjustment, 0.2)   # 高光
        
        # 保持亮度
        current_luminance = 0.2126 * img_copy[0, ..., 0] + 0.7152 * img_copy[0, ..., 1] + 0.0722 * img_copy[0, ..., 2]
        img_copy[0] *= (original_luminance / current_luminance).unsqueeze(-1)
        
        # 将tensor转换回PIL图像
        return self.tensor2pil(img_copy[0])