import torch
import numpy as np
from PIL import Image
from ..config.presets import PRESETS, get_size_from_preset

class ST_ImageMaskLatentSize:
    """图像尺寸节点 - 处理图像、遮罩和Latent的尺寸调整"""
    DISPLAY_NAME = "图像尺寸"
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        ratio_options = [name for name, size in PRESETS]
        
        return {
            "required": {
                "自定义尺寸": ("BOOLEAN", {"default": False}),
                "预设比例": (ratio_options, {"default": ratio_options[0] if ratio_options else "无可用比例"}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "缩放方式": (["中心裁剪", "等比缩放"], {"default": "中心裁剪"}),
                "缩放倍数": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT")
    RETURN_NAMES = ("图像", "遮罩", "latent", "宽度", "高度")
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具"
    DESCRIPTION = '处理优先级：\n 1. 尺寸设置优先级：自定义尺寸 > 预设尺寸 > 图像输出尺寸 > Latent 输入尺寸\n 2. 输入处理优先级：图像/遮罩输入 > Latent 输入\n处理方法：\n 中心裁剪：按目标比例裁剪中心调整大小\n 等比缩放：按目标比例乘以倍数等比缩放\n Latent：按目标比例调整大小（尺寸÷8）\n 遮罩不存在时：创建全白遮罩'

    def run(self, **kwargs):
        """执行尺寸调整操作"""
        try:
            # 获取输入参数
            use_custom = kwargs["自定义尺寸"]
            ratio_name = kwargs["预设比例"]
            width = kwargs["宽度"]
            height = kwargs["高度"]
            缩放方式 = kwargs["缩放方式"]
            缩放倍数 = kwargs["缩放倍数"]
            vae = kwargs.get("vae", None)
            
            # 检查输入类型
            has_image = "图像" in kwargs and kwargs["图像"] is not None
            has_latent = "latent" in kwargs and kwargs["latent"] is not None
            has_vae = vae is not None

            # 确定目标尺寸
            target_width, target_height = self._determine_target_size(kwargs, use_custom, ratio_name, width, height, has_image, has_latent)

            # 根据输入类型执行不同的处理
            if has_image:
                if has_vae:
                    # 有图像和VAE输入，使用VAE对图像进行编码
                    return self._process_image_with_vae(kwargs, target_width, target_height, 缩放方式, vae)
                else:
                    # 只有图像输入，使用基本处理方式
                    return self._process_image(kwargs, target_width, target_height, 缩放方式)
            elif has_latent:
                # 处理Latent输入
                return self._process_latent(kwargs, target_width, target_height, 缩放方式, 缩放倍数)
            else:
                # 无输入，仅返回尺寸
                latent = {
                    "samples": torch.zeros((1, 4, target_height // 8, target_width // 8)),
                    "downscale_ratio_spacial": 8
                }
                return (None, None, latent, target_width, target_height)
        except Exception as e:
            # 错误处理
            print(f"处理尺寸时出错: {e}")
            # 返回默认值
            latent = {
                "samples": torch.zeros((1, 4, 1024 // 8, 1024 // 8)),
                "downscale_ratio_spacial": 8
            }
            return (None, None, latent, 1024, 1024)

    def _determine_target_size(self, kwargs, use_custom, ratio_name, width, height, has_image, has_latent):
        """根据优先级确定目标尺寸"""
        if use_custom:
            # 优先使用自定义尺寸
            target_width = width
            target_height = height
        elif ratio_name != "关闭":
            # 其次使用预设尺寸
            target_width, target_height = get_size_from_preset(ratio_name)
        elif has_image:
            # 再次使用图像本身尺寸
            batch_size, img_height, img_width, channels = kwargs["图像"].shape
            target_width = img_width
            target_height = img_height
        elif has_latent:
            # 然后使用latent尺寸（乘以8，因为latent尺寸是实际尺寸的1/8）
            latent = kwargs["latent"]
            batch_size, channels, latent_height, latent_width = latent["samples"].shape
            target_width = latent_width * 8
            target_height = latent_height * 8
        else:
            # 最后使用默认尺寸
            target_width, target_height = 1024, 1024
        
        # 确保尺寸有效
        target_width = max(64, min(8192, target_width))
        target_height = max(64, min(8192, target_height))
        
        return target_width, target_height

    def _process_image(self, kwargs, target_width, target_height, 缩放方式):
        """处理图像和遮罩"""
        image = kwargs["图像"]
        mask = kwargs.get("遮罩", None)
        缩放倍数 = kwargs.get("缩放倍数", 1.0)

        batch_size, height1, width1, channels = image.shape
        new_images = []
        new_masks = []

        # 计算缩放后的尺寸
        if 缩放方式 == "等比缩放":
            # 等比缩放：以原始尺寸为基准
            scaled_width = int(width1 * 缩放倍数)
            scaled_height = int(height1 * 缩放倍数)
        else:
            # 中心裁剪：使用目标尺寸
            scaled_width = target_width
            scaled_height = target_height

        for i in range(batch_size):
            # 处理图像
            img = image[i]
            pil_img = Image.fromarray(np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8))
            
            # 执行缩放或裁剪
            if 缩放方式 == "中心裁剪":
                # 中心裁剪并调整大小
                pil_img = self._center_crop(pil_img, scaled_width, scaled_height, width1, height1)
                pil_img = pil_img.resize((scaled_width, scaled_height), Image.LANCZOS)
            elif 缩放方式 == "等比缩放":
                # 等比缩放
                pil_img = pil_img.resize((scaled_width, scaled_height), Image.LANCZOS)
            
            new_img = np.array(pil_img).astype(np.float32) / 255.0
            new_images.append(new_img)

            # 处理遮罩
            if mask is not None:
                # 处理现有遮罩
                m = mask[i]
                pil_mask = Image.fromarray(np.clip(255. * m.cpu().numpy(), 0, 255).astype(np.uint8))
                
                # 执行与图像相同的处理
                if 缩放方式 == "中心裁剪":
                    pil_mask = self._center_crop(pil_mask, scaled_width, scaled_height, width1, height1)
                    pil_mask = pil_mask.resize((scaled_width, scaled_height), Image.LANCZOS)
                elif 缩放方式 == "等比缩放":
                    pil_mask = pil_mask.resize((scaled_width, scaled_height), Image.LANCZOS)
                
                new_mask = np.array(pil_mask).astype(np.float32) / 255.0
            else:
                # 创建全白遮罩
                new_mask = np.ones((scaled_height, scaled_width), dtype=np.float32)
            
            new_masks.append(new_mask)

        # 转换为张量
        new_image = torch.tensor(np.stack(new_images, axis=0))
        new_mask = torch.tensor(np.stack(new_masks, axis=0))
        
        # 创建latent结构
        batch_size = new_image.shape[0]
        latent = {
            "samples": torch.zeros((batch_size, 4, scaled_height // 8, scaled_width // 8), device=new_image.device),
            "downscale_ratio_spacial": 8
        }
        
        return (new_image, new_mask, latent, scaled_width, scaled_height)

    def _process_image_with_vae(self, kwargs, target_width, target_height, 缩放方式, vae):
        """处理图像并使用VAE编码"""
        # 先处理图像
        new_image, new_mask, _, scaled_width, scaled_height = self._process_image(kwargs, target_width, target_height, 缩放方式)
        
        # 使用VAE编码
        encoded_latent = vae.encode(new_image)
        
        # 创建latent结构
        latent = {
            "samples": encoded_latent,
            "downscale_ratio_spacial": 8
        }
        
        return (new_image, new_mask, latent, scaled_width, scaled_height)

    def _process_latent(self, kwargs, target_width, target_height, 缩放方式, 缩放倍数):
        """处理Latent尺寸"""
        latent = kwargs["latent"]
        new_latent = {}
        
        if 缩放方式 == "等比缩放":
            # 等比缩放Latent
            batch_size, channels, latent_height, latent_width = latent["samples"].shape
            original_width = latent_width * 8
            original_height = latent_height * 8
            
            # 计算缩放后的尺寸
            scaled_width = int(original_width * 缩放倍数)
            scaled_height = int(original_height * 缩放倍数)
            
            # 调整latent尺寸
            new_latent["samples"] = torch.nn.functional.interpolate(
                latent["samples"],
                size=(scaled_height // 8, scaled_width // 8),
                mode="bilinear",
                align_corners=False
            )
            new_latent["downscale_ratio_spacial"] = 8
            
            return (None, None, new_latent, scaled_width, scaled_height)
        else:
            # 中心裁剪Latent
            batch_size, channels, latent_height, latent_width = latent["samples"].shape
            original_width = latent_width * 8
            original_height = latent_height * 8
            
            # 计算目标宽高比
            target_aspect_ratio = target_width / target_height
            original_aspect_ratio = original_width / original_height
            
            # 裁剪Latent
            if original_aspect_ratio > target_aspect_ratio:
                # 宽度过大，裁剪宽度
                new_width = int(original_height * target_aspect_ratio)
                new_width_latent = new_width // 8
                start_x = (latent_width - new_width_latent) // 2
                cropped_latent = latent["samples"][:, :, :, start_x:start_x + new_width_latent]
            else:
                # 高度过大，裁剪高度
                new_height = int(original_width / target_aspect_ratio)
                new_height_latent = new_height // 8
                start_y = (latent_height - new_height_latent) // 2
                cropped_latent = latent["samples"][:, :, start_y:start_y + new_height_latent, :]
            
            # 调整到目标尺寸
            new_latent["samples"] = torch.nn.functional.interpolate(
                cropped_latent,
                size=(target_height // 8, target_width // 8),
                mode="bilinear",
                align_corners=False
            )
            new_latent["downscale_ratio_spacial"] = 8
            
            return (None, None, new_latent, target_width, target_height)

    def _center_crop(self, img, target_width, target_height, original_width, original_height):
        """中心裁剪图像到目标比例"""
        aspect_ratio = target_width / target_height
        img_ratio = original_width / original_height
        
        if img_ratio > aspect_ratio:
            # 宽度过大，裁剪宽度
            new_width = int(original_height * aspect_ratio)
            left = (original_width - new_width) // 2
            return img.crop((left, 0, left + new_width, original_height))
        else:
            # 高度过大，裁剪高度
            new_height = int(original_width / aspect_ratio)
            top = (original_height - new_height) // 2
            return img.crop((0, top, original_width, top + new_height))

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ST_ImageMaskLatentSize": ST_ImageMaskLatentSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ST_ImageMaskLatentSize": "图像尺寸"
}