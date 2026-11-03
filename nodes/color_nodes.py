import torch
import numpy as np
from PIL import Image
import os
import time

from .. import st_log

class ST_FilterShader:
    """图像滤镜节点 - 加载和应用.cube格式的滤镜文件，支持深度叠加和柔光混合调色模式"""
    DISPLAY_NAME = "图像滤镜"
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        # 默认滤镜文件路径
        default_shader_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Shader")
        shader_files = []
        
        # 如果默认路径存在，获取所有滤镜文件
        if os.path.exists(default_shader_path):
            shader_files = [f for f in os.listdir(default_shader_path) if f.endswith('.cube')]
        
        color_space_list = ['深度叠加', '柔光混合']
        
        return {
            "required": {
                "图像": ("IMAGE",),
                "加载滤镜": (shader_files, {"default": shader_files[0] if shader_files else ""}),
                "调色模式": (color_space_list, {"default": "深度叠加"}),
                "滤镜强度": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制每次重算（滤镜预览场景需要）"""
        return str(time.time())

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = '图像滤镜节点\n\n滤镜文件来自 Shader 目录下的 .cube 文件，列表随目录自动更新。\n\n调色模式：\n- 深度叠加：LUT 直接叠加到原片\n- 柔光混合：对数空间应用 LUT，暗部过渡更自然\n\n滤镜强度：1.0 = 原片不变，2.0 = 滤镜效果最强（混入 30% LUT），推荐 1.2-1.5'

    def run(self, **kwargs):
        """执行滤镜应用操作"""
        try:
            image = kwargs["图像"]
            shader_file = kwargs["加载滤镜"]
            color_space = kwargs.get("调色模式", "深度叠加")
            
            # 处理旧的参数值
            if color_space in ['线性', 'linear']:
                color_space = '深度叠加'
            elif color_space in ['对数', 'log']:
                color_space = '柔光混合'
            
            intensity = kwargs["滤镜强度"]
            
            # 转换强度范围：1.0-2.0 映射到 0.0-0.3
            blend_factor = max(0.0, min(0.3, (intensity - 1.0) * 0.3))
            
            # 构建滤镜文件路径
            default_shader_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Shader")
            shader_full_path = os.path.join(default_shader_path, shader_file)
            
            # 加载滤镜文件
            lut = self._load_cube_lut(shader_full_path)
            
            # 应用滤镜
            result = self._apply_lut(image, lut, color_space, blend_factor)
            
            return (result,)
        except Exception as e:
            # 错误处理
            st_log(f"应用滤镜时出错: {e}")
            # 返回原始图像
            return (kwargs["图像"],)

    def _load_cube_lut(self, lut_path):
        """加载Iridas Cube格式的滤镜文件"""
        domain_min, domain_max = np.array([0, 0, 0]), np.array([1, 1, 1])
        dimensions = 3
        size = 2
        data = []
        
        with open(lut_path, encoding='utf-8') as cube_file:
            lines = cube_file.readlines()
            for line in lines:
                line = line.strip()
                
                if len(line) == 0 or line.startswith("#"):
                    continue
                
                tokens = line.split()
                if tokens[0] == "TITLE":
                    continue
                elif tokens[0] == "DOMAIN_MIN":
                    domain_min = np.array([float(x) for x in tokens[1:]])
                elif tokens[0] == "DOMAIN_MAX":
                    domain_max = np.array([float(x) for x in tokens[1:]])
                elif tokens[0] == "LUT_1D_SIZE":
                    dimensions = 2
                    size = int(tokens[1])
                elif tokens[0] == "LUT_3D_SIZE":
                    dimensions = 3
                    size = int(tokens[1])
                else:
                    data.append([float(x) for x in tokens])
        
        table = np.array(data)
        
        if dimensions == 3:
            # 3D LUT数据重塑
            table = table.reshape([size, size, size, 3], order="F")
        
        return {
            'table': table,
            'domain': np.vstack([domain_min, domain_max]),
            'dimensions': dimensions,
            'size': size
        }

    def _apply_lut(self, image, lut, color_space, blend_factor):
        """应用滤镜到图像"""
        batch_size, height, width, channels = image.shape
        result = []
        
        for i in range(batch_size):
            # 获取单张图像
            img = image[i]
            
            # 转换为numpy数组
            img_np = img.cpu().numpy().copy()
            original_img = img_np.copy()
            
            # 处理定义域
            is_non_default_domain = not np.array_equal(lut['domain'], np.array([[0., 0., 0.], [1., 1., 1.]]))
            dom_scale = None
            if is_non_default_domain:
                dom_scale = lut['domain'][1] - lut['domain'][0]
                img_np = img_np * dom_scale + lut['domain'][0]
            
            # 调色模式处理
            if color_space == "柔光混合":
                img_np = np.log1p(img_np * 10) / np.log1p(10)
            
            # 应用LUT
            if lut['dimensions'] == 3:
                # 3D LUT
                lut_applied = self._apply_3d_lut(img_np, lut['table'])
            else:
                # 1D LUT
                lut_applied = self._apply_1d_lut(img_np, lut['table'])
            
            # 反向定义域处理
            if is_non_default_domain:
                lut_applied = (lut_applied - lut['domain'][0]) / dom_scale
            
            # 混合原始图像和滤镜应用后的图像
            blended = original_img * (1 - blend_factor) + lut_applied * blend_factor
            
            # 转换回tensor
            blended_tensor = torch.from_numpy(blended).to(image.device)
            result.append(blended_tensor)
        
        return torch.stack(result, dim=0)

    def _apply_3d_lut(self, img, lut_table):
        """应用3D LUT"""
        height, width, channels = img.shape
        lut_size = lut_table.shape[0]
        
        # 计算LUT索引
        r_idx = np.clip((img[:, :, 0] * (lut_size - 1)).astype(int), 0, lut_size - 1)
        g_idx = np.clip((img[:, :, 1] * (lut_size - 1)).astype(int), 0, lut_size - 1)
        b_idx = np.clip((img[:, :, 2] * (lut_size - 1)).astype(int), 0, lut_size - 1)
        
        # 应用LUT
        result = lut_table[r_idx, g_idx, b_idx]
        
        return result

    def _apply_1d_lut(self, img, lut_table):
        """应用1D LUT"""
        height, width, channels = img.shape
        lut_size = lut_table.shape[0]
        
        # 计算LUT索引
        r_idx = np.clip((img[:, :, 0] * (lut_size - 1)).astype(int), 0, lut_size - 1)
        g_idx = np.clip((img[:, :, 1] * (lut_size - 1)).astype(int), 0, lut_size - 1)
        b_idx = np.clip((img[:, :, 2] * (lut_size - 1)).astype(int), 0, lut_size - 1)
        
        # 应用LUT
        result = np.zeros_like(img)
        result[:, :, 0] = lut_table[r_idx, 0]
        result[:, :, 1] = lut_table[g_idx, 1]
        result[:, :, 2] = lut_table[b_idx, 2]
        
        return result