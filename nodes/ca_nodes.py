import torch
import torch.nn.functional as F
import numpy as np

from .. import st_log


class ST_ChromaticAberration:
    """生成平均或径向色散，再按强度与原图混合。"""
    DISPLAY_NAME = "色散效果"
    FRINGE_PALETTES = {
        "淡暖黄 / 灰蓝": ((0.76, 0.66, 0.43), (0.40, 0.54, 0.72)),
        "低饱和青 / 暖橙": ((0.33, 0.65, 0.64), (0.78, 0.52, 0.33)),
        "灰蓝 / 淡紫": ((0.40, 0.56, 0.75), (0.69, 0.49, 0.75)),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入单张图像或一批图像，输出尺寸与输入一致。"}),
                "色散模式": (["平均色散", "径向色散"], {"default": "平均色散", "tooltip": "平均色散：全图红、蓝通道等量反向平移，径向色散：中心偏移小，越往外围偏移越大。两种模式都用 X/Y 和角度控制，X/Y 均为 0 时输出原图。"}),
                "色散强度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "控制彩边的颜色差异。0 显示原图，0.5 显示标准色散，1 将颜色变化放大至 2 倍。改变强度不会改变通道偏移量。 最小值 0.0，最大值 1.0，默认值 0.5，步长 0.01。"}),
                "偏移X轴": ("FLOAT", {"default": 0.15, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "旋转前的水平位移。红通道正值向右、负值向左，蓝通道反向。每 1.0 对应短边的 0.5%，两种模式均有效。 最小值 -1.0，最大值 1.0，默认值 0.15，步长 0.01。"}),
                "偏移Y轴": ("FLOAT", {"default": 0.15, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "旋转前的垂直位移。红通道正值向上、负值向下，蓝通道反向。每 1.0 对应短边的 0.5%，两种模式均有效。 最小值 -1.0，最大值 1.0，默认值 0.15，步长 0.01。"}),
                "偏移角度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0, "tooltip": "将合成位移逆时针旋转。0° 保持原方向，90° 把向右转为向上。两种模式都旋转由 X/Y 决定的偏移方向。 最小值 0.0°，最大值 360.0°，默认值 0.0°，步长 1.0°。"}),
            },
            "optional": {
                "色散色系": (["原始红蓝", *cls.FRINGE_PALETTES], {
                    "default": "原始红蓝",
                    "tooltip": "选择彩边配色。原始红蓝保留原效果，其余色系会柔化彩边，着色幅度为基础值的 4 倍，并尽量保持原图明暗。两种色散模式均可使用。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "🎯 石头工具/图像编辑"
    DESCRIPTION = """给画面中的物体轮廓加上彩边。

先选模式
• 平均色散：整张图的红、蓝通道按同一距离反向移动。
• 径向色散：中心偏移最小，越靠近外围，偏移越大。除了 X/Y 指定的方向，红通道还会向外、蓝通道向内。

再调方向和距离
两种模式在 X、Y 都为 0 时都输出原图。角度为 0° 时，X 正右负左，Y 正上负下，蓝通道与红通道反向。角度逆时针旋转 X/Y 合成的方向，例如正 X 转 90° 后朝上。
例如短边为 1000 像素，平均色散中 X=1、Y=0、角度为 0° 时，红、蓝通道各水平移动 5 像素。

选择彩边颜色
原始红蓝保留原来的通道错位效果。淡暖黄 / 灰蓝适合轻微冷暖彩边，低饱和青 / 暖橙的冷暖区别更明显，灰蓝 / 淡紫整体偏冷。柔和色系只柔化彩边并保持原图明暗，着色幅度为基础着色幅度的 4 倍。

最后调明显程度
强度 0 输出原图，0.5 显示标准色散，1 将颜色变化放大至 2 倍，输出颜色限制在可显示范围内。强度不改变采样位置。"""

    def run(self, 图像, 色散模式, 色散强度, 偏移X轴, 偏移Y轴, 偏移角度, 色散色系="原始红蓝"):
        if 色散强度 <= 0.0 or (偏移X轴 == 0 and 偏移Y轴 == 0):
            return (图像,)
        try:
            out_list = []
            for img in 图像:
                out_list.append(self._apply_single(img, 色散模式, 偏移X轴, 偏移Y轴, 偏移角度))
            dispersed = torch.stack(out_list, dim=0)
            gain = 2.0 * 色散强度
            palette = self.FRINGE_PALETTES.get(色散色系)
            if palette is not None:
                return (self._soft_fringe(图像, dispersed, palette, 4.0 * gain),)
            return (torch.lerp(图像, dispersed, gain).clamp_(0.0, 1.0),)
        except Exception as e:
            st_log(f"色散效果出错: {e}")
            return (图像,)

    def _soft_fringe(self, image, dispersed, palette, gain):
        source = image.float()
        difference = dispersed.float() - source
        # 从红蓝位移差提取彩边，只平滑这一层。
        fringe = (difference[..., 0] - difference[..., 2]).unsqueeze(1) * 0.5
        fringe = F.avg_pool2d(fringe, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        fringe = fringe.permute(0, 2, 3, 1)

        colors = source.new_tensor(palette)
        luma = source.new_tensor([0.2126, 0.7152, 0.0722])
        chroma = colors - (colors * luma).sum(dim=-1, keepdim=True)
        tint = (fringe.clamp_min(0) * chroma[0] + (-fringe).clamp_min(0) * chroma[1]) * gain

        # 统一缩小三个通道的增量，避免单独裁切破坏明暗。
        room = torch.where(tint > 0, 1.0 - source, source)
        limit = torch.where(tint != 0, room / tint.abs().clamp_min(1e-8), 1.0)
        scale = limit.amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
        return (source + tint * scale).clamp_(0.0, 1.0).to(image.dtype)

    def _apply_single(self, img, mode, offset_x, offset_y, angle_deg):
        t = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        H, W = t.shape[-2:]
        # align_corners=False 时，网格从首个像素中心开始。
        y_base = (torch.arange(H, device=t.device, dtype=torch.float32) + 0.5) * (2.0 / H) - 1.0
        x_base = (torch.arange(W, device=t.device, dtype=torch.float32) + 0.5) * (2.0 / W) - 1.0
        grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing='ij')
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        short_side = min(H, W)
        axis_pixels = short_side * 0.005
        x = offset_x * axis_pixels
        y = offset_y * axis_pixels

        angle_rad = np.deg2rad(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        # 在 Y 向上的坐标系旋转，再转换为 Y 向下的采样坐标。
        shift = base.new_tensor([x * cos_angle - y * sin_angle,
                                 -(x * sin_angle + y * cos_angle)])
        if mode == "径向色散":
            radial_pixels = short_side * 0.0015 * float(np.hypot(offset_x, offset_y))
            shift = self._radial_shift(base, H, W, shift, radial_pixels)
        shift = shift * base.new_tensor([2.0 / W, 2.0 / H])

        # 逆向采样：采样点往左，画面中的通道才向右移动。
        red = self._sample_channel(t[:, 0:1], base - shift)
        blue = self._sample_channel(t[:, 2:3], base + shift)
        return torch.cat([red, t[:, 1:2], blue], dim=1).squeeze(0).permute(1, 2, 0).contiguous()

    def _radial_shift(self, base, H, W, shift, pixel):
        # 以画面角点距离归一化半径，让位移从中心到外围持续增加。
        half_size = base.new_tensor([W / 2.0, H / 2.0])
        radial = base * half_size / float(np.hypot(W / 2.0, H / 2.0))
        radius = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
        return shift * radius + radial * pixel

    def _sample_channel(self, channel, grid):
        return F.grid_sample(
            channel.float(), grid,
            mode='bilinear',
            padding_mode="border",
            align_corners=False
        ).to(channel.dtype)
