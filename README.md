<img width="1917" height="964" alt="文生图像" src="https://github.com/user-attachments/assets/3759e117-81ca-4ab7-b27e-cffc07865f9c" />
<img width="1917" height="965" alt="图像编辑" src="https://github.com/user-attachments/assets/7419b53d-094a-4bb8-9dc7-8ef66f90e24b" />

# ST_tools
石头工具节点 - ComfyUI多功能插件，包含图像处理、模型管理、文本处理等多种功能

> 作者：石头
> QQ：34720803
> QQ群:33320584

## 节点列表

### 1. ST_ImageMaskLatentSize - 图像尺寸
**输入**：图像、遮罩、宽度、高度、缩放模式、裁剪方式、VAE（可选）
**输出**：图像、遮罩、latent（如有VAE）

**功能**：
- 统一处理图像、遮罩和latent的尺寸设置
- 支持自定义尺寸和预设尺寸
- 提供中心裁剪和等比缩放两种调整方式
- 支持与VAE配合使用，直接生成编码后的latent

### 2. ST_OfflineTranslator - 离线翻译
**输入**：待翻译文本、源语言、目标语言
**输出**：翻译结果

**功能**：
- 支持中文、英语之间的相互翻译
- 离线运行，无需网络连接，秒级基础翻译
- 首次使用时自动安装依赖

> 离线翻译模型下载地址：下载后存放于 `ComfyUI\custom_nodes\ST_tools\models` 中
> - 英文翻译中文模型：https://data.argosopentech.com/argospm/v1/translate-en_zh-1_9.argosmodel
> - 中文翻译英文模型：https://data.argosopentech.com/argospm/v1/translate-zh_en-1_9.argosmodel

### 3. ST_ImagePostProcessing - 图像调色
**输入**：图像、亮度、对比度、饱和度、Gamma、红通道、绿通道、蓝通道、模糊程度、锐化程度、HDR效果、自适应增强
**输出**：处理后图像

**功能**：
- 调整图像的亮度、对比度、饱和度
- 调整Gamma值和色彩平衡
- 调整模糊/锐化程度
- 应用HDR效果和自适应画质增强

### 4. ST_FilterShader - 图像滤镜
**输入**：图像、滤镜文件、混合模式、滤镜强度
**输出**：处理后图像

**功能**：
- 加载和应用`.cube`格式的滤镜文件
- 支持深度叠加和柔光混合两种调色模式
- 可调节滤镜强度
- 内置多种预设滤镜效果

### 5. ST_ImageEditor - 编辑图像
**输入**：CLIP模型、VAE模型、图片（动态1‑10张）、正面提示词、负面提示词、对齐模式、生成图像最长边、遮罩（可选）
**输出**：正面输出、负面输出、latent、补边信息、主图像、遮罩

**功能**：
- 动态输入：支持1‑10张图片，默认显示图片1，接入后自动显示新输入点
- 三种对齐模式：
  - `flux2klein`：仅通过VAE编码参考潜空间（适合保持图像结构）
  - `qwenedit`：视觉编码器+参考潜空间（适合理解图像内容并编辑）
  - `boogu`：视觉编码器+参考潜空间，正负条件均设置参考潜空间（适合保持图像身份）
- 自动处理：最长边缩放、补边对齐、兰佐斯插值
- 三种模式均支持自定义负面提示词
- 遮罩支持：单个遮罩输入，自动生成噪声掩码
- 主图像输出：处理后的主图像，与输入处理流程一致

### 6. ST_ImageSizeAligner - 编辑对齐
**输入**：补边信息、生成图像
**输出**：对齐后图像

**功能**：
- 接收编辑图像节点的补边信息
- 对K采样器生成的图像进行反向裁剪/对齐
- 让生成图像与原图尺寸完全一致

### 7. ST_ModelsLoader - 模型加载
**输入**：UNET模型、CLIP模型、VAE模型、CLIP类型、UNET权重类型、CLIP权重类型、VAE权重类型
**输出**：模型、CLIP、VAE

**功能**：
- 同时加载UNET、CLIP和VAE模型
- CLIP类型动态读取原生CLIPLoader，自动同步更新
- 支持不同权重类型设置

### 8. ST_LoraStack - Lora加载
**输入**：模型、CLIP、Lora1、Lora1强度、Lora1模型强度、Lora2、Lora2强度、Lora2模型强度、Lora3、Lora3强度、Lora3模型强度、Lora4、Lora4强度、Lora4模型强度
**输出**：模型、CLIP

**功能**：
- 同时加载多个Lora模型（最多4个）
- 支持调整每个Lora的强度
- 可同时应用到模型和CLIP

### 9. ST_KSamplerWithVAE - 解码采样器
**输入**：模型、正条件、负条件、VAE、种子、步数、CFG、采样器类型、调度器、latent、降噪强度、返回latent
**输出**：图像、latent（如有）

**功能**：
- 集成K采样器和VAE解码功能
- 一步完成采样和图像生成
- 支持多种采样器和调度器

### 10. ST_CLIPTextEncoder - CLIP编码器
**输入**：CLIP、正面提示词、负面提示词
**输出**：正面条件、负面条件、负面条件零化

**功能**：
- 同时处理正向和负向提示词
- 生成正面条件、负面条件和负面条件零化
- 支持多行提示词输入

## 安装方法
1. 下载或克隆本仓库到 ComfyUI 的 `custom_nodes` 目录
2. 重启 ComfyUI
3. 首次使用离线翻译节点时，会自动安装必要的依赖，可查阅 `ComfyUI\custom_nodes\ST_tools\requirements.txt`
