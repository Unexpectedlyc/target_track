# 目标跟踪项目 (Target Tracking) 🎯

本项目基于 YOLOv7 和 DeepSORT 实现目标检测与跟踪功能，可以实时检测并跟踪视频中的目标对象。🔍

## 项目简介 ℹ️

这是一个基于 YOLOv7 目标检测算法和 DeepSORT 跟踪算法的目标跟踪系统。它能够检测视频中的多个目标并为它们分配唯一 ID，实现持续跟踪。项目支持 COCO 数据集的 80 个类别目标检测和跟踪。📊

## 功能特点 ✨

- 🎯 基于 YOLOv7 的目标检测，具有高精度和较快的检测速度
- 🔄 使用 DeepSORT 算法实现多目标跟踪
- 📦 支持 COCO 数据集的 80 个类别检测与跟踪
- 🖼️ 可视化显示检测框、类别标签、置信度和跟踪 ID
- 📊 实时统计并显示画面中目标数量
- 🎥 支持视频文件处理和实时摄像头输入
- 🎯 可选择特定类别进行跟踪

## 目录结构 📁

```
.
├── cfg/ # 配置文件目录
│ ├── baseline/ # YOLO 基础配置文件
│ ├── deploy/ # 部署配置文件
│ └── training/ # 训练配置文件
├── data/ # 数据相关配置
├── deep_sort/ # DeepSORT 跟踪算法实现
├── models/ # YOLOv7 模型定义
├── utils/ # 工具函数
├── weights/ # 模型权重文件
├── config.py # 项目配置文件
├── demo.py # 演示程序入口
├── target_tracking.py # 核心跟踪逻辑
└── README.md # 项目说明文档
```

## 环境依赖 💻

- 🐍 Python 3.7+
- 🔥 PyTorch 1.7+
- 🎞️ OpenCV
- 其他依赖项详见 requirements.txt

## 安装步骤 ⚙️

1. 📥 克隆项目代码：

   ```bash
   git clone <repository-url>
   cd target_track
   ```

2. 📦 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. ⬇️ 下载预训练模型：
   - 下载 YOLOv7 权重文件(yolov7.pt)并放置在`weights/`目录下
   - DeepSORT 的预训练模型已在项目中提供

## 使用方法 🚀

### 基本使用

1. ✏️ 修改 `demo.py` 文件中的视频路径：

   ```python
   path=r'your_video_path.mp4'  # 设置要检测的视频路径
   ```

2. ▶️ 运行演示程序：

   ```bash
   python demo.py
   ```

3. 🛑 按 ESC 键停止程序

### 配置参数 ⚙️

在 config.py 中可以修改以下参数：

- conf_thres: 检测置信度阈值（默认: 0.62）📈
- iou_thres: NMS IOU 阈值（默认: 0.45）
- model_path: YOLOv7 模型路径（默认: "weights/yolov7.pt"）
- image_size: 输入图像尺寸，必须为 32 的倍数（默认: 640）

在 `deep_sort/configs/deep_sort.yaml` 中可以调整 DeepSORT 参数：

- `REID_CKPT`: ReID 模型路径
- `MAX_DIST`: 最大余弦距离
- `MIN_CONFIDENCE`: 最小置信度
- `MAX_IOU_DISTANCE`: 最大 IOU 距离
- `MAX_AGE`: 最大生存周期
- `N_INIT`: 初始化帧数

### 特定类别跟踪 🎯

如需跟踪特定类别的目标，可以在 [demo.py](file://d:\workplace\target_track\demo.py) 中指定类别名称：

```python
main(path, classname="person")  # 只跟踪"person"类别
```

### 保存结果视频 💾

如需保存处理后的视频结果，可以设置：

```python
main(path, IsvideoWriter=True)  # 保存结果为result.mp4
```

## 主要组件 🧩

### target_tracking.py

核心功能模块，包含：

- detect(): 目标检测函数
- update_tracker(): 目标跟踪更新函数
- plot_bboxes(): 检测框可视化函数
- main(): 主函数，处理视频流

### config.py

项目配置文件，包含模型路径、检测阈值等参数设置。

### deep_sort/

DeepSORT 算法实现，包括：

- 检测匹配算法
- 轨迹管理
- ReID 特征提取

## 注意事项 ⚠️

1. ⚡ 确保已正确安装 CUDA 和 cuDNN 以获得 GPU 加速（如果可用）
2. 📏 输入图像尺寸必须为 32 的倍数
3. 📁 模型权重文件需要单独下载并放置在正确位置
4. 🇨🇳 如需使用中文显示功能，确保系统中存在"simsun.ttc"字体文件

## 项目展示 📺

程序运行时会显示：

- 🟨 检测到的目标边界框
- 🏷️ 目标类别和置信度
- 🔢 每个目标的唯一 ID
- 📊 当前画面中目标总数

