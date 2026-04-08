# AutoLabel V3 — 设计文档

## 概述

一个基于 PyQt5 的桌面图像标注工具，集成 YOLOv8 自动标注、模型训练和推理功能。支持检测（bbox）、分类和关键点标注，提供人工修正确认自动标注结果的完整工作流。

## 技术栈

- **UI 框架**：PyQt5
- **深度学习**：Ultralytics YOLOv8 (8.2.69)
- **运行环境**：conda 环境 `yolov8`（Python, torch 2.8.0, opencv, numpy, pillow）
- **额外依赖**：PyQt5, pyqtgraph（训练曲线绘制）

## 架构

单体模块化 PyQt5 应用，三层结构：

```
┌─────────────────────────────────────────────────┐
│                 UI 层 (PyQt5)                    │
│   标注面板  |  训练面板  |  模型管理面板          │
├─────────────────────────────────────────────────┤
│                核心业务层                         │
│   项目管理  |  训练引擎  |  推理引擎  |  格式转换  │
├─────────────────────────────────────────────────┤
│              Ultralytics YOLOv8                  │
│         detect  |  classify  |  pose             │
└─────────────────────────────────────────────────┘
```

## 项目目录结构

```
auto-labeling-v3/
├── main.py                     # 入口
├── requirements.txt
├── src/
│   ├── app.py                  # MainWindow，Tab 切换三个面板
│   ├── ui/
│   │   ├── canvas.py           # 标注画布（绘制 bbox、关键点、缩放平移）
│   │   ├── label_panel.py      # 标注面板（文件列表、标注列表、属性栏、工具栏）
│   │   ├── train_panel.py      # 训练面板（参数配置、日志、曲线）
│   │   ├── model_panel.py      # 模型管理面板（模型列表、自动标注设置）
│   │   └── dialogs.py          # 对话框（新建项目、导入导出、类别管理）
│   ├── core/
│   │   ├── project.py          # 项目/数据集管理
│   │   ├── annotation.py       # 标注数据模型（bbox、关键点、分类）
│   │   ├── label_io.py         # 标注文件读写
│   │   ├── formats/
│   │   │   ├── yolo.py         # YOLO 格式导入/导出
│   │   │   ├── coco.py         # COCO 格式导入/导出
│   │   │   └── labelme.py      # labelme JSON 格式导入/导出
│   │   └── config.py           # 应用配置
│   ├── engine/
│   │   ├── trainer.py          # 训练引擎（封装 ultralytics YOLO.train）
│   │   ├── predictor.py        # 推理引擎（封装 ultralytics YOLO.predict）
│   │   └── model_manager.py    # 模型注册与管理
│   └── utils/
│       ├── image.py            # 图片加载与处理
│       ├── workers.py          # QThread 工作线程（训练、批量推理）
│       └── undo.py             # 撤销/重做栈
└── resources/
    └── icons/                  # 工具栏图标
```

## UI 设计

### 主题

暗色主题（Catppuccin Mocha 色系），减少长时间标注的视觉疲劳。

### 标注面板（主工作界面）

布局：左侧文件列表 | 中间画布 | 右侧标注列表+属性栏

**顶部工具栏：**
- 标注工具：矩形框、关键点、移动、缩放
- 自动标注按钮（单张 / 批量）
- 全部确认按钮
- 当前模型和置信度显示

**左侧 — 文件列表：**
- 显示项目内所有图片（支持 jpg/jpeg/png 格式），按文件名排序
- 颜色状态区分：绿色 ✓ 已确认 / 黄色 ⚡ 待确认（有自动标注未确认）/ 灰色 ○ 未标注
- 支持过滤（按状态筛选）
- 点击切换图片，自动保存当前标注
- 缩略图懒加载：使用 QListWidget 虚拟模式，仅加载可视区域的缩略图
- 图片预加载：当前图片 + 前后各 2 张预加载到内存，减少切换卡顿
- 提供"刷新文件列表"按钮，手动检测图片目录中新增的图片

**中间 — 画布（Canvas）：**
- 图片显示，支持 Ctrl+滚轮缩放和中键拖拽平移
- 打开图片时自动 fit-to-window 适应窗口大小
- 绘制矩形框（实线=已确认，虚线=自动标注待确认）
- 绘制关键点（彩色圆点，可拖拽编辑）
- 选中标注时显示控制点，可调整大小和位置
- 右键菜单：修改类别、删除、确认/取消确认
- 拖拽限制：bbox 和关键点不允许超出图片边界，保存时坐标自动 clamp 到 [0, 1]

**画框流程：**
先画框再选类别。鼠标拖拽创建矩形框，松开后弹出类别选择列表，默认选中上次使用的类别。按 Enter 确认或直接点击类别名。

**关键点与 bbox 的关联（参考 labelme）：**
关键点和 bbox 是独立的标注对象，不强制绑定。在 `Annotation` 数据模型中：
- 纯检测标注：`bbox` 有值，`keypoints` 为空
- 纯关键点标注：`bbox` 为 null，`keypoints` 有值
- 检测+关键点：`bbox` 有值且 `keypoints` 有值（用户手动在画框后追加关键点，或由 pose 模型推理产生）

用户可以在任何时候独立创建 bbox 或关键点，也可以通过右键菜单将已有关键点"关联到"某个 bbox（合并为同一个 Annotation）。

**类别颜色：**
每个类别自动分配颜色（预设 20 色 Catppuccin 调色板循环），在项目配置 `project.json` 的 `class_colors` 字段中可自定义。画布上 bbox 边框、标签背景、关键点颜色均使用类别对应颜色。

**右侧 — 标注列表 + 属性栏：**
- 标注列表：当前图片所有标注，点击选中对应标注高亮
- 属性栏：选中标注的类别、置信度、状态、关键点详情
- 分类标签：当前图片的图片级分类标签（下拉框选择，可多选）
- 标注统计：显示当前项目总体进度（已标注/未标注/待确认数量）

**底部状态栏：**
- 当前文件名、标注数量（确认/待确认）、图片索引、快捷键提示

### 启动页面

应用启动时显示欢迎页面：
- 最近打开的项目列表（点击快速打开）
- "新建项目"按钮
- "打开项目"按钮（浏览选择 project.json）

### 分类标注

分类标注作用于整张图片而非单个目标：
- 在标注面板右侧属性栏顶部区域，提供"图片分类"下拉框
- 从项目类别列表中选择，可分配一个或多个分类标签
- 分类标签独立于 bbox/关键点标注，两者可共存于同一图片
- 分类数据存储在元数据 JSON 中的 `image_tags` 字段
- 用于分类任务训练时，自动按 YOLO 分类目录结构导出（`root/class_name/img.jpg` 符号链接）

### 训练面板

布局：左侧参数配置 | 右侧训练曲线 + 日志

**左侧 — 参数配置：**
- 任务类型选择：检测 / 分类 / 关键点（Tab 或下拉框切换）
- 基础模型选择（yolov8n/s/m/l/x 预训练权重或自定义 .pt 文件）
- 数据集：当前项目（显示已确认标注数量），或指定外部数据集路径
- 训练/验证集比例（默认 8:2）

- **基础训练超参数：**
  - epochs, batch, imgsz
  - optimizer（SGD / Adam / AdamW）
  - lr0, lrf, momentum, weight_decay
  - warmup_epochs, warmup_momentum, warmup_bias_lr
  - device（GPU 设备选择：auto / 0 / 1 / cpu，多卡环境下可指定）

- **数据增强参数：**
  - hsv_h, hsv_s, hsv_v
  - degrees, translate, scale, shear, perspective
  - flipud, fliplr
  - mosaic, mixup, copy_paste

- **Pose 专用参数（关键点任务时显示）：**
  - kpt_shape（关键点数量和维度，如 [17, 3]）
  - pose 损失权重（pose）
  - kobj 权重

- **输出配置：**
  - 训练输出目录：默认为项目目录下 `models/<model_name>/`，用户可通过浏览按钮自定义输出路径
  - 通过设置 ultralytics 的 `project` 和 `name` 参数将输出重定向到指定目录，不使用 ultralytics 默认的 `runs/` 目录

- 所有参数提供 YOLOv8 默认值，用户可自由调整
- 开始训练 / 停止训练 / 恢复训练（resume）按钮
- 同一时间只允许一个训练任务运行，训练中时"开始训练"按钮置灰

**右侧 — 训练监控：**
- Loss 曲线（train loss + val loss，pyqtgraph 实时绘制）
- mAP 曲线
- 训练日志（滚动文本框，显示每个 epoch 的指标）

**训练指标获取机制：**
通过 Ultralytics 回调机制获取实时训练指标：注册 `on_fit_epoch_end` 回调，在回调中通过 Qt Signal 将 epoch 指标（loss、mAP 等）发送到 UI 线程更新曲线和日志。训练本身在 QThread 中运行，回调在训练线程中触发，通过 signal/slot 跨线程更新 UI。

### 模型管理面板

- 已注册模型列表（预训练 + 训练产出的模型）
- 每个模型显示：名称、任务类型、训练时间、指标
- 加载 / 切换当前推理模型
- 自动标注设置：置信度阈值、IoU 阈值
- 删除模型

## 标注数据模型

### 内部表示

```python
@dataclass
class Keypoint:
    x: float          # 归一化坐标 [0, 1]
    y: float
    visible: int      # 0=不可见 1=遮挡 2=可见
    label: str        # 关键点名称（如 "nose", "left_eye"）

@dataclass
class Annotation:
    id: str                       # UUID
    class_name: str               # 类别名
    class_id: int                 # 类别索引
    bbox: tuple[float, float, float, float] | None  # (x_center, y_center, w, h) 归一化
    keypoints: list[Keypoint]     # 关键点列表（可为空）
    confidence: float             # 置信度（手动标注为 1.0）
    confirmed: bool               # 是否已确认
    source: str                   # "manual" | "auto"

@dataclass
class ImageAnnotation:
    image_path: str
    image_size: tuple[int, int]   # (width, height)
    annotations: list[Annotation]
    image_tags: list[str]         # 图片级分类标签
```

### 存储格式

**统一存储方案 — 每张图片一个 JSON 元数据文件：**

每张图片对应一个同名 `.json` 文件（如 `img_001.json`），这是工具的内部主格式，保存所有标注信息：

```json
{
  "image_path": "img_001.jpg",
  "image_size": [1920, 1080],
  "image_tags": ["outdoor", "daytime"],
  "annotations": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "class_name": "person",
      "class_id": 0,
      "bbox": [0.5, 0.4, 0.3, 0.6],
      "keypoints": [
        {"x": 0.25, "y": 0.15, "visible": 2, "label": "nose"},
        {"x": 0.30, "y": 0.20, "visible": 2, "label": "left_eye"},
        {"x": 0.20, "y": 0.20, "visible": 1, "label": "right_eye"}
      ],
      "confidence": 0.95,
      "confirmed": true,
      "source": "auto"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "class_name": "car",
      "class_id": 1,
      "bbox": [0.6, 0.3, 0.25, 0.35],
      "keypoints": [],
      "confidence": 0.87,
      "confirmed": false,
      "source": "auto"
    }
  ]
}
```

此格式统一存储 bbox 标注、关键点标注（无论是否关联 bbox）、分类标签和确认状态，没有歧义。

**导出时**按目标格式转换（YOLO txt、COCO json、labelme json 等）。

### 格式导入/导出

| 格式 | 导入 | 导出 |
|------|------|------|
| YOLO (txt + data.yaml) | ✓ | ✓ |
| COCO (json) | ✓ | ✓ |
| labelme (json) | ✓ | ✓ |

- VOC 格式暂不支持，后续按需添加
- 导出时可选择只导出已确认的标注
- 导出 YOLO 检测格式时自动生成 `data.yaml`
- 导出 YOLO 分类格式时自动生成 `root/class_name/img.jpg` 符号链接目录结构
- 导入已有标注时，所有标注默认标记为 `confirmed: true, source: "manual"`

## 项目配置（project.json）

```json
{
  "name": "my_project",
  "image_dir": "/path/to/images",
  "label_dir": "/path/to/labels",
  "classes": ["person", "car", "dog"],
  "class_colors": {"person": "#a6e3a1", "car": "#89b4fa", "dog": "#f38ba8"},
  "keypoint_templates": {
    "person_pose": {
      "labels": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
      "skeleton": [[0, 1], [0, 2], [1, 3], [2, 4]]
    }
  },
  "default_model": "yolov8n.pt",
  "auto_label_conf": 0.5,
  "auto_label_iou": 0.45,
  "created_at": "2026-03-23T10:00:00",
  "version": "1.0"
}
```

- `class_colors`：可选，每个类别的显示颜色（十六进制）。未指定的类别从预设 20 色 Catppuccin 调色板中自动分配

- `keypoint_templates`：可选，用户可定义命名的关键点模板。标注时可选择模板辅助按顺序打点，但不强制——用户也可以自由打点不使用模板
- `default_model`、`auto_label_conf`、`auto_label_iou`：可选字段，未配置时自动标注相关按钮置灰，不影响手动标注功能
- 训练 pose 模型时，用户必须选择一个模板（或手动指定 kpt_shape），系统据此过滤和验证标注数据：只有关键点数量和名称匹配模板的标注才会被纳入训练集

### 关键点标注的顺序机制

关键点标注支持两种模式：

**模板模式（推荐用于需要训练的场景）：**
1. 用户在项目中预定义关键点模板（名称有序列表 + 骨架连接关系）
2. 选择模板后进入顺序标注模式，画布上高亮提示当前要标注的点位（如"正在标注: left_eye (2/17)"）
3. 点击画布放置当前点，自动跳到下一个点位
4. 按 `S` 键可跳过当前点（标记 visible=0，表示不可见/被遮挡）
5. 可以点击已标注的点重新定位
6. 标注完成后，关键点数组严格按模板定义的顺序存储，确保所有图片的关键点语义一致
7. 画布上按骨架连接关系绘制连线

**自由模式：**
1. 不选择模板，每次点击画布放置一个关键点
2. 弹出输入框让用户命名该点（或从历史标签列表中选择）
3. 关键点按标注时间顺序存储
4. 适合探索性标注或不需要训练的场景

## 模型注册表

模型信息存储在项目目录下的 `models/registry.json`：

```json
{
  "models": [
    {
      "id": "uuid",
      "name": "yolov8n-custom-det",
      "path": "models/yolov8n-custom-det/best.pt",
      "task": "detect",
      "base_model": "yolov8n.pt",
      "classes": ["person", "car", "dog"],
      "metrics": {"mAP50": 0.89, "mAP50-95": 0.67},
      "trained_at": "2026-03-23T14:30:00",
      "epochs": 100,
      "dataset_size": 156
    }
  ]
}
```

训练产出的模型权重保存在 `models/<model_name>/` 目录下，包含 `best.pt`、`last.pt` 和训练日志。

## 应用配置（config.json）

存储在用户目录 `~/.autolabel/config.json`，全局配置：

```json
{
  "recent_projects": ["/path/to/project1", "/path/to/project2"],
  "theme": "dark",
  "auto_save": true,
  "default_conf_threshold": 0.5,
  "default_iou_threshold": 0.45,
  "window_geometry": {"x": 100, "y": 100, "width": 1400, "height": 900}
}
```

## 核心工作流

### 1. 项目创建与导入

1. 新建项目：指定项目名、图片目录、初始类别列表
2. 或打开已有数据集：自动识别 YOLO/labelme/COCO 格式的现有标注并转为内部 JSON 格式
3. 项目配置保存为 `project.json`

### 1.1 类别管理

项目进行中可随时通过菜单 > 类别管理 对话框进行：
- **新增类别**：输入名称，自动分配颜色
- **重命名类别**：同步更新所有已有标注中的 `class_name`
- **删除类别**：弹出确认对话框，用户选择处理方式：
  - 删除该类别的所有标注
  - 将该类别标注重新指定到其他类别
- **调整颜色**：点击色块打开取色器

### 1.2 标注统计

在标注面板右侧提供可折叠的统计区域：
- 项目总进度：已标注 / 未标注 / 待确认图片数量和占比
- 各类别标注数量分布（横向柱状图）
- 可帮助判断数据集质量和训练就绪状态

### 2. 自动标注

**单张（快捷键 Shift+A）：**
1. 使用当前加载的模型对当前图片推理
2. 结果以虚线框 + 黄色标签显示在画布上，标记为"待确认"

**批量（快捷键 Ctrl+Shift+A）：**
1. 选择范围：全部未标注 / 全部 / 选中的图片
2. 如果图片已有标注，提示选择：
   - **覆盖**：删除现有标注，替换为推理结果
   - **合并**：保留现有标注，新增推理结果中与现有标注 IoU < 0.5 的检测框（高重叠的丢弃，避免重复）
   - **跳过**：不处理该图片
3. 显示进度条，后台 QThread 执行，可随时取消
4. 完成后文件列表状态更新为 ⚡ 待确认
5. 模型类别名与项目类别名不一致时：只保留项目类别列表中存在的类别的检测结果，其余丢弃并在日志中提示

### 3. 人工修正与确认

- **确认**：Space 确认当前选中标注，Ctrl+Space 确认当前图片全部
- **修改**：拖拽调整 bbox 或关键点位置，自动变为已确认
- **删除**：Del 键或右键删除
- **修改类别**：右键菜单或属性栏下拉框
- **新增**：手动绘制新的 bbox 或关键点，直接为已确认状态
- 切换图片时自动保存

### 4. 模型训练

1. 切换到训练面板
2. 选择任务类型（检测/分类/关键点）
3. 选择基础模型
4. 调整训练超参数（epochs、batch、lr、数据增强参数等）
5. 关键点任务时额外显示 pose 参数（kpt_shape、pose 权重等）
6. 关键点训练时必须选择一个关键点模板或手动指定 kpt_shape；系统自动过滤不符合模板的标注并警告
7. 训练前自动检查：
   - 已确认标注数量 >= 10 张（软警告，用户可忽略继续）
   - 类别分布是否严重不均衡（最多类 / 最少类 > 10x 时警告）
8. 自动划分 train/val：按类别分层随机拆分（stratified split），默认 8:2 比例，可调整。使用符号链接组织目录结构，生成 `data.yaml`
9. 后台 QThread 执行训练，通过 `on_fit_epoch_end` 回调 + Qt Signal 实时更新 Loss/mAP 曲线和日志
10. 训练过程中可继续标注操作，互不阻塞
11. 训练完成后模型自动注册到模型管理列表
12. 支持 resume 断点续训

### 5. 数据导出

1. 菜单 > 导出
2. 选择格式：YOLO / COCO / labelme
3. 可选：只导出已确认标注
4. YOLO 检测格式自动生成 `data.yaml`
5. 导出到指定目录

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| W | 创建矩形框 |
| K | 创建关键点 |
| S | 跳过当前关键点（模板模式，标记 visible=0） |
| V | 移动工具 |
| D / → | 下一张图片 |
| A / ← | 上一张图片 |
| Space | 确认当前选中标注 |
| Ctrl+Space | 确认当前图片全部标注 |
| Del | 删除选中标注 |
| Ctrl+Z | 撤销 |
| Ctrl+Y | 重做 |
| Ctrl+S | 保存 |
| Shift+A | 单张自动标注 |
| Ctrl+Shift+A | 批量自动标注 |
| Ctrl+滚轮 | 缩放画布 |
| 中键拖拽 | 平移画布 |
| F5 | 刷新文件列表（检测新增图片） |

## 撤销/重做

- 作用范围：**每张图片独立**，每张图片有自己的撤销栈，不跨图片
- 可撤销的操作：新增标注、删除标注、移动/调整标注、修改类别、确认/取消确认
- 最大栈深度：50 步
- 关闭项目后撤销栈清空；切换图片时当前图片的撤销栈保留在内存中，返回该图片后可继续撤销

## 自动标注确认机制

- **已确认标注**：实线边框，绿色/蓝色类别标签
- **自动标注（待确认）**：虚线边框，黄色标签，显示"⚡待确认"
- 任何手动修改（移动、调整大小、修改类别）自动将标注状态变为已确认
- 文件列表通过颜色反映整体状态：
  - 绿色 ✓：所有标注已确认
  - 黄色 ⚡：有待确认标注
  - 灰色 ○：未标注

## 错误处理

- **图片加载失败**（损坏/不支持格式）：在文件列表中标红，跳过该图片继续浏览，状态栏提示错误
- **模型与项目类别不匹配**：自动标注时只保留匹配的类别，丢弃不匹配的，在状态栏和日志中提示被丢弃的类别
- **训练失败**（OOM / CUDA 错误）：捕获异常，在训练日志中显示错误信息，训练面板恢复到可重新开始的状态。提示用户：减小 batch size 或 imgsz
- **标注文件读取失败**：跳过损坏的标注文件，在日志中记录，图片显示为"未标注"状态
- **磁盘空间不足**：训练前检查可用磁盘空间，空间不足时阻止训练并提示

## 约束与边界

- 不支持视频标注，仅处理图片
- 不支持实例分割（polygon mask），聚焦 bbox + 关键点 + 分类
- 不支持多用户协作，单机单用户使用
- 训练仅支持 YOLOv8 系列模型，不集成其他框架
- 关键点标注为自由格式（参考 labelme），不强制固定骨架模板；训练 pose 模型时需要选择模板或指定 kpt_shape 来确保数据一致性

## 纯手动标注模式

本工具可作为独立的手动标注工具使用，不依赖任何模型或 GPU：

- **新建项目时**：模型相关字段（`default_model`、`auto_label_conf`、`auto_label_iou`）全部可选，不填写即可
- **UI 行为**：未加载模型时，自动标注按钮（⚡单张、Shift+A、Ctrl+Shift+A）和训练面板的"开始训练"按钮置灰不可用，其他所有标注功能正常
- **完整可用的功能**：手动画框、手动打关键点（自由模式和模板模式）、分类标签、修改/删除标注、撤销重做、格式导入/导出
- **确认机制**：手动标注直接为"已确认"状态，无需额外确认步骤。确认机制仅在自动标注流程中有意义
- **文件列表状态**：纯手动标注时只有绿色 ✓（已标注）和灰色 ○（未标注）两种状态，不会出现黄色 ⚡
