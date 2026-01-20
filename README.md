# 认知负荷数据采集系统 (Cognitive Load Data Collection System)

基于 **L2CS-Net** 深度学习视线追踪的认知负荷研究数据采集工具，专为算法题解题过程中的认知负荷分析设计。

## 功能特性

- 🎯 **高精度视线追踪** - 使用 L2CS-Net 深度学习模型
- 📊 **AOI 区域检测** - 自动识别洛谷题目页面的不同区域（标题、题目描述、输入输出、示例等）
- 🔄 **注视检测** - 自动检测并记录注视事件
- 📈 **AOI 转换记录** - 记录视线在不同区域间的转移
- 🖥️ **智能窗口检测** - 自动区分浏览器和代码编辑器
- 📁 **完整数据导出** - CSV 格式，便于后续分析

## 系统要求

- Windows 10/11
- Python 3.10
- NVIDIA GPU (推荐，支持 CUDA)
- 摄像头

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/cognitive-load-tracker.git
cd cognitive-load-tracker
```

### 2. 创建虚拟环境

```bash
py -3.10 -m venv venv
.\venv\Scripts\activate
```

### 3. 安装 PyTorch

**有 NVIDIA 显卡（推荐，帧率 10-15fps）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**无显卡 / 仅 CPU（帧率 2-5fps，较慢）：**
```bash
pip install torch torchvision
```

**验证 GPU 是否可用：**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
> 输出 `CUDA: True` 表示 GPU 可用

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 5. 安装 L2CS-Net

```bash
pip install git+https://github.com/Ahmednull/L2CS-Net.git
```

### 6. 下载模型权重

从 [Google Drive](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd) 下载模型文件：

1. 进入 **Gaze360** 文件夹
2. 下载 `L2CSNet_gaze360.pkl`
3. 放到 `models/` 目录

### 7. 安装浏览器插件

1. 打开 Edge 浏览器，访问 `edge://extensions/`
2. 开启"开发者模式"
3. 点击"加载解压缩的扩展"
4. 选择 `cognitive_study/luogu_extension` 文件夹

## 使用方法

### 启动数据采集

```bash
py -3.10 cognitive_study/aoi_collector_v3.py
```

> **注意**: 如果从 GitHub 下载，根目录就是项目根目录，直接运行上述命令即可。

### 快捷键

| 按键 | 功能 |
|------|------|
| `C` | 校准视线（推荐首次使用时校准）|
| `N` | 开始新任务 |
| `1/2/3/4` | 设置任务难度 (Easy/Medium/Hard/Expert) |
| `A` | 结束任务 - AC (通过) |
| `W` | 结束任务 - WA (错误) |
| `T` | 结束任务 - TLE (超时) |
| `G` | 结束任务 - 放弃 |
| `+/-` | 调整视线平滑度 |
| `P` | 暂停/继续记录 |
| `Q` | 退出并保存数据 |

### 工作流程

1. 启动程序后，按 `C` 进行 9 点校准
2. 打开洛谷题目页面（确保浏览器插件已连接）
3. 按 `N` 开始任务，设置难度
4. 开始做题，系统自动记录视线数据
5. 完成后按 `A/W/T/G` 结束任务
6. 按 `Q` 退出，数据自动保存

## 数据输出

数据保存在 `cognitive_data/YYYYMMDD_HHMMSS/` 目录下：

| 文件 | 内容 |
|------|------|
| `gaze_data.csv` | 原始视线数据（时间戳、坐标、AOI 等）|
| `fixations.csv` | 注视事件（位置、时长、AOI）|
| `aoi_transitions.csv` | AOI 转换记录 |
| `tasks.csv` | 任务信息（题号、难度、结果）|
| `events.csv` | 事件日志 |
| `session_meta.json` | 会话元数据 |

### 数据字段说明

**gaze_data.csv:**
- `timestamp`: 时间戳
- `screen_x/y`: 屏幕坐标
- `yaw/pitch`: 视线角度
- `aoi_region`: AOI 区域 ID
- `aoi_name`: AOI 区域名称
- `is_fixation`: 是否为注视状态

**AOI 区域:**
- `A_TITLE`: 题目标题
- `B_PROBLEM`: 题目描述
- `C_IO_FORMAT`: 输入输出格式
- `D_EXAMPLES`: 示例
- `E_CONSTRAINTS`: 提示/约束
- `F_CODE_EDITOR`: 代码编辑区

## 项目结构

```
cognitive_study/
├── aoi_collector_v3.py      # 主程序
├── luogu_extension/         # 浏览器插件
│   ├── manifest.json
│   ├── content.js
│   └── icon.png
└── README.md

models/
└── L2CSNet_gaze360.pkl      # L2CS-Net 模型权重

cognitive_data/              # 数据输出目录
└── YYYYMMDD_HHMMSS/
    ├── gaze_data.csv
    ├── fixations.csv
    ├── aoi_transitions.csv
    ├── tasks.csv
    ├── events.csv
    └── session_meta.json
```

## 技术栈

- **视线追踪**: L2CS-Net (ResNet50)
- **数据平滑**: One Euro Filter
- **浏览器通信**: HTTP Server
- **GUI**: OpenCV

## 已知问题

- 帧率约 8-15 fps（受 GPU 性能影响）
- 眨眼检测暂时禁用（MediaPipe 兼容性问题）
- 视线精度约 3-5° 角度误差

## 后续计划

- [ ] 修复眨眼检测
- [ ] 添加瞳孔直径检测
- [ ] 支持更多题目平台
- [ ] 实时认知负荷预测模型

## 许可证

MIT License

## 致谢

- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) - 视线估计模型
- [洛谷](https://www.luogu.com.cn/) - 算法题平台


