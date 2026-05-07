# TransGlass PaddleOCR GPU

**屏幕翻译叠加工具** — 截图 → PaddleOCR(GPU) 识别 → Ollama 翻译 → 透明覆盖层显示

---

## 功能概述

- **截图翻译**：截取当前屏幕 → PaddleOCR 文字识别 → Ollama 本地翻译 → 翻译结果以半透明覆盖层显示在原文字上方
- **自动翻译模式**：监测屏幕变化，自动触发截图翻译（dHash 变化检测）
- **多屏幕支持**：在多显示器环境下切换识别目标屏幕
- **OCR 精度调节**：托盘菜单中可实时调整检测阈值和框置信度阈值
- **合并规则可调**：4 种合并规则（严格/适度/宽松/不合并），适应不同排版
- **快捷键自定义**：所有快捷键可在托盘菜单中自由设置
- **原始识别框开关**：`Ctrl+Alt+5` 切换绿色检测框显示，方便调试
- **双屏适配**：支持主屏横屏 + 副屏竖屏等异形布局
- **图表征增强**：Top-hat 形态学预处理，提升暗色场景（游戏/动漫）的文字检测率
- **鼠标穿透**：覆盖层不阻挡鼠标操作

---

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| CUDA | 12.6 |
| PaddlePaddle GPU | 3.3.0 |
| PaddleOCR | 3.2.0 |
| Ollama | 任意版本（需安装并运行翻译模型） |
| OS | Windows 10/11 64-bit |
| GPU | NVIDIA RTX 4090（测试），计算能力 ≥ 7.0 |

### Python 依赖

```
paddlepaddle-gpu==3.3.0
paddleocr==3.2.0
PySide6
mss
pynput
Pillow
numpy
opencv-python
requests
urllib3
```

---

## 快速开始

### 1. 启动 Ollama

确保 Ollama 已在后台运行，并安装了翻译用模型（如 `qwen2.5:7b`）：

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 2. 启动 TransGlass

**推荐方式** — 双击 `run_gpu.bat`

**手动启动**：
```bash
.\venv_gpu\Scripts\activate
python TransGlass_PaddleOCR_GPU.py
```

---

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Alt+1` | 识别翻译当前屏幕 |
| `Ctrl+Alt+2` | 切换目标屏幕 |
| `Ctrl+Alt+3` | 自动翻译开/关 |
| `Ctrl+Alt+4` | 退出程序 |
| `Ctrl+Alt+5` | 原始识别框（绿框）开/关 |

所有快捷键可在托盘菜单 →「快捷键设置」中自定义。

---

## 使用说明

### 托盘菜单

右键点击系统托盘的 TransGlass 图标，可以：

- **识别翻译** — 手动触发一次截图翻译
- **切换屏幕** — 在多显示器间切换目标屏
- **自动翻译** — 开启/关闭屏幕变化自动检测翻译
- **合并规则** — 选择 4 种合并规则之一（严格/适度/宽松/不合并）
- **OCR精度** — 打开滑块面板调节检测参数
  - `det_thresh`：检测阈值（0.05~1.0），越低检出越多
  - `box_thresh`：文本框置信度（0.05~1.0），越低保留越多
- **快捷键设置** — 自定义快捷键组合
- **退出** — 关闭程序

### 自动翻译模式

开启后，程序以固定间隔（默认 1.5 秒）检测屏幕变化。只有当检测到画面变动时，才会触发 OCR + 翻译。检测算法使用 dHash（差异哈希），对亮度变化不敏感，专注检测内容变化。

### 覆盖层

翻译结果以浅灰色（RGB 180,180,180）半透明背景框显示在原文字上方，鼠标可穿透覆盖层操作底层窗口。

---

## 项目结构

```
TransGlass_PaddleOCR_GPU/
├── TransGlass_PaddleOCR_GPU.py    # 主程序
├── run_gpu.bat                     # GPU 启动脚本
├── README.md                       # 本文件
├── .gitignore
├── venv_gpu/                       # GPU 虚拟环境（不提交）
└── test_*.py                       # 测试脚本
```

---

## 技术架构

### 主要流程

```
截图 (mss) → 图片预处理(四边扩展+Top-hat) → PaddleOCR GPU 识别
  → ComicBubbleMerger 合并 → Ollama 翻译 → 覆盖层渲染(PySide6)
```

### 核心模块

| 模块 | 类/方法 | 功能 |
|------|---------|------|
| OCR 引擎 | `ComicOCR` | 封装 PaddleOCR，支持语言切换和预处理增强 |
| 文本合并 | `ComicBubbleMerger` | 链式合并算法，支持 k/g 乘数调节 |
| 翻译引擎 | `OllamaTranslator` | 调用本地 Ollama API 翻译 |
| 截图线程 | `RecognizeThread` | 异步截图+识别+翻译，不阻塞 UI |
| 覆盖层 | `TranslucentOverlayWidget` | PySide6 透明窗口，鼠标穿透 |
| 快捷键 | `GlobalHotkeyListener` | pynput 全局监听，支持录制自定义 |
| 信号总线 | `SignalBus` | Qt Signal 解耦各模块通信 |

### OCR 预处理增强

对于暗色调场景（游戏、动漫），PaddleOCR 的默认检测模型可能无法识别亮色艺术字体。程序内置 Top-hat 形态学预处理（OpenCV），在识别前自动提取亮色文字区域，显著提升检测率。

### 合并算法

使用 ComicBubbleMerger 链式合并，基于文字块间距（横向 gap、纵向 gap）和字体大小统计，将同一气泡内的多行文字智能合并为一个对话框。

---

## 配置说明

配置文件路径：`~/.transglass_config.json`

主要配置项：
- `hotkeys` — 自定义快捷键映射
- `ocr_det_thresh` — 检测阈值（默认 0.30）
- `ocr_det_box_thresh` — 框置信度（默认 0.60）
- `merge_rule` — 合并规则名称
- `screen_index` — 目标屏幕索引
- `last_window_position` — 覆盖层窗口位置

---

## 常见问题

### Q: GPU 不工作，回退到 CPU 模式？

```bash
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
```

如果输出 `False`，请重新安装 paddlepaddle-gpu：
```bash
pip install paddlepaddle-gpu==3.3.0 -f https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

### Q: OCR 识别不到文字？

1. 在托盘菜单中调低「OCR精度」的检测阈值（det_thresh）和框置信度（box_thresh）
2. 切换合并规则为「宽松合并」
3. 按 `Ctrl+Alt+5` 查看绿色检测框是否覆盖文字区域

### Q: 翻译质量不理想？

在 Ollama 中更换翻译模型（如 `qwen2.5:14b`、`llama3.1` 等更大模型效果更好）。

### Q: 竖屏 / 异形屏幕坐标错乱？

程序自动匹配 DPI 缩放因子（50px 容差），支持横竖屏混合布局。如果仍有问题，尝试重新切换屏幕（托盘菜单）。

---

## 版本历史

- **v1.0** (2026-05)
  - 初始版本
  - PaddleOCR GPU 加速
  - Ollama 本地翻译
  - 透明覆盖层显示
  - 多屏幕支持
  - 自动翻译模式
  - OCR 精度滑块
  - Top-hat 预处理增强
  - 快捷键自定义
  - dHash 变化检测
