# vid2dataset

Smart video-to-image extractor for LoRA training sets, with sane defaults
tuned for Anima/SDXL training.

Turn a folder of videos (MMD dance, gameplay, animations) into a clean,
deduplicated, properly bucketed image dataset — no coding required.

---

智能视频转图片训练集工具，专为 LoRA / SDXL 训练优化。

将一个视频文件夹（MMD 舞蹈、游戏录像、动画）转换为干净、去重、分辨率对齐的图片数据集 — 无需编程。

---

## Quick Start / 快速开始

### Option A: Download .exe / 下载 .exe（最简单）

1. Go to [Releases](https://github.com/peter119lee/vid2dataset/releases) / 前往 [Releases](https://github.com/peter119lee/vid2dataset/releases)
2. Download `vid2dataset.exe` / 下载 `vid2dataset.exe`
3. Double-click to run / 双击运行

No Python needed. / 不需要 Python。

### Option B: One-click install / 一键安装

1. Download ZIP → extract / 下载 ZIP → 解压
2. Double-click `install.bat` / 双击 `install.bat`
3. A Desktop shortcut will be created / 桌面会出现快捷方式
4. Double-click the shortcut / 双击快捷方式启动

---

## Features / 功能

| Feature | 功能 |
|---|---|
| Scene-aware sampling (PySceneDetect) | 场景感知采样 |
| Blur + brightness quality gate | 模糊 + 亮度质量过滤 |
| Letterbox auto-crop | 黑边自动裁切 |
| Bucket-aware resize (Anima/SDXL 64-step grid) | 分辨率桶对齐 |
| SSIM diversity filter | SSIM 姿态多样性过滤 |
| Color/lighting diversity | 色彩/光照多样性 |
| Cross-video pHash dedup | 跨视频感知哈希去重 |
| Auto blur threshold detection | 自动模糊阈值检测 |
| Per-video minimum guarantee | 每视频最低输出保证 |
| Subject size filter | 主体大小过滤 |
| Contact sheet + HTML gallery | 缩略图总览 + HTML 画廊 |
| ETA estimation | 剩余时间预估 |
| Chinese/English UI | 中文/英文界面切换 |
| Remember last settings | 记住上次设置 |

---

## Built-in Presets / 内置预设

| Preset / 预设 | Use case / 用途 |
|---|---|
| `anima-style` | Style LoRA — broad sampling, auto-quality / 风格 LoRA — 广泛采样 |
| `anima-character` | Character LoRA — strict quality / 角色 LoRA — 严格质量 |
| `fast-preview` | Quick preview — JPG @ 768px / 快速预览 |

---

## Output / 输出结构

```
output/
├── video_one/
│   ├── video_one_00001.png
│   ├── video_one_00002.png
│   └── _stats.json
├── video_two/
│   └── ...
├── _contact_sheet.png
├── _gallery.html
└── _run_summary.json
```

---

## For Developers / 开发者

```bash
git clone https://github.com/peter119lee/vid2dataset.git
cd vid2dataset
pip install -e ".[dev]"
pytest
ruff check src/ tests/

# Build .exe
python build_exe.py
```

### CLI

```bash
vid2dataset extract path/to/videos -o output --preset anima-style
vid2dataset extract path/to/videos --keyframe --auto-quality
vid2dataset presets
vid2dataset app
```

---

## Defaults (Anima-aligned) / 默认参数

| Parameter / 参数 | Default / 默认 | Why / 原因 |
|---|---|---|
| resolution | 1024 | SDXL/Anima standard / 标准分辨率 |
| min_pixels | 500,000 | Anima auto-drops below / Anima 自动丢弃 |
| bucket_step | 64 | Anima bucket grid / 桶网格步长 |
| blur_threshold | auto | Auto-detect recommended / 推荐自动检测 |
| phash_distance | 5 | Near-duplicate detection / 近似重复检测 |
| min_per_video | 3 | Never 0 output / 保证不为零 |

---

## License / 许可证

MIT
