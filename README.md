# vid2dataset

[![CI](https://github.com/peter119lee/vid2dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/peter119lee/vid2dataset/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
2. Download `vid2dataset.exe` (151 MB) / 下载 `vid2dataset.exe`
3. Double-click to run / 双击运行

No Python needed. / 不需要 Python。

### Option B: GPU version (NVIDIA) / GPU 加速版

1. Download `vid2dataset-gpu.7z.001` + `vid2dataset-gpu.7z.002` (2.4 GB total)
2. Right-click `.001` → 7-Zip → Extract Here
3. Double-click `vid2dataset-gpu.exe`, tick **GPU 加速**
4. Bundles PyTorch + CUDA 12.1 — works on any modern NVIDIA GPU

### Option C: From source / 从源码安装

```bash
git clone https://github.com/peter119lee/vid2dataset.git
cd vid2dataset
pip install -e .[dev]              # CPU only
pip install -e .[dev,gpu]          # adds torch (CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade  # CUDA
```

Then `vid2dataset app` to launch the GUI, or `vid2dataset extract --help` for CLI.

**Need help choosing parameters? See [USAGE.md](USAGE.md) for a parameter-by-parameter guide with recommended values for common scenarios.**

---

## Platform support / 平台支持

| OS / 操作系统 | CPU pipeline | GPU pipeline | .exe |
|---|---|---|---|
| **Windows 10/11** | ✅ Verified / 已验证 | ✅ NVIDIA CUDA verified | ✅ |
| **Linux** | ⚠️ Code supports / 程式碼支援，未测 | ⚠️ NVIDIA / AMD ROCm via PyTorch | ❌ |
| **macOS Apple Silicon** | ⚠️ Code supports / 程式碼支援，未测 | ⚠️ MPS via PyTorch (auto-detected) | ❌ |
| **macOS Intel** | ⚠️ Code supports / 程式碼支援，未测 | ❌ | ❌ |

| GPU vendor / 显卡 | Decode (ffmpeg) | Filter pipeline (PyTorch) |
|---|---|---|
| **NVIDIA (any GeForce/RTX/Quadro)** | ✅ NVDEC verified | ✅ CUDA verified on RTX 3090 |
| **Apple Silicon M1/M2/M3** | ⚠️ VideoToolbox compiled in, untested | ⚠️ MPS code path exists, untested |
| **AMD Radeon** | ⚠️ AMF (Win) / VAAPI (Linux), untested | ⚠️ ROCm only on Linux, untested |
| **Intel Arc / iGPU** | ⚠️ QSV compiled in, untested | ❌ Not supported by PyTorch |

⚠️ = code is designed to work but I do not own the hardware to verify. Please open a GitHub issue if it doesn't.

---

## Features / 功能

| Feature | 功能 | Notes |
|---|---|---|
| Scene-aware sampling (PySceneDetect) | 场景感知采样 | Skipped in keyframe mode for speed |
| Blur + brightness quality gate | 模糊 + 亮度质量过滤 | |
| Letterbox auto-crop | 黑边自动裁切 | |
| Bucket-aware resize (Anima/SDXL 64-step grid) | 分辨率桶对齐 | |
| SSIM diversity filter | SSIM 姿态多样性过滤 | GPU-accelerated since v0.4 |
| Color/lighting diversity | 色彩/光照多样性 | |
| Cross-video pHash dedup | 跨视频感知哈希去重 | |
| Auto blur threshold detection | 自动模糊阈值检测 | |
| Per-video minimum guarantee | 每视频最低输出保证 | |
| Subject size filter | 主体大小过滤 | |
| Watermark detection | 浮水印偵測 (v0.6) | warn-only by default |
| Watermark cropping (opt-in) | 浮水印裁切 (v0.7) | |
| Pre-flight HTML report | 訓練前 HTML 報告 (v0.6) | |
| Gallery with hover info | 畫廊滑鼠 hover 資訊 (v0.7) | |
| Contact sheet + HTML gallery | 缩略图总览 + HTML 画廊 | |
| ETA estimation | 剩余时间预估 | |
| Chinese/English UI | 中文/英文界面切换 | |
| Remember last settings | 记住上次设置 | |
| Cancel mid-extraction | 中途取消 | |

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
├── _gallery.html         <- now with per-image hover info
├── _report.html          <- new in v0.6
└── _run_summary.json
```

Tick **Flatten output** in the GUI (or `flatten_output = true`) to drop all images into a single folder for kohya-ss-style trainers.

---

## Performance / 性能

Real-world benchmark on 34 × 4K MMD videos:

| version | time | speedup |
|---|---|---|
| v0.2.0 | 50.0 min | 1.0× |
| v0.3.1 | 16.0 min | 3.1× |
| v0.4.0 | 14.1 min | 3.5× |
| v0.5.0 | 8.21 min | 6.1× |
| v0.6.0 | 9.31 min | 5.4× (watermark scan) |
| **v0.7.0** | **~9 min** | **~5.5×** |

---

## For Developers / 开发者

```bash
git clone https://github.com/peter119lee/vid2dataset.git
cd vid2dataset
pip install -e ".[dev]"
pytest                        # 64+ tests
ruff check src/ tests/

# Build .exes (Windows)
python build_exe.py           # CPU .exe (151 MB)
python build_exe_gpu.py       # GPU .exe (2.4 GB)
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