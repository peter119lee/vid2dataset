# vid2dataset Usage Guide / 使用指南

A practical, parameter-by-parameter guide. **English first, 中文 in italics.**

> **Quick decision tree:**
> - Just want it to work? → Pick `anima-style` preset, click Extract.
> - Output too few images? → Lower `blur_threshold` or `color_distance`.
> - Output too many similar images? → Lower `ssim_threshold` (e.g. 0.80).
> - Background polluted with watermark? → Tick `Crop watermarks`.
> - PC weak / OOM? → Set workers manually to 1 or 2 in config.

---

## Table of Contents

1. [Choosing a preset](#1-choosing-a-preset)
2. [Filter parameters](#2-filter-parameters)
3. [Resolution & bucketing](#3-resolution--bucketing)
4. [GPU & performance](#4-gpu--performance)
5. [Watermarks](#5-watermarks)
6. [Output structure](#6-output-structure)
7. [Common scenarios](#7-common-scenarios)
8. [Troubleshooting](#8-troubleshooting)
9. [Auto-tagging](#9-auto-tagging)

---

## 1. Choosing a preset

| Preset | Best for | Key differences |
|---|---|---|
| **`anima-style`** (default) | Style LoRAs, MMD style, art style | Broad sampling, auto blur, GPU SSIM, keyframe mode |
| **`anima-character`** | Character likeness LoRAs | Strict quality (blur 130), looser dedup (more shots of same pose), completeness filter ON |
| **`fast-preview`** | Just checking what's in your videos | JPG @ 768px, 1 frame per scene, ~10x faster |

*预设选择：风格 LoRA 用 `anima-style`，角色 LoRA 用 `anima-character`，先看看片子内容用 `fast-preview`。*

**My recommendation for a first run:** start with `anima-style`. Look at the `_gallery.html` and `_report.html`. If the dataset looks too sparse, lower `blur_threshold` to 30. If too repetitive, lower `ssim_threshold` to 0.80.

---

## 2. Filter parameters

### Blur threshold (`blur_threshold`)

What it does: rejects frames with Laplacian variance below this. Lower = looser.

| Value | Effect | When to use |
|---|---|---|
| `0` | Accept everything | If you have very few candidates |
| `30-50` | Lenient (default in `anima-style`) | MMD content with motion blur |
| `80-100` | Balanced | General-purpose |
| `130+` | Strict (default in `anima-character`) | Character LoRAs, need sharp faces |

**Auto-quality** (`auto_quality = true`, default ON): completely overrides `blur_threshold`. The tool samples 50 frames per video and picks a per-video threshold at the 40th percentile of blur scores. **This is almost always what you want** — different cameras / renderers have wildly different baseline sharpness.

*模糊阈值：每秒动作快的舞蹈影片用 30-50，普通用 80-100，角色 LoRA 用 130+。开了自动模糊阈值（预设）就会忽略此参数，建议保持开。*

### SSIM diversity (`ssim_threshold`)

What it does: rejects frames too visually similar to recently accepted ones (different pose? different angle?).

| Value | Effect |
|---|---|
| `0.70` | Very strict — every frame must look quite different |
| `0.80` | Strict — good for character LoRAs |
| `0.85` | Default — good balance |
| `0.92` | Loose — keeps more frames, useful for tiny datasets |
| `1.00` | Disabled — no diversity check |

*SSIM 多样性：值越小要求姿态变化越大。0.85 平衡，0.80 严格，0.92 宽松。*

### Color diversity (`color_distance`)

What it does: rejects frames too similar in lighting / color palette to recently accepted ones.

| Value | Effect |
|---|---|
| `0.05` | Loose — keeps lots of similar-lit frames |
| `0.08` | Default — a bit aggressive on uniform-lit MMD |
| `0.15` | Strict — forces real lighting variation |

If your video has mostly fixed lighting (typical MMD), the default 0.08 sometimes drops too many frames. **Lower to 0.05 if you only get 3-5 images per video.**

*色彩距离：值越小越宽松。固定光照的 MMD 影片建议 0.05，不然会被砍很多。*

### pHash dedup distance (`phash_distance`)

What it does: cross-video deduplication via perceptual hash.

| Value | Effect |
|---|---|
| `0` | Only block exact duplicates |
| `5` | Default — blocks very similar frames |
| `10` | Aggressive — blocks loosely similar frames |

For style LoRAs: keep at 5. For character LoRAs: 8-10 (you want some repetition to reinforce the character).

*感知哈希去重距离：0=完全相同, 5=很像, 10=宽松。风格 LoRA 用 5，角色 LoRA 用 8-10。*

### Min / max per video (`min_per_video` / `max_per_video`)

| Setting | What it does |
|---|---|
| `min_per_video = 3` (default) | Guarantees ≥3 images even if all filters fire |
| `max_per_video = 0` | No upper cap |
| `max_per_video = 50` | Hard limit per video (useful for balance) |

*每影片最小数：保证至少几张（即使所有过滤器都生效）。每影片最大数：单支影片输出上限，0 = 不限。*

### Frames per scene (`frames_per_scene`)

How many candidate frames to sample per detected scene before filtering. 6 (default) is fine. Raise to 10 for very short videos that have few scenes; lower to 3 for very long videos to skip faster.

*每场景帧数：每个场景采样多少候选帧（过滤前）。预设 6。*

### Subject size & completeness filters

Both default OFF. Turn ON for **character LoRAs** to skip frames where the character is too small or partially out of frame:

- **`subject_size_filter`**: rejects frames where the foreground subject occupies < 15% of the frame.
- **`completeness_filter`**: rejects frames where the character is cut off at edges.

**Don't enable for style LoRAs** — wide shots and partial cuts contribute to learning the style.

*主体大小 & 完整性过滤：角色 LoRA 建议开（过滤太远 / 切到边的帧），风格 LoRA 建议关。*

---

## 3. Resolution & bucketing

### `resolution` (long edge)

| Value | Use case |
|---|---|
| `768` | SD 1.5 LoRA |
| `1024` | SDXL / Anima (default) — recommended |
| `1280` | High-res Anima training |

This is the bucket pixel budget (≈ `resolution × resolution` total pixels). Not the literal long edge. The actual output sizes depend on the source aspect ratio.

*分辨率：bucket 像素预算（不是字面上的长边）。SDXL/Anima 用 1024，SD1.5 用 768。*

### `min_bucket` / `max_bucket` / `bucket_step`

Constraints on the bucket grid. **Defaults match Anima**: 512 / 2048 / 64. Don't change unless you know your trainer wants different alignment.

### `min_pixels` (default 500_000 = 0.5 MP)

Reject crops below this pixel count. **Anima auto-drops images < 0.5 MP**, so matching that prevents wasted I/O.

### `resize_mode`

| Mode | Effect |
|---|---|
| `cover` (default) | Anima-standard: fill the bucket, center-crop the excess. No padding, no distortion. |
| `contain` | Fit fully inside the bucket, pad with black. Preserves every pixel but adds black bars. |
| `longest` | Just scale the long edge to `resolution`. **Not bucket-aligned** — only use for casual extraction. |

*缩放模式：`cover`（Anima 标准）= 填满裁切多余的；`contain` = 缩到框内补黑边；`longest` = 仅按长边缩放。*

---

## 4. GPU & performance

### `gpu_accel`

Default OFF. Tick **GPU acceleration (experimental)** in the GUI:
- ffmpeg uses NVDEC for decoding (5-10x faster decode on NVIDIA)
- SSIM filter runs on CUDA via PyTorch (4.6x faster than CPU)
- Self-validates: compares one frame CPU vs GPU; auto-disables on mismatch

**Real speedup on 34 4K MMD videos: 1.13x.** Decode is not the bottleneck — pipeline is filter-bound. Don't expect miracles.

**Required for the GPU .exe variant:** download `vid2dataset-gpu.7z.001` + `.7z.002` from Releases, extract with 7-Zip. The default `vid2dataset.exe` doesn't bundle PyTorch (would balloon to 2 GB) so the GPU checkbox is a no-op there.

*GPU 加速：勾上后用 NVDEC 解码 + CUDA 跑 SSIM。仅 GPU 版 .exe 有效，普通版的勾选无效（不会 crash）。整体加速约 1.13x。*

### `decode_mode`

| Mode | Effect |
|---|---|
| `keyframe` (default) | Stream only I-frames via ffmpeg. **5-20x faster decode**. Fine for ~99% of cases. |
| `accurate` | Frame-exact OpenCV seek. Slower but exact. Use only if you need specific frame numbers. |

*解码模式：`keyframe`（预设）= 只读关键帧，5-20x 快；`accurate` = 帧精确定位，慢但精准。*

### `workers` (auto-detected by default)

Number of parallel video workers. **Default is auto** — the tool detects your CPU cores and free RAM, picks a safe value:

| Machine | Auto picks |
|---|---|
| 8 GB / 4-core laptop | 1-2 workers |
| 16 GB / 8-core mid-tier | ~6 workers |
| 32 GB / 16-core (RTX 3090 host) | ~14 workers |
| 64 GB / 24-core (RTX 5090 host) | ~16 workers |

**Override only if you hit OOM** (`workers = 1`) or want to maximize a workstation (`workers = 16`).

*Worker 数：预设自动检测，根据 CPU 核数和可用 RAM 调整。机器很弱手动设 1，机器很猛可手动调大。*

---

## 5. Watermarks

### `detect_watermark` (default ON)

Scans every video for static text/logo overlays. Logs warnings + saves to `_stats.json` and `_report.html`. **Does not modify images** unless you also enable cropping.

Cost: ~2 seconds per video. Turn off if you trust your sources.

*侦测浮水印：扫描静态文字/标志，仅警告不修改图片。每视频 +2 秒。*

### `crop_watermark` (default OFF)

When ticked alongside `detect_watermark`:
- Watermarks at corners / edges → bucket crop is shrunk to exclude them
- Watermarks in the middle → **never cropped** (would slice the subject)

**Use when:** you've reviewed `_report.html` and confirmed real watermarks are flagged.
**Don't use when:** detection is flagging false positives (review the report first).

*浮水印裁切：勾上后边缘的浮水印会被裁掉。中央的不裁（会切到主角）。先看报告确认真假再开。*

---

## 6. Output structure

### `flatten_output`

| Off (default) | On |
|---|---|
| `output/video1/img_001.png` | `output/img_001.png` |
| `output/video1/_stats.json` | `output/img_002.png` |
| `output/video2/img_001.png` | `output/img_003.png` |
| ... | ... |

- **Off**: easier to debug, preserves source attribution.
- **On**: drops everything into one flat folder. **kohya-ss style trainers expect this.**

*扁平输出：勾上后所有图片在同一资料夹（kohya-ss 训练用），不勾则每影片一子资料夹。*

### `output_format`

| Format | When |
|---|---|
| `png` (default) | Lossless, recommended for training |
| `jpg` | Smaller files, slight quality loss |
| `webp` | Smallest files, broad support |

### Files always generated

- `_contact_sheet.png` — single overview image
- `_gallery.html` — browser gallery (hover over images for blur/bucket info)
- `_report.html` — pre-flight report with bucket distribution, blur histogram, watermark warnings
- `_run_summary.json` + per-video `_stats.json`

---

## 7. Common scenarios

### "Style LoRA from MMD videos" (most common)

```
Preset: anima-style
+ Auto blur threshold: ON
+ Keyframe mode: ON
+ GPU acceleration: ON (if you have GPU .exe)
+ Detect watermarks: ON
+ Crop watermarks: OFF (review report first)
+ Subject size filter: OFF
+ Completeness filter: OFF
+ Flatten output: depends on trainer (kohya-ss = ON)
```

After first run:
1. Open `_report.html`
2. If watermark warnings → spot-check the gallery → if real, tick `Crop watermarks` and re-run
3. If under 5 images per video → lower `color_distance` to 0.05 in `anima-style.toml`

### "Character LoRA from one source"

```
Preset: anima-character
+ Subject size filter: ON
+ Completeness filter: ON
+ ssim_threshold: 0.80 (stricter pose diversity)
+ phash_distance: 8 (allow some near-duplicates)
+ max_per_video: 50 (cap to avoid imbalance)
```

### "Quick first look at a folder"

```
Preset: fast-preview
```

That's it. JPG @ 768px, fast. Use to decide whether the videos are worth doing a full extraction on.

### "Weak laptop (8 GB RAM)"

```
Preset: anima-style
+ workers: 1 (manually, in the TOML or override)
+ Detect watermarks: OFF (skip the 2 s/video overhead)
+ Subject size filter: OFF
+ Completeness filter: OFF
```

Don't use the GPU .exe (too much RAM). Stick with `vid2dataset.exe`.

---

## 8. Troubleshooting

### "Output is empty / 0 images"

1. Check input path actually has video files (`.mp4`, `.mkv`, `.webm`, etc.)
2. Lower `blur_threshold` to 30
3. Lower `color_distance` to 0.05
4. Disable `Detect watermarks` (rare bug fallback)
5. Last resort: `min_per_video = 5` forces at least 5 frames regardless of filters

### "Output is way too few per video"

Likely `color_distance` or `ssim_threshold` is too strict. Try:
- `color_distance: 0.05` (down from default 0.08)
- `ssim_threshold: 0.92` (up from default 0.85)

### "Output is too repetitive / lots of similar frames"

- `ssim_threshold: 0.80`
- `color_distance: 0.12`
- `phash_distance: 8`

### "Crashes / OOM mid-extraction"

Override workers manually:
```toml
workers = 1
```
Or reduce `frames_per_scene` from 6 → 3.

### "GPU acceleration tick doesn't speed things up"

Two reasons:
1. You're using `vid2dataset.exe` (CPU-only build). Use the `vid2dataset-gpu.7z` variant.
2. The pipeline is filter-bound. GPU SSIM is 4.6x faster but it's only 30% of total time. The remaining bottleneck is PySceneDetect (CPU).

### "中文路径 / Chinese paths can't open"

Should work since v0.2.0. If it doesn't:
- Check the path doesn't have characters outside Unicode BMP (extreme edge case)
- Run from a path without spaces or special chars as a workaround

---

## Reference: every config field

For the full list with types and validators, see [`src/vid2dataset/config.py`](src/vid2dataset/config.py).

---

## 9. Auto-tagging

**(v1.0)** Tick **Auto-tag images** in the GUI (or `tag_images = true`) and every
output image gets a `.txt` caption sidecar: `trigger word, character tags, general tags`.
Drop the folder straight into kohya / OneTrainer.

*勾选**自动打标**（或 `tag_images = true`），每张输出图片会得到一个 `.txt` 标签文件：
`触发词, 角色标签, 通用标签`，可直接用于 kohya / OneTrainer。*

| Setting | Default | Notes |
|---|---|---|
| `tagger_model` | `wd-eva02-large-tagger-v3` | Most accurate, ~1.2 GB. `wd-swinv2-tagger-v3` is ~450 MB and faster on CPU. |
| `trigger_word` | *(empty)* | Written first in every caption. Use one rare token, e.g. `mychar_v1`. |
| `tag_general_threshold` | 0.35 | Lower = more (noisier) tags. 0.25-0.30 for style LoRAs wanting rich captions. |
| `tag_character_threshold` | 0.85 | Keep high — false character tags actively hurt training. |
| `kohya_repeats` | 0 (off) | With **Flatten output**: images land in `output/<N>_<trigger>/` (kohya dreambooth folder convention). |

**First enable** downloads the model + onnxruntime (~25 MB) once to
`%LOCALAPPDATA%/vid2dataset/`. Inference uses DirectML — GPU-accelerated on any
Windows GPU vendor (NVIDIA/AMD/Intel), automatic CPU fallback. Delete
`%LOCALAPPDATA%/vid2dataset/tagger_models` / `tagger_runtime` to reclaim space or
force a re-download.

**Standalone CLI** — caption a folder you already have (any source):

```bash
vid2dataset tag path/to/images --trigger mychar
vid2dataset tag path/to/images -m wd-swinv2-tagger-v3 --threshold 0.30 --cpu
```

**Check your captions**: `_report.html` gains an *Auto-tagging* section with a
top-30 tag frequency table — if 90% of images share `full_body`, you know to add
close-up sources. Hover any image in `_gallery.html` to see its tags.

*rating 标签（general/sensitive/…）永远不会写进 caption；它们只出现在统计里。*
