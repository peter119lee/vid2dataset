# vid2dataset Testing Plan / 测试计划

This document describes the test scenarios for verifying vid2dataset releases.
本文档描述发布前需要验证的测试场景。

## 1. Unit Tests / 单元测试

Location: `tests/`. Run: `pytest -q`.

| Module | Test File | Coverage |
|---|---|---|
| `config` | `test_config.py` | Defaults, TOML loading, validators |
| `quality` | `test_quality.py` | Laplacian variance, luma stats, blur rejection |
| `crop` | `test_crop.py` | Letterbox detection, pillarbox, no-bar fallback |
| `resize` | `test_resize.py` | Bucket grid, cover/contain modes, longest-edge |
| `dedup` | `test_dedup.py` | pHash matching, persistence, distance threshold |
| `io_utils` | `test_io_utils.py` | sanitize_stem, video discovery |

**Target: 100% pass rate. >80% coverage on pure-logic modules.**

## 2. Integration Test Scenarios

Run each scenario manually before release. Each one validates a specific code path.

### 2.1 Single video, ASCII path

```powershell
vid2dataset extract "test_videos/sample.mp4" -o "out/single" --preset anima-style --max-per-video 10
```

**Expected:**
- ≥3 images written (min guarantee)
- `_stats.json` and `_run_summary.json` created
- Contact sheet and gallery generated
- Time: <60s for a 1-minute video at 1080p

### 2.2 Single video, Chinese path

```powershell
vid2dataset extract "D:\测试\心海.mp4" -o "out/chinese" --preset anima-style
```

**Expected:**
- Reads the file successfully (verifies `\\?\` prefix workaround)
- Writes images to `out/chinese/心海/`
- No "could not open video" errors

**Failure mode this catches:** OpenCV silently failing on non-ANSI Windows paths.

### 2.3 Mixed orientations

Folder containing portrait (9:16) + landscape (16:9) + square (1:1) videos.

**Expected:**
- Each video gets a bucket appropriate to its aspect ratio
- Verify in `_stats.json`: `bucket` field shows different dimensions per video
- All buckets are multiples of 64

### 2.4 Folder with one corrupt video

Place a 0-byte `.mp4` and a real video in the same folder.

**Expected:**
- Real video processed normally
- Corrupt video logged as error, skipped
- Final summary shows N-1 successful videos

### 2.5 Very short video (<10 frames)

**Expected:**
- `auto_quality` skips its sampling (returns 50.0 fallback)
- `min_per_video` may be unmet (acceptable for tiny videos)
- No crash

### 2.6 Very long video (>1 hour 4K)

**Expected:**
- Memory peak <2GB (verify with Task Manager)
- `backup_heap` stays bounded at `min_per_video × 3`
- Diversity filter lists trimmed to `max_compare`

**Failure mode this catches:** Unbounded list growth (fixed in v0.2).

### 2.7 4K video with keyframe mode

```powershell
vid2dataset extract "4k_video.mp4" -o "out/4k" --preset anima-style --keyframe
```

**Expected:**
- Time per video: <2 minutes (vs ~3-4min OpenCV-only)
- Uses ffmpeg if available (check log for "ffmpeg" mentions)

### 2.8 Folder with 30+ videos (parallel)

```powershell
vid2dataset extract "big_folder" -o "out/big" --preset anima-style --workers 4
```

**Expected:**
- 4 videos processed concurrently
- Total time <40min for 34 videos × 4K (vs 119min sequential in v0.2)
- Output is correct (no corrupted images)

### 2.9 Resume after interruption

1. Start extraction on big folder
2. Kill process at ~50% progress
3. Re-run the same command

**Expected:**
- Already-completed videos (with `_stats.json`) are skipped
- Remaining videos processed normally
- No duplicate filenames

### 2.10 Cross-video dedup persistence

```powershell
vid2dataset extract "folder_A" -o "out" --dedup-index ".dedup.json"
vid2dataset extract "folder_B" -o "out" --dedup-index ".dedup.json"
```

**Expected:**
- Second run rejects images that match anything in folder_A's results
- `.dedup.json` grows after each run

## 3. UI Test Scenarios

### 3.1 Launch .exe

- Double-click `vid2dataset.exe` from File Explorer
- **Expected:** Window opens within 5 seconds, shows "vid2dataset 0.x.0" title

### 3.2 Browse buttons

- Click 输入 Browse → folder picker opens
- Select a folder → path appears in Entry
- Same for 输出

### 3.3 Preset auto-fill

- Change Preset dropdown to `anima-character`
- **Expected:** All parameter Entry values update to the preset's values

### 3.4 Language toggle

- Click 中文/EN button
- **Expected:** ALL labels switch language (header, I/O, params, checkboxes, buttons, status)

**Failure mode this catches:** Forgetting to refresh some labels (fixed in v0.2).

### 3.5 Cancel mid-run (v0.3)

- Start extraction on folder with 5+ videos
- Wait until 1-2 videos done
- Click Cancel button
- **Expected:**
  - Current video finishes its current frame, then stops
  - Already-written images remain
  - Status shows "Cancelled"
  - Run button re-enabled

### 3.6 Drag-and-drop folder (v0.3)

- Drag a folder from File Explorer onto the app window
- **Expected:** Input path Entry populates with the dropped folder's path

### 3.7 Open Output Folder

- Run a small extraction to completion
- Click 打开输出文件夹 button
- **Expected:** File Explorer opens the output directory

### 3.8 Update checker

- Click 检查更新
- **Expected:**
  - On latest version: "已是最新版本"
  - On outdated version: prompt to download + auto-restart

## 4. Performance Benchmarks

Run on a typical 4-core CPU with HDD (SSD will be faster).

| Scenario | v0.2 baseline | v0.3 target | How to measure |
|---|---|---|---|
| 34 × 4K videos (~60s each) | 119min | **<40min** | `time` of full run |
| Single 1080p video (60s) | ~30s | **<15s** | log shows elapsed_s |
| Memory peak (1hr 4K video) | ~2GB | **<1GB** | Task Manager Working Set |
| .exe startup | ~3s | **<5s** | manual stopwatch |
| Update check round-trip | n/a | **<3s** | click button → response |

## 5. Manual Smoke Test (release checklist)

Before tagging a release, run this 10-item check:

- [ ] `pytest -q` passes (44+ tests)
- [ ] `ruff check src/ tests/` clean
- [ ] `python build_exe.py` produces a .exe (~120MB)
- [ ] Double-click .exe → window opens
- [ ] Browse Input → pick a folder with 1 short video
- [ ] Click Extract → completes in <60s
- [ ] Output folder contains ≥3 PNGs and `_gallery.html`
- [ ] Open `_gallery.html` in browser → images load
- [ ] Toggle language → labels change
- [ ] Click 检查更新 → returns latest release info

## 6. Known Limitations

- **Windows-only .exe**: Linux/Mac users must install from source
- **No GPU acceleration**: extraction is CPU-bound
- **OpenCV path encoding**: extreme edge cases (e.g. surrogate pairs) may still fail. The `\\?\` prefix covers most cases.
- **PySceneDetect speed**: even with downscaling, scene detection adds ~10-30s per video. To skip it, use `--sampling interval` mode.
- **First-run ffmpeg download**: if `imageio-ffmpeg` is fresh, it downloads the binary on first call (~30MB). Subsequent runs are instant.
- **Cancel granularity**: cancel takes effect at the next frame boundary, not instantly.

## 7. Regression Tests for Past Bugs

These specific scenarios reproduce bugs we've fixed. They should NEVER fail again.

| Bug | Scenario | Fixed in |
|---|---|---|
| 0-output for Chinese paths | Run on `D:\路径\video.mp4` | v0.2 |
| Fake min_per_video guarantee | Video with all SSIM-rejected frames → still ≥3 | v0.2 |
| OOM on long videos | Run on >1hr 4K → mem stays <1GB | v0.2 |
| Broken .exe (missing imports) | Double-click .exe → window opens | v0.2 |
| Wrong widget for errors | Trigger error (empty input) → see proper messagebox | v0.2 |
| Missing param label translation | Toggle language → all labels change | v0.2 |