# Roadmap — approved "do all" (2026-07-11)

User approved the full PM assessment plus their own feature request (Advanced
mode). Execution order below. Each release follows the repo culture: implement
→ unit tests → real E2E on this machine → rebuild .exe → release.

## v1.1 — Caption quality (P0, small, ships first)

New `ExtractConfig` fields (all also `vid2dataset tag` CLI flags + GUI row):
- `tag_blacklist: str` — comma-separated tags never written to captions
  (match on formatted form, case-insensitive).
- `tag_always: str` — comma-separated tags appended right after the trigger
  word in every caption (e.g. `anime screencap`).
- `trait_prune_threshold: float` (0 = off) — after tagging all images, any tag
  whose frequency ≥ threshold (e.g. 0.8 = 80% of tagged images) is removed
  from ALL captions, so constant traits are absorbed by the trigger word
  (standard character-LoRA practice). Trigger + tag_always are never pruned.
- `tag_require: str` — image must have ALL of these tags (AND), else it is
  moved to `output/_rejected/` (no sidecar written).
- `tag_exclude: str` — image having ANY of these is moved to `_rejected/`.
  (blacklist = remove tag from caption; exclude = reject whole image.)

Implementation notes:
- tag_folder becomes two-phase: tag everything → compute formatted-token
  frequencies → filter/move/write. TagSummary gains `rejected: list[str]`
  (rel paths) + counts; extractor must drop rejected paths from
  all_image_paths/gallery_meta so the gallery has no dead links.
- Matching happens on formatted tokens (underscore→space), case-insensitive,
  against character+general tags only (never rating).
- Report: show rejected-by-tags count + pruned tag list.

## v1.2 — Advanced mode (USER REQUEST: segment cut + manual capture)

"Not just one click": a CTkToplevel window opened from a new **Advanced**
button in the app.
- **Video scrubber**: video list (from input folder) + preview canvas
  (cv2.VideoCapture seek → CTkImage) + timeline slider + step buttons
  (−10s/−1s/−1f/+1f/+1s/+10s). No realtime playback needed — scrubbing only.
- **Segment cut**: [Set In]/[Set Out] add a (start_s, end_s) range to a
  per-video segment list (editable/removable). Extraction then samples ONLY
  inside segments. Config: `segments: dict[str, list[tuple[float, float]]]`
  keyed by video filename; extractor filters candidate frames by
  frame_index/fps at one choke point (works for scene/interval/keyframe
  modes). CLI: `--segment "name.mp4:30-135"` repeatable.
- **Manual capture**: [Capture] runs the CURRENT frame through the same
  output path (letterbox crop → bucket resize → save) — bypasses
  quality/diversity filters (user chose the frame deliberately), still
  participates in tagging (tag_folder rescans output). Needs a small
  extractor refactor exposing `process_single_frame(cfg, frame_bgr,
  video_stem, seq) -> Path | None`.
- pHash dedup for manual captures: OFF (explicit user intent wins).

## v1.3 — Review mode (approve/reject inside the app)

Thumbnail wall (CTkScrollableFrame) over the output folder; keyboard
keep/reject; reject deletes image + .txt sidecar together and updates
_run_summary; optional move-to-`_rejected/` instead of delete.

## v1.4 — Face suite

Anime face detection ONNX (small model, same on-demand download pattern):
- reject no-face frames (opt-in, character presets)
- face-centered bucket cropping
- auto closeup-variant folder (full body + portrait mix is standard practice)

## v1.5 — Character match + smart selection

- Reference-image character filtering: user supplies ~3 reference images;
  keep only frames whose detected face embedding matches (ccip-style).
- Aesthetic scoring + "give me the best N" target-count auto-tuning.

## v2.0 — Strategic

- **Video-clip mode**: output short clips + captions for video-LoRA training
  (WAN etc.). New pipeline branch; the tool is literally named vid2dataset.
- **Kill the torch GPU runtime**: reimplement GPU SSIM on
  onnxruntime-DirectML (or accept CPU); removes the 2.5 GB download and a
  whole downloader; unifies on one 25 MB runtime for every GPU vendor.

## Deliberate NOs (documented in README Scope)

- yt-dlp / video downloading (legal + maintenance surface; users bring files)
- full tag editor (that's sd-image-sorter's job — do not rebuild it here)
- training integration, upscaling, image editing, prompt tools
- Code signing: desirable (SmartScreen friction) but costs money — user's call.
