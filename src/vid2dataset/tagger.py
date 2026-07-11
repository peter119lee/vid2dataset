"""WD tagger: anime image tagging that turns extracted frames into LoRA captions.

Slim port of the WD14 tagger from the author's sd-image-sorter project,
keeping the hard-won correctness bits (exact SmilingWolf preprocessing,
corrupt-model self-healing, GPU->CPU fallback) and dropping the web-app
machinery. Design + boundary rules: docs/v1.0_tagging_design.md.

Boundary contract: this module is a strictly optional post-pipeline pass.
The extraction core never imports it; deleting it leaves extraction working.

Caption format follows kohya's reference wd14 script: underscores become
spaces (except kaomoji tags), parentheses are kept literal (training-side
caption readers do not parse prompt weighting), captions are a single
comma-joined line, and rating tags never enter captions.

onnxruntime is provided by ``tagger_runtime`` (pip install for source users,
on-demand download in the .exe). numpy + PIL are already app dependencies.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────
# All WD v3 models share one contract: NHWC 448x448 BGR input over a white
# letterbox, sigmoid probabilities out, selected_tags.csv with categories
# 0=general, 3=copyright, 4=character, 9=rating. No per-model branching.

MODELS_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
    / "vid2dataset"
    / "tagger_models"
)
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"

# huggingface.co first, hf-mirror.com for users behind the GFW (same layout).
HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]


@dataclass(frozen=True)
class TaggerModelSpec:
    repo_id: str
    approx_mb: int  # measured model.onnx size, shown in the download prompt


TAGGER_MODELS: dict[str, TaggerModelSpec] = {
    # Highest accuracy of the WD v3 family; heavier download + slower on CPU.
    "wd-eva02-large-tagger-v3": TaggerModelSpec("SmilingWolf/wd-eva02-large-tagger-v3", 1202),
    # Lighter alternative, still strong.
    "wd-swinv2-tagger-v3": TaggerModelSpec("SmilingWolf/wd-swinv2-tagger-v3", 446),
}
DEFAULT_TAGGER_MODEL = "wd-eva02-large-tagger-v3"

# Recreate the ONNX session periodically on GPU: onnxruntime exposes no VRAM
# release API and long DirectML/CUDA runs slowly leak device memory (learned
# in sd-image-sorter, where it eventually BSOD'd a machine around ~300 images).
_SESSION_REFRESH_IMAGES = 200

ProgressCallback = Callable[[str, int, int], None]
"""``cb(stage, done, total)`` -- ``total`` may be 0 when unknown."""


# ── Caption formatting (kohya conventions) ─────────────────────────────

# Danbooru kaomoji tags whose underscores are structural, not word breaks.
# Converting these (e.g. ^_^ -> "^ ^") corrupts the tag — audit finding F3.
KAOMOJI = frozenset(
    {
        "0_0",
        "(o)_(o)",
        "+_+",
        "+_-",
        "._.",
        "<o>_<o>",
        "<|>_<|>",
        "=_=",
        ">_<",
        "3_3",
        "6_9",
        ">_o",
        "@_@",
        "^_^",
        "o_o",
        "u_u",
        "x_x",
        "|_|",
        "||_||",
    }
)


def format_tag(tag: str) -> str:
    """Booru tag -> caption token: underscores to spaces, kaomoji untouched."""
    if tag in KAOMOJI:
        return tag
    return tag.replace("_", " ")


def compose_caption(
    trigger_word: str,
    character_tags: list[tuple[str, float]],
    general_tags: list[tuple[str, float]],
) -> str:
    """Build a single-line caption: trigger, character tags, general tags.

    Tags arrive (name, confidence) sorted by confidence descending. The
    result is guaranteed single-line and case-insensitively deduplicated
    (first occurrence wins, so the trigger word is never repeated as a tag).
    """
    # Collapse any whitespace (incl. newlines) the user typed into the trigger.
    trigger = " ".join(trigger_word.split())
    parts: list[str] = []
    seen: set[str] = set()
    if trigger:
        parts.append(trigger)
        seen.add(trigger.lower())
    for name, _conf in [*character_tags, *general_tags]:
        token = format_tag(name)
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(token)
    return ", ".join(parts)


def write_sidecar(image_path: Path, caption: str) -> Path:
    """Write the caption next to the image as UTF-8 with LF line endings.

    Trainers read only the first line of a sidecar; keeping the file to one
    LF-terminated line avoids the CRLF / multi-line traps (audit F2/F6).
    """
    txt = image_path.with_suffix(".txt")
    txt.write_text(caption + "\n", encoding="utf-8", newline="\n")
    return txt


# ── Model files: status + download ─────────────────────────────────────


def _model_file_valid(model_path: Path) -> bool:
    """Cheap corruption check: a real WD v3 ONNX graph is hundreds of MB."""
    try:
        return model_path.stat().st_size > 1024 * 1024
    except OSError:
        return False


def model_status(model_name: str = DEFAULT_TAGGER_MODEL) -> tuple[bool, Path]:
    """Return (is_downloaded_and_plausible, model_dir) without touching the network."""
    model_dir = MODELS_DIR / model_name
    ok = (
        (model_dir / TAGS_FILE).exists()
        and (model_dir / MODEL_FILE).exists()
        and _model_file_valid(model_dir / MODEL_FILE)
    )
    return ok, model_dir


def download_size_mb(model_name: str = DEFAULT_TAGGER_MODEL) -> int:
    spec = TAGGER_MODELS.get(model_name)
    return spec.approx_mb if spec else 1300


# Serializes model downloads: two threads (e.g. GUI first-enable + a run's
# tagging pass) writing the same .part file interleave into a corrupt model.
_download_lock = threading.Lock()


def download_model(
    model_name: str = DEFAULT_TAGGER_MODEL,
    progress: ProgressCallback | None = None,
) -> Path:
    """Fetch model.onnx + selected_tags.csv into the cache, with mirror fallback.

    Thread-safe: concurrent callers serialize, and the second caller finds the
    files already present and returns immediately.
    """
    spec = TAGGER_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown tagger model: {model_name}. Available: {list(TAGGER_MODELS)}")
    with _download_lock:
        return _download_model_locked(spec, model_name, progress)


def _download_model_locked(
    spec: TaggerModelSpec,
    model_name: str,
    progress: ProgressCallback | None,
) -> Path:
    # Reuse the battle-tested downloader (.part + rename, progress callback).
    from vid2dataset.gpu_runtime import _download_with_progress

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    # Tags file first: it is tiny, so connectivity problems fail fast before
    # the multi-hundred-MB model download starts.
    for fname in (TAGS_FILE, MODEL_FILE):
        target = model_dir / fname
        if target.exists():
            if fname == MODEL_FILE and not _model_file_valid(target):
                target.unlink(missing_ok=True)  # partial/corrupt: re-fetch
            else:
                continue
        last_error: Exception | None = None
        for endpoint in HF_ENDPOINTS:
            url = f"{endpoint}/{spec.repo_id}/resolve/main/{fname}"
            try:
                _download_with_progress(url, target, fname, progress)
                last_error = None
                break
            except Exception as e:  # try next mirror
                last_error = e
                log.warning("Download of %s failed via %s: %s", fname, endpoint, e)
        if last_error is not None:
            raise RuntimeError(
                f"Could not download {fname} for {model_name} from any mirror: {last_error}"
            )
    if not _model_file_valid(model_dir / MODEL_FILE):
        raise RuntimeError(f"Downloaded {MODEL_FILE} looks invalid (too small); please retry.")
    return model_dir


# ── Tag vocabulary ─────────────────────────────────────────────────────


@dataclass
class TagVocabulary:
    general: list[tuple[int, str]] = field(default_factory=list)
    character: list[tuple[int, str]] = field(default_factory=list)
    copyright: list[tuple[int, str]] = field(default_factory=list)
    rating: list[tuple[int, str]] = field(default_factory=list)


def load_tag_vocabulary(tags_path: Path) -> TagVocabulary:
    """Parse selected_tags.csv (index order == model output order)."""
    vocab = TagVocabulary()
    with tags_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return vocab
    header = [str(c or "").strip().lower() for c in rows[0]]
    has_header = "name" in header and "category" in header
    name_i = header.index("name") if has_header else 1
    cat_i = header.index("category") if has_header else 2
    data = rows[1:] if has_header else rows
    for idx, parts in enumerate(data):
        if not parts or len(parts) <= max(name_i, cat_i):
            continue
        try:
            category = int(parts[cat_i])
        except ValueError:
            continue
        name = parts[name_i]
        if category == 0:
            vocab.general.append((idx, name))
        elif category == 3:
            vocab.copyright.append((idx, name))
        elif category == 4:
            vocab.character.append((idx, name))
        elif category == 9:
            vocab.rating.append((idx, name))
    return vocab


# ── Preprocessing (exact SmilingWolf-compatible pipeline) ──────────────


def preprocess_image(image: Image.Image, size: int = 448) -> np.ndarray:
    """PIL image -> HWC float32 BGR array on a white letterbox.

    Exact port of the audited implementation: transparent pixels are
    composited onto white (bare convert("RGB") turns them black), the
    letterbox resample is BICUBIC (LANCZOS shifts tag confidences), and
    channels are BGR in the 0-255 range with no normalization.
    """
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        image = canvas.convert("RGB")
    else:
        image = image.convert("RGB")

    old_w, old_h = image.size
    ratio = min(float(size) / max(1, old_w), float(size) / max(1, old_h))
    new_size = (int(old_w * ratio), int(old_h * ratio))
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    boxed = Image.new("RGB", (size, size), (255, 255, 255))
    boxed.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))

    arr = np.asarray(boxed, dtype=np.float32)
    return arr[:, :, ::-1]  # RGB -> BGR


# ── Tagger ─────────────────────────────────────────────────────────────


@dataclass
class ImageTags:
    character: list[tuple[str, float]]  # (raw booru tag, confidence), conf desc
    general: list[tuple[str, float]]
    rating: str | None
    error: str | None = None


class WDTagger:
    """ONNX WD v3 tagger with GPU->CPU fallback and corrupt-model self-healing.

    ``session`` and ``vocabulary`` are injectable so tests can run the full
    tagging flow without onnxruntime or a real model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TAGGER_MODEL,
        *,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        use_gpu: bool = True,
        session: Any | None = None,
        vocabulary: TagVocabulary | None = None,
    ) -> None:
        self.model_name = model_name
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.use_gpu = use_gpu
        self._session: Any = session
        self._vocab = vocabulary
        self._input_name = "input"
        self._input_size = 448
        self._images_since_create = 0
        self._lock = threading.Lock()
        if session is not None:
            self._refresh_session_metadata()

    # -- session lifecycle ------------------------------------------------

    def load(self) -> None:
        """Idempotently load vocabulary + ONNX session (model must be downloaded)."""
        with self._lock:
            if self._session is not None and self._vocab is not None:
                return
            ok, model_dir = model_status(self.model_name)
            if not ok:
                raise RuntimeError(
                    f"Tagger model {self.model_name} is not downloaded. "
                    "Call download_model() first."
                )
            if self._vocab is None:
                self._vocab = load_tag_vocabulary(model_dir / TAGS_FILE)
            if self._session is None:
                self._session = self._create_session(model_dir / MODEL_FILE)
                self._refresh_session_metadata()

    def _create_session(self, model_path: Path, *, force_cpu: bool = False) -> Any:
        import onnxruntime as ort  # provided by tagger_runtime before load()

        options = ort.SessionOptions()
        cpu_count = max(1, os.cpu_count() or 4)
        gpu = self.use_gpu and not force_cpu
        # GPU mode barely uses CPU threads; CPU mode leaves headroom instead of
        # pinning all cores (a long all-core run can trip marginal hardware).
        threads = 2 if gpu else min(cpu_count, max(2, (cpu_count // 2) - 1))
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = max(1, threads // 2)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # DirectML covers every Windows GPU vendor (incl. NVIDIA) without the
        # cuDNN baggage of the CUDA provider; elsewhere the list collapses to CPU.
        wanted = (
            ["DmlExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        )
        available = ort.get_available_providers()
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=providers
            )
        except Exception as e:
            msg = str(e)
            if "INVALID_PROTOBUF" in msg or "Protobuf parsing failed" in msg:
                # Corrupt download: delete, re-fetch once, retry (ported self-heal).
                log.warning("Tagger model corrupted, re-downloading: %s", model_path)
                model_path.unlink(missing_ok=True)
                download_model(self.model_name)
                session = ort.InferenceSession(
                    str(model_path), sess_options=options, providers=providers
                )
            else:
                raise
        log.info("Tagger session ready, providers: %s", session.get_providers())
        return session

    def _refresh_session_metadata(self) -> None:
        session = self._session
        if session is None or not hasattr(session, "get_inputs"):
            return
        info = session.get_inputs()[0]
        self._input_name = info.name
        shape = list(info.shape or [])
        # WD v3 is NHWC; index 1 is height when concrete.
        if len(shape) == 4 and isinstance(shape[1], int) and shape[1] > 0:
            self._input_size = int(shape[1])

    def _session_uses_gpu(self) -> bool:
        session = self._session
        if session is None or not hasattr(session, "get_providers"):
            return False
        return "DmlExecutionProvider" in session.get_providers()

    def _fallback_to_cpu(self, error: Exception) -> None:
        log.warning("GPU tagging failed, rebuilding session on CPU: %s", error)
        _, model_dir = model_status(self.model_name)
        self._session = self._create_session(model_dir / MODEL_FILE, force_cpu=True)
        self._refresh_session_metadata()
        self.use_gpu = False

    def _maybe_refresh_session(self) -> None:
        if not self._session_uses_gpu():
            return
        if self._images_since_create < _SESSION_REFRESH_IMAGES:
            return
        log.info("Recreating tagger session after %d images", self._images_since_create)
        _, model_dir = model_status(self.model_name)
        self._session = None
        gc.collect()
        self._session = self._create_session(model_dir / MODEL_FILE)
        self._refresh_session_metadata()
        self._images_since_create = 0

    # -- inference ---------------------------------------------------------

    def _probs_to_tags(self, probs: np.ndarray) -> ImageTags:
        assert self._vocab is not None
        values = np.asarray(probs, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
        n = len(values)

        def pick(pairs: list[tuple[int, str]], threshold: float) -> list[tuple[str, float]]:
            hits = [
                (name, float(values[i])) for i, name in pairs if i < n and values[i] >= threshold
            ]
            hits.sort(key=lambda item: item[1], reverse=True)
            return hits

        general = pick(self._vocab.general, self.general_threshold)
        character = pick(self._vocab.character, self.character_threshold)
        rating = None
        rated = [(name, float(values[i])) for i, name in self._vocab.rating if i < n]
        if rated:
            rating = max(rated, key=lambda item: item[1])[0]
        return ImageTags(character=character, general=general, rating=rating)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        assert self._session is not None
        return self._session.run(None, {self._input_name: batch})[0]

    def tag_paths(
        self,
        paths: list[Path],
        *,
        batch_size: int = 8,
        progress_cb: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[ImageTags]:
        """Tag images; per-image failures become ImageTags(error=...), never raise.

        Images the loop never reached (cancelled runs) come back with
        ``error="cancelled"`` — callers must NOT write sidecars for them, or a
        trigger word would produce caption files for images the model never saw.
        """
        self.load()
        results: list[ImageTags] = [
            ImageTags(character=[], general=[], rating=None, error="cancelled") for _ in paths
        ]

        def preprocess_one(path: Path) -> np.ndarray:
            with Image.open(path) as img:
                return preprocess_image(img, self._input_size)

        cursor = 0
        chunk = max(1, batch_size)
        # PIL decode/resize releases the GIL, so a small pool keeps the GPU fed.
        pool = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2))
        try:
            while cursor < len(paths):
                if cancel_event is not None and cancel_event.is_set():
                    break
                group = paths[cursor : cursor + chunk]
                # A retry after a failed batch re-preprocesses the same group
                # (rare, and a group is at most `batch_size` images), so
                # progress is counted only when the cursor advances below.
                prepared: list[tuple[int, Any]] = []
                for offset, item in enumerate(pool.map(_swallow(preprocess_one), group)):
                    idx = cursor + offset
                    if isinstance(item, Exception):
                        log.error("Preprocess failed for %s: %s", group[offset], item)
                        results[idx] = ImageTags([], [], None, error=str(item))
                    else:
                        prepared.append((idx, item))

                if prepared:
                    try:
                        batch = np.stack([arr for _, arr in prepared], axis=0)
                        output = self._run(batch)
                        for row, (idx, _arr) in enumerate(prepared):
                            results[idx] = self._probs_to_tags(output[row])
                        self._images_since_create += len(prepared)
                        self._maybe_refresh_session()
                    except Exception as error:
                        if chunk > 1:
                            log.warning("Batch of %d failed (%s); halving batch size", chunk, error)
                            chunk = max(1, chunk // 2)
                            continue  # retry same cursor with smaller chunk
                        if self._session_uses_gpu():
                            self._fallback_to_cpu(error)
                            continue  # retry same cursor on CPU
                        idx = prepared[0][0]
                        log.error("Tagging failed for %s: %s", paths[idx], error)
                        results[idx] = ImageTags([], [], None, error=str(error))
                # Every image in `group` now has a result: advance and report.
                cursor += len(group)
                if progress_cb:
                    progress_cb("tagging", cursor, len(paths))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        return results


def _swallow(fn: Callable[[Path], np.ndarray]) -> Callable[[Path], Any]:
    """Wrap fn so pool.map yields exceptions as values (per-image isolation)."""

    def inner(path: Path) -> Any:
        try:
            return fn(path)
        except Exception as e:
            return e

    return inner


# ── Folder-level API (what the pipeline and CLI call) ──────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class TagSummary:
    tagged: int = 0
    failed: int = 0
    total: int = 0
    cancelled: bool = False
    tag_counts: Counter = field(default_factory=Counter)
    per_image: dict[str, list[str]] = field(default_factory=dict)
    """relative image path -> formatted caption tokens (for gallery/report)."""


def collect_images(folder: Path) -> list[Path]:
    """All taggable images under folder, skipping tool artifacts (_contact_sheet...)."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("_")
    )


def tag_folder(
    folder: Path,
    *,
    model_name: str = DEFAULT_TAGGER_MODEL,
    trigger_word: str = "",
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    use_gpu: bool = True,
    download_if_missing: bool = True,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    tagger: WDTagger | None = None,
) -> TagSummary:
    """Tag every image under ``folder`` and write .txt sidecars beside them.

    ``tagger`` is injectable for tests; by default this ensures onnxruntime
    (via tagger_runtime) and the model files, then runs WDTagger.
    """
    folder = Path(folder)
    images = collect_images(folder)
    summary = TagSummary(total=len(images))
    if not images:
        return summary

    if tagger is None:
        from vid2dataset import tagger_runtime

        tagger_runtime.ensure_onnxruntime(
            progress=progress_cb, download_if_missing=download_if_missing
        )
        ok, _ = model_status(model_name)
        if not ok:
            if not download_if_missing:
                raise RuntimeError(f"Tagger model {model_name} is not downloaded.")
            download_model(model_name, progress=progress_cb)
        tagger = WDTagger(
            model_name,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            use_gpu=use_gpu,
        )

    results = tagger.tag_paths(images, progress_cb=progress_cb, cancel_event=cancel_event)
    for image_path, tags in zip(images, results, strict=True):
        if tags.error == "cancelled":
            # Never reached the model (user cancelled): writing a sidecar here
            # would stamp a trigger-only caption onto an untagged image.
            continue
        if tags.error:
            summary.failed += 1
            continue
        caption = compose_caption(trigger_word, tags.character, tags.general)
        if not caption:
            summary.failed += 1
            continue
        try:
            write_sidecar(image_path, caption)
        except OSError as e:
            log.error("Could not write sidecar for %s: %s", image_path, e)
            summary.failed += 1
            continue
        summary.tagged += 1
        tokens = [t for t in caption.split(", ") if t]
        rel = image_path.relative_to(folder).as_posix()
        summary.per_image[rel] = tokens
        # Frequency counts exclude the trigger word (it is on every image).
        trigger = " ".join(trigger_word.split()).lower()
        summary.tag_counts.update(t for t in tokens if t.lower() != trigger)

    summary.cancelled = bool(cancel_event is not None and cancel_event.is_set())
    return summary
