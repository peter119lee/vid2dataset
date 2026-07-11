"""On-demand onnxruntime for the WD tagger.

Same architecture as gpu_runtime (which is battle-tested end-to-end): the
.exe ships without onnxruntime; the first time the user enables tagging we
download the onnxruntime-directml wheel (~60 MB), extract it into a per-user
cache, and append it to sys.path. DirectML accelerates every Windows GPU
vendor (NVIDIA included) and falls back to CPU automatically.

Source installs should just ``pip install vid2dataset[tag]`` — if onnxruntime
imports, none of this machinery runs. Non-Windows platforms are pip-only
(the DirectML wheel is Windows-specific).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from vid2dataset.gpu_runtime import ProgressCallback, _download_with_progress, _query_pypi_url

log = logging.getLogger(__name__)

RUNTIME_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
    / "vid2dataset"
    / "tagger_runtime"
)
MANIFEST_FILE = "manifest.json"

ORT_PACKAGE = "onnxruntime-directml"
ORT_VERSION = "1.24.4"
RUNTIME_VERSION = f"{ORT_PACKAGE}=={ORT_VERSION}"
RUNTIME_DOWNLOAD_MB = 25  # measured wheel size (23 MB), for confirm dialogs


def onnxruntime_available() -> bool:
    """True when ``import onnxruntime`` already works (pip install or activated cache)."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def runtime_cached() -> bool:
    manifest_path = RUNTIME_DIR / MANIFEST_FILE
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("version") == RUNTIME_VERSION
        and (RUNTIME_DIR / "onnxruntime" / "__init__.py").exists()
    )


def _wheel_target(url: str) -> Path:
    fn = unquote(Path(urlparse(url).path).name)
    if not fn.endswith(".whl"):
        fn = f"{ORT_PACKAGE}.whl"
    return RUNTIME_DIR / "_wheels" / fn


def _clear_stale(keep_wheel: Path) -> None:
    """Wipe leftovers from a different runtime version (mirrors gpu_runtime)."""
    if not RUNTIME_DIR.exists():
        return
    if runtime_cached():
        return
    for p in RUNTIME_DIR.iterdir():
        if p.name == "_wheels":
            for w in list(p.iterdir()):
                if w != keep_wheel:
                    if w.is_dir():
                        shutil.rmtree(w, ignore_errors=True)
                    else:
                        w.unlink(missing_ok=True)
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# Serializes runtime installs: two threads extracting the same wheel into
# RUNTIME_DIR interleave into a half-written package that runtime_cached()
# would then wrongly report as valid.
_download_lock = threading.Lock()


def download_runtime(progress: ProgressCallback | None = None) -> bool:
    """Download + extract the onnxruntime wheel into the cache. Thread-safe."""
    with _download_lock:
        if runtime_cached():
            return True  # another caller finished the install while we waited
        return _download_runtime_locked(progress)


def _download_runtime_locked(progress: ProgressCallback | None) -> bool:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    url = _query_pypi_url(ORT_PACKAGE, ORT_VERSION)
    target = _wheel_target(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    _download_with_progress(url, target, "onnxruntime", progress)

    _clear_stale(keep_wheel=target)
    try:
        with zipfile.ZipFile(target) as z:
            z.extractall(RUNTIME_DIR)
    except Exception as e:
        raise RuntimeError(f"Extract failed for onnxruntime wheel: {e}") from e
    shutil.rmtree(RUNTIME_DIR / "_wheels", ignore_errors=True)
    (RUNTIME_DIR / MANIFEST_FILE).write_text(
        json.dumps({"version": RUNTIME_VERSION}, indent=2), encoding="utf-8"
    )
    log.info("Tagger runtime ready at %s", RUNTIME_DIR)
    return True


def activate_runtime() -> tuple[bool, str]:
    """Append the cache to sys.path and import onnxruntime from it."""
    if not (RUNTIME_DIR / "onnxruntime" / "__init__.py").exists():
        return False, "onnxruntime/__init__.py not found in cache directory"

    runtime_str = str(RUNTIME_DIR)
    if runtime_str not in sys.path:
        sys.path.append(runtime_str)

    capi_dir = RUNTIME_DIR / "onnxruntime" / "capi"
    if sys.platform == "win32" and capi_dir.exists():
        try:
            os.add_dll_directory(str(capi_dir))
        except Exception as e:
            log.warning("Could not add DLL directory: %s", e)

    for mod in list(sys.modules):
        if mod == "onnxruntime" or mod.startswith("onnxruntime."):
            del sys.modules[mod]

    try:
        import onnxruntime  # type: ignore[import-not-found]

        providers = onnxruntime.get_available_providers()
        log.info("onnxruntime %s active, providers: %s", onnxruntime.__version__, providers)
        return True, ""
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def total_first_enable_mb(model_mb: int) -> int:
    """Approximate download for the first enable: model + runtime if needed."""
    if onnxruntime_available() or runtime_cached():
        return model_mb
    return model_mb + RUNTIME_DOWNLOAD_MB


def ensure_onnxruntime(
    progress: ProgressCallback | None = None,
    *,
    download_if_missing: bool = True,
) -> None:
    """Make ``import onnxruntime`` work, downloading the runtime if allowed.

    Raises RuntimeError with a user-appropriate message when it cannot.
    """
    if onnxruntime_available():
        return
    if sys.platform != "win32":
        raise RuntimeError(
            "onnxruntime is not installed. Install tagging support with: "
            "pip install vid2dataset[tag]"
        )
    if not runtime_cached():
        if not download_if_missing:
            raise RuntimeError("Tagger runtime (onnxruntime) is not downloaded yet.")
        download_runtime(progress)
    ok, err = activate_runtime()
    if not ok:
        raise RuntimeError(
            f"Could not load the tagger runtime: {err}. "
            f"Try deleting {RUNTIME_DIR} and re-enabling tagging."
        )
