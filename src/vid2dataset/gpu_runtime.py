"""On-demand GPU runtime downloader.

When the user enables GPU acceleration, this module makes sure PyTorch
(with CUDA support) is available. If not, downloads the wheel files from
PyPI / PyTorch CDN to a per-user cache and adds them to ``sys.path``.

This lets the .exe stay small (~150 MB) while still offering GPU
acceleration for users who want it. The 2.4 GB CUDA payload is downloaded
once and cached at ``%LOCALAPPDATA%/vid2dataset/gpu_runtime``.

Cross-platform notes:
- Windows: works for NVIDIA via CUDA 12.1 wheels.
- Linux: would work with the cu121 wheels too (untested).
- macOS: torch+MPS uses different wheels (we currently do not auto-download
  on macOS; user should pip-install manually).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Where to cache the downloaded runtime
RUNTIME_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local"))) / "vid2dataset" / "gpu_runtime"
)
MANIFEST_FILE = "manifest.json"
RUNTIME_VERSION = "torch2.5.1+cu121"  # bump when we change wheel URLs


# ── Wheel URLs ─────────────────────────────────────────────────────────
# PyPI hosts standard package wheels; PyTorch CDN hosts the CUDA-suffixed torch.
# The python-version part of the URL is filled in at runtime.

def _py_tag() -> str:
    """Return the Python wheel ABI tag for the running interpreter."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


# Spec list: package + version constraint. URLs are looked up on PyPI JSON API.
_PYPI_DEPS = [
    ("typing_extensions", "4.12.2"),
    ("filelock", "3.16.1"),
    ("fsspec", "2024.10.0"),
    ("sympy", "1.13.1"),
    ("mpmath", "1.3.0"),
    ("networkx", "3.4.2"),
    ("jinja2", "3.1.4"),
    ("MarkupSafe", "2.1.5"),
]


def _query_pypi_url(package: str, version: str) -> str:
    """Look up the right wheel URL on PyPI JSON API.

    Picks the platform-appropriate wheel: prefers py3-none-any (universal),
    falls back to cp{py}-cp{py}-win_amd64 for C-extension packages.
    """
    pyt = _py_tag()
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = [u for u in data.get("urls", []) if u.get("packagetype") == "bdist_wheel"]
    # Universal wheel first
    for u in candidates:
        if "py3-none-any" in u["filename"]:
            return u["url"]
    # Platform-specific wheel
    for u in candidates:
        fn = u["filename"]
        if pyt in fn and "win_amd64" in fn:
            return u["url"]
    raise RuntimeError(f"No suitable wheel for {package}=={version} (Python {pyt})")


def _wheel_urls() -> dict[str, str]:
    pyt = _py_tag()
    urls: dict[str, str] = {
        "torch": (
            f"https://download.pytorch.org/whl/cu121/"
            f"torch-2.5.1+cu121-{pyt}-{pyt}-win_amd64.whl"
        ),
    }
    for pkg, ver in _PYPI_DEPS:
        urls[pkg] = _query_pypi_url(pkg, ver)
    return urls


# ── Public API ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeStatus:
    available: bool
    cached: bool
    version: str | None
    cache_dir: Path
    size_mb: float


def runtime_status() -> RuntimeStatus:
    """Inspect the cache without modifying anything."""
    manifest_path = RUNTIME_DIR / MANIFEST_FILE
    cached = False
    version = None
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            version = str(data.get("version", ""))
            cached = version == RUNTIME_VERSION and (RUNTIME_DIR / "torch" / "__init__.py").exists()
        except Exception:
            pass

    # Quick check: can we already import torch with CUDA?
    available = False
    try:
        import torch  # type: ignore[import-not-found]
        available = bool(getattr(torch.cuda, "is_available", lambda: False)())
    except ImportError:
        pass

    size_mb = 0.0
    if RUNTIME_DIR.exists():
        size_mb = sum(p.stat().st_size for p in RUNTIME_DIR.rglob("*") if p.is_file()) / (1024 * 1024)

    return RuntimeStatus(
        available=available,
        cached=cached,
        version=version,
        cache_dir=RUNTIME_DIR,
        size_mb=size_mb,
    )


def total_download_size_mb() -> int:
    """Approximate total download size (for showing in confirm dialog)."""
    return 2400  # ~2.4 GB, dominated by torch+cu121


ProgressCallback = Callable[[str, int, int], None]
"""``cb(stage, bytes_done, bytes_total)`` -- bytes_total may be 0 if unknown."""


def download_runtime(progress: ProgressCallback | None = None) -> bool:
    """Download wheels into the cache and extract them.

    Returns True on success. On failure, leaves the cache directory in
    a partial state which the next call can clean up and retry.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    wheels = _wheel_urls()

    for name, url in wheels.items():
        log.info("Downloading %s from %s", name, url)
        target = RUNTIME_DIR / "_wheels" / f"{name}.whl"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_with_progress(url, target, name, progress)
        except Exception as e:
            log.error("Download failed for %s: %s", name, e)
            return False

    # Extract each wheel to RUNTIME_DIR (so it ends up like a site-packages tree)
    for name in wheels:
        whl = RUNTIME_DIR / "_wheels" / f"{name}.whl"
        if not whl.exists():
            return False
        try:
            with zipfile.ZipFile(whl) as z:
                z.extractall(RUNTIME_DIR)
        except Exception as e:
            log.error("Extract failed for %s: %s", name, e)
            return False

    # Clean up wheel files (saved ~2 GB after extraction)
    shutil.rmtree(RUNTIME_DIR / "_wheels", ignore_errors=True)

    # Write manifest
    (RUNTIME_DIR / MANIFEST_FILE).write_text(
        json.dumps({"version": RUNTIME_VERSION}, indent=2),
        encoding="utf-8",
    )
    log.info("GPU runtime ready at %s", RUNTIME_DIR)
    return True


def _download_with_progress(
    url: str,
    target: Path,
    label: str,
    progress: ProgressCallback | None,
) -> None:
    if target.exists() and target.stat().st_size > 0:
        log.debug("%s already downloaded (%d bytes)", label, target.stat().st_size)
        return

    req = urllib.request.Request(url, headers={"User-Agent": "vid2dataset/0.8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 256 * 1024
        with target.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress:
                    progress(label, done, total)


def activate_runtime() -> bool:
    """Add the runtime to ``sys.path`` and configure DLL loading.

    Returns True if torch becomes importable after this call.
    """
    if not (RUNTIME_DIR / "torch" / "__init__.py").exists():
        return False

    runtime_str = str(RUNTIME_DIR)
    if runtime_str not in sys.path:
        sys.path.insert(0, runtime_str)

    # CUDA dlls live under torch/lib. Tell Windows to look there.
    if sys.platform == "win32":
        cuda_dll_dir = RUNTIME_DIR / "torch" / "lib"
        if cuda_dll_dir.exists():
            try:
                os.add_dll_directory(str(cuda_dll_dir))
            except Exception as e:
                log.warning("Could not add DLL directory: %s", e)

    # Verify
    try:
        import torch  # noqa: F401  type: ignore[import-not-found]
        return True
    except ImportError as e:
        log.error("Failed to import torch from runtime: %s", e)
        return False


def remove_runtime() -> bool:
    """Delete the cache. Useful for re-downloading after a corrupt install."""
    if RUNTIME_DIR.exists():
        try:
            shutil.rmtree(RUNTIME_DIR)
            return True
        except Exception as e:
            log.error("Failed to remove runtime: %s", e)
    return False
