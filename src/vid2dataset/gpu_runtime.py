"""On-demand GPU runtime downloader.

When the user enables GPU acceleration, this module makes sure PyTorch
(with CUDA support) is available. If not, downloads the wheel files from
PyPI / PyTorch CDN to a per-user cache and adds them to ``sys.path``.

This lets the .exe stay small (~150 MB) while still offering GPU
acceleration for users who want it. The ~2.5 GB CUDA payload is downloaded
once and cached at ``%LOCALAPPDATA%/vid2dataset/gpu_runtime``.

Cross-platform notes:
- Windows: works for NVIDIA via CUDA 12.6 wheels (12.8 for Blackwell / RTX 50xx).
- Linux: would work with the cu126/cu128 wheels too (untested).
- macOS: torch+MPS uses different wheels (we currently do not auto-download
  on macOS; user should pip-install manually).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

log = logging.getLogger(__name__)

# Where to cache the downloaded runtime
RUNTIME_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local")))
    / "vid2dataset"
    / "gpu_runtime"
)
MANIFEST_FILE = "manifest.json"
RUNTIME_VERSION = (
    "torch2.11.0+numpy2.4.6"  # bump when wheel pins change; mismatch invalidates cache
)


# ── Wheel URLs ─────────────────────────────────────────────────────────
# PyPI hosts standard package wheels; PyTorch CDN hosts the CUDA-suffixed torch.
# The python-version part of the URL is filled in at runtime.


def _py_tag() -> str:
    """Return the Python wheel ABI tag for the running interpreter."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


# Spec list: package + version constraint. URLs are looked up on PyPI JSON API.
# Keep these in sync with the versions bundled into the .exe (the build venv):
# numpy especially — see comment below.
_PYPI_DEPS = [
    # numpy first - torch's C extension binds against a specific ABI.
    # Without our own numpy, torch falls back to PyInstaller's bundled
    # numpy (different version) and torch.cuda.is_available() returns False.
    # MUST match the numpy version in the build venv (and RUNTIME_VERSION).
    ("numpy", "2.4.6"),
    # torch >= 2.9 declares setuptools as a hard runtime dep on Python 3.12
    # (distutils removal); torch 2.11 caps it at < 82.
    ("setuptools", "80.9.0"),
    ("typing_extensions", "4.15.0"),
    ("filelock", "3.29.0"),
    ("fsspec", "2026.4.0"),
    ("sympy", "1.14.0"),  # torch >= 2.6 requires sympy >= 1.13.3
    ("mpmath", "1.3.0"),
    ("networkx", "3.6.1"),
    ("jinja2", "3.1.6"),
    ("MarkupSafe", "3.0.3"),
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


# 2.11.0 is the newest torch published for BOTH cu126 and cu128 Windows
# wheels (cu128 builds stopped at 2.11.0; 2.12+ moved to cu130/cu132 which
# require an R580+ driver). One version = one dep matrix.
_TORCH_VERSIONS = {
    "cu126": "2.11.0",
    "cu128": "2.11.0",
}
_DEFAULT_TORCH_VERSION = "2.11.0"


def _resolve_torch_url(cuda_tag: str, torch_ver: str) -> str:
    """Look up the actual torch wheel URL on PyTorch's CDN.

    PyTorch publishes an HTML index at /whl/{cuda_tag}/torch/ with anchors
    to the real files (currently hosted on download-r2.pytorch.org). We
    parse the index, find the matching wheel, and return its URL.
    """
    pyt = _py_tag()
    index_url = f"https://download.pytorch.org/whl/{cuda_tag}/torch/"
    req = urllib.request.Request(
        index_url,
        headers={"User-Agent": "vid2dataset/0.9"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Could not fetch PyTorch index for {cuda_tag}: {e}") from e

    # Filename we want, with + URL-encoded as %2B in the href
    target_fn_plain = f"torch-{torch_ver}+{cuda_tag}-{pyt}-{pyt}-win_amd64.whl"
    target_fn_enc = f"torch-{torch_ver}%2B{cuda_tag}-{pyt}-{pyt}-win_amd64.whl"
    pattern = re.compile(r'href="([^"]+)"')
    for m in pattern.finditer(html):
        href = m.group(1)
        if target_fn_enc in href or target_fn_plain in href:
            # Strip integrity fragment (#sha256=...) so urllib can use the URL
            return href.split("#")[0]
    raise RuntimeError(
        f"Could not find torch wheel for {target_fn_plain} in {index_url}. "
        f"PyTorch may have changed format or the version doesn't exist for "
        f"this CUDA tag + Python version."
    )


def _wheel_urls(cuda_tag: str = "cu126") -> dict[str, str]:
    """Build the wheel URL map for the given CUDA tag."""
    torch_ver = _TORCH_VERSIONS.get(cuda_tag, _DEFAULT_TORCH_VERSION)
    urls: dict[str, str] = {
        "torch": _resolve_torch_url(cuda_tag, torch_ver),
    }
    for pkg, ver in _PYPI_DEPS:
        urls[pkg] = _query_pypi_url(pkg, ver)
    return urls


# ── Hardware / OS detection ────────────────────────────────────────────


@dataclass(frozen=True)
class HardwareProfile:
    vendor: str  # NVIDIA / AMD / Intel / Apple / Unknown
    gpu_name: str  # e.g. "NVIDIA GeForce RTX 3090"
    arch: str  # ampere / ada / blackwell / hopper / turing / etc / "" if unknown
    compute_cap: float  # e.g. 8.6 (NVIDIA only), 0.0 if unknown
    os_name: str  # windows / linux / macos
    os_arch: str  # x86_64 / arm64

    def __str__(self) -> str:
        parts = [self.os_name, self.os_arch]
        if self.gpu_name:
            parts.append(self.gpu_name)
        if self.arch:
            parts.append(f"({self.arch})")
        return " ".join(parts)


def detect_os() -> tuple[str, str]:
    import platform

    plat = platform.system().lower()
    if plat == "darwin":
        plat = "macos"
    arch = platform.machine().lower()
    if arch in ("amd64", "x64"):
        arch = "x86_64"
    return plat, arch


def _run_cmd(args: list[str], timeout: float = 5.0) -> str:
    """Run a command silently and return stdout. Empty string on failure."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="ignore",
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        return result.stdout
    except Exception:
        return ""


def _nvidia_smi() -> dict:
    """Query nvidia-smi for GPU name + compute capability + memory."""
    out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out.strip():
        return {}
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return {}
    name = parts[0]
    try:
        cap = float(parts[1])
    except ValueError:
        cap = 0.0
    return {"name": name, "compute_cap": cap}


def _wmic_gpus() -> list[str]:
    """Windows fallback: list all GPU device names via wmic."""
    out = _run_cmd(["wmic", "path", "win32_VideoController", "get", "name"])
    return [line.strip() for line in out.splitlines()[1:] if line.strip()]


def _classify_nvidia(name: str, compute_cap: float) -> str:
    """Map GPU name / compute capability to a CUDA architecture nickname."""
    n = name.lower()
    # Blackwell consumer: RTX 50xx (CC 10.0 expected)
    if "rtx 50" in n or compute_cap >= 10.0:
        return "blackwell"
    # Hopper: H100, H200 (CC 9.0)
    if "h100" in n or "h200" in n or compute_cap == 9.0:
        return "hopper"
    # Ada Lovelace: RTX 40xx (CC 8.9)
    if "rtx 40" in n or (compute_cap and 8.85 <= compute_cap < 9.0):
        return "ada"
    # Ampere: RTX 30xx, A100 (CC 8.0-8.6)
    if "rtx 30" in n or "a100" in n or (compute_cap and 8.0 <= compute_cap < 8.85):
        return "ampere"
    # Turing: RTX 20xx, GTX 16xx (CC 7.5)
    if "rtx 20" in n or "gtx 16" in n or compute_cap == 7.5:
        return "turing"
    return ""


def detect_gpu() -> HardwareProfile:
    """Inspect the running machine for GPU + OS info."""
    os_name, os_arch = detect_os()

    # Try NVIDIA first
    nv = _nvidia_smi()
    if nv:
        arch = _classify_nvidia(nv["name"], nv["compute_cap"])
        return HardwareProfile(
            vendor="NVIDIA",
            gpu_name=nv["name"],
            arch=arch,
            compute_cap=nv["compute_cap"],
            os_name=os_name,
            os_arch=os_arch,
        )

    # Apple Silicon: arm64 macOS
    if os_name == "macos" and os_arch == "arm64":
        return HardwareProfile(
            vendor="Apple",
            gpu_name="Apple Silicon (MPS)",
            arch="apple",
            compute_cap=0.0,
            os_name=os_name,
            os_arch=os_arch,
        )

    # Windows fallback: enumerate via wmic
    if os_name == "windows":
        gpus = _wmic_gpus()
        for gpu in gpus:
            g = gpu.lower()
            if "amd" in g or "radeon" in g:
                return HardwareProfile(
                    vendor="AMD",
                    gpu_name=gpu,
                    arch="rdna",
                    compute_cap=0.0,
                    os_name=os_name,
                    os_arch=os_arch,
                )
            if "intel" in g and ("arc" in g or "iris" in g or "uhd" in g):
                return HardwareProfile(
                    vendor="Intel",
                    gpu_name=gpu,
                    arch="xe",
                    compute_cap=0.0,
                    os_name=os_name,
                    os_arch=os_arch,
                )

    # Linux fallback: try lspci
    if os_name == "linux":
        out = _run_cmd(["lspci"])
        for line in out.splitlines():
            ll = line.lower()
            if "vga" in ll or "3d controller" in ll or "display" in ll:
                if "nvidia" in ll:
                    return HardwareProfile(
                        vendor="NVIDIA",
                        gpu_name=line.split(":")[-1].strip(),
                        arch="",
                        compute_cap=0.0,
                        os_name=os_name,
                        os_arch=os_arch,
                    )
                if "amd" in ll or "radeon" in ll:
                    return HardwareProfile(
                        vendor="AMD",
                        gpu_name=line.split(":")[-1].strip(),
                        arch="rdna",
                        compute_cap=0.0,
                        os_name=os_name,
                        os_arch=os_arch,
                    )

    return HardwareProfile(
        vendor="Unknown",
        gpu_name="",
        arch="",
        compute_cap=0.0,
        os_name=os_name,
        os_arch=os_arch,
    )


def cuda_version_for_profile(hw: HardwareProfile) -> str | None:
    """Return the right PyTorch CUDA tag for the detected GPU.

    Returns None if the machine can't run CUDA wheels at all (Apple, AMD, Intel,
    no NVIDIA, etc.).
    """
    if hw.vendor != "NVIDIA":
        return None
    if hw.os_name == "macos":
        return None  # macOS NVIDIA is unsupported by modern torch
    # Map architecture -> CUDA wheel tag.
    # cu128 wheels carry sm 7.5-12.0 (Turing through Blackwell); Blackwell
    # (sm_100/sm_120) is NOT in cu126, so RTX 50xx must get cu128.
    if hw.arch == "blackwell":
        return "cu128"
    # cu126 wheels carry sm 5.0-9.0 (Maxwell through Hopper) — the widest
    # legacy coverage, and the same CUDA 12.x driver family as the old cu121.
    return "cu126"


def runtime_supported(hw: HardwareProfile) -> tuple[bool, str]:
    """Can we offer GPU acceleration to this user? Returns (yes/no, reason)."""
    if hw.vendor == "NVIDIA":
        if hw.os_name == "macos":
            return False, "NVIDIA on macOS is not supported by modern PyTorch."
        return True, ""
    if hw.vendor == "Apple":
        return (
            False,
            "Apple Silicon needs PyTorch+MPS via pip install (auto-download not supported on macOS).",
        )
    if hw.vendor == "AMD":
        return (
            False,
            "AMD GPU detected. PyTorch+ROCm only works on Linux and is not auto-downloaded. Use pip install on Linux.",
        )
    if hw.vendor == "Intel":
        return False, "Intel GPU detected. PyTorch does not support Intel Arc/iGPU acceleration."
    return False, "No NVIDIA GPU detected. GPU acceleration requires NVIDIA + CUDA."


# ── Mirror speed selection ─────────────────────────────────────────────


# Verified mirrors that actually serve PyTorch CUDA wheels.
# Tsinghua / Aliyun PyPI mirrors do NOT host the +cuXXX builds, only the CPU
# torch — they're useless for this purpose. SJTU mirrors PyTorch's CDN for
# China users.
PYTORCH_MIRRORS = {
    "official": "https://download.pytorch.org/whl/{cuda}/",
    "r2": "https://download-r2.pytorch.org/whl/{cuda}/",
    "sjtu": "https://mirror.sjtu.edu.cn/pytorch-wheels/{cuda}/",
}


def pick_fastest_mirror(
    cuda_tag: str,
    *,
    test_kb: int = 256,
    timeout: float = 5.0,
) -> tuple[str, str, float]:
    """Race the mirrors with a small partial download from the actual wheel.

    Each mirror is probed by requesting bytes 0..test_kb-1 of the torch wheel
    file. This proves both connectivity AND that the file actually exists on
    that mirror (not just that an index page loads).

    Returns (mirror_name, base_url_template, seconds). Falls back to
    PyTorch official if every mirror times out or 404s.
    """
    pyt = _py_tag()
    torch_ver = _TORCH_VERSIONS.get(cuda_tag, _DEFAULT_TORCH_VERSION)
    test_filename = f"torch-{torch_ver}+{cuda_tag}-{pyt}-{pyt}-win_amd64.whl"
    candidates: list[tuple[str, str, float]] = []
    for name, template in PYTORCH_MIRRORS.items():
        base = template.format(cuda=cuda_tag)
        # PyTorch official redirects to r2 for actual files; we hit r2 directly
        # but for the index probe we pass through the public hostname.
        if name == "official":
            try:
                resolved = _resolve_torch_url(cuda_tag, torch_ver)
                test_url = resolved
            except Exception:
                continue
        else:
            test_url = base.rstrip("/") + "/" + test_filename.replace("+", "%2B")
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(
                test_url,
                headers={
                    "User-Agent": "vid2dataset/0.9",
                    "Range": f"bytes=0-{test_kb * 1024 - 1}",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            elapsed = time.perf_counter() - t0
            candidates.append((name, test_url, elapsed))
        except Exception as e:
            log.debug("Mirror %s failed: %s", name, e)
            continue

    if not candidates:
        # All failed — fall back to PyTorch official, even if untested
        try:
            url = _resolve_torch_url(cuda_tag, torch_ver)
        except Exception:
            url = PYTORCH_MIRRORS["official"].format(cuda=cuda_tag)
        return ("official", url, -1.0)
    candidates.sort(key=lambda x: x[2])
    log.info("Mirror race: %s", [(n, f"{t:.2f}s") for n, _, t in candidates])
    return candidates[0]


# ── Public API ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuntimeStatus:
    available: bool
    cached: bool
    version: str | None
    cache_dir: Path
    size_mb: float
    cuda_tag: str | None = None  # CUDA tag the cache was built for (from manifest)


def runtime_status() -> RuntimeStatus:
    """Inspect the cache without modifying anything."""
    manifest_path = RUNTIME_DIR / MANIFEST_FILE
    cached = False
    version = None
    cuda_tag = None
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            version = str(data.get("version", ""))
            raw_tag = data.get("cuda_tag")
            cuda_tag = str(raw_tag) if raw_tag else None
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
        size_mb = sum(p.stat().st_size for p in RUNTIME_DIR.rglob("*") if p.is_file()) / (
            1024 * 1024
        )

    return RuntimeStatus(
        available=available,
        cached=cached,
        version=version,
        cache_dir=RUNTIME_DIR,
        size_mb=size_mb,
        cuda_tag=cuda_tag,
    )


def total_download_size_mb(cuda_tag: str = "cu126") -> int:
    """Approximate total download size (for showing in confirm dialog)."""
    # torch wheel: cu126 ~2.42 GB, cu128 ~2.57 GB; PyPI deps add ~50 MB.
    return 2700 if cuda_tag == "cu128" else 2500


ProgressCallback = Callable[[str, int, int], None]
"""``cb(stage, bytes_done, bytes_total)`` -- bytes_total may be 0 if unknown."""


def _wheel_target(name: str, url: str) -> Path:
    """Cache path for a wheel, keyed by its real filename.

    Using the actual wheel filename (which embeds version + CUDA tag +
    platform) instead of a generic ``{name}.whl`` means a leftover wheel
    from an older torch version or a different CUDA tag is never reused.
    """
    fn = unquote(Path(urlparse(url).path).name)
    if not fn.endswith(".whl"):
        fn = f"{name}.whl"
    return RUNTIME_DIR / "_wheels" / fn


def _clear_stale_cache(wheels: dict[str, str], cuda_tag: str) -> None:
    """Remove leftovers from a previous runtime before installing.

    A cache is stale when its manifest version differs from RUNTIME_VERSION
    (older torch / pins) or it was built for a different CUDA tag (e.g. the
    user upgraded to a Blackwell card). Extracting a new torch on top of a
    stale one leaves orphaned modules and DLLs behind (files that no longer
    exist in the new version), which breaks imports in subtle ways. Wheels in
    ``_wheels`` whose filenames match the current targets are kept so an
    interrupted download can be retried without re-fetching completed files.
    """
    if not RUNTIME_DIR.exists():
        return
    manifest_path = RUNTIME_DIR / MANIFEST_FILE
    current = None
    current_tag = None
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            current = data.get("version")
            current_tag = data.get("cuda_tag")
        except Exception:
            current = None
    if current == RUNTIME_VERSION and current_tag == cuda_tag:
        return  # cache already holds the target build; nothing stale
    keep = {_wheel_target(n, u) for n, u in wheels.items()}
    for p in RUNTIME_DIR.iterdir():
        if p.name == "_wheels":
            for w in list(p.iterdir()):
                if w in keep:
                    continue
                if w.is_dir():
                    shutil.rmtree(w, ignore_errors=True)
                else:
                    w.unlink(missing_ok=True)
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def download_runtime(
    progress: ProgressCallback | None = None,
    *,
    cuda_tag: str | None = None,
    torch_url: str | None = None,
) -> bool:
    """Download wheels into the cache and extract them.

    Returns True on success. On failure, leaves the cache directory in
    a partial state which the next call can clean up and retry.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if cuda_tag is None:
        hw = detect_gpu()
        cuda_tag = cuda_version_for_profile(hw) or "cu126"
    wheels = _wheel_urls(cuda_tag)
    # If caller provided a specific torch URL (e.g. from a mirror race),
    # override the auto-resolved URL.
    if torch_url:
        wheels["torch"] = torch_url

    for name, url in wheels.items():
        log.info("Downloading %s from %s", name, url)
        target = _wheel_target(name, url)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_with_progress(url, target, name, progress)
        except Exception as e:
            log.error("Download failed for %s: %s", name, e)
            raise RuntimeError(f"Download failed for {name}: {e}") from e

    # Every wheel is on disk now — only at this point wipe leftovers from an
    # older runtime version (or a different CUDA tag) before extracting on
    # top of them. A failed download therefore never destroys a runtime that
    # still works.
    _clear_stale_cache(wheels, cuda_tag)

    # Extract each wheel to RUNTIME_DIR (so it ends up like a site-packages tree)
    for name, url in wheels.items():
        whl = _wheel_target(name, url)
        if not whl.exists():
            raise RuntimeError(f"Wheel file missing after download: {whl.name}")
        try:
            with zipfile.ZipFile(whl) as z:
                z.extractall(RUNTIME_DIR)
        except Exception as e:
            log.error("Extract failed for %s: %s", name, e)
            raise RuntimeError(f"Extract failed for {name}: {e}") from e

    # Clean up wheel files (saved ~2 GB after extraction)
    shutil.rmtree(RUNTIME_DIR / "_wheels", ignore_errors=True)

    # Write manifest
    (RUNTIME_DIR / MANIFEST_FILE).write_text(
        json.dumps({"version": RUNTIME_VERSION, "cuda_tag": cuda_tag}, indent=2),
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

    # Download to a .part file and rename on completion, so an interrupted
    # download can never be mistaken for a finished wheel.
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "vid2dataset/0.9"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 256 * 1024
        with tmp.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress:
                    progress(label, done, total)
    tmp.replace(target)


def activate_runtime() -> tuple[bool, str]:
    """Add the runtime to `sys.path` and configure DLL loading.

    Returns (success, error_message). On success error_message is empty;
    on failure it contains the underlying error text so the GUI can show
    it to the user.
    """
    if not (RUNTIME_DIR / "torch" / "__init__.py").exists():
        return False, "torch/__init__.py not found in cache directory"

    runtime_str = str(RUNTIME_DIR)
    if runtime_str not in sys.path:
        # Append (not insert at 0): PyInstaller-bundled deps win for shared
        # packages like typing_extensions (newer in bundle than what torch
        # would pin). torch is excluded from the .exe so it's only in cache,
        # which means it gets found there regardless of position.
        sys.path.append(runtime_str)

    if sys.platform == "win32":
        cuda_dll_dir = RUNTIME_DIR / "torch" / "lib"
        if cuda_dll_dir.exists():
            try:
                os.add_dll_directory(str(cuda_dll_dir))
            except Exception as e:
                log.warning("Could not add DLL directory: %s", e)

    # Clear any partially-loaded torch first so we definitely use the cache copy.
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]

    try:
        import torch  # type: ignore[import-not-found]

        try:
            available = bool(torch.cuda.is_available())
        except Exception as e:
            return False, f"torch loaded but cuda check failed: {type(e).__name__}: {e}"
        if not available:
            return False, (
                "torch loaded but torch.cuda.is_available() is False. "
                "Likely missing or outdated NVIDIA driver, or the downloaded "
                "CUDA build does not match your GPU."
            )
        return True, ""
    except ImportError as e:
        log.error("Failed to import torch from runtime: %s", e)
        return False, f"ImportError: {e}"
    except Exception as e:
        log.error("Unexpected error activating runtime: %s", e)
        return False, f"{type(e).__name__}: {e}"


def remove_runtime() -> bool:
    """Delete the cache. Useful for re-downloading after a corrupt install."""
    if RUNTIME_DIR.exists():
        try:
            shutil.rmtree(RUNTIME_DIR)
            return True
        except Exception as e:
            log.error("Failed to remove runtime: %s", e)
    return False
