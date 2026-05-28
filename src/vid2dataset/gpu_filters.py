"""GPU-accelerated batch versions of the slow CPU filters.

Uses PyTorch when available. Auto-detects best device (cuda > mps > cpu).
Includes self-validation: each batch op compares its output to the CPU
implementation on a small probe and falls back to CPU if results diverge.

Designed to drop into the existing pipeline:
- ``BatchSSIMFilter``: replaces ``DiversityFilter`` when GPU is enabled.
- ``BatchColorFilter``: replaces ``ColorDiversityFilter`` when GPU is enabled.
- ``is_gpu_pipeline_available()``: tells caller whether to use this path.

Both batch filters keep the same API as their CPU counterparts:
``is_diverse(frame)``, ``accept(frame)``, ``reset()``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False


def is_torch_available() -> bool:
    return _HAS_TORCH


def best_device() -> str:
    """Pick the best available torch device. Returns 'cuda' / 'mps' / 'cpu'."""
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_gpu_pipeline_available() -> bool:
    """True if torch is installed AND a non-CPU device is available."""
    return _HAS_TORCH and best_device() != "cpu"


def device_summary() -> str:
    """One-line description of the GPU pipeline state."""
    if not _HAS_TORCH:
        return "PyTorch not installed (CPU pipeline only)"
    dev = best_device()
    if dev == "cpu":
        return "PyTorch installed but no GPU detected (CPU pipeline)"
    if dev == "cuda":
        return f"GPU pipeline: CUDA on {torch.cuda.get_device_name(0)}"
    if dev == "mps":
        return "GPU pipeline: Apple Metal (MPS)"
    return f"GPU pipeline: {dev}"


# ── Batch SSIM diversity ──────────────────────────────────────────────


class BatchSSIMFilter:
    """GPU-accelerated diversity filter with the same API as DiversityFilter.

    Stores accepted thumbnails on GPU (or CPU if no GPU). When asked
    ``is_diverse(frame)``, computes SSIM in a single batch matmul against
    all accepted thumbnails, returns True if max SSIM <= threshold.

    Self-validates on first call by comparing GPU vs CPU SSIM on a sample.
    Falls back to CPU forever if outputs diverge.
    """

    THUMB_SIZE = 128

    def __init__(self, *, ssim_threshold: float = 0.85, max_compare: int = 20) -> None:
        self.threshold = ssim_threshold
        self.max_compare = max_compare
        self.device = best_device() if _HAS_TORCH else "cpu"
        # Validated on first use; if False we fall back to CPU SSIM
        self._validated: bool | None = None
        # Stored thumbnails as a single tensor [N, 1, H, W] on device, float32
        self._thumbs = None  # type: ignore

    @staticmethod
    def _to_gray128(frame_bgr: np.ndarray) -> np.ndarray:
        thumb = cv2.resize(frame_bgr, (BatchSSIMFilter.THUMB_SIZE, BatchSSIMFilter.THUMB_SIZE),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float64)

    def _ssim_cpu_pair(self, a: np.ndarray, b: np.ndarray) -> float:
        mu_a, mu_b = a.mean(), b.mean()
        sa, sb = a.std(), b.std()
        sab = ((a - mu_a) * (b - mu_b)).mean()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        return float(((2 * mu_a * mu_b + c1) * (2 * sab + c2))
                     / ((mu_a**2 + mu_b**2 + c1) * (sa**2 + sb**2 + c2)))

    def _ssim_gpu_batch(self, target: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between target [1, 1, H, W] and refs [N, 1, H, W].

        Returns a 1-D tensor of N SSIM values.
        """
        # Mean / std per image
        # target.mean() is scalar; refs.mean(dim=(1,2,3)) is per-ref
        mu_t = target.mean()
        mu_r = refs.mean(dim=(1, 2, 3))
        std_t = target.std(unbiased=False)
        std_r = refs.std(dim=(1, 2, 3), unbiased=False)
        # Cross covariance per ref: mean((t-mu_t) * (r-mu_r))
        t_centered = target - mu_t
        r_centered = refs - mu_r.view(-1, 1, 1, 1)
        sab = (t_centered * r_centered).mean(dim=(1, 2, 3))
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        num = (2 * mu_t * mu_r + c1) * (2 * sab + c2)
        den = (mu_t ** 2 + mu_r ** 2 + c1) * (std_t ** 2 + std_r ** 2 + c2)
        return num / den

    def _validate_once(self, frame_bgr: np.ndarray) -> bool:
        """Compare GPU SSIM to CPU SSIM on the same pair. Mark filter as validated."""
        if not _HAS_TORCH or self.device == "cpu":
            self._validated = False
            return False
        try:
            a = self._to_gray128(frame_bgr)
            # Slightly perturbed version
            b = a + np.random.default_rng(0).normal(0, 5, a.shape)
            cpu = self._ssim_cpu_pair(a, b)
            ta = torch.from_numpy(a.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            tb = torch.from_numpy(b.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            gpu = float(self._ssim_gpu_batch(ta, tb).item())
            ok = abs(gpu - cpu) < 0.01
            self._validated = ok
            if not ok:
                log.warning("GPU SSIM diverges from CPU (%.3f vs %.3f); using CPU", gpu, cpu)
            else:
                log.info("BatchSSIMFilter validated on %s (gpu=%.3f cpu=%.3f)",
                         self.device, gpu, cpu)
            return ok
        except Exception as e:
            log.warning("BatchSSIMFilter validation failed: %s; using CPU", e)
            self._validated = False
            return False

    def is_diverse(self, frame_bgr: np.ndarray) -> bool:
        if self._validated is None:
            self._validate_once(frame_bgr)
        gray = self._to_gray128(frame_bgr)
        if self._thumbs is None or (not _HAS_TORCH or not self._validated):
            # CPU fallback path
            return self._is_diverse_cpu(gray)
        target = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        ssims = self._ssim_gpu_batch(target, self._thumbs)
        return bool((ssims <= self.threshold).all().item())

    def _is_diverse_cpu(self, gray: np.ndarray) -> bool:
        # Used only on first call (no thumbs stored yet) or when GPU fails validation
        if self._thumbs is None:
            return True
        if _HAS_TORCH and isinstance(self._thumbs, torch.Tensor):
            refs = self._thumbs.cpu().numpy().squeeze(1)
        else:
            refs = self._thumbs  # list of np arrays
        return all(self._ssim_cpu_pair(gray, r) <= self.threshold for r in refs)

    def accept(self, frame_bgr: np.ndarray) -> None:
        gray = self._to_gray128(frame_bgr).astype(np.float32)
        if _HAS_TORCH and self._validated:
            new_thumb = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
            if self._thumbs is None:
                self._thumbs = new_thumb
            else:
                self._thumbs = torch.cat([self._thumbs, new_thumb], dim=0)
                if self._thumbs.shape[0] > self.max_compare:
                    self._thumbs = self._thumbs[-self.max_compare:]
        else:
            # CPU list fallback
            if self._thumbs is None:
                self._thumbs = []
            self._thumbs.append(gray)
            if len(self._thumbs) > self.max_compare:
                self._thumbs = self._thumbs[-self.max_compare:]

    def reset(self) -> None:
        self._thumbs = None


# ── Batch HSV color diversity ──────────────────────────────────────────


class BatchColorFilter:
    """GPU-accelerated color diversity filter (HSV histogram + chi-squared).

    Same API as ColorDiversityFilter on the CPU side.
    """

    H_BINS = 16
    S_BINS = 8
    V_BINS = 4

    def __init__(self, *, min_distance: float = 0.08, max_compare: int = 15) -> None:
        self.min_distance = min_distance
        self.max_compare = max_compare
        self.device = best_device() if _HAS_TORCH else "cpu"
        self._validated: bool | None = None
        # Stored fingerprints: tensor [N, 512] on device or list of np arrays
        self._fps = None  # type: ignore

    @staticmethod
    def _hsv(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    def _hist_cpu(self, hsv: np.ndarray) -> np.ndarray:
        h = cv2.calcHist([hsv], [0, 1, 2], None,
                         [self.H_BINS, self.S_BINS, self.V_BINS],
                         [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
        s = h.sum()
        if s > 0:
            h /= s
        return h

    def _chi2_cpu(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = a + b
        mask = denom > 0
        if not mask.any():
            return 0.0
        return float(((a[mask] - b[mask]) ** 2 / denom[mask]).sum())

    def _hist_gpu(self, hsv: np.ndarray) -> torch.Tensor:
        # Compute histogram via torch.histc-style bucketing on H, S, V channels.
        # Build joint 16x8x4 histogram by indexing.
        t = torch.from_numpy(hsv).to(self.device)
        h = (t[..., 0].float() * self.H_BINS / 180.0).long().clamp_(0, self.H_BINS - 1)
        s = (t[..., 1].float() * self.S_BINS / 256.0).long().clamp_(0, self.S_BINS - 1)
        v = (t[..., 2].float() * self.V_BINS / 256.0).long().clamp_(0, self.V_BINS - 1)
        idx = h * (self.S_BINS * self.V_BINS) + s * self.V_BINS + v
        idx = idx.flatten()
        flat = torch.zeros(self.H_BINS * self.S_BINS * self.V_BINS,
                           device=self.device, dtype=torch.float32)
        flat.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        total = flat.sum()
        if total > 0:
            flat = flat / total
        return flat

    def _chi2_gpu_batch(self, target: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        """Compute chi-squared distance between target [D] and each ref [N, D]."""
        denom = refs + target.unsqueeze(0)
        diff = (refs - target.unsqueeze(0)) ** 2
        eps = 1e-12
        return (diff / (denom + eps)).sum(dim=1)

    def _validate_once(self, frame_bgr: np.ndarray) -> bool:
        if not _HAS_TORCH or self.device == "cpu":
            self._validated = False
            return False
        try:
            hsv = self._hsv(frame_bgr)
            cpu = self._hist_cpu(hsv)
            gpu = self._hist_gpu(hsv).cpu().numpy()
            diff = float(np.abs(cpu - gpu).max())
            ok = diff < 1e-3
            self._validated = ok
            if not ok:
                log.warning("GPU color hist diverges from CPU (max abs diff=%.4f)", diff)
            else:
                log.info("BatchColorFilter validated on %s (max diff=%.5f)", self.device, diff)
            return ok
        except Exception as e:
            log.warning("BatchColorFilter validation failed: %s", e)
            self._validated = False
            return False

    def is_diverse(self, frame_bgr: np.ndarray) -> bool:
        if self._validated is None:
            self._validate_once(frame_bgr)
        if self._fps is None:
            return True
        if not _HAS_TORCH or not self._validated:
            return self._is_diverse_cpu(frame_bgr)
        fp = self._hist_gpu(self._hsv(frame_bgr))
        dists = self._chi2_gpu_batch(fp, self._fps)
        return bool((dists >= self.min_distance).all().item())

    def _is_diverse_cpu(self, frame_bgr: np.ndarray) -> bool:
        if self._fps is None:
            return True
        fp = self._hist_cpu(self._hsv(frame_bgr))
        if _HAS_TORCH and isinstance(self._fps, torch.Tensor):
            refs = self._fps.cpu().numpy()
        else:
            refs = self._fps
        return all(self._chi2_cpu(fp, r) >= self.min_distance for r in refs)

    def accept(self, frame_bgr: np.ndarray) -> None:
        if _HAS_TORCH and self._validated:
            fp = self._hist_gpu(self._hsv(frame_bgr)).unsqueeze(0)
            if self._fps is None:
                self._fps = fp
            else:
                self._fps = torch.cat([self._fps, fp], dim=0)
                if self._fps.shape[0] > self.max_compare:
                    self._fps = self._fps[-self.max_compare:]
        else:
            fp = self._hist_cpu(self._hsv(frame_bgr))
            if self._fps is None:
                self._fps = []
            self._fps.append(fp)
            if len(self._fps) > self.max_compare:
                self._fps = self._fps[-self.max_compare:]

    def reset(self) -> None:
        self._fps = None
