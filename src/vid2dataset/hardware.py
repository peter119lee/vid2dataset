"""Hardware-aware safety: auto-detect safe worker count + memory monitoring.

Goal: never crash a user's machine due to OOM. Always pick a worker count
that fits in available RAM, given the resolution of the videos to process.

Strategy:
1. Probe a few videos to get representative resolution.
2. Estimate per-worker peak RAM (decoded frame + filter buffers + diversity thumbnails).
3. Pick workers = min(cpu_cores // 2, available_ram_gb / per_worker_gb, video_count, 4).
4. Always >= 1.

Memory model (per worker, approximate):
- One full-res decoded BGR frame: w * h * 3 bytes
- backup_heap of 9 frames at bucket size: ~30MB for 1024-bucket
- diversity filter thumbnails: max_compare * 49KB ~= 1MB
- ffmpeg subprocess buffers: ~50MB
- Python overhead: ~100MB
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Conservative per-worker RAM cost in GB based on source resolution.
# Includes decode buffer, backup heap, filters, ffmpeg subprocess, Python overhead.
# Per-worker RAM cost in GB by source LONG-edge resolution (16:9 reference).
# Includes decoded frame, backup heap, diversity buffers, ffmpeg subprocess,
# Python overhead, plus ~30%% safety margin.
RAM_PER_WORKER_GB = {
    854:  0.2,    # 480p
    1280: 0.3,    # 720p
    1920: 0.5,    # 1080p
    2560: 0.8,    # 1440p
    4096: 1.2,    # 4K
    7680: 2.5,    # 8K
}


@dataclass(frozen=True)
class HardwareInfo:
    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float

    def __str__(self) -> str:
        return (
            f"{self.cpu_cores} cores, "
            f"{self.available_ram_gb:.1f}/{self.total_ram_gb:.1f} GB RAM"
        )


def detect_hardware() -> HardwareInfo:
    """Probe the running machine for CPU + RAM info."""
    cpu = os.cpu_count() or 2
    try:
        import psutil
        vm = psutil.virtual_memory()
        return HardwareInfo(
            cpu_cores=cpu,
            total_ram_gb=vm.total / (1024**3),
            available_ram_gb=vm.available / (1024**3),
        )
    except ImportError:
        # Fallback: assume modest 8GB / 4-core machine
        log.warning("psutil not installed; assuming 8GB RAM / %d cores", cpu)
        return HardwareInfo(cpu_cores=cpu, total_ram_gb=8.0, available_ram_gb=4.0)


def estimate_per_worker_ram(video_long_edge: int) -> float:
    """Return estimated peak RAM (GB) per worker for a given source resolution."""
    if video_long_edge <= 0:
        return 1.0  # unknown, assume 1080p-ish
    # Find the closest tier (round up to be safe).
    for tier_long_edge in sorted(RAM_PER_WORKER_GB.keys()):
        if video_long_edge <= tier_long_edge:
            return RAM_PER_WORKER_GB[tier_long_edge]
    # Larger than our biggest tier: extrapolate.
    return RAM_PER_WORKER_GB[max(RAM_PER_WORKER_GB.keys())]


def probe_max_resolution(videos: list[Path], sample_count: int = 3) -> int:
    """Return the longest edge among a sample of videos. 0 if all probes fail."""
    import cv2  # local import to avoid forcing heavy dep at module load
    sampled = videos[:sample_count] if len(videos) > sample_count else videos
    max_edge = 0
    for v in sampled:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        max_edge = max(max_edge, w, h)
    return max_edge


def auto_detect_workers(
    videos: list[Path],
    *,
    max_cap: int | None = None,
    user_override: int | None = None,
) -> tuple[int, str]:
    """Compute a safe number of parallel workers.

    Args:
        videos: List of videos to process (used for resolution probe).
        max_cap: Hard upper bound on worker count.
        user_override: If not None, just clamp this value to [1, len(videos)]
            and return it (with a note explaining no auto-detection ran).

    Returns:
        (worker_count, explanation_string) - explanation suitable for showing
        in the GUI / log so the user understands why this number was picked.
    """
    if not videos:
        return 1, "no videos"

    if user_override is not None:
        wc = max(1, min(user_override, len(videos), max_cap * 2))
        return wc, f"user override: {wc}"

    hw = detect_hardware()
    long_edge = probe_max_resolution(videos)
    per_worker_gb = estimate_per_worker_ram(long_edge)

    # Hardware-scaled cap: leave 2 cores for system+GUI, never exceed 16
    # (disk I/O bottleneck plateaus past ~12 workers on most SSDs).
    if max_cap is None:
        max_cap = max(1, min(hw.cpu_cores - 2, 16))

    # Reserve 2GB RAM for system + GUI
    usable_ram = max(0.0, hw.available_ram_gb - 2.0)
    ram_limit = max(1, int(usable_ram / per_worker_gb))
    # Use most CPU cores but keep 2 for system on weak machines
    cpu_limit = max(1, hw.cpu_cores - 2 if hw.cpu_cores >= 4 else hw.cpu_cores)
    video_limit = len(videos)

    workers = min(ram_limit, cpu_limit, video_limit, max_cap)
    workers = max(1, workers)

    explanation = (
        f"auto: {workers} workers "
        f"({hw.cpu_cores} cores, {hw.available_ram_gb:.1f}GB free, "
        f"~{per_worker_gb:.1f}GB/worker @ {long_edge}p)"
    )
    log.info("Hardware probe: %s", explanation)
    return workers, explanation


def memory_pressure() -> float:
    """Return current memory pressure as 0.0..1.0 (1.0 = critically full).

    Used to trigger graceful degradation if memory gets tight mid-run.
    Returns 0.0 if psutil isn't available.
    """
    try:
        import psutil
        return psutil.virtual_memory().percent / 100.0
    except ImportError:
        return 0.0
