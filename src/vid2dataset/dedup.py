"""Perceptual-hash based deduplication with a global, persistable index.

Uses ``imagehash.phash`` (8x8 DCT-based, 64-bit by default). Tested
extensively in the deduplication literature; cheap and robust to small
crops, brightness shifts, and recompression.

The index supports cross-video deduplication so a near-identical pose
appearing in two different MMD videos only ends up in the dataset once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


def hash_image(
    image_bgr: np.ndarray,
    *,
    hash_size: int = 8,
) -> imagehash.ImageHash:
    """Compute a pHash for an OpenCV BGR image."""
    rgb = image_bgr[:, :, ::-1]
    return imagehash.phash(Image.fromarray(rgb), hash_size=hash_size)


@dataclass
class DedupIndex:
    """Linear-scan pHash index. Fine for ~10-100k images.

    For larger sets swap in a BK-tree; the API stays the same.
    """

    hash_size: int = 8
    distance: int = 5
    hashes: list[imagehash.ImageHash] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hashes)

    def is_duplicate(self, h: imagehash.ImageHash) -> str | None:
        """Return the source path of the first match, or None."""
        for existing, src in zip(self.hashes, self.sources, strict=True):
            if (existing - h) <= self.distance:
                return src
        return None

    def add(self, h: imagehash.ImageHash, source: str) -> None:
        self.hashes.append(h)
        self.sources.append(source)

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: Path | str) -> None:
        """Persist the index as JSON. Hashes are stored as hex strings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hash_size": self.hash_size,
            "distance": self.distance,
            "entries": [
                {"hash": str(h), "source": s}
                for h, s in zip(self.hashes, self.sources, strict=True)
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> DedupIndex:
        path = Path(path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(hash_size=payload.get("hash_size", 8), distance=payload.get("distance", 5))
        for e in payload.get("entries", []):
            idx.hashes.append(imagehash.hex_to_hash(e["hash"]))
            idx.sources.append(e["source"])
        return idx

    @classmethod
    def load_or_new(
        cls,
        path: Path | str | None,
        *,
        hash_size: int,
        distance: int,
    ) -> DedupIndex:
        if path is None:
            return cls(hash_size=hash_size, distance=distance)
        idx = cls.load(path)
        # Honour the *current* run's distance threshold even when reloading.
        idx.hash_size = hash_size
        idx.distance = distance
        return idx
