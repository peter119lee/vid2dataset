"""Contact sheet and HTML gallery generation.

After extraction, generate visual QA outputs so users can quickly
inspect the entire dataset without opening hundreds of files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def generate_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    thumb_size: int = 192,
    cols: int = 8,
    max_images: int = 200,
    border: int = 2,
) -> Path | None:
    """Create a single PNG contact sheet from a list of images.

    Returns the output path, or None if no images.
    """
    if not image_paths:
        return None

    paths = image_paths[:max_images]
    rows = (len(paths) + cols - 1) // cols
    cell = thumb_size + border * 2
    sheet_w = cols * cell
    sheet_h = rows * cell
    sheet = np.full((sheet_h, sheet_w, 3), 32, dtype=np.uint8)

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_size / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        row, col = divmod(i, cols)
        y = row * cell + border + (thumb_size - nh) // 2
        x = col * cell + border + (thumb_size - nw) // 2
        sheet[y : y + nh, x : x + nw] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    log.info("Contact sheet: %s (%d images, %dx%d)", output_path, len(paths), sheet_w, sheet_h)
    return output_path


def generate_html_gallery(
    image_paths: list[Path],
    output_path: Path,
    *,
    title: str = "vid2dataset output",
) -> Path | None:
    """Generate a self-contained HTML gallery with lazy-loaded thumbnails.

    Uses relative paths so the HTML works when opened from the output dir.
    """
    if not image_paths:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output_path.parent

    rows: list[str] = []
    for p in image_paths:
        try:
            rel = p.relative_to(base_dir)
        except ValueError:
            rel = p
        rows.append(
            f'<div class="card"><img loading="lazy" src="{rel.as_posix()}" '
            f'alt="{p.stem}"><span>{p.stem}</span></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ background: #1a1a1a; color: #eee; font-family: system-ui; margin: 0; padding: 16px; }}
h1 {{ font-size: 1.4rem; margin-bottom: 12px; }}
.info {{ color: #888; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }}
.card {{ background: #2a2a2a; border-radius: 6px; overflow: hidden; }}
.card img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
.card span {{ display: block; padding: 4px 8px; font-size: 0.7rem; color: #aaa;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="info">{len(image_paths)} images</p>
<div class="grid">
{"".join(rows)}
</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    log.info("HTML gallery: %s (%d images)", output_path, len(image_paths))
    return output_path
