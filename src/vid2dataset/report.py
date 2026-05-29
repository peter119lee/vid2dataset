"""HTML pre-flight report generator.

Produces ``_report.html`` next to the gallery + contact sheet, summarising
the run for trainers: per-video stats, bucket distribution, blur histogram,
watermark warnings, rejection reasons. Purely additive — does not affect
the extracted images at all.
"""

from __future__ import annotations

import html
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


def _bar(label: str, value: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"{label}: {value}"
    pct = value * 100.0 / total
    filled = int(round(value * width / max(1, total)))
    bar = "&#9608;" * filled + "&#9617;" * (width - filled)
    return f"{label} {bar} {value} ({pct:.1f}%)"


def _bucket_histogram(records: list[dict]) -> str:
    buckets = Counter()
    for r in records:
        b = tuple(r.get("bucket") or [])
        if len(b) == 2:
            buckets[f"{b[0]}x{b[1]}"] += 1
    if not buckets:
        return "<p>No bucket data.</p>"
    total = sum(buckets.values())
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td style='text-align:right'>{v}</td>"
        f"<td><div style='width:{v*300//total}px;height:14px;background:#3b82f6;border-radius:3px'></div></td></tr>"
        for k, v in buckets.most_common()
    )
    return (
        "<table><thead><tr><th>Bucket</th><th>Count</th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _blur_histogram(records: list[dict], n_bins: int = 12) -> str:
    blurs = [float(r.get("blur") or 0) for r in records]
    if not blurs:
        return "<p>No blur data.</p>"
    lo, hi = min(blurs), max(blurs)
    if hi <= lo:
        return f"<p>All frames have blur score {lo:.1f}.</p>"
    bins = [0] * n_bins
    for b in blurs:
        idx = min(n_bins - 1, int((b - lo) / (hi - lo) * n_bins))
        bins[idx] += 1
    max_count = max(bins) or 1
    rows = ""
    for i, count in enumerate(bins):
        edge_lo = lo + (hi - lo) * i / n_bins
        edge_hi = lo + (hi - lo) * (i + 1) / n_bins
        bar_w = int(count * 300 / max_count)
        rows += (
            f"<tr><td>{edge_lo:.0f}-{edge_hi:.0f}</td>"
            f"<td style='text-align:right'>{count}</td>"
            f"<td><div style='width:{bar_w}px;height:14px;background:#10b981;border-radius:3px'></div></td></tr>"
        )
    return (
        "<table><thead><tr><th>Blur range (Laplacian variance)</th><th>Count</th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _watermark_table(videos: list[dict]) -> str:
    flagged = [v for v in videos if v.get("watermarks")]
    if not flagged:
        return (
            "<p style='color:#10b981'>No suspected watermarks detected. "
            "(If a watermark is visible, it may not have been detected; "
            "spot-check the gallery.)</p>"
        )
    rows = ""
    for v in flagged:
        name = html.escape(Path(v["video"]).name)
        for wm in v["watermarks"]:
            rows += (
                f"<tr><td>{name}</td><td>{html.escape(wm['location'])}</td>"
                f"<td>{wm['x']},{wm['y']}</td><td>{wm['w']}x{wm['h']}</td>"
                f"<td>{wm['confidence']:.2f}</td></tr>"
            )
    return (
        "<p style='color:#f59e0b'>Suspected static overlays detected. "
        "Pipeline did NOT crop them. Review and decide whether to re-extract "
        "with manual crop or accept as-is.</p>"
        "<table><thead><tr><th>Video</th><th>Location</th><th>Origin</th>"
        "<th>Size</th><th>Confidence</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _per_video_table(videos: list[dict]) -> str:
    rows = ""
    for v in videos:
        name = html.escape(Path(v["video"]).name)
        rows += (
            f"<tr><td>{name}</td>"
            f"<td style='text-align:right'>{v.get('written', 0)}</td>"
            f"<td style='text-align:right'>{v.get('candidates', 0)}</td>"
            f"<td style='text-align:right'>{v.get('rejected_blur', 0)}</td>"
            f"<td style='text-align:right'>{v.get('rejected_ssim', 0)}</td>"
            f"<td style='text-align:right'>{v.get('rejected_color', 0)}</td>"
            f"<td style='text-align:right'>{v.get('rejected_dup', 0)}</td>"
            f"<td style='text-align:right'>{v.get('elapsed_s', 0):.1f}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>Video</th><th>Kept</th><th>Cand.</th>"
        "<th>Blur drop</th><th>SSIM drop</th><th>Color drop</th>"
        "<th>Dup drop</th><th>Time (s)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def generate_report(
    summary: dict,
    output_path: Path,
    *,
    title: str = "vid2dataset extraction report",
) -> Path | None:
    """Build an HTML report. ``summary`` is the run summary dict that
    extractor writes to _run_summary.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    videos = summary.get("videos", [])
    total_written = summary.get("total_written", 0)
    total_cand = summary.get("total_candidates", 0)
    elapsed = summary.get("elapsed_s", 0.0)

    # Aggregate FrameRecords from all videos for histograms
    all_records: list[dict] = []
    for v in videos:
        # Per-video _stats.json has full records; we only have the summary view.
        # Fall back: reconstruct minimal records from per-video data when available.
        for r in v.get("records", []):
            all_records.append(r)

    bucket_html = _bucket_histogram(all_records)
    blur_html = _blur_histogram(all_records)
    watermark_html = _watermark_table(videos)
    table_html = _per_video_table(videos)

    css = """
    body { background:#0f172a; color:#e2e8f0; font-family:system-ui,-apple-system,sans-serif;
           margin:0; padding:24px; line-height:1.5; }
    h1 { font-size:1.6rem; margin:0 0 6px; }
    h2 { font-size:1.15rem; margin:20px 0 8px; color:#93c5fd; border-bottom:1px solid #334155; padding-bottom:4px; }
    .info { color:#94a3b8; font-size:.9rem; }
    .stat { display:inline-block; background:#1e293b; padding:8px 14px; margin:4px 6px 4px 0;
            border-radius:6px; }
    .stat .v { font-size:1.3rem; font-weight:bold; color:#fff; }
    .stat .l { font-size:.8rem; color:#94a3b8; display:block; }
    table { border-collapse:collapse; margin:8px 0; font-size:.85rem; }
    th, td { padding:4px 10px; text-align:left; }
    thead th { color:#94a3b8; border-bottom:1px solid #334155; }
    tbody tr:hover { background:#1e293b; }
    """
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="info">{len(videos)} videos &middot; {elapsed:.1f}s elapsed</p>

<div>
  <span class="stat"><span class="l">Images written</span><span class="v">{total_written}</span></span>
  <span class="stat"><span class="l">Candidates inspected</span><span class="v">{total_cand}</span></span>
  <span class="stat"><span class="l">Acceptance rate</span><span class="v">{(total_written * 100 / max(1, total_cand)):.1f}%</span></span>
</div>

<h2>Watermark check</h2>
{watermark_html}

<h2>Bucket distribution</h2>
{bucket_html}

<h2>Blur score distribution</h2>
{blur_html}

<h2>Per-video breakdown</h2>
{table_html}

<p class="info" style="margin-top:24px">
Generated by vid2dataset. See <code>_gallery.html</code> for the visual gallery.
</p>
</body>
</html>"""

    output_path.write_text(html_doc, encoding="utf-8")
    log.info("Pre-flight report: %s", output_path)
    return output_path
