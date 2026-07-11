"""Advanced mode window: per-video segment marking + manual frame capture.

GUI-only module, imported lazily by app.py. The logic it relies on lives in
extractor.py (``process_single_frame``, ``ExtractConfig.segments``) and is
unit-tested there — this window stays a thin interactive shell:

- scrub any input video with a slider + frame/second step buttons
- [Set In] / [Set Out] mark (start, end) segments; extraction then samples
  only inside them (the ``segments`` dict is shared with the App and applied
  on the next Extract run)
- [Capture frame] writes the CURRENT frame through the same crop/bucket
  pipeline immediately, bypassing quality/diversity gates — the user chose
  this exact frame. Captures are tagged along with everything else.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk
import cv2
from PIL import Image

from vid2dataset.config import ExtractConfig
from vid2dataset.i18n import t
from vid2dataset.io_utils import discover_videos, sanitize_stem

log = logging.getLogger(__name__)

_PREVIEW_W, _PREVIEW_H = 720, 405
_SEEK_DEBOUNCE_MS = 60


def _open_cap(path: Path) -> cv2.VideoCapture:
    """Persistent VideoCapture with the non-ASCII Windows path workaround."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap = cv2.VideoCapture(f"\\\\?\\{path.resolve()}")
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    return cap


class AdvancedWindow(ctk.CTkToplevel):
    """Scrub videos, mark segments, capture frames manually."""

    def __init__(
        self,
        master,
        *,
        input_dir: Path,
        segments: dict[str, list[tuple[float, float]]],
        build_config: Callable[[], ExtractConfig],
        lang: str = "en",
    ) -> None:
        super().__init__(master)
        self.lang = lang
        self.segments = segments  # shared with the App; mutated in place
        self._build_config = build_config
        self._cap: cv2.VideoCapture | None = None
        self._video: Path | None = None
        self._frame_count = 0
        self._fps = 30.0
        self._frame_idx = 0
        self._frame_bgr = None
        self._pending_in: float | None = None
        self._seek_job: str | None = None
        self._preview_ref = None  # keep CTkImage alive

        self.title(t("adv_title", lang))
        self.geometry("880x680")
        self.grid_columnconfigure(0, weight=1)

        self._videos = discover_videos(input_dir) if input_dir.exists() else []
        names = [v.name for v in self._videos] or [t("adv_no_videos", lang)]
        self.video_var = ctk.StringVar(value=names[0])
        self.video_menu = ctk.CTkOptionMenu(
            self, variable=self.video_var, values=names, width=420, command=self._on_video_pick
        )
        self.video_menu.grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        self.preview = ctk.CTkLabel(self, text="", width=_PREVIEW_W, height=_PREVIEW_H)
        self.preview.grid(row=1, column=0, padx=16, pady=4)

        self.time_var = ctk.StringVar(value="0.00s")
        ctk.CTkLabel(self, textvariable=self.time_var, font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, pady=(2, 0)
        )
        self.slider = ctk.CTkSlider(self, from_=0, to=1, command=self._on_slider)
        self.slider.set(0)
        self.slider.grid(row=3, column=0, padx=16, pady=4, sticky="ew")

        steps = ctk.CTkFrame(self, fg_color="transparent")
        steps.grid(row=4, column=0, pady=2)
        for label, delta_s, delta_f in (
            ("-10s", -10.0, 0),
            ("-1s", -1.0, 0),
            ("-1f", 0.0, -1),
            ("+1f", 0.0, 1),
            ("+1s", 1.0, 0),
            ("+10s", 10.0, 0),
        ):
            ctk.CTkButton(
                steps,
                text=label,
                width=52,
                command=lambda ds=delta_s, df=delta_f: self._step(ds, df),
            ).pack(side="left", padx=3)

        segrow = ctk.CTkFrame(self, fg_color="transparent")
        segrow.grid(row=5, column=0, padx=16, pady=(8, 2), sticky="ew")
        ctk.CTkButton(segrow, text=t("adv_set_in", lang), width=90, command=self._set_in).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(segrow, text=t("adv_set_out", lang), width=90, command=self._set_out).pack(
            side="left", padx=(0, 10)
        )
        self.pending_var = ctk.StringVar(value="")
        ctk.CTkLabel(segrow, textvariable=self.pending_var, font=ctk.CTkFont(size=11)).pack(
            side="left", padx=(0, 14)
        )
        ctk.CTkButton(segrow, text=t("adv_capture", lang), width=130, command=self._capture).pack(
            side="right"
        )

        listrow = ctk.CTkFrame(self, fg_color="transparent")
        listrow.grid(row=6, column=0, padx=16, pady=2, sticky="ew")
        listrow.grid_columnconfigure(0, weight=1)
        self.seg_list = tk.Listbox(
            listrow, height=4, bg="#1e1e1e", fg="#e2e8f0", selectbackground="#2b5f8f"
        )
        self.seg_list.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            listrow, text=t("adv_remove", lang), width=80, command=self._remove_segment
        ).grid(row=0, column=1, padx=(8, 0))

        self.status_var = ctk.StringVar(value=t("adv_hint", lang))
        ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=11),
            text_color="#888",
            wraplength=820,
            justify="left",
        ).grid(row=7, column=0, padx=16, pady=(4, 12), sticky="w")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if self._videos:
            self._load_video(self._videos[0])

    # ── Video handling ────────────────────────────────────────────

    def _on_video_pick(self, name: str) -> None:
        for v in self._videos:
            if v.name == name:
                self._load_video(v)
                return

    def _load_video(self, video: Path) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        try:
            self._cap = _open_cap(video)
        except OSError as e:
            self.status_var.set(str(e))
            return
        self._video = video
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        self._frame_count = max(1, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self._pending_in = None
        self.pending_var.set("")
        self.slider.configure(from_=0, to=self._frame_count - 1)
        self.slider.set(0)
        self._refresh_segment_list()
        self._seek(0)

    def _on_slider(self, value: float) -> None:
        # Debounce: decoding on every pixel of slider movement floods cv2.
        self._frame_idx = int(value)
        if self._seek_job is not None:
            self.after_cancel(self._seek_job)
        self._seek_job = self.after(_SEEK_DEBOUNCE_MS, lambda: self._seek(self._frame_idx))

    def _step(self, delta_s: float, delta_f: int) -> None:
        idx = self._frame_idx + int(round(delta_s * self._fps)) + delta_f
        idx = min(max(0, idx), self._frame_count - 1)
        self.slider.set(idx)
        self._seek(idx)

    def _seek(self, idx: int) -> None:
        self._seek_job = None
        if self._cap is None:
            return
        idx = min(max(0, idx), self._frame_count - 1)
        self._frame_idx = idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        self._frame_bgr = frame
        self.time_var.set(f"{idx / self._fps:.2f}s  ({idx}/{self._frame_count - 1})")
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil.thumbnail((_PREVIEW_W, _PREVIEW_H))
        self._preview_ref = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.preview.configure(image=self._preview_ref)

    # ── Segments ──────────────────────────────────────────────────

    def _current_time(self) -> float:
        return self._frame_idx / self._fps

    def _set_in(self) -> None:
        self._pending_in = self._current_time()
        self.pending_var.set(t("adv_pending", self.lang, t=f"{self._pending_in:.2f}"))

    def _set_out(self) -> None:
        now = self._current_time()
        if self._video is None or self._pending_in is None or now <= self._pending_in:
            self.status_var.set(t("adv_need_in", self.lang))
            return
        self.segments.setdefault(self._video.name, []).append((self._pending_in, now))
        self._pending_in = None
        self.pending_var.set("")
        self.status_var.set(t("adv_hint", self.lang))
        self._refresh_segment_list()

    def _refresh_segment_list(self) -> None:
        self.seg_list.delete(0, "end")
        if self._video is None:
            return
        for a, b in self.segments.get(self._video.name, []):
            self.seg_list.insert("end", f"{a:.2f}s  →  {b:.2f}s")

    def _remove_segment(self) -> None:
        if self._video is None:
            return
        sel = self.seg_list.curselection()
        if not sel:
            return
        segs = self.segments.get(self._video.name, [])
        if 0 <= sel[0] < len(segs):
            del segs[sel[0]]
        self._refresh_segment_list()

    # ── Manual capture ────────────────────────────────────────────

    def _capture(self) -> None:
        if self._video is None or self._frame_bgr is None:
            return
        try:
            cfg = self._build_config()
        except Exception as e:  # config from GUI fields can be invalid
            self.status_var.set(f"{t('error', self.lang)}: {e}")
            return
        from vid2dataset.extractor import _output_dir_for, process_single_frame

        out_dir = _output_dir_for(cfg, self._video)
        prefix = f"{sanitize_stem(self._video.stem)}_manual_"
        seq = (
            max(
                (
                    int(p.stem[len(prefix) :])
                    for p in out_dir.glob(f"{prefix}*")
                    if p.stem[len(prefix) :].isdigit()
                ),
                default=0,
            )
            if out_dir.exists()
            else 0
        ) + 1
        try:
            out_path = process_single_frame(cfg, self._frame_bgr, self._video, seq)
        except Exception as e:  # noqa: BLE001 — surfaced in the status line
            log.error("Manual capture failed: %s", e)
            self.status_var.set(f"{t('error', self.lang)}: {e}")
            return
        if out_path is None:
            self.status_var.set(t("adv_capture_failed", self.lang))
        else:
            self.status_var.set(t("adv_saved", self.lang, name=out_path.name))

    def _on_close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.destroy()
