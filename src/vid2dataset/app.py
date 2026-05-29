"""vid2dataset desktop app — CustomTkinter, i18n, ETA, remember paths."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time

try:
    import winsound
except ImportError:
    winsound = None
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from pathlib import Path

import customtkinter as ctk

from vid2dataset import __version__
from vid2dataset.config import ExtractConfig
from vid2dataset.extractor import run_pipeline
from vid2dataset.i18n import t
from vid2dataset.presets import list_presets, load_preset
from vid2dataset.tooltip import Tooltip

try:
    import windnd
except ImportError:
    windnd = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PRESETS = dict(list_presets())
PRESET_NAMES = list(PRESETS.keys())
PREFS_PATH = Path.home() / ".vid2dataset.json"


def _load_prefs() -> dict:
    if PREFS_PATH.exists():
        try:
            return json.loads(PREFS_PATH.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _save_prefs(prefs: dict) -> None:
    PREFS_PATH.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), "utf-8")


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.prefs = _load_prefs()
        self.lang = self.prefs.get("lang", "en")
        self._running = False
        self._cancel_event: threading.Event | None = None
        self._build()
        self._apply_preset(self.prefs.get("preset", "anima-style"))
        # Restore paths
        if self.prefs.get("input"):
            self.input_entry.insert(0, self.prefs["input"])
        # Enable drag-and-drop of folders onto the window
        if windnd is not None:
            with contextlib.suppress(Exception):
                windnd.hook_dropfiles(self, func=self._on_drop)
        if self.prefs.get("output"):
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, self.prefs["output"])

    def _build(self) -> None:
        self.title(f"vid2dataset {__version__}")
        self.geometry("900x760")
        self.minsize(750, 620)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(12, 4))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="vid2dataset", font=ctk.CTkFont(size=26, weight="bold")).grid(row=0, column=0, sticky="w")
        self.update_btn = ctk.CTkButton(
            header, text=t("check_update", self.lang), width=130,
            command=self._check_update_async,
        )
        self.update_btn.grid(row=0, column=2, sticky="e", padx=(0, 6))
        self.lang_btn = ctk.CTkButton(header, text=t("lang_btn", self.lang), width=50, command=self._toggle_lang)
        self.lang_btn.grid(row=0, column=3, sticky="e")
        self.subtitle_lbl = ctk.CTkLabel(header, text=t("subtitle", self.lang), font=ctk.CTkFont(size=12), text_color="#888")
        self.subtitle_lbl.grid(row=1, column=0, columnspan=4, sticky="w")

        # I/O
        io = ctk.CTkFrame(self)
        io.grid(row=1, column=0, sticky="ew", padx=24, pady=4)
        io.grid_columnconfigure(1, weight=1)
        self.input_lbl = ctk.CTkLabel(io, text=t("input", self.lang), font=ctk.CTkFont(weight="bold"))
        self.input_lbl.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")
        self.input_entry = ctk.CTkEntry(io, placeholder_text="D:\\videos\\...")
        self.input_entry.grid(row=0, column=1, padx=4, pady=(10, 4), sticky="ew")
        self.input_btn = ctk.CTkButton(io, text=t("browse", self.lang), width=70, command=self._browse_input)
        self.input_btn.grid(row=0, column=2, padx=(4, 12), pady=(10, 4))

        self.output_lbl = ctk.CTkLabel(io, text=t("output", self.lang), font=ctk.CTkFont(weight="bold"))
        self.output_lbl.grid(row=1, column=0, padx=12, pady=(4, 10), sticky="w")
        self.output_entry = ctk.CTkEntry(io, placeholder_text="D:\\datasets\\output")
        self.output_entry.insert(0, "output")
        self.output_entry.grid(row=1, column=1, padx=4, pady=(4, 10), sticky="ew")
        self.output_btn = ctk.CTkButton(io, text=t("browse", self.lang), width=70, command=self._browse_output)
        self.output_btn.grid(row=1, column=2, padx=(4, 12), pady=(4, 10))

        # Settings
        settings = ctk.CTkFrame(self)
        settings.grid(row=2, column=0, sticky="ew", padx=24, pady=4)
        settings.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Preset row
        self.preset_lbl = ctk.CTkLabel(settings, text=t("preset", self.lang), font=ctk.CTkFont(weight="bold"))
        self.preset_lbl.grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")
        self.preset_var = ctk.StringVar(value="anima-style")
        ctk.CTkOptionMenu(settings, variable=self.preset_var, values=PRESET_NAMES, command=self._apply_preset, width=170).grid(row=0, column=1, padx=4, pady=(10, 4), sticky="w")
        self.preset_desc = ctk.CTkLabel(settings, text="", text_color="#888", font=ctk.CTkFont(size=11))
        self.preset_desc.grid(row=0, column=2, columnspan=2, padx=8, pady=(10, 4), sticky="w")

        # Params
        self._params: dict[str, ctk.CTkEntry] = {}
        self._param_labels: dict[str, ctk.CTkLabel] = {}
        params = [
            ("resolution", "1024", "tip_resolution", 1, 0),
            ("blur_threshold", "50", "tip_blur", 1, 1),
            ("max_per_video", "0", "tip_max", 1, 2),
            ("min_per_video", "3", "tip_min", 1, 3),
            ("phash_distance", "5", "tip_phash", 2, 0),
            ("ssim_threshold", "0.85", "tip_ssim", 2, 1),
            ("color_distance", "0.08", "tip_color", 2, 2),
            ("frames_per_scene", "6", "tip_frames", 2, 3),
        ]
        self._param_tooltips: dict[str, Tooltip] = {}
        self._param_tip_keys: dict[str, str] = {}
        for key, default, tip_key, row, col in params:
            f = ctk.CTkFrame(settings, fg_color="transparent")
            f.grid(row=row, column=col, padx=10, pady=3, sticky="w")
            lbl = ctk.CTkLabel(f, text=t(key, self.lang), font=ctk.CTkFont(size=11))
            lbl.pack(anchor="w")
            self._param_labels[key] = lbl
            entry = ctk.CTkEntry(f, width=90)
            entry.insert(0, default)
            entry.pack(anchor="w")
            self._params[key] = entry
            self._param_tip_keys[key] = tip_key
            tk_widget = entry
            tip = Tooltip(tk_widget, lambda k=tip_key: t(k, self.lang))
            self._param_tooltips[key] = tip
            tip_lbl = Tooltip(lbl, lambda k=tip_key: t(k, self.lang))
            self._param_tooltips[key + '_lbl'] = tip_lbl

        # Checkboxes
        chk = ctk.CTkFrame(settings, fg_color="transparent")
        chk.grid(row=3, column=0, columnspan=4, padx=8, pady=(4, 10), sticky="w")
        self.auto_quality_var = ctk.BooleanVar(value=True)
        self.chk_auto = ctk.CTkCheckBox(chk, text=t("auto_quality", self.lang), variable=self.auto_quality_var)
        self.chk_auto.pack(side="left", padx=(4, 14))
        self.keyframe_var = ctk.BooleanVar(value=True)
        self.chk_kf = ctk.CTkCheckBox(chk, text=t("keyframe", self.lang), variable=self.keyframe_var)
        self.chk_kf.pack(side="left", padx=(0, 14))
        self.subject_var = ctk.BooleanVar(value=False)
        self.chk_subj = ctk.CTkCheckBox(chk, text=t("subject_size", self.lang), variable=self.subject_var)
        self.chk_subj.pack(side="left", padx=(0, 14))
        self.watermark_var = ctk.BooleanVar(value=True)
        self.chk_wm = ctk.CTkCheckBox(chk, text=t("watermark", self.lang), variable=self.watermark_var)
        self.chk_wm.pack(side="left", padx=(0, 14))
        Tooltip(self.chk_wm, lambda: t("tip_watermark", self.lang), wraplength=380)
        self.crop_wm_var = ctk.BooleanVar(value=False)
        self.chk_crop_wm = ctk.CTkCheckBox(chk, text=t("crop_watermark", self.lang), variable=self.crop_wm_var)
        self.chk_crop_wm.pack(side="left", padx=(0, 14))
        Tooltip(self.chk_crop_wm, lambda: t("tip_crop_watermark", self.lang), wraplength=380)
        self.flatten_var = ctk.BooleanVar(value=False)
        self.chk_flatten = ctk.CTkCheckBox(chk, text=t("flatten_output", self.lang), variable=self.flatten_var)
        self.chk_flatten.pack(side="left", padx=(0, 14))
        Tooltip(self.chk_flatten, lambda: t("tip_flatten", self.lang), wraplength=380)
        self.gpu_var = ctk.BooleanVar(value=False)
        self.chk_gpu = ctk.CTkCheckBox(
            chk, text=t("gpu_accel", self.lang), variable=self.gpu_var,
            command=self._on_gpu_toggle,
        )
        self.chk_gpu.pack(side="left", padx=(0, 14))
        Tooltip(self.chk_gpu, lambda: t("tip_gpu", self.lang), wraplength=380)

        # Run
        run_frame = ctk.CTkFrame(self, fg_color="transparent")
        run_frame.grid(row=3, column=0, sticky="ew", padx=24, pady=6)
        run_frame.grid_columnconfigure(2, weight=1)
        self.run_btn = ctk.CTkButton(run_frame, text=t("extract", self.lang), font=ctk.CTkFont(size=14, weight="bold"), height=42, command=self._start)
        self.run_btn.grid(row=0, column=0, padx=(0, 8))
        self.cancel_btn = ctk.CTkButton(run_frame, text=t("cancel", self.lang), height=42, width=80, command=self._cancel, state="disabled", fg_color="#8b3a3a", hover_color="#a04848")
        self.cancel_btn.grid(row=0, column=1, padx=(0, 12))
        self.progress = ctk.CTkProgressBar(run_frame, mode="indeterminate", height=8)
        self.progress.grid(row=0, column=2, sticky="ew")
        self.progress.set(0)
        self.open_btn = ctk.CTkButton(run_frame, text=t("open_output", self.lang), width=140, command=self._open_output, state="disabled")
        self.open_btn.grid(row=0, column=3, padx=(12, 0))

        # Status
        self.status_var = ctk.StringVar(value=t("ready", self.lang))
        ctk.CTkLabel(self, textvariable=self.status_var, font=ctk.CTkFont(size=11), text_color="#888").grid(row=4, column=0, sticky="w", padx=28, pady=(2, 0))

        # Log
        self.log_box = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=11), corner_radius=8, state="disabled")
        self.log_box.grid(row=5, column=0, sticky="nsew", padx=24, pady=(4, 14))

    # ── Actions ──────────────────────────────────────────────────

    def _on_drop(self, files: list[bytes]) -> None:
        """Handle dropped files/folders. Take the first directory."""
        if not files:
            return
        for raw in files:
            try:
                path = raw.decode("gbk") if isinstance(raw, bytes) else str(raw)
            except Exception:
                path = str(raw)
            from pathlib import Path as _P
            p = _P(path)
            if p.is_dir() or p.is_file():
                self.input_entry.delete(0, "end")
                self.input_entry.insert(0, str(p))
                break

    def _browse_input(self) -> None:
        p = filedialog.askdirectory(title=t("select_video_folder", self.lang))
        if p:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, p)

    def _browse_output(self) -> None:
        p = filedialog.askdirectory(title=t("select_output_folder", self.lang))
        if p:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, p)

    def _on_gpu_toggle(self) -> None:
        """Called when user clicks the GPU checkbox.

        If torch is already available, keep the tick. If runtime is
        cached but not loaded, activate it. Otherwise prompt to download.
        """
        if not self.gpu_var.get():
            return

        from vid2dataset.gpu_runtime import (
            activate_runtime,
            runtime_status,
            total_download_size_mb,
        )
        status = runtime_status()
        if status.available:
            return

        if status.cached:
            if activate_runtime():
                messagebox.showinfo("vid2dataset", t("gpu_activated", self.lang))
            else:
                self.gpu_var.set(False)
                messagebox.showerror(t("error", self.lang), t("gpu_activate_failed", self.lang))
            return

        ans = messagebox.askyesno(
            "vid2dataset",
            t("gpu_download_prompt", self.lang, mb=total_download_size_mb()),
        )
        if not ans:
            self.gpu_var.set(False)
            return
        self._download_gpu_runtime()

    def _download_gpu_runtime(self) -> None:
        """Download torch+cuda wheels in a background thread."""
        from vid2dataset.gpu_runtime import download_runtime

        self.chk_gpu.configure(state="disabled")
        self.status_var.set(t("gpu_downloading", self.lang))
        self.progress.start()

        def progress_cb(label: str, done: int, total: int) -> None:
            if total > 0:
                pct = done * 100 // total
                txt = t("gpu_downloading_pkg", self.lang, pkg=label, pct=pct)
                self.after(0, lambda s=txt: self.status_var.set(s))

        def worker() -> None:
            ok = False
            err = ""
            try:
                ok = download_runtime(progress=progress_cb)
            except Exception as exc:
                err = str(exc)
            self.after(0, lambda: self._on_gpu_download_done(ok, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_gpu_download_done(self, ok: bool, err: str) -> None:
        from vid2dataset.gpu_runtime import activate_runtime

        self.progress.stop()
        self.chk_gpu.configure(state="normal")
        if not ok:
            self.gpu_var.set(False)
            self.status_var.set(t("gpu_download_failed", self.lang))
            messagebox.showerror(
                t("error", self.lang),
                f"{t('gpu_download_failed', self.lang)}: {err}",
            )
            return
        if activate_runtime():
            self.status_var.set(t("gpu_ready", self.lang))
            messagebox.showinfo("vid2dataset", t("gpu_ready", self.lang))
        else:
            self.gpu_var.set(False)
            messagebox.showerror(t("error", self.lang), t("gpu_activate_failed", self.lang))

    def _open_output(self) -> None:
        p = self.output_entry.get().strip()
        if not p or not Path(p).exists():
            return
        import subprocess
        import sys
        try:
            if sys.platform == "win32":
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            log = __import__("logging").getLogger(__name__)
            log.warning("open output folder failed: %s", e)

    def _check_update_async(self) -> None:
        """Check for updates in a background thread."""
        self.update_btn.configure(state="disabled")
        threading.Thread(target=self._check_update_worker, daemon=True).start()

    def _check_update_worker(self) -> None:
        from vid2dataset.updater import (
            fetch_latest_release,
            is_newer,
            is_running_as_exe,
        )

        try:
            release = fetch_latest_release()
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda: messagebox.showerror(
                t("error", self.lang), f"{t('update_check_failed', self.lang)}: {err}"
            ))
            self.after(0, lambda: self.update_btn.configure(state="normal"))
            return

        if release is None:
            self.after(0, lambda: messagebox.showerror(
                t("error", self.lang), t("update_check_failed", self.lang)
            ))
            self.after(0, lambda: self.update_btn.configure(state="normal"))
            return

        if not is_newer(release.version):
            self.after(0, lambda: messagebox.showinfo(
                "vid2dataset", t("update_latest", self.lang)
            ))
            self.after(0, lambda: self.update_btn.configure(state="normal"))
            return

        # Newer version available
        msg = t("update_available", self.lang, version=release.version)
        if release.notes:
            msg += "\n\n" + release.notes[:500]

        if is_running_as_exe():
            # Offer one-click install
            if messagebox.askyesno("vid2dataset", msg + "\n\nDownload and install now?"):
                self._do_update(release)
            else:
                self.after(0, lambda: self.update_btn.configure(state="normal"))
        else:
            # Dev mode: just inform the user
            messagebox.showinfo(
                "vid2dataset",
                msg + "\n\nRun `git pull` or download the new release from GitHub.",
            )
            self.after(0, lambda: self.update_btn.configure(state="normal"))

    def _do_update(self, release) -> None:
        """Download new exe and stage replacement."""
        from vid2dataset.updater import download_exe, install_update

        if not release.exe_url:
            self.after(0, lambda: messagebox.showerror(
                t("error", self.lang), "No .exe asset in release."
            ))
            self.after(0, lambda: self.update_btn.configure(state="normal"))
            return

        self.after(0, lambda: self.status_var.set(t("update_downloading", self.lang)))
        self.after(0, lambda: self.progress.start())

        import sys
        target = Path(sys.executable).parent / "vid2dataset_new.exe"

        def progress_cb(done: int, total: int) -> None:
            if total > 0:
                pct = done * 100 // total
                self.after(0, lambda p=pct: self.status_var.set(
                    f"{t('update_downloading', self.lang)} {p}%"
                ))

        try:
            download_exe(release.exe_url, target, progress_cb=progress_cb)
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: messagebox.showerror(
                t("error", self.lang), f"Download failed: {err}"
            ))
            self.after(0, lambda: self.update_btn.configure(state="normal"))
            return

        self.after(0, lambda: self.progress.stop())
        self.after(0, lambda: self.status_var.set(t("update_ready", self.lang)))
        self.after(0, lambda: messagebox.showinfo(
            "vid2dataset",
            t("update_ready", self.lang) + "\n\nThe app will now restart.",
        ))

        try:
            install_update(target)
            # Exit the app — the .bat will swap and restart
            self.after(500, self.destroy)
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda: messagebox.showerror(
                t("error", self.lang), f"Install failed: {err}"
            ))
            self.after(0, lambda: self.update_btn.configure(state="normal"))

    def _toggle_lang(self) -> None:
        self.lang = "zh" if self.lang == "en" else "en"
        self.prefs["lang"] = self.lang
        _save_prefs(self.prefs)
        # Refresh key labels
        self.lang_btn.configure(text=t("lang_btn", self.lang))
        self.subtitle_lbl.configure(text=t("subtitle", self.lang))
        self.input_lbl.configure(text=t("input", self.lang))
        self.output_lbl.configure(text=t("output", self.lang))
        self.input_btn.configure(text=t("browse", self.lang))
        self.output_btn.configure(text=t("browse", self.lang))
        self.preset_lbl.configure(text=t("preset", self.lang))
        self.run_btn.configure(text=t("extract", self.lang))
        self.cancel_btn.configure(text=t("cancel", self.lang))
        self.open_btn.configure(text=t("open_output", self.lang))
        self.chk_auto.configure(text=t("auto_quality", self.lang))
        self.chk_kf.configure(text=t("keyframe", self.lang))
        self.chk_subj.configure(text=t("subject_size", self.lang))
        self.chk_wm.configure(text=t("watermark", self.lang))
        self.chk_crop_wm.configure(text=t("crop_watermark", self.lang))
        self.chk_flatten.configure(text=t("flatten_output", self.lang))
        # Refresh all parameter labels
        for key, lbl in self._param_labels.items():
            lbl.configure(text=t(key, self.lang))
        if hasattr(self, "update_btn"):
            self.update_btn.configure(text=t("check_update", self.lang))
        if not self._running:
            self.status_var.set(t("ready", self.lang))

    def _apply_preset(self, name: str) -> None:
        self.preset_desc.configure(text=PRESETS.get(name, ""))
        cfg = load_preset(name)
        for key, entry in self._params.items():
            if key in cfg:
                entry.delete(0, "end")
                entry.insert(0, str(cfg[key]))
        self.auto_quality_var.set(bool(cfg.get("auto_quality", False)))
        self.keyframe_var.set(cfg.get("decode_mode", "accurate") == "keyframe")
        self.subject_var.set(bool(cfg.get("subject_size_filter", False)))
        self.watermark_var.set(bool(cfg.get("detect_watermark", True)))
        self.crop_wm_var.set(bool(cfg.get("crop_watermark", False)))
        self.flatten_var.set(bool(cfg.get("flatten_output", False)))

    def _log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start(self) -> None:
        if self._running:
            return
        inp = self.input_entry.get().strip()
        out = self.output_entry.get().strip() or "output"
        if not inp:
            messagebox.showerror(t("error", self.lang), t("no_input", self.lang))
            return
        if not Path(inp).exists():
            messagebox.showerror(
                t("error", self.lang),
                f"{t('not_found', self.lang)}\n{inp}",
            )
            return
        # Save paths
        self.prefs.update({"input": inp, "output": out, "preset": self.preset_var.get()})
        _save_prefs(self.prefs)

        self._running = True
        self._cancel_event = threading.Event()
        self.run_btn.configure(state="disabled", text=t("running", self.lang))
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.progress.start()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        threading.Thread(target=self._worker, args=(inp, out), daemon=True).start()

    def _worker(self, inp: str, out: str) -> None:
        handler = _LogHandler(self)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        try:
            base = load_preset(self.preset_var.get()) if self.preset_var.get() in PRESETS else {}
            base.update({
                "input": Path(inp), "output": Path(out),
                "auto_quality": self.auto_quality_var.get(),
                "decode_mode": "keyframe" if self.keyframe_var.get() else "accurate",
                "subject_size_filter": self.subject_var.get(),
                "detect_watermark": self.watermark_var.get(),
                "crop_watermark": self.crop_wm_var.get(),
                "flatten_output": self.flatten_var.get(),
                "gpu_accel": self.gpu_var.get(),
            })
            for key, entry in self._params.items():
                val = entry.get().strip()
                if not val:
                    continue
                if key in ("resolution", "max_per_video", "min_per_video", "phash_distance", "frames_per_scene"):
                    parsed = int(val)
                    if key == "max_per_video" and parsed <= 0:
                        continue
                    base[key] = parsed
                elif key in ("blur_threshold", "ssim_threshold", "color_distance"):
                    base[key] = float(val)

            cfg = ExtractConfig(**base)

            # ETA: count videos first
            from vid2dataset.io_utils import discover_videos
            videos = discover_videos(cfg.input)
            total_videos = len(videos)
            video_times: list[float] = []
            last_video_start = time.perf_counter()

            def progress_cb(stage: str, current: int, total: int) -> None:
                nonlocal last_video_start
                if stage == "video":
                    now = time.perf_counter()
                    if current > 0:
                        video_times.append(now - last_video_start)
                    last_video_start = now
                    if video_times:
                        avg = sum(video_times) / len(video_times)
                        remaining = avg * (total_videos - current - 1)
                        mins, secs = divmod(int(remaining), 60)
                        eta_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                        self.after(0, lambda s=eta_str, c=current: self.status_var.set(
                            t("processing", self.lang, name=Path(videos[c]).stem[:20], current=c + 1, total=total_videos)
                            + "  " + t("eta", self.lang, remaining=s)
                        ))
                    else:
                        self.after(0, lambda c=current: self.status_var.set(
                            t("processing", self.lang, name=Path(videos[c]).stem[:20], current=c + 1, total=total_videos)
                        ))

            result = run_pipeline(cfg, progress=progress_cb, cancel_event=self._cancel_event)

            done_msg = t("done", self.lang, count=result.total_written, time=result.elapsed_s)
            self.after(0, lambda: self.status_var.set(done_msg))
            self.after(0, lambda: self._log(f"\n{'='*50}\n{done_msg}"))
            for vs in result.videos:
                self.after(0, lambda v=vs: self._log(
                    f"  {Path(v.video).name}: {v.written} kept"
                ))
            self.after(0, lambda: self.open_btn.configure(state="normal"))

        except Exception as exc:
            err = str(exc)
            self.after(0, lambda e=err: self.status_var.set(f"{t('error', self.lang)}: {e}"))
            self.after(0, lambda e=err: self._log(f"\nERROR: {e}"))
        finally:
            logging.getLogger().removeHandler(handler)
            self.after(0, self._done)

    def _cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_btn.configure(state="disabled", text=t("cancelling", self.lang))

    def _done(self) -> None:
        self._running = False
        self._cancel_event = None
        self.run_btn.configure(state="normal", text=t("extract", self.lang))
        self.cancel_btn.configure(state="disabled", text=t("cancel", self.lang))
        self.progress.stop()
        self.progress.set(0)
        if winsound is not None:
            with contextlib.suppress(Exception):
                winsound.MessageBeep(winsound.MB_OK)


class _LogHandler(logging.Handler):
    def __init__(self, app: App) -> None:
        super().__init__()
        self.app = app
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.app.after(0, lambda m=msg: self.app._log(m))


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
