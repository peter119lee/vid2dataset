"""Hover tooltip widget compatible with tkinter / customtkinter.

Usage:
    from vid2dataset.tooltip import attach_tooltip
    attach_tooltip(my_entry, "Help text...")
    # or with a callable (re-evaluated each time the tooltip shows):
    attach_tooltip(my_entry, lambda: t("tip_resolution", current_lang))
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable

TooltipText = str | Callable[[], str]


class Tooltip:
    """A small dark popup that appears below a widget on hover."""

    def __init__(
        self,
        widget: tk.Widget,
        text: TooltipText,
        *,
        delay_ms: int = 500,
        wraplength: int = 320,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._tip_window: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _resolve_text(self) -> str:
        if callable(self.text):
            try:
                return str(self.text())
            except Exception:
                return ""
        return str(self.text)

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            with contextlib.suppress(Exception):
                self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        text = self._resolve_text()
        if not text or self._tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        # Keep on screen
        screen_w = self.widget.winfo_screenwidth()
        if x + self.wraplength + 20 > screen_w:
            x = max(0, screen_w - self.wraplength - 20)

        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        with contextlib.suppress(tk.TclError):
            tw.attributes("-alpha", 0.95)

        label = tk.Label(
            tw,
            text=text,
            justify="left",
            background="#2b2b2b",
            foreground="#dcdcdc",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
            padx=8,
            pady=5,
            font=("Segoe UI", 9),
        )
        label.pack()
        self._tip_window = tw

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip_window is not None:
            with contextlib.suppress(Exception):
                self._tip_window.destroy()
            self._tip_window = None


def attach_tooltip(widget: tk.Widget, text: TooltipText, **kwargs) -> Tooltip:
    """Convenience: create a Tooltip and return it (for later text updates)."""
    return Tooltip(widget, text, **kwargs)


if __name__ == "__main__":  # pragma: no cover
    root = tk.Tk()
    root.geometry("400x200")
    root.configure(bg="#1a1a2e")
    btn = tk.Button(root, text="Hover me")
    btn.pack(pady=40)
    attach_tooltip(btn, "This is a tooltip.\nIt supports multiple lines.")
    root.mainloop()
