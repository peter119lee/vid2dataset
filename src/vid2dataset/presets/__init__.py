"""Built-in presets bundled with the wheel."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PRESETS_PACKAGE = "vid2dataset.presets"


def _preset_dir() -> Path:
    """Return the on-disk directory containing preset TOMLs.

    Handles normal installs, editable installs, and PyInstaller bundles.
    """
    # PyInstaller: check sys._MEIPASS first
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / "vid2dataset" / "presets"
        if p.exists() and any(p.glob("*.toml")):
            return p

    # importlib.resources (normal install)
    try:
        from importlib import resources
        p = Path(str(resources.files(PRESETS_PACKAGE)))
        if p.exists() and any(p.glob("*.toml")):
            return p
    except Exception:
        pass

    # Fallback: relative to this file
    p = Path(__file__).parent
    if any(p.glob("*.toml")):
        return p

    raise FileNotFoundError("Cannot locate preset TOML files")


def list_presets() -> list[tuple[str, str]]:
    """Return [(name, description), ...] for every bundled preset."""
    out: list[tuple[str, str]] = []
    for p in sorted(_preset_dir().glob("*.toml")):
        with p.open("rb") as f:
            data = tomllib.load(f)
        desc = str(data.get("description", "")).strip()
        out.append((p.stem, desc))
    return out


def load_preset(name: str) -> dict:
    """Return the merged config dict from a named preset.

    The preset's ``description`` field is stripped — it's metadata only.
    Anything else is treated as ``ExtractConfig`` overrides.
    """
    path = _preset_dir() / f"{name}.toml"
    if not path.exists():
        available = ", ".join(n for n, _ in list_presets())
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available: {available}"
        )
    with path.open("rb") as f:
        data = tomllib.load(f)
    data.pop("description", None)
    return data
