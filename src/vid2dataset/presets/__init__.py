"""Built-in presets bundled with the wheel."""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PRESETS_PACKAGE = "vid2dataset.presets"


def _preset_dir() -> Path:
    """Return the on-disk directory containing preset TOMLs."""
    return Path(resources.files(PRESETS_PACKAGE))  # type: ignore[arg-type]


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
