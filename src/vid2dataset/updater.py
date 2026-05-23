"""Update checker and one-click installer for vid2dataset.

Queries the GitHub releases API to detect newer versions, downloads
the new .exe, and writes a small batch script that swaps it on next
launch (you can't replace a running .exe on Windows directly).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from vid2dataset import __version__

log = logging.getLogger(__name__)

GITHUB_OWNER = "peter119lee"
GITHUB_REPO = "vid2dataset"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str            # e.g. "v0.2.0"
    version: str        # e.g. "0.2.0"
    name: str           # release title
    notes: str          # markdown body
    exe_url: str | None # download URL of vid2dataset.exe asset
    exe_size: int       # bytes


def _parse_version(s: str) -> tuple[int, ...]:
    """'v0.2.0' -> (0, 2, 0). Returns (0,) on parse failure."""
    s = s.lstrip("v")
    parts = s.split(".")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p.split("-")[0]))  # strip pre-release suffixes
        except ValueError:
            break
    return tuple(out) if out else (0,)


def fetch_latest_release(timeout: float = 10.0) -> ReleaseInfo | None:
    """Query GitHub for the latest release. Returns None on network failure."""
    try:
        req = urllib.request.Request(
            RELEASES_API,
            headers={"User-Agent": f"vid2dataset/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        log.warning("Update check failed: %s", e)
        return None

    tag = data.get("tag_name", "")
    version = tag.lstrip("v")
    exe_url = None
    exe_size = 0
    for asset in data.get("assets", []):
        if asset.get("name", "").lower().endswith(".exe"):
            exe_url = asset.get("browser_download_url")
            exe_size = int(asset.get("size", 0))
            break

    return ReleaseInfo(
        tag=tag,
        version=version,
        name=data.get("name", tag),
        notes=data.get("body", ""),
        exe_url=exe_url,
        exe_size=exe_size,
    )


def is_newer(remote_version: str, local_version: str = __version__) -> bool:
    """Return True if remote_version is newer than local_version."""
    return _parse_version(remote_version) > _parse_version(local_version)


def is_running_as_exe() -> bool:
    """True if we are running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def download_exe(
    url: str,
    dest: Path,
    progress_cb: callable | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download an .exe to ``dest``. Calls ``progress_cb(bytes_done, total)``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": f"vid2dataset/{__version__}"})

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)

    tmp.replace(dest)
    return dest


def install_update(new_exe: Path) -> None:
    """Stage an update for the currently-running .exe.

    On Windows you can't replace a running .exe. We write a small .bat that:
      1) waits for our process to exit
      2) replaces the old .exe with the new one
      3) starts the new .exe
      4) deletes itself

    The .bat is started detached, then this process should exit.
    """
    if not is_running_as_exe():
        raise RuntimeError("install_update only works when running from a .exe")

    current_exe = Path(sys.executable).resolve()
    bat_path = current_exe.parent / "vid2dataset_update.bat"
    pid = os.getpid()

    bat = f"""@echo off
:wait
tasklist /FI "PID eq {pid}" 2>NUL | find "{pid}" >NUL
if not errorlevel 1 (
    timeout /t 1 /nobreak >NUL
    goto wait
)
move /Y "{new_exe}" "{current_exe}" >NUL
start "" "{current_exe}"
del "%~f0"
"""
    bat_path.write_text(bat, encoding="utf-8")

    # Launch the updater detached and exit our process
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    import subprocess
    subprocess.Popen(
        ["cmd", "/c", str(bat_path)],
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
