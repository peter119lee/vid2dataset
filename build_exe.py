"""Build a standalone .exe with PyInstaller.

Run this script once to produce dist/vid2dataset.exe (~150MB).
Upload it to GitHub Releases.

Usage:
    python build_exe.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    # Ensure PyInstaller is available (skip if already installed)
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Hidden imports — PyInstaller's import tracer misses some.
    hidden = [
        "vid2dataset",
        "vid2dataset.config",
        "vid2dataset.extractor",
        "vid2dataset.io_utils",
        "vid2dataset.quality",
        "vid2dataset.crop",
        "vid2dataset.resize",
        "vid2dataset.dedup",
        "vid2dataset.diversity",
        "vid2dataset.color_diversity",
        "vid2dataset.completeness",
        "vid2dataset.auto_quality",
        "vid2dataset.scene",
        "vid2dataset.gallery",
        "vid2dataset.i18n",
        "vid2dataset.updater",
        "vid2dataset.presets",
        "customtkinter",
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "scenedetect",
        "scenedetect.detectors",
        "scenedetect.detectors.content_detector",
        "imagehash",
        "pydantic",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--name", "vid2dataset",
        "--add-data", f"{HERE / 'src' / 'vid2dataset' / 'presets'};vid2dataset/presets",
        "--collect-all", "customtkinter",
        "--collect-all", "scenedetect",
        "--collect-submodules", "vid2dataset",
    ]
    for h in hidden:
        cmd.extend(["--hidden-import", h])
    cmd.append(str(HERE / "src" / "vid2dataset" / "app.py"))

    print("Running PyInstaller...")
    subprocess.check_call(cmd, cwd=str(HERE))
    out = HERE / "dist" / "vid2dataset.exe"
    if out.exists():
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\nDone! {out} ({size_mb:.1f} MB)")
    else:
        print("\nBuild failed — vid2dataset.exe not produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
