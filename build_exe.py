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
    # Ensure PyInstaller is available
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "-q"])

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "vid2dataset",
        "--add-data", f"{HERE / 'src' / 'vid2dataset' / 'presets'};vid2dataset/presets",
        "--hidden-import", "vid2dataset",
        "--hidden-import", "customtkinter",
        "--collect-all", "customtkinter",
        str(HERE / "src" / "vid2dataset" / "app.py"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(HERE))
    print(f"\nDone! Executable at: {HERE / 'dist' / 'vid2dataset.exe'}")


if __name__ == "__main__":
    main()
