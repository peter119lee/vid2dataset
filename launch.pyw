"""Double-click this file to launch vid2dataset."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Find the venv python
venv_python = HERE / ".venv" / "Scripts" / "python.exe"
if not venv_python.exists():
    venv_python = HERE / ".venv" / "bin" / "python"

if not venv_python.exists():
    # Fallback: try system python
    venv_python = sys.executable

subprocess.Popen(
    [str(venv_python), "-m", "vid2dataset", "app"],
    cwd=str(HERE),
    creationflags=0x08000000,  # CREATE_NO_WINDOW — hide console
)
