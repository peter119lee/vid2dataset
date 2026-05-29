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
    # torch's stdlib needs (159 modules, AST-scanned from torch + numpy + sympy
    # in the cached runtime). When --exclude-module torch removes torch from
    # the graph, PyInstaller stops tracing what torch imports. Since the
    # runtime download brings torch back at activation time, but its stdlib
    # deps must already be in the .exe, we list them all explicitly.
    # Re-generate via:
    #   python -c "import ast, sys, pathlib; ..." (see scan in repo notes)
    torch_stdlib_deps = [
        "_codecs", "_collections", "_collections_abc", "_compat_pickle",
        "_ctypes", "_operator", "_string", "_thread", "_warnings",
        "_weakrefset", "abc", "argparse", "array", "ast", "asyncio",
        "asyncio.events", "atexit", "base64", "binascii", "bisect",
        "builtins", "bz2", "cProfile", "cmath", "code", "codecs",
        "collections", "collections.abc", "colorsys", "concurrent",
        "concurrent.futures", "concurrent.futures._base",
        "concurrent.futures.process", "concurrent.futures.thread",
        "configparser", "contextlib", "contextvars", "copy", "copyreg",
        "csv", "ctypes", "ctypes.wintypes", "ctypes.util", "dataclasses",
        "datetime", "decimal", "difflib", "dis", "doctest", "enum",
        "errno", "faulthandler", "fileinput", "fnmatch", "fractions",
        "ftplib", "functools", "gc", "getpass", "gettext", "glob", "gzip",
        "hashlib", "heapq", "html.entities", "importlib", "importlib.abc",
        "importlib.machinery", "importlib.metadata", "importlib.resources",
        "importlib.util", "inspect", "io", "ipaddress", "itertools", "json",
        "keyword", "linecache", "locale", "logging", "lzma", "marshal",
        "math", "mmap", "msvcrt", "multiprocessing",
        "multiprocessing.connection", "multiprocessing.pool",
        "multiprocessing.process", "multiprocessing.queues",
        "multiprocessing.reduction", "multiprocessing.resource_sharer",
        "multiprocessing.synchronize", "multiprocessing.util", "numbers",
        "operator", "optparse", "os", "os.path", "pathlib", "pdb",
        "pickle", "pickletools", "pkgutil", "platform", "posixpath",
        "pprint", "pstats", "pydoc", "queue", "random", "re",
        "rlcompleter", "runpy", "secrets", "select", "selectors", "shlex",
        "shutil", "signal", "site", "socket", "sqlite3", "stat",
        "statistics", "string", "struct", "subprocess", "sys", "sysconfig",
        "tarfile", "tempfile", "textwrap", "threading", "time", "timeit",
        "tokenize", "trace", "traceback", "types", "typing",
        "unicodedata", "unittest", "unittest.case", "unittest.mock",
        "urllib", "urllib.error", "urllib.parse", "urllib.request", "uuid",
        "warnings", "wave", "weakref", "winreg", "xml.dom.minidom",
        "xml.etree.ElementTree", "zipfile", "zipimport", "zlib",
    ]
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
        "vid2dataset.watermark",
        "vid2dataset.report",
        "vid2dataset.async_writer",
        "vid2dataset.i18n",
        "vid2dataset.updater",
        "vid2dataset.presets",
        "vid2dataset.gpu_filters",
        "vid2dataset.gpu_runtime",
        "vid2dataset.hardware",
        "vid2dataset.tooltip",
        "vid2dataset.keyframe_decoder",
        "vid2dataset.cli",  # has gpu-test command
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
        "imageio_ffmpeg",
        "pydantic",
        *torch_stdlib_deps,
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        # Exclude torch from .exe to keep size small. Users wanting GPU pipeline
        # should "pip install vid2dataset[gpu]" \u2014 see release notes.
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
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
