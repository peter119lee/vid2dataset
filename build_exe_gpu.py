"""Build the GPU-enabled .exe variant.

Bundles PyTorch + CUDA, results in ~1.5-2 GB .exe.
Output: dist/vid2dataset-gpu.exe

Usage:
    python build_exe_gpu.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Verify torch+CUDA is in the env
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: torch.cuda not available in this env. The GPU .exe")
            print("will only have CPU torch, which won't actually accelerate anything.")
            print("Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: torch not installed. Install first:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    hidden = [
        "vid2dataset",
        "vid2dataset.config", "vid2dataset.extractor", "vid2dataset.io_utils",
        "vid2dataset.quality", "vid2dataset.crop", "vid2dataset.resize",
        "vid2dataset.dedup", "vid2dataset.diversity", "vid2dataset.color_diversity",
        "vid2dataset.completeness", "vid2dataset.auto_quality", "vid2dataset.scene",
        "vid2dataset.gallery", "vid2dataset.i18n", "vid2dataset.updater",
        "vid2dataset.presets", "vid2dataset.hardware", "vid2dataset.tooltip",
        "vid2dataset.keyframe_decoder", "vid2dataset.gpu_filters",
        "customtkinter", "tkinter", "tkinter.filedialog", "tkinter.messagebox",
        "cv2", "numpy", "PIL", "PIL.Image",
        "scenedetect", "scenedetect.detectors", "scenedetect.detectors.content_detector",
        "imagehash", "pydantic", "imageio_ffmpeg",
        "torch", "torch.cuda", "torch.backends", "torch.backends.cuda",
        "torch.backends.cudnn", "torch.backends.mps",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--name", "vid2dataset-gpu",
        "--add-data", f"{HERE / 'src' / 'vid2dataset' / 'presets'};vid2dataset/presets",
        "--collect-all", "customtkinter",
        "--collect-all", "scenedetect",
        # Trim torch bundle to fit GitHub's 2GB asset limit
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "torch.distributed",
        "--exclude-module", "torch.onnx",
        "--exclude-module", "torch.utils.tensorboard",
        "--exclude-module", "torch.testing",
        "--collect-submodules", "torch",
        "--collect-submodules", "vid2dataset",
    ]
    for h in hidden:
        cmd.extend(["--hidden-import", h])
    cmd.append(str(HERE / "src" / "vid2dataset" / "app.py"))

    print("Running PyInstaller (this may take 5-10 minutes)...")
    subprocess.check_call(cmd, cwd=str(HERE))
    out = HERE / "dist" / "vid2dataset-gpu.exe"
    if out.exists():
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\nDone! {out} ({size_mb:.1f} MB)")
    else:
        print("\nBuild failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()