"""Unit tests for vid2dataset.gpu_runtime (pure logic, no network)."""

from __future__ import annotations

import json

from vid2dataset import gpu_runtime as gr
from vid2dataset.gpu_runtime import (
    _DEFAULT_TORCH_VERSION,
    _PYPI_DEPS,
    _TORCH_VERSIONS,
    RUNTIME_VERSION,
    HardwareProfile,
    _classify_nvidia,
    _wheel_target,
    cuda_version_for_profile,
    runtime_supported,
    total_download_size_mb,
)


def _nvidia(arch: str, os_name: str = "windows") -> HardwareProfile:
    return HardwareProfile(
        vendor="NVIDIA",
        gpu_name="test GPU",
        arch=arch,
        compute_cap=0.0,
        os_name=os_name,
        os_arch="x86_64",
    )


def _vendor(vendor: str) -> HardwareProfile:
    return HardwareProfile(
        vendor=vendor,
        gpu_name="test GPU",
        arch="",
        compute_cap=0.0,
        os_name="windows",
        os_arch="x86_64",
    )


# ── GPU name/CC classification ─────────────────────────────────────────


def test_classify_blackwell_by_name_and_cc() -> None:
    assert _classify_nvidia("NVIDIA GeForce RTX 5090", 12.0) == "blackwell"
    assert _classify_nvidia("NVIDIA GeForce RTX 5070 Ti", 12.0) == "blackwell"
    # CC alone should be enough even if the name is unrecognised
    assert _classify_nvidia("NVIDIA Whatever", 12.0) == "blackwell"
    assert _classify_nvidia("NVIDIA B200", 10.0) == "blackwell"


def test_classify_older_generations() -> None:
    assert _classify_nvidia("NVIDIA H100 PCIe", 9.0) == "hopper"
    assert _classify_nvidia("NVIDIA GeForce RTX 4090", 8.9) == "ada"
    assert _classify_nvidia("NVIDIA GeForce RTX 3090", 8.6) == "ampere"
    assert _classify_nvidia("NVIDIA A100", 8.0) == "ampere"
    assert _classify_nvidia("NVIDIA GeForce RTX 2060", 7.5) == "turing"
    assert _classify_nvidia("NVIDIA GeForce GTX 1660", 7.5) == "turing"
    # Pascal and older are unclassified -> generic default path
    assert _classify_nvidia("NVIDIA GeForce GTX 1080", 6.1) == ""


# ── CUDA tag selection ─────────────────────────────────────────────────


def test_blackwell_gets_cu128() -> None:
    # cu126 wheels have no sm_100/sm_120 kernels; RTX 50xx must get cu128.
    assert cuda_version_for_profile(_nvidia("blackwell")) == "cu128"


def test_everything_else_gets_cu126() -> None:
    for arch in ("hopper", "ada", "ampere", "turing", ""):
        assert cuda_version_for_profile(_nvidia(arch)) == "cu126", arch


def test_non_nvidia_gets_none() -> None:
    for vendor in ("AMD", "Apple", "Intel", "Unknown"):
        assert cuda_version_for_profile(_vendor(vendor)) is None, vendor


def test_nvidia_on_macos_gets_none() -> None:
    assert cuda_version_for_profile(_nvidia("ampere", os_name="macos")) is None


def test_runtime_supported_matrix() -> None:
    ok, _ = runtime_supported(_nvidia("ampere"))
    assert ok is True
    for vendor in ("AMD", "Apple", "Intel", "Unknown"):
        ok, reason = runtime_supported(_vendor(vendor))
        assert ok is False and reason, vendor
    ok, reason = runtime_supported(_nvidia("ampere", os_name="macos"))
    assert ok is False and reason


# ── Pin consistency (guards against bumping one constant and not the rest) ─


def test_every_selectable_tag_has_a_pinned_torch() -> None:
    selectable = {
        cuda_version_for_profile(_nvidia(arch))
        for arch in ("blackwell", "hopper", "ada", "ampere", "turing", "")
    }
    assert selectable <= set(_TORCH_VERSIONS), selectable


def test_runtime_version_tracks_pins() -> None:
    # RUNTIME_VERSION must change whenever the torch or numpy pin changes,
    # otherwise stale caches would be treated as current.
    assert _DEFAULT_TORCH_VERSION in RUNTIME_VERSION
    numpy_pin = dict(_PYPI_DEPS)["numpy"]
    assert numpy_pin in RUNTIME_VERSION


def test_torch_versions_are_uniform() -> None:
    # One torch version across all tags = one _PYPI_DEPS matrix. If this is
    # ever relaxed, _PYPI_DEPS must become per-torch-version too.
    assert set(_TORCH_VERSIONS.values()) == {_DEFAULT_TORCH_VERSION}


def test_download_size_estimates() -> None:
    assert total_download_size_mb("cu126") > 2000
    assert total_download_size_mb("cu128") > total_download_size_mb("cu126")
    # default (no tag) must not crash and stays in a sane range
    assert 2000 < total_download_size_mb() < 4000


# ── Wheel cache naming ─────────────────────────────────────────────────


def test_wheel_target_uses_real_filename() -> None:
    url = "https://download.pytorch.org/whl/cu126/torch-2.11.0%2Bcu126-cp312-cp312-win_amd64.whl"
    target = _wheel_target("torch", url)
    assert target.name == "torch-2.11.0+cu126-cp312-cp312-win_amd64.whl"


def test_wheel_target_differs_across_tags_and_versions() -> None:
    # A leftover wheel from another CUDA tag or torch version must never be
    # reused, so the cache filename has to differ.
    a = _wheel_target("torch", "https://x/whl/torch-2.11.0%2Bcu126-cp312-cp312-win_amd64.whl")
    b = _wheel_target("torch", "https://x/whl/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl")
    c = _wheel_target("torch", "https://x/whl/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl")
    assert len({a, b, c}) == 3


def test_wheel_target_falls_back_to_generic_name() -> None:
    target = _wheel_target("torch", "https://example.com/download?id=123")
    assert target.name == "torch.whl"


# ── Stale-cache wipe + manifest cuda_tag ───────────────────────────────


def test_clear_stale_cache_keeps_matching_build(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gr, "RUNTIME_DIR", tmp_path)
    extracted = tmp_path / "torch" / "some_module.py"
    extracted.parent.mkdir()
    extracted.write_text("current", encoding="utf-8")
    (tmp_path / gr.MANIFEST_FILE).write_text(
        json.dumps({"version": gr.RUNTIME_VERSION, "cuda_tag": "cu126"}),
        encoding="utf-8",
    )
    gr._clear_stale_cache({}, "cu126")
    assert extracted.exists()  # same version + same tag: nothing wiped


def test_clear_stale_cache_wipes_on_tag_mismatch(tmp_path, monkeypatch) -> None:
    # GPU swap scenario: cache built for cu126, user now needs cu128.
    monkeypatch.setattr(gr, "RUNTIME_DIR", tmp_path)
    extracted = tmp_path / "torch" / "some_module.py"
    extracted.parent.mkdir()
    extracted.write_text("cu126 build", encoding="utf-8")
    (tmp_path / gr.MANIFEST_FILE).write_text(
        json.dumps({"version": gr.RUNTIME_VERSION, "cuda_tag": "cu126"}),
        encoding="utf-8",
    )
    gr._clear_stale_cache({}, "cu128")
    assert not extracted.exists()


def test_clear_stale_cache_wipes_old_version_and_keeps_target_wheels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gr, "RUNTIME_DIR", tmp_path)
    orphan = tmp_path / "torch" / "_orphan.py"
    orphan.parent.mkdir()
    orphan.write_text("v0.8 leftover", encoding="utf-8")
    (tmp_path / gr.MANIFEST_FILE).write_text(
        json.dumps({"version": "torch2.5.1+cu121+numpy"}), encoding="utf-8"
    )
    url = "https://x/whl/torch-2.11.0%2Bcu126-cp312-cp312-win_amd64.whl"
    wanted_wheel = gr._wheel_target("torch", url)
    wanted_wheel.parent.mkdir(parents=True)
    wanted_wheel.write_bytes(b"downloaded")
    foreign_wheel = wanted_wheel.parent / "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"
    foreign_wheel.write_bytes(b"old")

    gr._clear_stale_cache({"torch": url}, "cu126")

    assert not orphan.exists()  # old extracted tree wiped
    assert wanted_wheel.exists()  # current-target wheel kept for retry
    assert not foreign_wheel.exists()  # cross-version wheel removed


def test_runtime_status_exposes_cuda_tag(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gr, "RUNTIME_DIR", tmp_path)
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / gr.MANIFEST_FILE).write_text(
        json.dumps({"version": gr.RUNTIME_VERSION, "cuda_tag": "cu128"}),
        encoding="utf-8",
    )
    st = gr.runtime_status()
    assert st.cached is True
    assert st.cuda_tag == "cu128"


def test_runtime_status_old_manifest_not_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gr, "RUNTIME_DIR", tmp_path)
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / gr.MANIFEST_FILE).write_text(
        json.dumps({"version": "torch2.5.1+cu121+numpy"}), encoding="utf-8"
    )
    st = gr.runtime_status()
    assert st.cached is False  # version bump invalidates the v0.8 cache
    assert st.cuda_tag is None
