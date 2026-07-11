"""Unit tests for vid2dataset.tagger (pure logic + FakeSession flow, no network)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vid2dataset.tagger import (
    DEFAULT_TAGGER_MODEL,
    TAGGER_MODELS,
    TagVocabulary,
    WDTagger,
    collect_images,
    compose_caption,
    format_tag,
    load_tag_vocabulary,
    preprocess_image,
    tag_folder,
    write_sidecar,
)

# ── Caption formatting ─────────────────────────────────────────────────


def test_format_tag_underscores_to_spaces() -> None:
    assert format_tag("long_hair") == "long hair"
    assert format_tag("hair_between_eyes") == "hair between eyes"
    assert format_tag("1girl") == "1girl"


def test_format_tag_preserves_kaomoji() -> None:
    # Audit F3: ^_^ must not become "^ ^".
    assert format_tag("^_^") == "^_^"
    assert format_tag(">_<") == ">_<"
    assert format_tag("(o)_(o)") == "(o)_(o)"


def test_compose_caption_order_and_dedup() -> None:
    caption = compose_caption(
        "mychar",
        [("hatsune_miku", 0.99)],
        [("1girl", 0.9), ("long_hair", 0.8), ("1girl", 0.7), ("Mychar", 0.5)],
    )
    # Trigger first, character before general, case-insensitive dedup.
    assert caption == "mychar, hatsune miku, 1girl, long hair"


def test_compose_caption_is_single_line_even_with_hostile_trigger() -> None:
    caption = compose_caption("my\nchar\t v1", [], [("1girl", 0.9)])
    assert "\n" not in caption and "\t" not in caption
    assert caption == "my char v1, 1girl"


def test_compose_caption_empty_trigger() -> None:
    assert compose_caption("", [], [("1girl", 0.9)]) == "1girl"
    assert compose_caption("", [], []) == ""


def test_write_sidecar_single_lf_line(tmp_path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    txt = write_sidecar(img, "mychar, 1girl")
    raw = txt.read_bytes()
    assert raw == b"mychar, 1girl\n"  # LF only, no CRLF (audit F2/F6)
    assert txt.name == "a.txt"


# ── Vocabulary parsing ─────────────────────────────────────────────────


def test_load_tag_vocabulary(tmp_path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "tag_id,name,category,count\n"
        "0,general,9,10\n"
        "1,sensitive,9,10\n"
        "2,1girl,0,100\n"
        "3,long_hair,0,90\n"
        "4,hatsune_miku,4,50\n"
        "5,vocaloid,3,40\n",
        encoding="utf-8",
    )
    vocab = load_tag_vocabulary(csv_path)
    assert vocab.rating == [(0, "general"), (1, "sensitive")]
    assert vocab.general == [(2, "1girl"), (3, "long_hair")]
    assert vocab.character == [(4, "hatsune_miku")]
    assert vocab.copyright == [(5, "vocaloid")]


# ── Preprocessing ──────────────────────────────────────────────────────


def test_preprocess_shape_dtype_and_bgr() -> None:
    img = Image.new("RGB", (448, 448), (255, 0, 0))  # pure red
    arr = preprocess_image(img, 448)
    assert arr.shape == (448, 448, 3)
    assert arr.dtype == np.float32
    # BGR order: red pixel -> B channel 0, R channel 255.
    assert arr[224, 224, 0] == 0.0
    assert arr[224, 224, 2] == 255.0


def test_preprocess_letterbox_pads_white() -> None:
    img = Image.new("RGB", (400, 100), (0, 0, 0))  # wide black bar
    arr = preprocess_image(img, 448)
    assert arr.shape == (448, 448, 3)
    # Top rows are padding -> white (255 in every channel).
    assert float(arr[0, 224].min()) == 255.0


def test_preprocess_transparency_composited_to_white() -> None:
    img = Image.new("RGBA", (448, 448), (0, 0, 0, 0))  # fully transparent
    arr = preprocess_image(img, 448)
    # Bare convert("RGB") would give black; the audited pipeline gives white.
    assert float(arr.min()) == 255.0


# ── Registry sanity ────────────────────────────────────────────────────


def test_default_model_registered_with_sane_size() -> None:
    assert DEFAULT_TAGGER_MODEL in TAGGER_MODELS
    for spec in TAGGER_MODELS.values():
        assert spec.approx_mb > 100
        assert "/" in spec.repo_id


# ── Inference flow with a fake ONNX session ────────────────────────────


class _FakeInput:
    name = "input"
    shape = [None, 448, 448, 3]


class _FakeSession:
    """Returns the same probability vector for every image in the batch."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.asarray(probs, dtype=np.float32)

    def get_inputs(self):
        return [_FakeInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, feed):
        n = feed["input"].shape[0]
        return [np.stack([self._probs] * n, axis=0)]


def _fake_tagger() -> WDTagger:
    vocab = TagVocabulary(
        general=[(0, "1girl"), (1, "long_hair"), (2, "^_^")],
        character=[(3, "hatsune_miku")],
        copyright=[(4, "vocaloid")],
        rating=[(5, "general"), (6, "sensitive")],
    )
    #                 1girl lhair  ^_^  miku  voca  gen  sens
    probs = [0.90, 0.40, 0.20, 0.95, 0.50, 0.80, 0.10]
    return WDTagger(session=_FakeSession(probs), vocabulary=vocab)


def _write_png(path, size=(64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 64, 32)).save(path)


def test_tag_paths_applies_thresholds(tmp_path) -> None:
    img = tmp_path / "a.png"
    _write_png(img)
    tagger = _fake_tagger()
    results = tagger.tag_paths([img])
    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert [t for t, _ in r.general] == ["1girl", "long_hair"]  # ^_^ at 0.20 < 0.35
    assert [t for t, _ in r.character] == ["hatsune_miku"]  # 0.95 >= 0.85
    assert r.rating == "general"


def test_tag_paths_isolates_bad_images(tmp_path) -> None:
    good = tmp_path / "good.png"
    _write_png(good)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not a png")
    results = _fake_tagger().tag_paths([broken, good])
    assert results[0].error is not None
    assert results[1].error is None


def test_collect_images_skips_artifacts(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "sub" / "b.jpg")
    _write_png(tmp_path / "_contact_sheet.png")
    (tmp_path / "_gallery.html").write_text("<html>", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    names = [p.name for p in collect_images(tmp_path)]
    assert names == ["a.png", "b.jpg"]


def test_parse_tag_list_normalizes() -> None:
    from vid2dataset.tagger import _parse_tag_list

    assert _parse_tag_list("long_hair, Long Hair, , watermark") == ["long hair", "watermark"]
    assert _parse_tag_list("^_^") == ["^_^"]
    assert _parse_tag_list("") == []


def test_compose_caption_always_and_drop() -> None:
    caption = compose_caption(
        "mychar",
        [("hatsune_miku", 0.99)],
        [("1girl", 0.9), ("long_hair", 0.8)],
        always=["anime screencap"],
        drop={"long hair"},
    )
    # always follows the trigger; drop filters model tags only.
    assert caption == "mychar, anime screencap, hatsune miku, 1girl"


def test_tag_folder_blacklist_removes_tokens(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    summary = tag_folder(
        tmp_path, trigger_word="mychar", blacklist="long_hair", tagger=_fake_tagger()
    )
    caption = (tmp_path / "a.txt").read_text(encoding="utf-8").strip()
    assert caption == "mychar, hatsune miku, 1girl"
    assert "long hair" not in summary.tag_counts


def test_tag_folder_require_rejects_and_moves(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    summary = tag_folder(tmp_path, trigger_word="mychar", require="2girls", tagger=_fake_tagger())
    # The fake tagger never emits 2girls -> image moves to _rejected/.
    assert summary.tagged == 0
    assert summary.failed == 0
    assert summary.rejected == ["a.png"]
    assert not (tmp_path / "a.png").exists()
    assert (tmp_path / "_rejected" / "a.png").exists()
    assert not (tmp_path / "_rejected" / "a.txt").exists()  # no sidecar for rejects


def test_tag_folder_require_pass_keeps_image(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    summary = tag_folder(tmp_path, trigger_word="m", require="1girl", tagger=_fake_tagger())
    assert summary.tagged == 1
    assert summary.rejected == []


def test_tag_folder_exclude_rejects(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    summary = tag_folder(tmp_path, exclude="1girl", tagger=_fake_tagger())
    assert summary.rejected == ["a.png"]
    assert (tmp_path / "_rejected" / "a.png").exists()


def test_tag_folder_trait_pruning(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png")
    # The fake tagger emits identical tags for every image -> 100% frequency.
    summary = tag_folder(
        tmp_path, trigger_word="mychar", trait_prune_threshold=0.9, tagger=_fake_tagger()
    )
    caption = (tmp_path / "a.txt").read_text(encoding="utf-8").strip()
    assert caption == "mychar"  # every constant trait absorbed by the trigger
    assert set(summary.pruned_tags) == {"hatsune miku", "1girl", "long hair"}


def test_tag_folder_cancel_writes_no_fake_captions(tmp_path) -> None:
    # Regression: a cancelled run must NOT stamp trigger-only sidecars onto
    # images the model never saw (they would poison the training set).
    import threading

    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png")
    ev = threading.Event()
    ev.set()  # cancelled before anything is processed
    summary = tag_folder(tmp_path, trigger_word="mychar", tagger=_fake_tagger(), cancel_event=ev)
    assert summary.cancelled is True
    assert summary.tagged == 0
    assert summary.failed == 0
    assert list(tmp_path.glob("**/*.txt")) == []


def test_tag_folder_writes_sidecars_and_summary(tmp_path) -> None:
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "sub" / "b.png")
    _write_png(tmp_path / "_contact_sheet.png")  # must not be tagged

    summary = tag_folder(tmp_path, trigger_word="mychar", tagger=_fake_tagger())

    assert summary.total == 2
    assert summary.tagged == 2
    assert summary.failed == 0
    assert summary.cancelled is False

    caption = (tmp_path / "a.txt").read_text(encoding="utf-8").strip()
    assert caption == "mychar, hatsune miku, 1girl, long hair"
    assert (tmp_path / "sub" / "b.txt").exists()
    assert not (tmp_path / "_contact_sheet.txt").exists()

    assert set(summary.per_image) == {"a.png", "sub/b.png"}
    # Trigger word is excluded from frequency counts.
    assert summary.tag_counts["1girl"] == 2
    assert "mychar" not in summary.tag_counts
