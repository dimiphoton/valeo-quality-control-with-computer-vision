"""Tests du prétraitement (split, poids, crop) — images synthétiques."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from valeo_qc.preprocessing import (
    ROT_CROP,
    class_weights,
    crop_frame,
    prepare_dataset,
    rotate_and_crop,
    stratified_split,
)


def test_class_weights_inverses_a_la_frequence() -> None:
    """La classe rare pèse plus que la classe fréquente."""
    labels = pd.Series([0, 0, 0, 1])
    weights = class_weights(labels)
    assert weights[1] > weights[0]
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 2.0


def test_stratified_split_conserve_toutes_les_classes() -> None:
    """Train et val voient chaque label, et l'union reconstitue le jeu."""
    rows = []
    for label, n in [("GOOD", 20), ("Missing", 20), ("Boucle plate", 10)]:
        for i in range(n):
            rows.append({"filename": f"{label}_{i}.png", "Label": label})
    frame = pd.DataFrame(rows)
    train, val = stratified_split(frame, val_fraction=0.2, seed=0)
    assert set(train["Label"]) == set(frame["Label"])
    assert set(val["Label"]) == set(frame["Label"])
    assert len(train) + len(val) == len(frame)
    assert set(train["filename"]).isdisjoint(set(val["filename"]))


def test_stratified_split_reproductible() -> None:
    """La même graine redonne le même split."""
    frame = pd.DataFrame(
        {"filename": [f"{i}.png" for i in range(30)], "Label": ["A"] * 15 + ["B"] * 15}
    )
    a_train, a_val = stratified_split(frame, seed=7)
    b_train, b_val = stratified_split(frame, seed=7)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_val, b_val)


def test_rotate_and_crop_n_ecrase_pas_la_source(tmp_path: Path) -> None:
    """Le PNG brut reste intact ; le crop est écrit ailleurs."""
    source = tmp_path / "raw.png"
    dest = tmp_path / "processed" / "out.png"
    Image.new("L", (800, 900), color=128).save(source)
    before = source.read_bytes()
    # Crop Die01 : (340, 120, 500, 680) après rotation expand — image assez grande.
    rotate_and_crop(source, dest, lib="Die01")
    assert dest.is_file()
    assert source.read_bytes() == before
    cropped = Image.open(dest)
    left, upper, right, lower = ROT_CROP["Die01"][1]
    assert cropped.size == (right - left, lower - upper)


def test_rotate_and_crop_lib_inconnu(tmp_path: Path) -> None:
    """Un die hors table lève KeyError."""
    source = tmp_path / "raw.png"
    Image.new("L", (10, 10)).save(source)
    with pytest.raises(KeyError, match="lib inconnu"):
        rotate_and_crop(source, tmp_path / "out.png", lib="Die99")


def _write_gray_png(path: Path) -> None:
    """PNG assez grand pour le crop Die01 après rotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (800, 900), color=128).save(path)


def test_crop_frame_saute_existant_et_liste_les_manquants(tmp_path: Path) -> None:
    """Un PNG déjà là est sauté ; une source absente est signalée."""
    source_dir = tmp_path / "raw"
    dest_dir = tmp_path / "processed"
    present = "ok.png"
    _write_gray_png(source_dir / present)
    frame = pd.DataFrame(
        {"filename": [present, "absent.png"], "lib": ["Die01", "Die01"]}
    )
    first = crop_frame(frame, source_dir, dest_dir)
    assert first["written"] == 1
    assert first["missing"] == ["absent.png"]
    before = (dest_dir / present).read_bytes()
    second = crop_frame(frame, source_dir, dest_dir, overwrite=False)
    assert second["skipped"] == 1
    assert (dest_dir / present).read_bytes() == before


def test_prepare_dataset_n_ecrase_pas_raw(tmp_path: Path) -> None:
    """Pipeline complet : split, poids sur le train, crop hors de raw/."""
    raw_train = tmp_path / "raw_train"
    raw_test = tmp_path / "raw_test"
    processed = tmp_path / "processed"
    rows = []
    for label in ("GOOD", "Missing"):
        for i in range(6):
            name = f"{label}_{i}.png"
            _write_gray_png(raw_train / name)
            rows.append(
                {"filename": name, "window": 2003, "lib": "Die01", "Label": label}
            )
    labels_csv = tmp_path / "y_train.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=True)
    test_name = "test0.png"
    _write_gray_png(raw_test / test_name)
    test_meta = tmp_path / "win_and_lib.csv"
    pd.DataFrame(
        [{"filename": test_name, "window": 2003, "lib": "Die01"}]
    ).to_csv(test_meta, index=False)

    raw_bytes = {p.name: p.read_bytes() for p in raw_train.glob("*.png")}
    summary = prepare_dataset(
        labels_csv=labels_csv,
        train_images_dir=raw_train,
        test_images_dir=raw_test,
        test_meta_csv=test_meta,
        processed_dir=processed,
        val_fraction=0.25,
        seed=0,
        workers=1,
    )
    assert summary["n_train"] + summary["n_val"] == len(rows)
    assert summary["n_val"] > 0
    split = pd.read_csv(summary["split_path"])
    assert set(split["split"]) == {"train", "val"}
    payload = json.loads(summary["weights_path"].read_text(encoding="utf-8"))
    assert payload["computed_on"] == "train_split"
    assert payload["n_train"] == summary["n_train"]
    assert summary["train_crop"]["written"] == len(rows)
    assert summary["test_crop"]["written"] == 1
    assert (processed / "test" / test_name).is_file()
    for name, content in raw_bytes.items():
        assert (raw_train / name).read_bytes() == content
    # Les poids viennent du split train, pas du jeu entier.
    train_ids = split.loc[split["split"] == "train", "label_id"]
    expected = class_weights(train_ids)
    for class_id, weight in expected.items():
        assert payload["weights"][str(class_id)] == pytest.approx(weight)
