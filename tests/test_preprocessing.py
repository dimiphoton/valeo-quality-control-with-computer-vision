"""Tests du prétraitement (split, poids, crop) — images synthétiques."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from valeo_qc.preprocessing import (
    ROT_CROP,
    class_weights,
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
