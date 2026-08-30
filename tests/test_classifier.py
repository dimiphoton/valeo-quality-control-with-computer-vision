"""Tests du classifieur — petit CNN, pas le checkpoint officiel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from valeo_qc.classifier import (
    SplitImageDataset,
    classification_metrics,
    evaluate_classifier,
    load_split,
    load_weight_tensor,
    train_classifier,
)


class TinyNet(nn.Module):
    """3 → 6 classes via pooling, assez petit pour un test CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_classification_metrics_parfait_et_macro() -> None:
    """Accuracy 1 si tout est juste ; F1 macro pénalise une classe oubliée."""
    y_true = np.array([0, 0, 1, 1])
    perfect = classification_metrics(y_true, y_true, n_classes=2)
    assert perfect["accuracy"] == 1.0
    assert perfect["macro_f1"] == pytest.approx(1.0)
    always_zero = classification_metrics(y_true, np.zeros_like(y_true), n_classes=2)
    assert always_zero["recall"]["1"] == 0.0
    assert always_zero["macro_f1"] < 1.0


def test_split_image_dataset(tmp_path: Path) -> None:
    """Un PNG + une ligne de split se chargent en tenseur 3×224×224."""
    Image.new("RGB", (40, 40), color=(10, 20, 30)).save(tmp_path / "a.png")
    frame = pd.DataFrame([{"filename": "a.png", "label_id": 4}])
    dataset = SplitImageDataset(frame, tmp_path)
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 224, 224)
    assert label == 4


def test_load_split_filtre(tmp_path: Path) -> None:
    """Le filtre train/val lit le CSV ; une valeur inconnue lève."""
    csv = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "filename": ["a.png", "b.png"],
            "label_id": [0, 1],
            "split": ["train", "val"],
        }
    ).to_csv(csv, index=False)
    val = load_split(csv, which="val")
    assert list(val["filename"]) == ["b.png"]
    with pytest.raises(ValueError, match="which"):
        load_split(csv, which="test")


def test_load_weight_tensor(tmp_path: Path) -> None:
    """Les poids JSON deviennent un tenseur de longueur 6."""
    path = tmp_path / "w.json"
    path.write_text(
        json.dumps({"weights": {"0": 0.5, "4": 2.0}}),
        encoding="utf-8",
    )
    tensor = load_weight_tensor(path)
    assert tensor.shape == (6,)
    assert float(tensor[0]) == pytest.approx(0.5)
    assert float(tensor[4]) == pytest.approx(2.0)
    assert float(tensor[1]) == pytest.approx(1.0)


def test_train_classifier_une_epoch_sans_mlflow(tmp_path: Path) -> None:
    """Une epoch sur 8 images synthétiques écrit un checkpoint."""
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for split_name in ("train", "val"):
        for i in range(4):
            name = f"{split_name}_{i}.png"
            Image.new("RGB", (32, 32), color=i * 40).save(images / name)
            rows.append(
                {
                    "filename": name,
                    "label_id": i % 2,
                    "split": split_name,
                }
            )
    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    weights_json = tmp_path / "w.json"
    weights_json.write_text(json.dumps({"weights": {"0": 1.0, "1": 1.0}}), encoding="utf-8")
    out = tmp_path / "best.pt"
    best = train_classifier(
        split_csv=split_csv,
        images_dir=images,
        weights_json=weights_json,
        epochs=1,
        batch_size=2,
        model=TinyNet(),
        checkpoint_out=out,
        log_mlflow=False,
        pretrained=False,
    )
    assert out.is_file()
    assert best["n"] == 4
    assert "macro_f1" in best


def test_evaluate_classifier_tiny() -> None:
    """L'éval d'un TinyNet sur un batch ne plante pas."""
    batch = [(torch.zeros(2, 3, 224, 224), torch.tensor([0, 1]))]
    metrics = evaluate_classifier(TinyNet(), batch, torch.device("cpu"))
    assert metrics["n"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0
