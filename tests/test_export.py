"""Tests d'export ONNX — petits réseaux, pas les checkpoints officiels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from valeo_qc.export import (
    _ForwardOnly,
    export_onnx,
    run_onnx,
    torch_vs_onnx,
    write_manifest,
)
from valeo_qc.padim import embedding_concat


class TinyClassifier(nn.Module):
    """3 → 6 classes, assez petit pour un test CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


class TinyPadim(nn.Module):
    """Deux cartes de features concaténées, comme PaDiM en miniature."""

    def __init__(self, idx: torch.Tensor) -> None:
        super().__init__()
        self.l1 = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.l2 = nn.Conv2d(4, 8, 3, padding=1)
        self.register_buffer("idx", idx.long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1 = self.l1(x)
        layer2 = self.l2(self.pool(layer1))
        concatenated = embedding_concat(layer1, layer2)
        return torch.index_select(concatenated, 1, self.idx)


def test_export_tiny_classifier_match_torch(tmp_path: Path) -> None:
    """ONNX et PyTorch donnent les mêmes probas sur un dummy."""
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    onnx_path = tmp_path / "clf.onnx"
    export_onnx(
        _ForwardOnly(model),
        onnx_path,
        (1, 3, 16, 16),
        output_name="probs",
    )
    dummy = torch.rand(2, 3, 16, 16)
    stats = torch_vs_onnx(model, onnx_path, dummy)
    assert stats["ok"]
    assert stats["max_abs"] < 1e-5
    # batch dynamique : N=2 alors que le dummy d'export était N=1
    out = run_onnx(onnx_path, dummy.numpy())
    assert out.shape == (2, 6)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_export_tiny_padim_match_torch(tmp_path: Path) -> None:
    """Le concat vectorisé survit à l'export (unfold/fold + index_select)."""
    torch.manual_seed(1)
    idx = torch.tensor([0, 3, 7, 11])
    model = TinyPadim(idx).eval()
    onnx_path = tmp_path / "padim.onnx"
    export_onnx(
        model,
        onnx_path,
        (1, 3, 8, 8),
        output_name="embeddings",
    )
    dummy = torch.rand(2, 3, 8, 8)
    stats = torch_vs_onnx(model, onnx_path, dummy, rtol=1e-4, atol=1e-5)
    assert stats["ok"]
    out = run_onnx(onnx_path, dummy.numpy())
    assert out.shape == (2, 4, 8, 8)


def test_write_manifest_relatif(tmp_path: Path) -> None:
    """Le manifeste pointe vers les fichiers et rappelle la limite min-max."""
    clf = tmp_path / "c.onnx"
    pad = tmp_path / "p.onnx"
    clf.write_bytes(b"x")
    pad.write_bytes(b"yy")
    dest = tmp_path / "manifest.json"
    write_manifest(
        classifier_path=clf,
        padim_path=pad,
        opset=17,
        checks={"classifier": {"ok": True, "max_abs": 0.0}},
        out_path=dest,
    )
    import json

    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["opset"] == 17
    assert payload["classifier"]["bytes"] == 1
    assert payload["padim_backbone"]["d"] == 550
    assert "figer min/max" in payload["decision"]["score_norm"]
    assert payload["checks"]["classifier"]["ok"] is True


def test_torch_vs_onnx_leve_si_divergence(tmp_path: Path) -> None:
    """Un graphe volontairement différent fait échouer la comparaison."""
    torch.manual_seed(0)
    a = TinyClassifier().eval()
    b = TinyClassifier().eval()
    onnx_path = tmp_path / "a.onnx"
    export_onnx(a, onnx_path, (1, 3, 8, 8))
    dummy = torch.rand(1, 3, 8, 8)
    with pytest.raises(AssertionError, match="divergent"):
        torch_vs_onnx(b, onnx_path, dummy, rtol=1e-12, atol=1e-12)
