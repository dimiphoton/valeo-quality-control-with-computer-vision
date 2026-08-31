"""Tests PaDiM — tenseurs synthétiques, pas le pickle 1,2 Go."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from valeo_qc.padim import (
    dimension_index,
    embedding_concat,
    fit_gaussian,
    invert_covariance,
    load_padim_stats,
    mahalanobis_maps,
    minmax_image_scores,
    padim_transform,
    raw_image_scores,
    save_padim_stats,
    summarize_scores,
    upsample_and_smooth,
)


def _embedding_concat_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Copie bouclée du notebook, pour vérifier la version vectorisée."""
    import torch.nn.functional as F

    batch, c1, h1, w1 = x.size()
    _, c2, h2, _ = y.size()
    scale = int(h1 / h2)
    unfolded = F.unfold(x, kernel_size=scale, dilation=1, stride=scale)
    unfolded = unfolded.view(batch, c1, -1, h2, h2)
    stacked = torch.zeros(
        batch,
        c1 + c2,
        unfolded.size(2),
        h2,
        h2,
        device=x.device,
        dtype=x.dtype,
    )
    for patch in range(unfolded.size(2)):
        stacked[:, :, patch, :, :] = torch.cat((unfolded[:, :, patch, :, :], y), 1)
    stacked = stacked.view(batch, -1, h2 * h2)
    return F.fold(stacked, kernel_size=scale, output_size=(h1, w1), stride=scale)


def test_dimension_index_reproductible() -> None:
    """Seed 1024 → 550 indices uniques, identiques d'un appel à l'autre."""
    first = dimension_index()
    second = dimension_index()
    assert first.shape == (550,)
    assert len(set(first.tolist())) == 550
    assert int(first.min()) >= 0
    assert int(first.max()) < 1792
    assert torch.equal(first, second)


def test_embedding_concat_double_la_resolution() -> None:
    """layer2 (B, 4, 4, 4) se cale sur layer1 (B, 2, 8, 8) → 6 canaux 8×8."""
    low = torch.ones(2, 2, 8, 8)
    high = torch.ones(2, 4, 4, 4)
    out = embedding_concat(low, high)
    assert tuple(out.shape) == (2, 6, 8, 8)
    assert torch.allclose(out, torch.ones(2, 6, 8, 8))


def test_embedding_concat_vectorise_egal_boucle() -> None:
    """La version sans boucle Python colle au notebook officiel."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 5, 4, 4)
    looped = _embedding_concat_loop(x, y)
    assert torch.allclose(embedding_concat(x, y), looped, atol=1e-6)


def test_fit_gaussian_puis_mahalanobis_nul_sur_la_moyenne() -> None:
    """Un point égal à la moyenne a une distance ~0 (ridge petit)."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 4, 2, 2)).astype(np.float32)
    mean, cov = fit_gaussian(embeddings, ridge=1e-3)
    assert mean.shape == (4, 4)
    assert cov.shape == (4, 4, 4)
    cov_inv = invert_covariance(cov)
    at_mean = np.broadcast_to(mean.reshape(1, 4, 2, 2), (1, 4, 2, 2)).copy()
    dist = mahalanobis_maps(at_mean, mean, cov_inv)
    assert dist.shape == (1, 2, 2)
    assert float(dist.max()) < 0.05


def test_mahalanobis_identite_egal_l2() -> None:
    """Σ^{-1} = I → Mahalanobis² = ||x-μ||²."""
    mean = np.zeros((3, 2), dtype=np.float32)
    cov_inv = np.stack([np.eye(3, dtype=np.float32)] * 2, axis=-1)
    embeddings = np.zeros((1, 3, 1, 2), dtype=np.float32)
    embeddings[0, 0, 0, 0] = 2.0
    dist = mahalanobis_maps(embeddings, mean, cov_inv)
    assert dist[0, 0, 0] == pytest.approx(4.0)
    assert dist[0, 0, 1] == pytest.approx(0.0)


def test_minmax_image_scores_et_max_spatial() -> None:
    """Le score image est le max de la carte normalisée sur le lot."""
    maps = np.array(
        [
            [[0.0, 10.0], [0.0, 0.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ],
        dtype=np.float64,
    )
    scores, min_s, max_s = minmax_image_scores(maps)
    assert min_s == 0.0
    assert max_s == 10.0
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)
    raw = raw_image_scores(maps)
    assert raw[0] == pytest.approx(10.0)


def test_upsample_and_smooth_taille() -> None:
    """32×32 → 128×128, valeurs positives conservées."""
    maps = np.ones((2, 32, 32), dtype=np.float32)
    out = upsample_and_smooth(maps)
    assert out.shape == (2, 128, 128)
    assert float(out.min()) > 0.0


def test_save_load_checkpoint(tmp_path: Path) -> None:
    """Round-trip dict : mean, cov, idx."""
    mean = np.zeros((4, 8), dtype=np.float32)
    cov = np.stack([np.eye(4, dtype=np.float32)] * 8, axis=-1)
    idx = torch.arange(4)
    path = tmp_path / "padim.pkl"
    save_padim_stats(path, mean, cov, idx, extra={"n_train": 3})
    loaded_mean, loaded_cov, loaded_idx = load_padim_stats(path)
    assert loaded_mean.shape == (4, 8)
    assert loaded_cov.shape == (4, 4, 8)
    assert torch.equal(loaded_idx, idx)


def test_load_official_list_format(tmp_path: Path) -> None:
    """Le pickle officiel est une liste [mean, cov] — on régénère idx."""
    import pickle

    mean = np.ones((4, 2), dtype=np.float32)
    cov = np.stack([np.eye(4, dtype=np.float32)] * 2, axis=-1)
    path = tmp_path / "official.pkl"
    with path.open("wb") as handle:
        pickle.dump([mean, cov], handle)
    loaded_mean, loaded_cov, idx = load_padim_stats(path)
    assert loaded_mean.shape == (4, 2)
    assert loaded_cov.shape == (4, 4, 2)
    assert idx.shape == (550,)


def test_summarize_scores_par_classe() -> None:
    """Une classe absente n'apparaît pas ; GOOD a la moyenne attendue."""
    labels = np.array([0, 0, 4])
    raw = np.array([1.0, 3.0, 10.0])
    norm = np.array([0.1, 0.3, 0.9])
    summary = summarize_scores(raw, norm, labels)
    assert summary["n"] == 3
    assert summary["per_class"]["GOOD"]["n"] == 2
    assert summary["per_class"]["GOOD"]["raw_mean"] == pytest.approx(2.0)
    assert "Boucle plate" not in summary["per_class"]
    assert summary["frac_above_0_5"] == pytest.approx(1 / 3)


def test_padim_transform_taille() -> None:
    """Resize 128, tenseur 3×128×128."""
    from PIL import Image

    image = Image.new("RGB", (40, 80), color=(10, 20, 30))
    tensor = padim_transform()(image)
    assert tuple(tensor.shape) == (3, 128, 128)


def test_load_padim_absent(tmp_path: Path) -> None:
    """Fichier manquant → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_padim_stats(tmp_path / "missing.pkl")
