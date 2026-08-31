"""Scores PaDiM en numpy/scipy, sans PyTorch (runtime Lambda / ONNX)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

LOGGER = logging.getLogger(__name__)

PADIM_SIZE = 128
GAUSSIAN_SIGMA = 4.0


def invert_covariance(cov: np.ndarray, chunk: int = 32) -> np.ndarray:
    """Inverse chaque covariance spatiale ``(C, C, HW)`` → ``(C, C, HW)``.

    Par paquets pour ne pas allouer un ``(HW, C, C)`` float64 d'un coup
    (~2,5 Go si C=550).

    Parameters
    ----------
    cov
        Tenseur float ``(C, C, HW)``.
    chunk
        Nombre de patches inversés ensemble.

    Returns
    -------
    numpy.ndarray
        Inverses, même forme, float32. ``pinv`` si une tranche est singulière.
    """
    _channels, _, hw = cov.shape
    out = np.empty_like(cov, dtype=np.float32)
    for start in range(0, hw, chunk):
        end = min(start + chunk, hw)
        stacked = np.moveaxis(cov[:, :, start:end], -1, 0).astype(np.float64)
        try:
            inverted = np.linalg.inv(stacked)
        except np.linalg.LinAlgError:
            LOGGER.warning("covariance singulière patches %s-%s, repli pinv", start, end)
            inverted = np.linalg.pinv(stacked)
        out[:, :, start:end] = np.moveaxis(inverted.astype(np.float32), 0, -1)
    return out


def mahalanobis_maps(
    embeddings: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Cartes de distance ``(B, H, W)`` pour des embeddings ``(B, C, H, W)``.

    Parameters
    ----------
    embeddings
        Features réduites.
    mean
        Moyenne PaDiM ``(C, HW)``.
    cov_inv
        Précision ``(C, C, HW)``.

    Returns
    -------
    numpy.ndarray
        Distances de Mahalanobis par patch.
    """
    batch, _channels, height, width = embeddings.shape
    hw = height * width
    flat = embeddings.reshape(batch, embeddings.shape[1], hw)
    diff = flat - mean[np.newaxis, :, :]
    dist = np.empty((batch, hw), dtype=np.float64)
    chunk = 64
    for start in range(0, hw, chunk):
        end = min(start + chunk, hw)
        piece = diff[:, :, start:end]
        inv = cov_inv[:, :, start:end]
        precision_dot = np.einsum("cdi,bdi->bci", inv, piece, optimize=True)
        dist[:, start:end] = np.einsum("bci,bci->bi", piece, precision_dot, optimize=True)
    return dist.reshape(batch, height, width)


def minmax_image_scores(score_maps: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max sur tout le lot, puis max spatial → un score par image.

    Parameters
    ----------
    score_maps
        ``(N, H, W)`` déjà lissées.

    Returns
    -------
    tuple
        Scores ``(N,)``, min et max bruts du lot.
    """
    min_score = float(score_maps.min())
    max_score = float(score_maps.max())
    span = max_score - min_score
    if span <= 0:
        normalized = np.zeros(score_maps.shape, dtype=np.float64)
    else:
        normalized = (score_maps - min_score) / span
    image_scores = normalized.reshape(normalized.shape[0], -1).max(axis=1)
    return image_scores.astype(np.float64), min_score, max_score


def raw_image_scores(score_maps: np.ndarray) -> np.ndarray:
    """Max spatial sans min-max (échelle Mahalanobis, comparable d'un run à l'autre)."""
    return score_maps.reshape(score_maps.shape[0], -1).max(axis=1).astype(np.float64)


def upsample_and_smooth_numpy(
    dist_maps: np.ndarray,
    size: int = PADIM_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Bilinear (zoom) vers ``size×size`` puis Gaussienne σ, sans PyTorch."""
    maps = np.asarray(dist_maps, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[None, ...]
    planes = []
    for plane in maps:
        height, width = plane.shape
        zoomed = zoom(plane, (size / height, size / width), order=1)
        planes.append(zoomed)
    stacked = np.stack(planes, axis=0)
    return gaussian_filter(stacked, sigma=(0.0, sigma, sigma))


def frozen_image_score(
    score_map: np.ndarray,
    min_score: float,
    max_score: float,
) -> float:
    """Score image avec min-max figé (inférence unitaire).

    Parameters
    ----------
    score_map
        Carte lissée ``(H, W)`` ou ``(1, H, W)``.
    min_score, max_score
        Bornes calées une fois (val), pas sur l'image courante.

    Returns
    -------
    float
        Score dans ``[0, 1]`` (clip).
    """
    span = float(max_score) - float(min_score)
    peak = float(np.max(score_map))
    if span <= 0:
        return 0.0
    return float(np.clip((peak - min_score) / span, 0.0, 1.0))
