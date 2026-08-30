"""Logique de décision : fusion classifieur + anomalie, calée sur le coût.

La matrice et la métrique *penalty-weighted accuracy* (PWA) reproduisent
le notebook officiel du challenge (`Supp_files/Notebook_ENS.ipynb`).
Un score d'anomalie au-dessus du seuil → classe ``drift`` (6) ; sinon
argmax des 6 probabilités du classifieur.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

N_KNOWN_CLASSES = 6
N_CLASSES = 7  # 0–5 connues + 6 drift
DRIFT_CLASS = 6
DEFAULT_THRESHOLD = 0.5

# Noms officiels du notebook (train en texte, soumission en entiers).
ID_TO_LABEL: dict[int, str] = {
    0: "GOOD",
    1: "Boucle plate",
    2: "Lift-off blanc",
    3: "Lift-off noir",
    4: "Missing",
    5: "Short circuit MOS",
    6: "Drift",
}
LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in ID_TO_LABEL.items()}
# Variante minuscule vue dans l'énoncé public du challenge.
LABEL_TO_ID["drift"] = DRIFT_CLASS

# Lignes / colonnes = classes 0–6. Extraite du notebook officiel.
COST_MATRIX = np.array(
    [
        [0, 100, 100, 100, 100, 100, 10000],  # GOOD
        [10000, 0, 1, 1, 1, 1, 1000],  # Boucle plate
        [10000, 1, 0, 1, 1, 1, 1000],  # Lift-off blanc
        [10000, 1, 1, 0, 1, 1, 1000],  # Lift-off noir
        [10000, 1, 1, 1, 0, 1, 1000],  # Missing
        [10000, 1, 1, 1, 1, 0, 1000],  # Short circuit MOS
        [10000, 1000, 1000, 1000, 1000, 1000, 0],  # Drift
    ],
    dtype=np.float64,
)


def encode_label(label: str | int) -> int:
    """Convertit un label texte ou entier en identifiant 0–6.

    Parameters
    ----------
    label
        Nom officiel (``GOOD``, ``Drift``, …) ou entier déjà encodé.

    Returns
    -------
    int
        Identifiant de classe.

    Raises
    ------
    ValueError
        Si le label n'est pas dans le mapping.
    """
    if isinstance(label, (int, np.integer)):
        idx = int(label)
        if idx not in ID_TO_LABEL:
            raise ValueError(f"label entier hors 0–6 : {idx}")
        return idx
    if label not in LABEL_TO_ID:
        raise ValueError(f"label inconnu : {label!r}")
    return LABEL_TO_ID[label]


def decide_class(
    p_drift: float,
    class_probs: Iterable[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """Fusionne score d'anomalie et probabilités du classifieur.

    Parameters
    ----------
    p_drift
        Score d'anomalie (PaDiM ou équivalent), plus élevé = plus anomal.
    class_probs
        Six probabilités (classes 0–5), dans l'ordre du mapping officiel.
    threshold
        Au-dessus : on renvoie ``drift`` (6). En dessous : argmax.

    Returns
    -------
    int
        Classe prédite (0–6).

    Raises
    ------
    ValueError
        Si ``class_probs`` n'a pas 6 valeurs.
    """
    probs = np.asarray(list(class_probs), dtype=np.float64)
    if probs.shape != (N_KNOWN_CLASSES,):
        raise ValueError(
            f"class_probs doit avoir {N_KNOWN_CLASSES} valeurs, reçu {probs.shape}"
        )
    if p_drift > threshold:
        return DRIFT_CLASS
    return int(np.argmax(probs))


def penalty_weighted_accuracy(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    penalty_matrix: np.ndarray | None = None,
) -> float:
    """Accuracy pondérée par la matrice de pénalité du challenge.

    ``PWA = 1 - pénalité_totale / (n × max_pénalité)``. Une prédiction
    juste coûte 0.

    Parameters
    ----------
    y_true, y_pred
        Labels entiers 0–6, même longueur.
    penalty_matrix
        Matrice 7×7. Défaut : :data:`COST_MATRIX`.

    Returns
    -------
    float
        Score dans ``[-∞, 1]`` (1 = parfait).

    Raises
    ------
    ValueError
        Si les longueurs diffèrent.
    """
    matrix = COST_MATRIX if penalty_matrix is None else np.asarray(penalty_matrix)
    true = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)
    if true.shape != pred.shape:
        raise ValueError("y_true et y_pred doivent avoir la même longueur")
    n = true.size
    if n == 0:
        return 1.0
    total = 0.0
    for t, p in zip(true, pred, strict=True):
        if t != p:
            total += float(matrix[t, p])
    max_penalty = float(matrix.max())
    return 1.0 - total / (n * max_penalty)


def decide_on_frame(
    frame: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Applique :func:`decide_class` à un tableau ``p_drift, p0…p5``.

    Parameters
    ----------
    frame
        Colonnes ``p_drift`` et ``p0`` … ``p5``.
    threshold
        Seuil d'anomalie.

    Returns
    -------
    numpy.ndarray
        Prédictions (n,).
    """
    prob_cols = [f"p{i}" for i in range(N_KNOWN_CLASSES)]
    preds = [
        decide_class(row["p_drift"], row[prob_cols], threshold=threshold)
        for _, row in frame.iterrows()
    ]
    return np.asarray(preds, dtype=np.int64)


def find_optimal_threshold(
    frame: pd.DataFrame,
    y_true: Iterable[int],
    thresholds: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Cherche le seuil d'anomalie qui maximise la PWA.

    Parameters
    ----------
    frame
        Colonnes ``p_drift``, ``p0`` … ``p5``.
    y_true
        Labels 0–6 alignés sur les lignes de ``frame``.
    thresholds
        Grille à tester. Défaut : 0, 0.02, …, 1.

    Returns
    -------
    tuple[float, float]
        ``(seuil, pwa)`` du meilleur couple. En cas d'égalité, le plus
        petit seuil est conservé.
    """
    if thresholds is None:
        grid = np.linspace(0.0, 1.0, 51)
    else:
        grid = np.asarray(list(thresholds), dtype=np.float64)
    y = np.asarray(list(y_true), dtype=np.int64)
    best_threshold = float(grid[0])
    best_pwa = -np.inf
    for threshold in grid:
        pred = decide_on_frame(frame, threshold=float(threshold))
        pwa = penalty_weighted_accuracy(y, pred)
        if pwa > best_pwa:
            best_pwa = pwa
            best_threshold = float(threshold)
    return best_threshold, float(best_pwa)
