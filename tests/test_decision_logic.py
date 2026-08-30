"""Tests de la logique de décision (matrice de coût du challenge)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valeo_qc.decision_logic import (
    COST_MATRIX,
    DRIFT_CLASS,
    decide_class,
    decide_on_frame,
    encode_label,
    find_optimal_threshold,
    penalty_weighted_accuracy,
)


def test_encode_label_texte_et_entier() -> None:
    """Les noms officiels et les entiers 0–6 se recodent pareil."""
    assert encode_label("GOOD") == 0
    assert encode_label("Missing") == 4
    assert encode_label("drift") == 6
    assert encode_label(3) == 3


def test_encode_label_inconnu_leve() -> None:
    """Un libellé hors mapping est rejeté."""
    with pytest.raises(ValueError, match="inconnu"):
        encode_label("inconnu")


def test_decide_class_au_dessus_du_seuil_renvoie_drift() -> None:
    """Un score d'anomalie strictement au-dessus du seuil force la classe 6."""
    probs = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
    assert decide_class(0.51, probs, threshold=0.5) == DRIFT_CLASS


def test_decide_class_egal_au_seuil_garde_le_classifieur() -> None:
    """À égalité de seuil, on ne bascule pas en drift (comme le notebook : `>`)."""
    probs = [0.1, 0.1, 0.1, 0.1, 0.5, 0.1]
    assert decide_class(0.5, probs, threshold=0.5) == 4


def test_decide_class_en_dessous_du_seuil_prend_argmax() -> None:
    """Sous le seuil, la classe est l'argmax des 6 probabilités."""
    probs = [0.05, 0.05, 0.7, 0.05, 0.1, 0.05]
    assert decide_class(0.2, probs, threshold=0.5) == 2


def test_penalty_zero_si_tout_est_juste() -> None:
    """PWA = 1 si aucune erreur."""
    y = [0, 1, 4, 6]
    assert penalty_weighted_accuracy(y, y) == 1.0


def test_penalty_good_vers_drift_est_maximale() -> None:
    """GOOD prédit drift coûte 10 000, soit PWA = 0 sur un seul exemple."""
    pwa = penalty_weighted_accuracy([0], [6])
    assert COST_MATRIX[0, 6] == 10000
    assert pwa == pytest.approx(0.0)


def test_penalty_defaut_vers_good_est_maximale() -> None:
    """Un défaut classé GOOD coûte aussi 10 000."""
    assert penalty_weighted_accuracy([4], [0]) == pytest.approx(0.0)


def test_find_optimal_threshold_choisit_le_seuil_qui_isole_le_drift() -> None:
    """Sur un mini-jeu, le seuil retenu sépare drift et classes connues."""
    frame = pd.DataFrame(
        {
            "p_drift": [0.9, 0.1],
            "p0": [0.2, 0.8],
            "p1": [0.16, 0.04],
            "p2": [0.16, 0.04],
            "p3": [0.16, 0.04],
            "p4": [0.16, 0.04],
            "p5": [0.16, 0.04],
        }
    )
    y_true = [6, 0]
    threshold, pwa = find_optimal_threshold(
        frame, y_true, thresholds=[0.0, 0.5, 1.0]
    )
    assert pwa == pytest.approx(1.0)
    assert 0.0 < threshold <= 0.5
    pred = decide_on_frame(frame, threshold=threshold)
    np.testing.assert_array_equal(pred, [6, 0])
