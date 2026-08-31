"""Tests de la logique de décision (matrice de coût du challenge)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valeo_qc.decision_logic import (
    COST_MATRIX,
    DRIFT_CLASS,
    decide_class,
    decide_from_arrays,
    decide_on_frame,
    decision_stats,
    encode_label,
    find_optimal_threshold,
    penalty_weighted_accuracy,
    threshold_sweep,
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


def test_decision_stats_separe_faux_drift_et_confusion() -> None:
    """GOOD→drift et défaut→défaut n'ont pas la même pénalité."""
    stats = decision_stats([0, 4, 4], [6, 2, 4])
    assert stats["n_false_drift"] == 1
    assert stats["n_false_drift_good"] == 1
    assert stats["n_class_error"] == 1
    assert stats["penalty_false_drift"] == 10000
    assert stats["penalty_class_error"] == 1
    assert stats["n_missed_drift"] == 0


def test_decide_from_arrays_aligne_decide_class() -> None:
    """La version vectorisée colle à decide_class exemple par exemple."""
    p_drift = np.array([0.9, 0.1])
    probs = np.array(
        [
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
        ]
    )
    pred = decide_from_arrays(p_drift, probs, threshold=0.5)
    assert pred[0] == DRIFT_CLASS
    assert pred[1] == 4


def test_threshold_sweep_pwa_croit_quand_on_arrete_de_flagger() -> None:
    """Sans drift réel, un seuil trop bas fait baisser la PWA."""
    frame = pd.DataFrame(
        {
            "p_drift": [0.8, 0.1],
            "p0": [0.9, 0.9],
            "p1": [0.02, 0.02],
            "p2": [0.02, 0.02],
            "p3": [0.02, 0.02],
            "p4": [0.02, 0.02],
            "p5": [0.02, 0.02],
        }
    )
    y_true = [0, 0]
    sweep = threshold_sweep(frame, y_true, thresholds=[0.0, 0.5, 1.0])
    pwa_low = float(sweep.loc[sweep["threshold"] == 0.0, "pwa"].iloc[0])
    pwa_high = float(sweep.loc[sweep["threshold"] == 1.0, "pwa"].iloc[0])
    assert pwa_high > pwa_low
    assert int(sweep.loc[sweep["threshold"] == 0.0, "n_false_drift_good"].iloc[0]) == 2
    assert int(sweep.loc[sweep["threshold"] == 0.5, "n_false_drift_good"].iloc[0]) == 1

