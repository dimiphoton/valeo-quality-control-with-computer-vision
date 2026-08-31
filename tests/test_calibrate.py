"""Tests de calibration — tableaux synthétiques, pas les checkpoints GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from valeo_qc.calibrate import (
    build_decision_frame,
    calibrate,
    operating_points,
    protect_good_threshold,
)


def _known_probs(n: int, class_id: int = 0) -> np.ndarray:
    """One-hot approximatif sur 6 classes."""
    probs = np.full((n, 6), 0.02)
    probs[:, class_id] = 0.9
    return probs


def test_build_decision_frame_colonnes() -> None:
    """p_drift + p0…p5, une ligne par image."""
    frame = build_decision_frame(
        np.array([0.1, 0.9]),
        _known_probs(2, 4),
        filenames=["a.png", "b.png"],
    )
    assert list(frame.columns[:8]) == [
        "p_drift",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "filename",
    ]
    assert len(frame) == 2


def test_protect_good_threshold_ne_flag_pas_good() -> None:
    """Le seuil vaut le max des scores GOOD."""
    p_drift = np.array([0.2, 0.8, 0.4])
    y = np.array([0, 4, 0])
    assert protect_good_threshold(p_drift, y) == pytest.approx(0.4)


def test_operating_points_sans_drift_maximise_pwa_en_ne_flaggant_pas() -> None:
    """Sans drift, classifieur seul a une PWA ≥ seuil 0,5 (faux drift)."""
    frame = build_decision_frame(
        np.array([0.9, 0.1]),
        np.vstack([_known_probs(1, 0), _known_probs(1, 4)]),
    )
    y = np.array([0, 4])
    points = operating_points(frame, y)
    assert points["benchmark_0.5"]["n_false_drift_good"] == 1
    assert points["classifier_only"]["n_false_drift"] == 0
    assert points["val_pwa_max"]["pwa"] >= points["benchmark_0.5"]["pwa"]
    assert points["classifier_only"]["pwa"] == pytest.approx(1.0)


def test_calibrate_injecte_sans_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Le rapport exporte le seuil 0,5 même si le max PWA est plus haut."""
    import valeo_qc.calibrate as mod

    monkeypatch.setattr(mod, "THRESHOLD_JSON", tmp_path / "threshold.json")
    monkeypatch.setattr(mod, "SWEEP_CSV", tmp_path / "sweep.csv")
    monkeypatch.setattr(mod, "PLOT_PATH", tmp_path / "pwa.png")
    monkeypatch.setattr(mod, "MODELS_DIR", tmp_path)

    rng = np.random.default_rng(0)
    p_drift = rng.uniform(0, 1, size=20)
    p_drift[0] = 0.95
    labels = np.array([0] * 10 + [4] * 10)
    probs = np.vstack([_known_probs(10, 0), _known_probs(10, 4)])
    frame = build_decision_frame(p_drift, probs)

    report = calibrate(
        official_frame=frame,
        official_labels=labels,
        include_ours=False,
        log_mlflow=False,
        write_artifacts=True,
    )
    assert report["operating_point"]["name"] == "protect_good"
    assert report["operating_point"]["n_false_drift_good"] == 0
    assert report["val_has_drift"] is False
    assert (tmp_path / "threshold.json").is_file()
    assert (tmp_path / "pwa.png").is_file()
    assert report["pipelines"]["official"]["n"] == 20
