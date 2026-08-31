"""Calibration du seuil d'anomalie sur la matrice de coût.

Le split val n'a **pas** de ``drift``. Maximiser la PWA dessus revient à
ne jamais prédire la classe 6 (tout flag est un faux positif). On
chiffre ce choix, le 0,5 du notebook, et un seuil qui protège GOOD
(erreur à 10 000). C'est ce dernier qui est exporté : il reste du
rappel d'anomalie sans payer la case GOOD→drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from valeo_qc.decision_logic import (
    DEFAULT_THRESHOLD,
    ID_TO_LABEL,
    N_KNOWN_CLASSES,
    decide_from_arrays,
    decision_stats,
    find_optimal_threshold,
    threshold_sweep,
)
from valeo_qc.preprocessing import PROCESSED_DIR, PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "valeo-qc-decision"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"
SWEEP_CSV = MODELS_DIR / "threshold-sweep.csv"
PLOT_PATH = PROJECT_ROOT / "pictures" / "experiments" / "pwa-vs-threshold.png"

BENCHMARK_THRESHOLD = DEFAULT_THRESHOLD
OPERATING_REASON = (
    "Le val n'a pas de drift : le max PWA (seuil 1) éteint PaDiM. "
    "On exporte le plus petit seuil qui ne classe aucun GOOD en drift "
    "(case à 10 000), et on compare au 0,5 du notebook."
)


def build_decision_frame(
    p_drift: np.ndarray,
    class_probs: np.ndarray,
    filenames: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble ``p_drift, p0…p5`` (et ``filename`` si fourni)."""
    data: dict[str, Any] = {"p_drift": np.asarray(p_drift, dtype=np.float64)}
    probs = np.asarray(class_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != N_KNOWN_CLASSES:
        raise ValueError(f"class_probs doit être (n, 6), reçu {probs.shape}")
    for i in range(N_KNOWN_CLASSES):
        data[f"p{i}"] = probs[:, i]
    if filenames is not None:
        data["filename"] = list(filenames)
    return pd.DataFrame(data)


def _stats_at(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Stats de décision à un seuil donné."""
    pred = decide_from_arrays(
        frame["p_drift"].to_numpy(),
        frame[[f"p{i}" for i in range(N_KNOWN_CLASSES)]].to_numpy(),
        threshold=threshold,
    )
    stats = decision_stats(y_true, pred)
    stats["threshold"] = float(threshold)
    return stats


def protect_good_threshold(p_drift: np.ndarray, y_true: np.ndarray) -> float:
    """Plus petit seuil qui ne classe aucun GOOD du lot en drift.

    ``decide`` utilise ``>`` : on prend le max des scores GOOD (un GOOD
    à exactement ce seuil n'est pas flaggé). S'il n'y a pas de GOOD : 0.
    """
    p_drift = np.asarray(p_drift, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    good = p_drift[y_true == 0]
    if good.size == 0:
        return 0.0
    return float(good.max())


def operating_points(
    frame: pd.DataFrame,
    y_true: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Quatre points : classifieur seul, 0,5, max PWA val, protège-GOOD."""
    p_drift = frame["p_drift"].to_numpy(dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    never = float(max(p_drift.max() if p_drift.size else 1.0, 1.0))
    t_pwa, _ = find_optimal_threshold(frame, y)
    t_good = protect_good_threshold(p_drift, y)
    return {
        "classifier_only": _stats_at(frame, y, never),
        "benchmark_0.5": _stats_at(frame, y, BENCHMARK_THRESHOLD),
        "val_pwa_max": _stats_at(frame, y, t_pwa),
        "protect_good": _stats_at(frame, y, t_good),
    }


def _json_safe(value: Any) -> Any:
    """Convertit numpy en types JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


def plot_pwa_curves(
    sweeps: dict[str, pd.DataFrame],
    out_path: Path | None = None,
    extra_lines: dict[str, float] | None = None,
) -> Path:
    """Courbes PWA vs seuil (une courbe par pipeline)."""
    import matplotlib.pyplot as plt

    out_path = PLOT_PATH if out_path is None else Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#f7f3ea")
    for name, sweep in sweeps.items():
        ax.plot(sweep["threshold"], sweep["pwa"], label=name, linewidth=2)
    lines = {"seuil 0,5": BENCHMARK_THRESHOLD}
    if extra_lines:
        lines.update(extra_lines)
    styles = ["--", ":"]
    for idx, (label, xpos) in enumerate(lines.items()):
        ax.axvline(
            xpos,
            color="#333",
            linestyle=styles[idx % len(styles)],
            linewidth=1,
            label=label,
        )
    ax.set_xlabel("Seuil PaDiM (score min-max)")
    ax.set_ylabel("PWA (val, sans drift)")
    ax.set_title("Sans drift au val, flagger coûte ; le 0,5 du notebook n'est pas le max PWA")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, transparent=True)
    plt.close(fig)
    return out_path


def setup_mlflow(tracking_uri: str | None = None) -> None:
    """Expérience MLflow dédiée à la décision."""
    import mlflow

    if tracking_uri is None:
        db = MLFLOW_DB.resolve().as_posix()
        tracking_uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def collect_official_scores(
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    batch_size: int = 16,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Probas classifieur officiel + scores PaDiM officiels sur le val."""
    from valeo_qc.classifier import (
        get_device,
        load_official_classifier,
        load_split,
        make_loader,
        predict_proba,
    )
    from valeo_qc.padim import evaluate_official as eval_padim

    device = get_device()
    images = PROCESSED_DIR / "train" if images_dir is None else Path(images_dir)
    val = load_split(split_csv, which="val")
    loader = make_loader(val, images, batch_size=batch_size, shuffle=False)
    LOGGER.info("probas classifieur officiel")
    model = load_official_classifier(device=device)
    probs, labels = predict_proba(model, loader, device)
    LOGGER.info("scores PaDiM officiel")
    padim = eval_padim(
        split_csv=split_csv,
        images_dir=images,
        batch_size=batch_size,
        log_mlflow=False,
    )
    if not np.array_equal(labels, padim["labels"]):
        raise RuntimeError("ordre val classifieur / PaDiM désaligné")
    frame = build_decision_frame(padim["scores_norm"], probs, list(val["filename"]))
    return frame, labels


def collect_ours_scores(
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    batch_size: int = 16,
    classifier_ckpt: Path | None = None,
    padim_ckpt: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Même chose avec nos checkpoints locaux."""
    from valeo_qc.classifier import (
        MODELS_DIR as CLS_MODELS,
        get_device,
        load_split,
        load_trained_classifier,
        make_loader,
        predict_proba,
    )
    from valeo_qc.padim import MODELS_DIR as PADIM_MODELS
    from valeo_qc.padim import evaluate_official as eval_padim

    clf_path = CLS_MODELS / "classifier-best.pt" if classifier_ckpt is None else Path(classifier_ckpt)
    pad_path = PADIM_MODELS / "padim-best.pkl" if padim_ckpt is None else Path(padim_ckpt)
    if not clf_path.is_file() or not pad_path.is_file():
        raise FileNotFoundError(f"checkpoints locaux absents : {clf_path}, {pad_path}")
    device = get_device()
    images = PROCESSED_DIR / "train" if images_dir is None else Path(images_dir)
    val = load_split(split_csv, which="val")
    loader = make_loader(val, images, batch_size=batch_size, shuffle=False)
    LOGGER.info("probas classifieur local")
    model = load_trained_classifier(clf_path, device=device)
    probs, labels = predict_proba(model, loader, device)
    LOGGER.info("scores PaDiM local")
    padim = eval_padim(
        split_csv=split_csv,
        images_dir=images,
        checkpoint=pad_path,
        batch_size=batch_size,
        log_mlflow=False,
    )
    if not np.array_equal(labels, padim["labels"]):
        raise RuntimeError("ordre val classifieur / PaDiM désaligné")
    frame = build_decision_frame(padim["scores_norm"], probs, list(val["filename"]))
    return frame, labels


def calibrate(
    *,
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    batch_size: int = 16,
    include_ours: bool = True,
    tracking_uri: str | None = None,
    log_mlflow: bool = True,
    write_artifacts: bool = True,
    official_frame: pd.DataFrame | None = None,
    official_labels: np.ndarray | None = None,
    ours_frame: pd.DataFrame | None = None,
    ours_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calibre et compare officiel vs nôtre (si checkpoints présents).

    Parameters
    ----------
    split_csv, images_dir, batch_size
        Val recadré.
    include_ours
        Tente nos checkpoints ``models/*-best.*``.
    tracking_uri, log_mlflow, write_artifacts
        MLflow + JSON/PNG.
    official_frame, official_labels, ours_frame, ours_labels
        Injectés dans les tests (pas d'inférence).

    Returns
    -------
    dict
        Points de fonctionnement, seuil exporté, chemins d'artefacts.
    """
    if official_frame is None or official_labels is None:
        official_frame, official_labels = collect_official_scores(
            split_csv=split_csv, images_dir=images_dir, batch_size=batch_size
        )
    official_labels = np.asarray(official_labels, dtype=np.int64)
    pipelines: dict[str, Any] = {
        "official": {
            "points": operating_points(official_frame, official_labels),
            "n": int(len(official_labels)),
            "n_drift_labels": int((official_labels == 6).sum()),
        }
    }
    sweeps: dict[str, pd.DataFrame] = {
        "officiel": threshold_sweep(official_frame, official_labels),
    }
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    ours_ok = False
    if ours_frame is not None and ours_labels is not None:
        ours_ok = True
    elif include_ours:
        try:
            ours_frame, ours_labels = collect_ours_scores(
                split_csv=split_csv, images_dir=images_dir, batch_size=batch_size
            )
            ours_ok = True
        except FileNotFoundError as exc:
            LOGGER.warning("pipeline local ignoré (%s)", exc)

    if ours_ok and ours_frame is not None and ours_labels is not None:
        ours_labels = np.asarray(ours_labels, dtype=np.int64)
        pipelines["ours"] = {
            "points": operating_points(ours_frame, ours_labels),
            "n": int(len(ours_labels)),
            "n_drift_labels": int((ours_labels == 6).sum()),
        }
        sweeps["nôtre"] = threshold_sweep(ours_frame, ours_labels)

    exported = pipelines["official"]["points"]["protect_good"]
    report: dict[str, Any] = {
        "val_n": pipelines["official"]["n"],
        "val_has_drift": False,
        "operating_point": {
            "name": "protect_good",
            "threshold": float(exported["threshold"]),
            "pwa": float(exported["pwa"]),
            "n_false_drift": int(exported["n_false_drift"]),
            "n_false_drift_good": int(exported["n_false_drift_good"]),
            "reason": OPERATING_REASON,
        },
        "pipelines": pipelines,
        "class_names": ID_TO_LABEL,
    }

    plot_path: Path | None = None
    json_path: Path | None = None
    csv_path: Path | None = None
    if write_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = THRESHOLD_JSON
        json_path.write_text(
            json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        csv_path = SWEEP_CSV
        combined = []
        for name, sweep in sweeps.items():
            piece = sweep.copy()
            piece.insert(0, "pipeline", name)
            combined.append(piece)
        pd.concat(combined, ignore_index=True).to_csv(csv_path, index=False)
        plot_path = plot_pwa_curves(
            sweeps,
            extra_lines={"protège GOOD": float(exported["threshold"])},
        )

    if log_mlflow:
        import mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name="threshold-calibration"):
            mlflow.log_params(
                {
                    "operating_point": "protect_good",
                    "threshold": float(exported["threshold"]),
                    "val_has_drift": False,
                    "n_val": pipelines["official"]["n"],
                }
            )
            mlflow.log_metrics(
                {
                    "pwa_official_0.5": float(
                        pipelines["official"]["points"]["benchmark_0.5"]["pwa"]
                    ),
                    "pwa_official_classifier_only": float(
                        pipelines["official"]["points"]["classifier_only"]["pwa"]
                    ),
                    "false_drift_official_0.5": float(
                        pipelines["official"]["points"]["benchmark_0.5"]["n_false_drift"]
                    ),
                    "false_drift_good_official_0.5": float(
                        pipelines["official"]["points"]["benchmark_0.5"][
                            "n_false_drift_good"
                        ]
                    ),
                }
            )
            if "ours" in pipelines:
                mlflow.log_metrics(
                    {
                        "pwa_ours_0.5": float(
                            pipelines["ours"]["points"]["benchmark_0.5"]["pwa"]
                        ),
                        "pwa_ours_classifier_only": float(
                            pipelines["ours"]["points"]["classifier_only"]["pwa"]
                        ),
                    }
                )
            if json_path is not None:
                mlflow.log_artifact(str(json_path))
            if plot_path is not None:
                mlflow.log_artifact(str(plot_path))

    report["artifacts"] = {
        "json": str(json_path) if json_path else None,
        "sweep_csv": str(csv_path) if csv_path else None,
        "plot": str(plot_path) if plot_path else None,
    }
    return report
