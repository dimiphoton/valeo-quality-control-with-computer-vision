"""Chargement des labels, split, poids de classes, rotate-and-crop.

Les chemins pointent vers ``data/raw/`` (jamais modifié) et
``data/processed/`` (images recadrées). Le crop officiel dépend de
``lib`` (Die01–Die04) — voir le notebook du challenge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from valeo_qc.decision_logic import encode_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_IMAGES_DIR = RAW_DIR / "input_train"
TEST_IMAGES_DIR = RAW_DIR / "input_test_1a4aqAg" / "input_test"
TRAIN_LABELS_CSV = RAW_DIR / "Y_train_eVW9jym.csv"
SUBMISSION_EXAMPLE_CSV = RAW_DIR / "Y_random_nKwalR1.csv"

# Angle (degrés, antihoraire) et crop PIL (left, upper, right, lower).
ROT_CROP: dict[str, tuple[float, tuple[int, int, int, int]]] = {
    "Die01": (55, (340, 120, 500, 680)),
    "Die02": (-44, (480, 210, 640, 930)),
    "Die03": (134, (460, 200, 620, 920)),
    "Die04": (35, (310, 130, 470, 690)),
}


def load_train_labels(csv_path: Path | None = None) -> pd.DataFrame:
    """Charge ``Y_train`` et ajoute ``label_id``.

    Parameters
    ----------
    csv_path
        CSV du challenge. Défaut : :data:`TRAIN_LABELS_CSV`.

    Returns
    -------
    pandas.DataFrame
        Colonnes ``filename``, ``window``, ``lib``, ``Label``, ``label_id``.
    """
    path = TRAIN_LABELS_CSV if csv_path is None else Path(csv_path)
    frame = pd.read_csv(path, index_col=0)
    frame["label_id"] = frame["Label"].map(encode_label)
    return frame


def class_weights(label_ids: pd.Series | np.ndarray) -> dict[int, float]:
    """Poids inversement proportionnels à la fréquence (somme = n_classes).

    Parameters
    ----------
    label_ids
        Identifiants 0–5 présents dans le train (pas de drift).

    Returns
    -------
    dict[int, float]
        Poids par classe, utiles pour une loss pondérée.
    """
    ids = np.asarray(label_ids)
    unique, counts = np.unique(ids, return_counts=True)
    n_classes = unique.size
    weights = n_classes * (1.0 / counts) / np.sum(1.0 / counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights, strict=True)}


def stratified_split(
    frame: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/val stratifié sur le label (reproductible).

    Parameters
    ----------
    frame
        Table des images labellisées.
    val_fraction
        Fraction validation, dans ``(0, 1)``.
    seed
        Graine du tirage.
    label_col
        Colonne de stratification.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(train, val)``, index réinitialisé.

    Raises
    ------
    ValueError
        Si ``val_fraction`` est hors ``(0, 1)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction doit être dans (0, 1)")
    parts_train: list[pd.DataFrame] = []
    parts_val: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for _, group in frame.groupby(label_col, sort=True):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        parts_val.append(shuffled.iloc[:n_val])
        parts_train.append(shuffled.iloc[n_val:])
    train = pd.concat(parts_train).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(parts_val).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val


def rotate_and_crop(
    source: Path,
    dest: Path,
    lib: str,
    rot_crop: dict[str, tuple[float, tuple[int, int, int, int]]] | None = None,
) -> Path:
    """Tourne et recadre une image selon le die, sans toucher à la source.

    Parameters
    ----------
    source
        PNG brut (``data/raw``).
    dest
        Destination (``data/processed``). Les parents sont créés.
    lib
        ``Die01`` … ``Die04``.
    rot_crop
        Table angle / crop. Défaut : :data:`ROT_CROP`.

    Returns
    -------
    pathlib.Path
        Chemin écrit.

    Raises
    ------
    KeyError
        Si ``lib`` est inconnu.
    FileNotFoundError
        Si ``source`` n'existe pas.
    """
    table = ROT_CROP if rot_crop is None else rot_crop
    if lib not in table:
        raise KeyError(f"lib inconnu : {lib!r} (attendu {sorted(table)})")
    angle, crop_box = table[lib]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rotated = image.rotate(angle, expand=True)
        cropped = rotated.crop(crop_box)
        cropped.save(dest, format="PNG")
    return dest
