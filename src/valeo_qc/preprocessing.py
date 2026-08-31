"""Chargement des labels, split, poids de classes, rotate-and-crop.

Les chemins pointent vers ``data/raw/`` (jamais modifié) et
``data/processed/`` (images recadrées). Le crop officiel dépend de
``lib`` (Die01–Die04) — voir le notebook du challenge.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from valeo_qc.decision_logic import ID_TO_LABEL, encode_label

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_IMAGES_DIR = RAW_DIR / "input_train"
TEST_IMAGES_DIR = RAW_DIR / "input_test_1a4aqAg" / "input_test"
TRAIN_LABELS_CSV = RAW_DIR / "Y_train_eVW9jym.csv"
SUBMISSION_EXAMPLE_CSV = RAW_DIR / "Y_random_nKwalR1.csv"
TEST_META_CSV = RAW_DIR / "Supp_files" / "win_and_lib.csv"
SPLIT_CSV_NAME = "split.csv"
WEIGHTS_JSON_NAME = "class_weights.json"

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
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        cropped = crop_pil(image, lib, rot_crop=table)
        cropped.save(dest, format="PNG")
    return dest


def crop_pil(
    image: Image.Image,
    lib: str,
    rot_crop: dict[str, tuple[float, tuple[int, int, int, int]]] | None = None,
) -> Image.Image:
    """Même rotate+crop que :func:`rotate_and_crop`, en mémoire (API).

    Parameters
    ----------
    image
        Image source RGB.
    lib
        ``Die01`` … ``Die04``.
    rot_crop
        Table angle / crop. Défaut : :data:`ROT_CROP`.

    Returns
    -------
    PIL.Image.Image
        Image recadrée.

    Raises
    ------
    KeyError
        Si ``lib`` est inconnu.
    """
    table = ROT_CROP if rot_crop is None else rot_crop
    if lib not in table:
        raise KeyError(f"lib inconnu : {lib!r} (attendu {sorted(table)})")
    angle, crop_box = table[lib]
    return image.rotate(angle, expand=True).crop(crop_box)


def load_test_meta(csv_path: Path | None = None) -> pd.DataFrame:
    """Charge les métadonnées test (filename, window, lib), sans labels.

    Parameters
    ----------
    csv_path
        CSV officiel ``win_and_lib.csv``. Défaut : :data:`TEST_META_CSV`.

    Returns
    -------
    pandas.DataFrame
        Colonnes ``filename``, ``window``, ``lib``.
    """
    path = TEST_META_CSV if csv_path is None else Path(csv_path)
    return pd.read_csv(path)


def _crop_one(
    filename: str,
    lib: str,
    source_dir: Path,
    dest_dir: Path,
    overwrite: bool,
) -> str:
    """Recadre un fichier. Retourne ``written``, ``skipped`` ou ``missing``."""
    dest = dest_dir / filename
    if dest.is_file() and not overwrite:
        return "skipped"
    source = source_dir / filename
    if not source.is_file():
        return "missing"
    rotate_and_crop(source, dest, lib=lib)
    return "written"


def crop_frame(
    frame: pd.DataFrame,
    source_dir: Path,
    dest_dir: Path,
    *,
    overwrite: bool = False,
    workers: int = 1,
    filename_col: str = "filename",
    lib_col: str = "lib",
) -> dict[str, Any]:
    """Recadre toutes les images d'un tableau vers ``dest_dir``.

    Les PNG déjà présents sont sautés sauf si ``overwrite`` est vrai.
    Les sources absentes sont listées, jamais créées à partir de rien.

    Parameters
    ----------
    frame
        Table avec au moins ``filename`` et ``lib``.
    source_dir
        Dossier des PNG bruts (``data/raw/...``).
    dest_dir
        Dossier de sortie (``data/processed/train`` ou ``test``).
    overwrite
        Réécrire les PNG déjà recadrés.
    workers
        Nombre de threads. ``1`` = séquentiel (défaut des tests).
    filename_col, lib_col
        Noms de colonnes.

    Returns
    -------
    dict
        Compteurs ``written``, ``skipped``, ``missing`` (liste de noms).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir)
    rows = list(zip(frame[filename_col].astype(str), frame[lib_col].astype(str), strict=True))
    counts = {"written": 0, "skipped": 0, "missing": []}

    def _run(filename: str, lib: str) -> tuple[str, str]:
        return filename, _crop_one(filename, lib, source_dir, dest_dir, overwrite)

    if workers <= 1:
        results = [_run(name, lib) for name, lib in rows]
    else:
        # Threads : chaque PNG va dans un fichier distinct, pas de partage d'état.
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, name, lib) for name, lib in rows]
            for future in as_completed(futures):
                results.append(future.result())

    for filename, status in results:
        if status == "missing":
            counts["missing"].append(filename)
        else:
            counts[status] += 1
        done = counts["written"] + counts["skipped"] + len(counts["missing"])
        if done % 500 == 0:
            LOGGER.info("crop %s/%s -> %s", done, len(rows), dest_dir.name)
    return counts


def save_class_weights(
    weights: dict[int, float],
    path: Path,
    *,
    n_train: int,
    seed: int,
    val_fraction: float,
) -> Path:
    """Écrit les poids de classes (calculés sur le split train) en JSON.

    Parameters
    ----------
    weights
        Poids par ``label_id`` (0–5).
    path
        Fichier de sortie.
    n_train, seed, val_fraction
        Métadonnées du split, pour retracer le calcul.

    Returns
    -------
    pathlib.Path
        Chemin écrit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "computed_on": "train_split",
        "n_train": int(n_train),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "weights": {str(k): v for k, v in sorted(weights.items())},
        "labels": {str(k): ID_TO_LABEL[k] for k in sorted(weights)},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def prepare_dataset(
    *,
    labels_csv: Path | None = None,
    train_images_dir: Path | None = None,
    test_images_dir: Path | None = None,
    test_meta_csv: Path | None = None,
    processed_dir: Path | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
    crop_test: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Split stratifié, poids de classes, rotate-and-crop vers ``processed/``.

    Les poids sont calculés **sur le split train seulement** (pas de fuite
    vers la validation). ``raw/`` n'est jamais modifié.

    Parameters
    ----------
    labels_csv, train_images_dir, test_images_dir, test_meta_csv, processed_dir
        Chemins. Défaut : constantes du module (données du challenge).
    val_fraction, seed
        Paramètres du split.
    overwrite
        Réécrire les PNG déjà présents dans ``processed/``.
    crop_test
        Recadrer aussi les images test (``lib`` lu dans ``win_and_lib.csv``).
    workers
        Threads pour le crop.

    Returns
    -------
    dict
        Chemins écrits, effectifs du split, compteurs de crop.
    """
    processed = PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    labels = load_train_labels(labels_csv)
    train, val = stratified_split(labels, val_fraction=val_fraction, seed=seed)
    train = train.assign(split="train")
    val = val.assign(split="val")
    split = pd.concat([train, val], ignore_index=True)
    split_path = processed / SPLIT_CSV_NAME
    split.to_csv(split_path, index=False)

    weights = class_weights(train["label_id"])
    weights_path = save_class_weights(
        weights,
        processed / WEIGHTS_JSON_NAME,
        n_train=len(train),
        seed=seed,
        val_fraction=val_fraction,
    )

    src_train = TRAIN_IMAGES_DIR if train_images_dir is None else Path(train_images_dir)
    train_crop = crop_frame(
        labels,
        src_train,
        processed / "train",
        overwrite=overwrite,
        workers=workers,
    )

    test_crop: dict[str, Any] | None = None
    if crop_test:
        meta_path = TEST_META_CSV if test_meta_csv is None else Path(test_meta_csv)
        src_test = TEST_IMAGES_DIR if test_images_dir is None else Path(test_images_dir)
        test_meta = load_test_meta(meta_path)
        test_crop = crop_frame(
            test_meta,
            src_test,
            processed / "test",
            overwrite=overwrite,
            workers=workers,
        )

    return {
        "split_path": split_path,
        "weights_path": weights_path,
        "n_train": len(train),
        "n_val": len(val),
        "train_crop": train_crop,
        "test_crop": test_crop,
    }
