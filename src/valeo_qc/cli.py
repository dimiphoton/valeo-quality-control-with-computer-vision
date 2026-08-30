"""Point d'entrée en ligne de commande."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from valeo_qc.preprocessing import (
    PROCESSED_DIR,
    TRAIN_IMAGES_DIR,
    load_train_labels,
    rotate_and_crop,
    stratified_split,
)


def _cmd_split(args: argparse.Namespace) -> None:
    """Écrit un CSV train/val stratifié (chemins seulement, pas d'images)."""
    frame = load_train_labels()
    train, val = stratified_split(frame, val_fraction=args.val_fraction, seed=args.seed)
    train = train.assign(split="train")
    val = val.assign(split="val")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([train, val], ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"écrit {out} ({len(train)} train, {len(val)} val)")


def _cmd_crop(args: argparse.Namespace) -> None:
    """Recadre une image vers processed/, sans modifier raw/."""
    labels = load_train_labels().set_index("filename")
    filename = Path(args.image).name
    lib = args.lib or str(labels.loc[filename, "lib"])
    source = Path(args.image) if Path(args.image).is_file() else TRAIN_IMAGES_DIR / filename
    dest = Path(args.output) if args.output else PROCESSED_DIR / "train" / filename
    written = rotate_and_crop(source, dest, lib=lib)
    print(written)


def main() -> None:
    """Point d'entrée principal du CLI."""
    parser = argparse.ArgumentParser(
        description="Contrôle qualité Valeo — split et prétraitement"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    split_p = sub.add_parser("split", help="split train/val stratifié → CSV")
    split_p.add_argument(
        "--output",
        default=str(PROCESSED_DIR / "split.csv"),
        help="CSV de sortie",
    )
    split_p.add_argument("--val-fraction", type=float, default=0.2)
    split_p.add_argument("--seed", type=int, default=42)
    split_p.set_defaults(func=_cmd_split)

    crop_p = sub.add_parser("crop", help="rotate+crop d'une image vers processed/")
    crop_p.add_argument("image", help="chemin ou nom de fichier train")
    crop_p.add_argument("--lib", default=None, help="Die01–Die04 (sinon lu dans Y_train)")
    crop_p.add_argument("--output", default=None, help="PNG de sortie")
    crop_p.set_defaults(func=_cmd_crop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
