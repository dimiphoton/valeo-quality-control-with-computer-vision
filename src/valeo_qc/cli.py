"""Point d'entrée en ligne de commande."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from valeo_qc.preprocessing import (
    PROCESSED_DIR,
    TRAIN_IMAGES_DIR,
    load_train_labels,
    prepare_dataset,
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


def _fmt_crop(name: str, stats: dict | None) -> str:
    """Résumé d'un passage crop pour le terminal."""
    if stats is None:
        return f"{name}: ignoré"
    n_missing = len(stats["missing"])
    return (
        f"{name}: {stats['written']} écrits, {stats['skipped']} déjà là, "
        f"{n_missing} manquants"
    )


def _cmd_prepare(args: argparse.Namespace) -> None:
    """Split, poids de classes, rotate-and-crop vers processed/."""
    summary = prepare_dataset(
        val_fraction=args.val_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        crop_test=not args.skip_test,
        workers=args.workers,
    )
    print(
        f"split {summary['split_path']} "
        f"({summary['n_train']} train, {summary['n_val']} val)"
    )
    print(f"poids {summary['weights_path']}")
    print(_fmt_crop("train", summary["train_crop"]))
    print(_fmt_crop("test", summary["test_crop"]))
    missing = list(summary["train_crop"]["missing"])
    if summary["test_crop"] is not None:
        missing.extend(summary["test_crop"]["missing"])
    if missing:
        print(f"fichiers bruts introuvables ({len(missing)}) : {missing[:5]}", file=sys.stderr)
        raise SystemExit(1)


def _cmd_eval_classifier(args: argparse.Namespace) -> None:
    """Évalue le checkpoint officiel et journalise dans MLflow."""
    from valeo_qc.classifier import evaluate_official

    metrics = evaluate_official(batch_size=args.batch_size)
    print(
        f"officiel val n={metrics['n']} "
        f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )
    for class_id, rec in metrics["recall"].items():
        print(f"  recall {class_id}: {rec:.4f}")


def _cmd_train_classifier(args: argparse.Namespace) -> None:
    """Entraîne le classifieur avec suivi MLflow."""
    from valeo_qc.classifier import train_classifier

    best = train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )
    print(
        f"meilleur epoch {best['epoch']} "
        f"acc={best['accuracy']:.4f} macro_f1={best['macro_f1']:.4f} "
        f"-> {best['checkpoint']}"
    )


def _fmt_padim(summary: dict) -> str:
    """Résumé une ligne des scores PaDiM."""
    return (
        f"n={summary['n']} raw_mean={summary['raw_mean']:.3f} "
        f"p95={summary['raw_p95']:.3f} frac>0.5={summary['frac_above_0_5']:.3f}"
    )


def _cmd_eval_padim(args: argparse.Namespace) -> None:
    """Évalue le pickle PaDiM officiel sur le split val."""
    from valeo_qc.padim import evaluate_official

    summary = evaluate_official(batch_size=args.batch_size)
    print(f"officiel val {_fmt_padim(summary)}")
    for name, stats in summary["per_class"].items():
        print(f"  {name}: raw_mean={stats['raw_mean']:.3f} n={stats['n']}")


def _cmd_train_padim(args: argparse.Namespace) -> None:
    """Fit PaDiM sur le split train, scores val, pickle dans models/."""
    from valeo_qc.padim import train_padim

    summary = train_padim(batch_size=args.batch_size, seed=args.seed)
    print(f"notre PaDiM val {_fmt_padim(summary)} -> {summary['checkpoint']}")
    for name, stats in summary["per_class"].items():
        print(f"  {name}: raw_mean={stats['raw_mean']:.3f} n={stats['n']}")


def _fmt_point(name: str, point: dict) -> str:
    """Une ligne PWA / faux drift."""
    return (
        f"{name}: seuil={point['threshold']:.3f} PWA={point['pwa']:.6f} "
        f"faux_drift={point['n_false_drift']} "
        f"(GOOD={point['n_false_drift_good']}) "
        f"err_classe={point['n_class_error']}"
    )


def _cmd_calibrate(args: argparse.Namespace) -> None:
    """Seuil de coût sur le val + comparaison officiel / nôtre."""
    from valeo_qc.calibrate import calibrate

    report = calibrate(batch_size=args.batch_size, include_ours=not args.official_only)
    op = report["operating_point"]
    print(f"exporté {op['name']} seuil={op['threshold']} PWA={op['pwa']:.6f}")
    print(op["reason"])
    for pipe_name, pipe in report["pipelines"].items():
        print(f"— {pipe_name} (n={pipe['n']})")
        for key, point in pipe["points"].items():
            print(f"  {_fmt_point(key, point)}")
    artifacts = report.get("artifacts") or {}
    for key, path in artifacts.items():
        if path:
            print(f"{key}: {path}")


def _cmd_export(args: argparse.Namespace) -> None:
    """Exporte classifieur + backbone PaDiM en ONNX et écrit le manifeste."""
    from valeo_qc.export import export_pipeline

    report = export_pipeline(
        opset=args.opset,
        skip_classifier=args.skip_classifier,
        skip_padim=args.skip_padim,
        verify=not args.no_verify,
    )
    print(f"manifeste {report['manifest']}")
    if report.get("classifier"):
        print(f"  classifieur {report['classifier']['path']} ({report['classifier']['bytes']} octets)")
    if report.get("padim_backbone"):
        print(
            f"  padim {report['padim_backbone']['path']} "
            f"({report['padim_backbone']['bytes']} octets)"
        )
    for name, check in report.get("checks", {}).items():
        print(f"  verif {name}: max_abs={check['max_abs']:.3g}")


def _cmd_predict(args: argparse.Namespace) -> None:
    """Inférence locale (même runtime que Lambda)."""
    import json

    from PIL import Image

    from valeo_qc.serve import Runtime

    runtime = Runtime()
    image = Image.open(args.image)
    result = runtime.predict_pil(image, lib=args.lib)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _cmd_deploy(args: argparse.Namespace) -> None:
    """Alarme de facturation puis ECR/Lambda (dry-run par défaut)."""
    from valeo_qc.aws_deploy import DeployError, deploy

    try:
        text = deploy(
            email=args.email,
            dry_run=not args.apply,
            api_region=args.region,
            image_uri=args.image_uri,
            limit_usd=args.limit_usd,
            skip_docker=args.skip_docker,
        )
    except (DeployError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(text, end="")


def main() -> None:
    """Point d'entrée principal du CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Contrôle qualité Valeo — prétraitement, modèles, ONNX, API"
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

    prep_p = sub.add_parser(
        "prepare",
        help="split + poids + rotate-and-crop de tout le jeu vers processed/",
    )
    prep_p.add_argument("--val-fraction", type=float, default=0.2)
    prep_p.add_argument("--seed", type=int, default=42)
    prep_p.add_argument(
        "--overwrite",
        action="store_true",
        help="réécrire les PNG déjà recadrés",
    )
    prep_p.add_argument(
        "--skip-test",
        action="store_true",
        help="ne pas recadrer les images test",
    )
    prep_p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="threads pour le crop (défaut : 4)",
    )
    prep_p.set_defaults(func=_cmd_prepare)

    eval_p = sub.add_parser(
        "eval-classifier",
        help="évalue Classifier.pt officiel sur le split val (MLflow)",
    )
    eval_p.add_argument("--batch-size", type=int, default=16)
    eval_p.set_defaults(func=_cmd_eval_classifier)

    train_p = sub.add_parser(
        "train-classifier",
        help="entraîne resnest50d + poids de classes, journal MLflow",
    )
    train_p.add_argument("--epochs", type=int, default=15)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ne pas partir des poids ImageNet",
    )
    train_p.set_defaults(func=_cmd_train_classifier)

    eval_padim = sub.add_parser(
        "eval-padim",
        help="évalue PADIM.pkl officiel sur le split val (MLflow)",
    )
    eval_padim.add_argument("--batch-size", type=int, default=16)
    eval_padim.set_defaults(func=_cmd_eval_padim)

    train_padim_p = sub.add_parser(
        "train-padim",
        help="fit PaDiM (WRN-50-2) sur le split train, journal MLflow",
    )
    train_padim_p.add_argument("--batch-size", type=int, default=16)
    train_padim_p.add_argument("--seed", type=int, default=1024)
    train_padim_p.set_defaults(func=_cmd_train_padim)

    cal_p = sub.add_parser(
        "calibrate",
        help="seuil de coût (PWA) sur le val, compare officiel et nôtre",
    )
    cal_p.add_argument("--batch-size", type=int, default=16)
    cal_p.add_argument(
        "--official-only",
        action="store_true",
        help="ne pas scorer les checkpoints locaux",
    )
    cal_p.set_defaults(func=_cmd_calibrate)

    export_p = sub.add_parser(
        "export",
        help="exporte classifieur + backbone PaDiM en ONNX (gaussienne hors graphe)",
    )
    export_p.add_argument("--opset", type=int, default=18)
    export_p.add_argument(
        "--skip-classifier",
        action="store_true",
        help="ne pas exporter le resnest50d",
    )
    export_p.add_argument(
        "--skip-padim",
        action="store_true",
        help="ne pas exporter le WRN-50-2",
    )
    export_p.add_argument(
        "--no-verify",
        action="store_true",
        help="ne pas comparer PyTorch et onnxruntime sur un dummy",
    )
    export_p.set_defaults(func=_cmd_export)

    pred_p = sub.add_parser(
        "predict",
        help="inférence locale ONNX (même logique que le handler Lambda)",
    )
    pred_p.add_argument("image", help="PNG/JPEG (déjà recadré, sauf si --lib)")
    pred_p.add_argument(
        "--lib",
        default=None,
        help="Die01–Die04 : rotate+crop comme le notebook",
    )
    pred_p.set_defaults(func=_cmd_predict)

    dep_p = sub.add_parser(
        "deploy",
        help="AWS : alarme de facturation d'abord, puis ECR + Lambda (dry-run)",
    )
    dep_p.add_argument(
        "--email",
        required=True,
        help="destinataire SNS / budget (obligatoire même en dry-run)",
    )
    dep_p.add_argument(
        "--apply",
        action="store_true",
        help="exécuter AWS CLI (sinon affiche seulement le plan)",
    )
    dep_p.add_argument("--region", default="eu-west-3", help="région ECR/Lambda")
    dep_p.add_argument("--image-uri", default=None, help="URI ECR déjà poussée")
    dep_p.add_argument("--limit-usd", type=int, default=1, help="plafond alarme USD")
    dep_p.add_argument(
        "--skip-docker",
        action="store_true",
        help="ne pas builder/pusher l'image (déjà dans ECR)",
    )
    dep_p.set_defaults(func=_cmd_deploy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
