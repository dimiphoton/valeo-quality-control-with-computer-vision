"""Classifieur des 6 défauts connus (pas le drift).

Le checkpoint officiel ``Classifier.pt`` est un timm ``resnest50d``
sauvegardé après 15 époques (module entier, pas un state_dict). On le
charge avec un shim des anciens chemins ``timm.models.layers`` puis on
l'évalue sur le split val. L'entraînement local reprend la même
architecture, avec une loss pondérée et un suivi MLflow.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from valeo_qc.decision_logic import ID_TO_LABEL, N_KNOWN_CLASSES
from valeo_qc.preprocessing import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    SPLIT_CSV_NAME,
    WEIGHTS_JSON_NAME,
)

LOGGER = logging.getLogger(__name__)

OFFICIAL_CHECKPOINT = RAW_DIR / "Supp_files" / "Classifier.pt"
DEFAULT_MODEL_NAME = "resnest50d"
IMAGE_SIZE = 224
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "valeo-qc-classifier"


def get_device() -> torch.device:
    """CUDA si un tenseur d'essai passe, sinon CPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as exc:
            LOGGER.warning("CUDA indisponible (%s), repli CPU", exc)
    return torch.device("cpu")


def classifier_transform() -> transforms.Compose:
    """Même prétraitement que le notebook officiel (pas de normalise ImageNet)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


class SplitImageDataset(Dataset):
    """Images recadrées + labels du CSV de split."""

    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform or classifier_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        path = self.images_dir / str(row["filename"])
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, int(row["label_id"])


def load_split(
    split_csv: Path | None = None,
    which: str = "val",
) -> pd.DataFrame:
    """Charge le CSV de split et filtre ``train`` ou ``val``.

    Parameters
    ----------
    split_csv
        Défaut : ``data/processed/split.csv``.
    which
        ``train``, ``val`` ou ``all``.

    Returns
    -------
    pandas.DataFrame
        Lignes demandées.

    Raises
    ------
    FileNotFoundError
        Si le CSV n'existe pas (lancer ``prepare`` d'abord).
    ValueError
        Si ``which`` est inconnu.
    """
    path = PROCESSED_DIR / SPLIT_CSV_NAME if split_csv is None else Path(split_csv)
    if not path.is_file():
        raise FileNotFoundError(f"split introuvable : {path} (python -m valeo_qc.cli prepare)")
    frame = pd.read_csv(path)
    if which == "all":
        return frame
    if which not in {"train", "val"}:
        raise ValueError(f"which doit être train, val ou all, reçu {which!r}")
    return frame.loc[frame["split"] == which].reset_index(drop=True)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_KNOWN_CLASSES,
) -> dict[str, Any]:
    """Accuracy, F1 macro et rappel par classe (sans sklearn).

    Parameters
    ----------
    y_true, y_pred
        Labels entiers 0–5.
    n_classes
        Nombre de classes connues.

    Returns
    -------
    dict
        ``accuracy``, ``macro_f1``, ``recall`` (dict id → float).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 1.0
    f1_scores: list[float] = []
    recall: dict[str, float] = {}
    for class_id in range(n_classes):
        tp = int(np.sum((y_true == class_id) & (y_pred == class_id)))
        fp = int(np.sum((y_true != class_id) & (y_pred == class_id)))
        fn = int(np.sum((y_true == class_id) & (y_pred != class_id)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * rec / (precision + rec) if (precision + rec) else 0.0
        f1_scores.append(f1)
        recall[str(class_id)] = rec
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "recall": recall,
    }


def _patch_resnest_drop_path(module: nn.Module) -> int:
    """Les checkpoints timm anciens n'ont pas ``drop_path`` sur ResNestBottleneck."""
    patched = 0
    for child in module.modules():
        if child.__class__.__name__ == "ResNestBottleneck" and not hasattr(child, "drop_path"):
            child.drop_path = None
            patched += 1
    return patched


def _install_timm_shims() -> None:
    """Réécrit les imports ``timm.models.layers.*`` du pickle officiel."""
    import timm.layers
    import timm.layers.adaptive_avgmax_pool
    import timm.layers.split_attn

    sys.modules["timm.models.layers"] = timm.layers
    sys.modules["timm.models.layers.split_attn"] = timm.layers.split_attn
    sys.modules["timm.models.layers.adaptive_avgmax_pool"] = timm.layers.adaptive_avgmax_pool


class Classifier(nn.Module):
    """Même enveloppe que ``Supp_files/model.py`` (timm + softmax)."""

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        features = self.model(x)
        if logits:
            return features
        return self.activation(features)


def load_official_classifier(
    checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Charge ``Classifier.pt`` (module entier pickle, epoch 15).

    Parameters
    ----------
    checkpoint
        Défaut : :data:`OFFICIAL_CHECKPOINT`.
    device
        Cible. Défaut : :func:`get_device`.

    Returns
    -------
    torch.nn.Module
        Modèle en ``eval()``.

    Raises
    ------
    FileNotFoundError
        Si le fichier est absent.
    """
    path = OFFICIAL_CHECKPOINT if checkpoint is None else Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    device = get_device() if device is None else device
    _install_timm_shims()
    wrapper = types.ModuleType("model")
    wrapper.Classifier = Classifier
    sys.modules["model"] = wrapper
    loaded = torch.load(path, map_location=device, weights_only=False)
    _patch_resnest_drop_path(loaded)
    loaded.eval()
    return loaded.to(device)


def build_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = N_KNOWN_CLASSES,
    pretrained: bool = True,
) -> Classifier:
    """Instancie un classifieur timm neuf (entraînement)."""
    return Classifier(model_name, num_classes=num_classes, pretrained=pretrained)


def make_loader(
    frame: pd.DataFrame,
    images_dir: Path,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader Windows-friendly (``num_workers=0`` par défaut)."""
    dataset = SplitImageDataset(frame, images_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Prédit sur un loader et calcule accuracy / F1 macro / rappels.

    Parameters
    ----------
    model
        Sortie logits ou probabilités : on prend l'argmax.
    loader, device
        Données et device.

    Returns
    -------
    dict
        Métriques plus ``n``.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred))
    metrics["n"] = len(y_true)
    return metrics


def load_weight_tensor(
    weights_json: Path | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Lit ``class_weights.json`` → tenseur (6,) pour CrossEntropyLoss."""
    path = PROCESSED_DIR / WEIGHTS_JSON_NAME if weights_json is None else Path(weights_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["weights"]
    tensor = torch.ones(N_KNOWN_CLASSES, dtype=torch.float32)
    for key, value in raw.items():
        tensor[int(key)] = float(value)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def setup_mlflow(tracking_uri: str | None = None) -> None:
    """Pointe MLflow vers une SQLite locale (MLflow 3 n'aime plus ``./mlruns``)."""
    import mlflow

    if tracking_uri is None:
        db = MLFLOW_DB.resolve().as_posix()
        tracking_uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_metrics_to_mlflow(metrics: dict[str, Any], step: int | None = None) -> None:
    """Envoie accuracy / F1 / rappels nommés (MLflow n'aime pas les dicts)."""
    import mlflow

    scalars = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    for class_id, rec in metrics.get("recall", {}).items():
        label = ID_TO_LABEL[int(class_id)].replace(" ", "_")
        scalars[f"recall_{label}"] = float(rec)
    mlflow.log_metrics(scalars, step=step)


def evaluate_official(
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    checkpoint: Path | None = None,
    batch_size: int = 16,
    tracking_uri: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Évalue le ``Classifier.pt`` officiel sur le split val.

    Parameters
    ----------
    split_csv, images_dir, checkpoint, batch_size
        Chemins et batch. Images = ``data/processed/train`` (crops).
    tracking_uri
        URI MLflow. Défaut : ``mlruns/``.
    log_mlflow
        Désactiver dans les tests.

    Returns
    -------
    dict
        Métriques val.
    """
    device = get_device()
    images = PROCESSED_DIR / "train" if images_dir is None else Path(images_dir)
    val = load_split(split_csv, which="val")
    loader = make_loader(val, images, batch_size=batch_size, shuffle=False)
    model = load_official_classifier(checkpoint=checkpoint, device=device)
    metrics = evaluate_classifier(model, loader, device)
    LOGGER.info(
        "officiel val n=%s acc=%.4f macro_f1=%.4f",
        metrics["n"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    if log_mlflow:
        import mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name="official-resnest50d"):
            mlflow.log_params(
                {
                    "source": "Classifier.pt",
                    "architecture": DEFAULT_MODEL_NAME,
                    "split": "val",
                    "image_size": IMAGE_SIZE,
                    "normalize": "none",
                    "device": str(device),
                }
            )
            log_metrics_to_mlflow(metrics)
            mlflow.log_dict(
                {ID_TO_LABEL[int(k)]: v for k, v in metrics["recall"].items()},
                "recall_per_class.json",
            )
    return metrics


def train_classifier(
    *,
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    weights_json: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
    tracking_uri: str | None = None,
    checkpoint_out: Path | None = None,
    model: nn.Module | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Entraîne un classifieur, journal MLflow, sauve le meilleur F1 val.

    Parameters
    ----------
    split_csv, images_dir, weights_json
        Sorties de ``prepare``.
    model_name, epochs, batch_size, lr, seed, pretrained
        Hyperparamètres. 15 époques = le checkpoint officiel.
    tracking_uri, checkpoint_out
        MLflow et fichier ``.pt``.
    model
        Réseau déjà construit (tests : un tout petit CNN).
    log_mlflow
        Désactiver dans les tests.

    Returns
    -------
    dict
        Meilleures métriques val + chemin du checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    images = PROCESSED_DIR / "train" if images_dir is None else Path(images_dir)
    train_frame = load_split(split_csv, which="train")
    val_frame = load_split(split_csv, which="val")
    train_loader = make_loader(train_frame, images, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_frame, images, batch_size=batch_size, shuffle=False)

    net = model if model is not None else build_classifier(model_name, pretrained=pretrained)
    net = net.to(device)
    weights = load_weight_tensor(weights_json, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    out_path = MODELS_DIR / "classifier-best.pt" if checkpoint_out is None else Path(checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best: dict[str, Any] = {"macro_f1": -1.0}

    def _loop() -> dict[str, Any]:
        nonlocal best
        import mlflow

        for epoch in range(1, epochs + 1):
            net.train()
            running_loss = 0.0
            n_seen = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # logits : la CE de PyTorch softmax tout seul, contrairement au forward officiel
                logits = net(inputs, logits=True) if isinstance(net, Classifier) else net(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * labels.size(0)
                n_seen += int(labels.size(0))
            train_loss = running_loss / max(n_seen, 1)
            val_metrics = evaluate_classifier(net, val_loader, device)
            LOGGER.info(
                "epoch %s/%s loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch,
                epochs,
                train_loss,
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
            )
            if log_mlflow:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                log_metrics_to_mlflow(val_metrics, step=epoch)
            if val_metrics["macro_f1"] > best["macro_f1"]:
                best = {**val_metrics, "epoch": epoch, "train_loss": train_loss}
                torch.save(
                    {
                        "model_name": model_name,
                        "state_dict": net.state_dict(),
                        "metrics": {k: v for k, v in best.items() if k != "recall"}
                        | {"recall": best.get("recall", {})},
                    },
                    out_path,
                )
        return best

    if log_mlflow:
        import mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name=f"{model_name}-weighted"):
            mlflow.log_params(
                {
                    "architecture": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                    "pretrained": pretrained,
                    "class_weights": "train_split",
                    "image_size": IMAGE_SIZE,
                    "normalize": "none",
                    "device": str(device),
                    "n_train": len(train_frame),
                    "n_val": len(val_frame),
                }
            )
            best = _loop()
            mlflow.log_artifact(str(out_path))
            mlflow.log_metrics(
                {
                    "best_epoch": int(best["epoch"]),
                    "best_macro_f1": float(best["macro_f1"]),
                    "best_accuracy": float(best["accuracy"]),
                }
            )
    else:
        best = _loop()

    best["checkpoint"] = str(out_path)
    return best
