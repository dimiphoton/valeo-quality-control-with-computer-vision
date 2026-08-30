"""Détecteur d'anomalie PaDiM, aligné sur le pickle officiel.

Le notebook du challenge (``Supp_files/Notebook_ENS.ipynb``) n'utilise
pas anomalib : WideResNet-50-2 ImageNet, hooks sur ``layer1/2/3``,
réduction aléatoire 1792 → 550 (seed 1024), gaussienne par patch,
Mahalanobis, upsample 128, lissage σ=4, min-max sur le lot scoré.

``PADIM.pkl`` = ``[mean (550, 1024), cov (550, 550, 1024)]`` en float32.
Le val n'a pas de ``drift`` : on journalise la distribution des scores
par classe connue, pas un AUROC.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from valeo_qc.classifier import SplitImageDataset, get_device, load_split
from valeo_qc.decision_logic import ID_TO_LABEL, N_KNOWN_CLASSES
from valeo_qc.preprocessing import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

LOGGER = logging.getLogger(__name__)

OFFICIAL_PADIM = RAW_DIR / "Supp_files" / "PADIM.pkl"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "valeo-qc-padim"

IMAGE_SIZE = 128
T_D = 1792
D = 550
SEED = 1024
GAUSSIAN_SIGMA = 4.0
RIDGE = 0.01
SPATIAL = 32  # layer1 de WRN-50-2 à 128×128
HW = SPATIAL * SPATIAL


def padim_transform() -> transforms.Compose:
    """Même prétraitement que le notebook (Resize 128 + ToTensor, pas de normalise)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def dimension_index(t_d: int = T_D, d: int = D, seed: int = SEED) -> torch.Tensor:
    """Tiroir aléatoire des canaux, identique au notebook (``random.sample``).

    Parameters
    ----------
    t_d, d, seed
        Dimension concaténée, dimension retenue, graine.

    Returns
    -------
    torch.Tensor
        Indices ``(d,)`` int64.
    """
    rng = random.Random(seed)
    return torch.tensor(rng.sample(range(t_d), d), dtype=torch.int64)


def embedding_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concatène deux cartes de features en upsamplant ``y`` vers ``x``.

    Copie du notebook officiel (unfold / fold).
    """
    batch, c1, h1, w1 = x.size()
    _, c2, h2, _ = y.size()
    scale = int(h1 / h2)
    unfolded = F.unfold(x, kernel_size=scale, dilation=1, stride=scale)
    unfolded = unfolded.view(batch, c1, -1, h2, h2)
    stacked = torch.zeros(
        batch,
        c1 + c2,
        unfolded.size(2),
        h2,
        h2,
        device=x.device,
        dtype=x.dtype,
    )
    for patch in range(unfolded.size(2)):
        stacked[:, :, patch, :, :] = torch.cat((unfolded[:, :, patch, :, :], y), 1)
    stacked = stacked.view(batch, -1, h2 * h2)
    return F.fold(stacked, kernel_size=scale, output_size=(h1, w1), stride=scale)


class WideResNetFeatures:
    """WideResNet-50-2 avec hooks PaDiM (layer1, layer2, layer3)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model.to(device).eval()
        self._outputs: list[torch.Tensor] = []
        self.model.layer1[-1].register_forward_hook(self._hook)
        self.model.layer2[-1].register_forward_hook(self._hook)
        self.model.layer3[-1].register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self._outputs.append(output)

    @torch.no_grad()
    def reduced_embeddings(
        self,
        images: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Embeddings réduits ``(B, D, 32, 32)``."""
        self._outputs = []
        _ = self.model(images.to(self.device))
        layer1, layer2, layer3 = self._outputs
        self._outputs = []
        concatenated = embedding_concat(layer1, layer2)
        concatenated = embedding_concat(concatenated, layer3)
        idx = idx.to(concatenated.device)
        return torch.index_select(concatenated, 1, idx)


def invert_covariance(cov: np.ndarray, chunk: int = 32) -> np.ndarray:
    """Inverse chaque covariance spatiale ``(C, C, HW)`` → ``(C, C, HW)``.

    Par paquets pour ne pas allouer un ``(HW, C, C)`` float64 d'un coup
    (~2,5 Go si C=550).

    Parameters
    ----------
    cov
        Tenseur float ``(C, C, HW)``.
    chunk
        Nombre de patches inversés ensemble.

    Returns
    -------
    numpy.ndarray
        Inverses, même forme, float32. ``pinv`` si une tranche est singulière.
    """
    _channels, _, hw = cov.shape
    out = np.empty_like(cov, dtype=np.float32)
    for start in range(0, hw, chunk):
        end = min(start + chunk, hw)
        stacked = np.moveaxis(cov[:, :, start:end], -1, 0).astype(np.float64)
        try:
            inverted = np.linalg.inv(stacked)
        except np.linalg.LinAlgError:
            LOGGER.warning("covariance singulière patches %s-%s, repli pinv", start, end)
            inverted = np.linalg.pinv(stacked)
        out[:, :, start:end] = np.moveaxis(inverted.astype(np.float32), 0, -1)
    return out


def mahalanobis_maps(
    embeddings: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Cartes de distance ``(B, H, W)`` pour des embeddings ``(B, C, H, W)``.

    Parameters
    ----------
    embeddings
        Features réduites.
    mean
        Moyenne PaDiM ``(C, HW)``.
    cov_inv
        Précision ``(C, C, HW)``.

    Returns
    -------
    numpy.ndarray
        Distances de Mahalanobis par patch.
    """
    batch, channels, height, width = embeddings.shape
    hw = height * width
    flat = embeddings.reshape(batch, channels, hw)
    diff = flat - mean[np.newaxis, :, :]
    dist = np.empty((batch, hw), dtype=np.float64)
    chunk = 64
    for start in range(0, hw, chunk):
        end = min(start + chunk, hw)
        piece = diff[:, :, start:end]
        inv = cov_inv[:, :, start:end]
        precision_dot = np.einsum("cdi,bdi->bci", inv, piece, optimize=True)
        dist[:, start:end] = np.einsum("bci,bci->bi", piece, precision_dot, optimize=True)
    return dist.reshape(batch, height, width)


def _mahalanobis_from_tensors(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    cov_inv: torch.Tensor,
    chunk: int = 64,
) -> np.ndarray:
    """Mahalanobis GPU : ``mean`` et ``cov_inv`` déjà sur le device (un seul upload)."""
    batch, channels, height, width = embeddings.shape
    hw = height * width
    flat = embeddings.reshape(batch, channels, hw)
    diff = flat - mean.unsqueeze(0)
    dist = torch.empty(batch, hw, device=embeddings.device, dtype=embeddings.dtype)
    for start in range(0, hw, chunk):
        end = min(start + chunk, hw)
        piece = diff[:, :, start:end]
        precision_dot = torch.einsum("cdi,bdi->bci", cov_inv[:, :, start:end], piece)
        dist[:, start:end] = torch.einsum("bci,bci->bi", piece, precision_dot)
    return dist.reshape(batch, height, width).detach().cpu().numpy()


def upsample_and_smooth(
    dist_maps: np.ndarray,
    size: int = IMAGE_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Bilinear vers ``size×size`` puis Gaussienne σ (comme le notebook)."""
    tensor = torch.from_numpy(dist_maps.astype(np.float32)).unsqueeze(1)
    upsampled = F.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )
    maps = upsampled.squeeze(1).numpy()
    return gaussian_filter(maps, sigma=(0.0, sigma, sigma))


def minmax_image_scores(score_maps: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max sur tout le lot, puis max spatial → un score par image.

    Parameters
    ----------
    score_maps
        ``(N, H, W)`` déjà lissées.

    Returns
    -------
    tuple
        Scores ``(N,)``, min et max bruts du lot.
    """
    min_score = float(score_maps.min())
    max_score = float(score_maps.max())
    span = max_score - min_score
    if span <= 0:
        normalized = np.zeros(score_maps.shape, dtype=np.float64)
    else:
        normalized = (score_maps - min_score) / span
    image_scores = normalized.reshape(normalized.shape[0], -1).max(axis=1)
    return image_scores.astype(np.float64), min_score, max_score


def raw_image_scores(score_maps: np.ndarray) -> np.ndarray:
    """Max spatial sans min-max (échelle Mahalanobis, comparable d'un run à l'autre)."""
    return score_maps.reshape(score_maps.shape[0], -1).max(axis=1).astype(np.float64)


def fit_gaussian(
    embeddings: np.ndarray,
    ridge: float = RIDGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne et covariance empirique + ridge, sur un tenseur ``(N, C, H, W)``.

    Utilisé par les tests et les petits lots. L'entraînement réel passe par
    :func:`fit_gaussian_streaming` pour ne pas garder 15 Go d'embeddings.

    Parameters
    ----------
    embeddings
        Features d'apprentissage.
    ridge
        λ I ajouté à chaque covariance (évite les inverses instables).

    Returns
    -------
    tuple
        ``mean (C, HW)``, ``cov (C, C, HW)`` float32.
    """
    n_samples, channels, height, width = embeddings.shape
    flat = embeddings.reshape(n_samples, channels, height * width).astype(np.float64)
    mean = flat.mean(axis=0)
    diff = flat - mean
    cov = np.einsum("nci,ndi->cdi", diff, diff, optimize=True) / max(n_samples - 1, 1)
    eye = np.eye(channels, dtype=np.float64)[:, :, np.newaxis]
    cov = cov + ridge * eye
    return mean.astype(np.float32), cov.astype(np.float32)


def _accumulate_cov(cov: torch.Tensor, diff: torch.Tensor, chunk: int = 32) -> None:
    """Ajoute ``(x-μ)(x-μ)ᵀ`` à ``cov`` par paquets spatiaux (évite un (B,C,C,HW))."""
    hw = diff.size(-1)
    for start in range(0, hw, chunk):
        end = min(start + chunk, hw)
        piece = diff[:, :, start:end]
        cov[:, :, start:end] += torch.einsum("bci,bdi->cdi", piece, piece)


def load_padim_stats(
    path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Charge ``PADIM.pkl`` officiel (liste) ou un checkpoint dict maison.

    Parameters
    ----------
    path
        Défaut : :data:`OFFICIAL_PADIM`.

    Returns
    -------
    tuple
        ``mean``, ``cov``, indices de canaux.

    Raises
    ------
    FileNotFoundError
        Si le fichier est absent.
    ValueError
        Si le format est inconnu.
    """
    stats_path = OFFICIAL_PADIM if path is None else Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    with stats_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        mean, cov = payload[0], payload[1]
        return np.asarray(mean), np.asarray(cov), dimension_index()
    if isinstance(payload, dict) and "mean" in payload and "cov" in payload:
        idx = torch.as_tensor(payload["idx"], dtype=torch.int64)
        return np.asarray(payload["mean"]), np.asarray(payload["cov"]), idx
    raise ValueError(f"format PaDiM inconnu dans {stats_path}")


def save_padim_stats(
    path: Path,
    mean: np.ndarray,
    cov: np.ndarray,
    idx: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Écrit un checkpoint dict (mean, cov, idx, hyperparamètres)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mean": np.asarray(mean),
        "cov": np.asarray(cov),
        "idx": idx.detach().cpu().tolist(),
        "t_d": T_D,
        "d": D,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "ridge": RIDGE,
    }
    if extra:
        payload.update(extra)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def make_padim_loader(
    frame: pd.DataFrame,
    images_dir: Path,
    batch_size: int = 16,
    shuffle: bool = False,
) -> DataLoader:
    """DataLoader 128×128, ``num_workers=0`` (Windows)."""
    dataset = SplitImageDataset(frame, images_dir, transform=padim_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def summarize_scores(
    scores_raw: np.ndarray,
    scores_norm: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Stats globales et par classe (val sans drift)."""
    summary: dict[str, Any] = {
        "n": int(scores_raw.size),
        "raw_mean": float(scores_raw.mean()) if scores_raw.size else 0.0,
        "raw_std": float(scores_raw.std()) if scores_raw.size else 0.0,
        "raw_p95": float(np.quantile(scores_raw, 0.95)) if scores_raw.size else 0.0,
        "norm_mean": float(scores_norm.mean()) if scores_norm.size else 0.0,
        "frac_above_0_5": float((scores_norm > 0.5).mean()) if scores_norm.size else 0.0,
        "per_class": {},
    }
    for class_id in range(N_KNOWN_CLASSES):
        mask = labels == class_id
        if not np.any(mask):
            continue
        name = ID_TO_LABEL[class_id]
        summary["per_class"][name] = {
            "n": int(mask.sum()),
            "raw_mean": float(scores_raw[mask].mean()),
            "raw_p95": float(np.quantile(scores_raw[mask], 0.95)),
            "norm_mean": float(scores_norm[mask].mean()),
        }
    return summary


def score_loader(
    extractor: WideResNetFeatures,
    loader: DataLoader,
    mean: np.ndarray,
    cov_inv: np.ndarray,
    idx: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score un loader : cartes lissées, labels, scores image min-max.

    Returns
    -------
    tuple
        ``score_maps (N, 128, 128)``, ``labels (N,)``, ``scores_norm (N,)``.
    """
    maps: list[np.ndarray] = []
    labels: list[int] = []
    n_seen = 0
    device = extractor.device
    mean_t: torch.Tensor | None = None
    inv_t: torch.Tensor | None = None
    if device.type == "cuda":
        try:
            LOGGER.info("upload mean/cov_inv sur GPU")
            mean_t = torch.from_numpy(np.ascontiguousarray(mean)).to(device)
            inv_t = torch.from_numpy(np.ascontiguousarray(cov_inv)).to(device)
        except RuntimeError as exc:
            LOGGER.warning("GPU trop petit pour cov_inv (%s), repli CPU", exc)
            mean_t, inv_t = None, None
    for images, batch_labels in loader:
        reduced = extractor.reduced_embeddings(images, idx)
        if mean_t is not None and inv_t is not None:
            dist = _mahalanobis_from_tensors(reduced, mean_t, inv_t)
        else:
            embeddings = reduced.detach().cpu().numpy().astype(np.float32)
            dist = mahalanobis_maps(embeddings, mean, cov_inv)
        maps.append(upsample_and_smooth(dist))
        labels.extend(int(x) for x in batch_labels.tolist())
        n_seen += int(images.size(0))
        if n_seen % 256 == 0:
            LOGGER.info("PaDiM inférence %s images", n_seen)
    score_maps = np.concatenate(maps, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    scores_norm, _, _ = minmax_image_scores(score_maps)
    return score_maps, y, scores_norm


def setup_mlflow(tracking_uri: str | None = None) -> None:
    """Pointe MLflow vers la SQLite locale, expérience PaDiM."""
    import mlflow

    if tracking_uri is None:
        db = MLFLOW_DB.resolve().as_posix()
        tracking_uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_padim_to_mlflow(summary: dict[str, Any], prefix: str = "") -> None:
    """Scalaires globaux + moyenne brute par classe."""
    import mlflow

    key = f"{prefix}raw_mean" if prefix else "raw_mean"
    mlflow.log_metrics(
        {
            key: float(summary["raw_mean"]),
            f"{prefix}raw_p95": float(summary["raw_p95"]),
            f"{prefix}norm_mean": float(summary["norm_mean"]),
            f"{prefix}frac_above_0_5": float(summary["frac_above_0_5"]),
        }
    )
    for name, stats in summary.get("per_class", {}).items():
        slug = name.replace(" ", "_")
        mlflow.log_metric(f"{prefix}raw_mean_{slug}", float(stats["raw_mean"]))


def evaluate_official(
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    checkpoint: Path | None = None,
    batch_size: int = 16,
    tracking_uri: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Score le ``PADIM.pkl`` officiel sur le split val.

    Parameters
    ----------
    split_csv, images_dir, checkpoint, batch_size
        Chemins et batch. Images = crops ``data/processed/train``.
    tracking_uri, log_mlflow
        Journal MLflow (désactiver dans les tests).

    Returns
    -------
    dict
        Résumé des scores val.
    """
    device = get_device()
    images = PROCESSED_DIR / "train" if images_dir is None else Path(images_dir)
    val = load_split(split_csv, which="val")
    loader = make_padim_loader(val, images, batch_size=batch_size)
    mean, cov, idx = load_padim_stats(checkpoint)
    LOGGER.info("inversion des covariances officielles (%s patches)", cov.shape[-1])
    cov_inv = invert_covariance(cov)
    extractor = WideResNetFeatures(device)
    score_maps, labels, scores_norm = score_loader(extractor, loader, mean, cov_inv, idx)
    scores_raw = raw_image_scores(score_maps)
    summary = summarize_scores(scores_raw, scores_norm, labels)
    LOGGER.info(
        "officiel val n=%s raw_mean=%.3f p95=%.3f frac>0.5=%.3f",
        summary["n"],
        summary["raw_mean"],
        summary["raw_p95"],
        summary["frac_above_0_5"],
    )
    if log_mlflow:
        import mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name="official-padim"):
            mlflow.log_params(
                {
                    "source": "PADIM.pkl",
                    "backbone": "wide_resnet50_2",
                    "image_size": IMAGE_SIZE,
                    "d": D,
                    "seed": SEED,
                    "normalize": "none",
                    "device": str(device),
                }
            )
            log_padim_to_mlflow(summary)
            mlflow.log_dict(summary["per_class"], "scores_per_class.json")
    summary["scores_raw"] = scores_raw
    summary["scores_norm"] = scores_norm
    summary["labels"] = labels
    return summary


def train_padim(
    *,
    split_csv: Path | None = None,
    images_dir: Path | None = None,
    batch_size: int = 16,
    ridge: float = RIDGE,
    seed: int = SEED,
    tracking_uri: str | None = None,
    checkpoint_out: Path | None = None,
    log_mlflow: bool = True,
    extractor: WideResNetFeatures | None = None,
) -> dict[str, Any]:
    """Fit PaDiM sur le split train (toutes classes connues), évalue sur val.

    Deux passes GPU : moyenne puis covariance + ridge. Pas de LedoitWolf
    (il exigerait de garder tous les embeddings, ~15 Go).

    Parameters
    ----------
    split_csv, images_dir, batch_size, ridge, seed
        Données et hyperparamètres.
    tracking_uri, checkpoint_out, log_mlflow
        MLflow et pickle de sortie.
    extractor
        Backbone déjà construit (tests).

    Returns
    -------
    dict
        Résumé val + chemin du checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    images = PROCESSED_DIR / "train" if images_dir is None else Path(images_dir)
    train_frame = load_split(split_csv, which="train")
    val_frame = load_split(split_csv, which="val")
    train_loader = make_padim_loader(train_frame, images, batch_size=batch_size)
    val_loader = make_padim_loader(val_frame, images, batch_size=batch_size)

    net = extractor if extractor is not None else WideResNetFeatures(device)
    idx = dimension_index(seed=seed)

    LOGGER.info("PaDiM passe 1/2 — moyenne sur %s images", len(train_frame))
    total = torch.zeros(D, HW, dtype=torch.float64)
    n_seen = 0
    for inputs, _labels in train_loader:
        reduced = net.reduced_embeddings(inputs, idx)
        flat = reduced.detach().cpu().reshape(reduced.size(0), D, HW).double()
        total += flat.sum(dim=0)
        n_seen += int(reduced.size(0))
        if n_seen % 256 == 0 or n_seen == len(train_frame):
            LOGGER.info("moyenne %s / %s", n_seen, len(train_frame))
    mean_t = (total / max(n_seen, 1)).float()

    LOGGER.info("PaDiM passe 2/2 — covariance")
    cov_t = torch.zeros(D, D, HW, dtype=torch.float32)
    n_seen = 0
    for inputs, _labels in train_loader:
        reduced = net.reduced_embeddings(inputs, idx)
        flat = reduced.detach().cpu().reshape(reduced.size(0), D, HW)
        diff = flat - mean_t
        # Σ_i += (x-μ)(x-μ)ᵀ  par patch, en float32
        _accumulate_cov(cov_t, diff)
        n_seen += int(reduced.size(0))
        if n_seen % 256 == 0 or n_seen == len(train_frame):
            LOGGER.info("covariance %s / %s", n_seen, len(train_frame))
    cov_t = cov_t / max(n_seen - 1, 1)
    cov_t = cov_t + ridge * torch.eye(D).unsqueeze(-1)

    mean = mean_t.numpy()
    cov = cov_t.numpy()
    out_path = MODELS_DIR / "padim-best.pkl" if checkpoint_out is None else Path(checkpoint_out)
    save_padim_stats(
        out_path,
        mean,
        cov,
        idx,
        extra={"n_train": n_seen, "ridge": ridge, "fit_on": "train_split_all_classes"},
    )

    LOGGER.info("inversion des covariances apprises")
    cov_inv = invert_covariance(cov)
    score_maps, labels, scores_norm = score_loader(net, val_loader, mean, cov_inv, idx)
    scores_raw = raw_image_scores(score_maps)
    summary = summarize_scores(scores_raw, scores_norm, labels)
    summary["checkpoint"] = str(out_path)
    summary["n_train"] = n_seen
    LOGGER.info(
        "notre PaDiM val n=%s raw_mean=%.3f p95=%.3f frac>0.5=%.3f",
        summary["n"],
        summary["raw_mean"],
        summary["raw_p95"],
        summary["frac_above_0_5"],
    )

    if log_mlflow:
        import mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name="padim-wrn50-2"):
            mlflow.log_params(
                {
                    "backbone": "wide_resnet50_2",
                    "image_size": IMAGE_SIZE,
                    "d": D,
                    "seed": seed,
                    "ridge": ridge,
                    "fit_on": "train_split_all_classes",
                    "n_train": n_seen,
                    "device": str(device),
                }
            )
            log_padim_to_mlflow(summary)
            mlflow.log_dict(summary["per_class"], "scores_per_class.json")
            mlflow.log_artifact(str(out_path))

    summary["scores_raw"] = scores_raw
    summary["scores_norm"] = scores_norm
    summary["labels"] = labels
    return summary
