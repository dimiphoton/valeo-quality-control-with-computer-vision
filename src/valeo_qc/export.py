"""Export ONNX du classifieur et du backbone PaDiM.

La gaussienne PaDiM (moyenne + covariance, ~1,2 Go) reste un pickle :
ce n'est pas un réseau. En inférence unitaire, le min-max du notebook
(sur le *lot* scoré) n'est pas utilisable — une image seule aurait
toujours un score normalisé de 1. Le manifeste note qu'il faudra figer
min/max (étape Lambda).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from valeo_qc.preprocessing import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"
CLASSIFIER_ONNX = MODELS_DIR / "classifier.onnx"
PADIM_ONNX = MODELS_DIR / "padim-backbone.onnx"
MANIFEST_JSON = MODELS_DIR / "onnx-manifest.json"
DEFAULT_OPSET = 18
OFFICIAL_PADIM_REL = "data/raw/Supp_files/PADIM.pkl"
CLS_SIZE = 224
PADIM_SIZE = 128
D = 550


class _ForwardOnly(nn.Module):
    """Enveloppe sans kwargs — ``torch.onnx.export`` n'aime que ``forward(x)``."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


def _rel(path: Path) -> str:
    """Chemin posix relatif à la racine du repo, sinon absolu."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def export_onnx(
    model: nn.Module,
    path: Path,
    input_shape: tuple[int, ...],
    *,
    opset: int = DEFAULT_OPSET,
    input_name: str = "image",
    output_name: str = "output",
) -> Path:
    """Trace un ``nn.Module`` vers un fichier ONNX (batch dynamique).

    Parameters
    ----------
    model
        Réseau en ``eval()``. Forcé sur CPU.
    path
        Fichier de sortie.
    input_shape
        Forme du dummy, ex. ``(1, 3, 224, 224)``.
    opset
        Version d'opset ONNX.
    input_name, output_name
        Noms des I/O dans le graphe.

    Returns
    -------
    pathlib.Path
        Chemin écrit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    LOGGER.info("ONNX écrit %s (%s octets)", path, path.stat().st_size)
    return path


def run_onnx(
    path: Path,
    images: np.ndarray,
    input_name: str | None = None,
) -> np.ndarray:
    """Infère un ONNX avec onnxruntime CPU dans un sous-processus.

    Un process séparé évite un access violation Windows quand
    ``torch`` (CUDA) et ``onnxruntime`` cohabitent (pytest surtout).

    Parameters
    ----------
    path
        Fichier ``.onnx``.
    images
        Batch float32 NCHW.
    input_name
        Nom de l'entrée. Défaut : première entrée du graphe.

    Returns
    -------
    numpy.ndarray
        Première sortie.

    Raises
    ------
    RuntimeError
        Si onnxruntime échoue dans le sous-processus.
    """
    import subprocess
    import sys
    import tempfile

    batch = np.asarray(images, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_file = tmp_dir / "in.npy"
        out_file = tmp_dir / "out.npy"
        np.save(in_file, batch)
        script = (
            "import sys, numpy as np, onnxruntime as ort; "
            "path, inf, outf, name = sys.argv[1:5]; "
            "batch = np.load(inf); "
            "sess = ort.InferenceSession(path, providers=['CPUExecutionProvider']); "
            "key = name if name else sess.get_inputs()[0].name; "
            "np.save(outf, sess.run(None, {key: batch})[0])"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(in_file),
                str(out_file),
                input_name or "",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"onnxruntime a échoué ({result.returncode}): {result.stderr[-2000:]}"
            )
        return np.load(out_file)


def torch_vs_onnx(
    model: nn.Module,
    onnx_path: Path,
    tensor: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Compare une passe PyTorch et onnxruntime sur le même tenseur.

    Parameters
    ----------
    model, onnx_path, tensor
        Réseau CPU, fichier ONNX, entrée ``(N, C, H, W)``.
    rtol, atol
        Tolérances ``numpy.allclose``.

    Returns
    -------
    dict
        ``max_abs`` et ``ok``.

    Raises
    ------
    AssertionError
        Si les sorties divergent au-delà des tolérances.
    """
    model = model.eval().cpu()
    tensor = tensor.detach().cpu().float()
    with torch.no_grad():
        torch_out = model(tensor).detach().cpu().numpy()
    onnx_out = run_onnx(onnx_path, tensor.numpy())
    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    if not np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol):
        raise AssertionError(
            f"PyTorch et ONNX divergent (max_abs={max_abs:.4g}) pour {onnx_path}"
        )
    return {"max_abs": max_abs, "ok": True}


def _load_classifier_cpu(checkpoint: Path | None) -> nn.Module:
    """Charge le pickle officiel ou un ``state_dict`` local, sur CPU."""
    from valeo_qc.classifier import (
        OFFICIAL_CHECKPOINT,
        load_official_classifier,
        load_trained_classifier,
    )

    ckpt = OFFICIAL_CHECKPOINT if checkpoint is None else Path(checkpoint)
    device = torch.device("cpu")
    if ckpt.name != "Classifier.pt":
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            return load_trained_classifier(ckpt, device=device)
    return load_official_classifier(checkpoint=ckpt, device=device)


def export_classifier(
    checkpoint: Path | None = None,
    out_path: Path | None = None,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Exporte le classifieur officiel (ou un ``.pt`` local) en ONNX.

    Parameters
    ----------
    checkpoint
        Défaut : ``Classifier.pt`` officiel.
    out_path
        Défaut : ``models/classifier.onnx``.
    opset
        Opset ONNX.

    Returns
    -------
    pathlib.Path
        Fichier écrit.
    """
    wrapper = _ForwardOnly(_load_classifier_cpu(checkpoint))
    dest = CLASSIFIER_ONNX if out_path is None else Path(out_path)
    return export_onnx(
        wrapper,
        dest,
        (1, 3, CLS_SIZE, CLS_SIZE),
        opset=opset,
        output_name="probs",
    )


def export_padim_backbone(
    idx: torch.Tensor | None = None,
    out_path: Path | None = None,
    opset: int = DEFAULT_OPSET,
    pretrained: bool = True,
) -> Path:
    """Exporte le WRN-50-2 PaDiM (embeddings 550×32×32), pas la gaussienne.

    Parameters
    ----------
    idx
        Indices de canaux. Défaut : seed 1024 du notebook.
    out_path
        Défaut : ``models/padim-backbone.onnx``.
    opset
        Opset ONNX.
    pretrained
        Poids ImageNet (comme le notebook).

    Returns
    -------
    pathlib.Path
        Fichier écrit.
    """
    from torchvision.models import Wide_ResNet50_2_Weights

    from valeo_qc.padim import PadimBackbone, dimension_index

    index = dimension_index() if idx is None else idx
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
    net = PadimBackbone(index, weights=weights)
    dest = PADIM_ONNX if out_path is None else Path(out_path)
    return export_onnx(
        net,
        dest,
        (1, 3, PADIM_SIZE, PADIM_SIZE),
        opset=opset,
        output_name="embeddings",
    )


def _load_threshold() -> dict[str, Any] | None:
    """Lit ``models/threshold.json`` s'il existe."""
    if not THRESHOLD_JSON.is_file():
        return None
    return json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))


def write_manifest(
    *,
    classifier_path: Path | None,
    padim_path: Path | None,
    opset: int,
    checks: dict[str, Any],
    out_path: Path | None = None,
) -> Path:
    """Écrit le manifeste JSON (chemins relatifs, seuil, limites min-max)."""
    threshold_payload = _load_threshold()
    threshold_value = None
    if threshold_payload and "operating_point" in threshold_payload:
        threshold_value = threshold_payload["operating_point"].get("threshold")
    manifest: dict[str, Any] = {
        "opset": opset,
        "classifier": None
        if classifier_path is None
        else {
            "path": _rel(classifier_path),
            "bytes": classifier_path.stat().st_size if classifier_path.is_file() else 0,
            "input_shape": [None, 3, CLS_SIZE, CLS_SIZE],
            "output": "probs",
            "architecture": "resnest50d",
        },
        "padim_backbone": None
        if padim_path is None
        else {
            "path": _rel(padim_path),
            "bytes": padim_path.stat().st_size if padim_path.is_file() else 0,
            "input_shape": [None, 3, PADIM_SIZE, PADIM_SIZE],
            "output": "embeddings",
            "architecture": "wide_resnet50_2",
            "d": D,
            "stats": OFFICIAL_PADIM_REL,
            "stats_note": (
                "La gaussienne (mean/cov) reste un pickle hors ONNX : trop volumineuse "
                "et ce n'est pas un réseau."
            ),
        },
        "decision": {
            "threshold_json": _rel(THRESHOLD_JSON) if THRESHOLD_JSON.is_file() else None,
            "threshold": threshold_value,
            "operating_point": "protect_good",
            "score_norm": (
                "Le min-max du notebook est calculé sur le lot scoré. "
                "En inférence unitaire il faut figer min/max (sinon le score vaut 1). "
                "À brancher à l'étape Lambda."
            ),
        },
        "checks": checks,
    }
    dest = MANIFEST_JSON if out_path is None else Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def export_pipeline(
    *,
    classifier_checkpoint: Path | None = None,
    classifier_out: Path | None = None,
    padim_out: Path | None = None,
    manifest_out: Path | None = None,
    opset: int = DEFAULT_OPSET,
    skip_classifier: bool = False,
    skip_padim: bool = False,
    verify: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Exporte les deux graphes, vérifie un tenseur aléatoire, écrit le manifeste.

    Parameters
    ----------
    classifier_checkpoint
        ``Classifier.pt`` par défaut.
    classifier_out, padim_out, manifest_out
        Destinations. Défauts sous ``models/``.
    opset
        Opset ONNX.
    skip_classifier, skip_padim
        N'exporter qu'une des deux briques (debug).
    verify
        Comparer PyTorch vs onnxruntime sur un dummy.
    seed
        Graine du dummy de vérif.

    Returns
    -------
    dict
        Manifeste (déjà sérialisé sur disque).
    """
    rng = np.random.default_rng(seed)
    checks: dict[str, Any] = {}
    clf_path: Path | None = None
    padim_path: Path | None = None

    if not skip_classifier:
        LOGGER.info("export classifieur ONNX")
        wrapper = _ForwardOnly(_load_classifier_cpu(classifier_checkpoint))
        clf_path = CLASSIFIER_ONNX if classifier_out is None else Path(classifier_out)
        export_onnx(
            wrapper,
            clf_path,
            (1, 3, CLS_SIZE, CLS_SIZE),
            opset=opset,
            output_name="probs",
        )
        if verify:
            dummy = torch.from_numpy(
                rng.random((1, 3, CLS_SIZE, CLS_SIZE), dtype=np.float32)
            )
            checks["classifier"] = torch_vs_onnx(wrapper, clf_path, dummy)

    if not skip_padim:
        LOGGER.info("export backbone PaDiM ONNX")
        from torchvision.models import Wide_ResNet50_2_Weights

        from valeo_qc.padim import PadimBackbone, dimension_index

        net = PadimBackbone(dimension_index(), weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        padim_path = PADIM_ONNX if padim_out is None else Path(padim_out)
        export_onnx(
            net,
            padim_path,
            (1, 3, PADIM_SIZE, PADIM_SIZE),
            opset=opset,
            output_name="embeddings",
        )
        if verify:
            dummy = torch.from_numpy(
                rng.random((1, 3, PADIM_SIZE, PADIM_SIZE), dtype=np.float32)
            )
            checks["padim_backbone"] = torch_vs_onnx(net, padim_path, dummy)

    dest = write_manifest(
        classifier_path=clf_path,
        padim_path=padim_path,
        opset=opset,
        checks=checks,
        out_path=manifest_out,
    )
    LOGGER.info("manifeste %s", dest)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    payload["manifest"] = str(dest)
    return payload
