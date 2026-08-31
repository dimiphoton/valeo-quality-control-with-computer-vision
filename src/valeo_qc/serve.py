"""Inférence ONNX sans PyTorch : classifieur, PaDiM optionnel, décision.

Conçu pour Lambda (image conteneur) et ``valeo-qc predict``. La
gaussienne 1,2 Go et le min-max figé sont optionnels : sans eux on
classe seulement les 6 défauts connus (p_drift = 0).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from valeo_qc.anomaly_numpy import (
    frozen_image_score,
    invert_covariance,
    mahalanobis_maps,
    upsample_and_smooth_numpy,
)
from valeo_qc.decision_logic import (
    DEFAULT_THRESHOLD,
    DRIFT_CLASS,
    ID_TO_LABEL,
    decide_class,
)
from valeo_qc.preprocessing import crop_pil

LOGGER = logging.getLogger(__name__)

CLS_SIZE = 224
PADIM_SIZE = 128
SCORE_SCALE_NAME = "score-scale.json"
THRESHOLD_NAME = "threshold.json"


class PredictError(ValueError):
    """Entrée HTTP / image invalide (→ 400)."""


def models_dir() -> Path:
    """Répertoire des artefacts (env ``VALEO_MODELS_DIR`` ou ``<repo>/models``)."""
    env = os.environ.get("VALEO_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"


def image_to_nchw(image: Image.Image, size: int) -> np.ndarray:
    """Resize + ToTensor (0–1), comme le notebook officiel.

    Parameters
    ----------
    image
        RGB.
    size
        Côté du carré.

    Returns
    -------
    numpy.ndarray
        ``(1, 3, size, size)`` float32.
    """
    rgb = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def decode_image_bytes(raw: bytes) -> Image.Image:
    """Décode PNG/JPEG depuis des octets."""
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise PredictError("image illisible (PNG/JPEG attendu)") from exc
    return image.convert("RGB")


def _load_threshold(models: Path) -> float:
    path = models / THRESHOLD_NAME
    if not path.is_file():
        return float(DEFAULT_THRESHOLD)
    payload = json.loads(path.read_text(encoding="utf-8"))
    point = payload.get("operating_point") or {}
    value = point.get("threshold")
    return float(value) if value is not None else float(DEFAULT_THRESHOLD)


def _load_score_scale(models: Path) -> tuple[float, float] | None:
    path = models / SCORE_SCALE_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["min"]), float(payload["max"])


def _load_padim_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge pickle officiel ou dict maison → mean, cov (pas encore inversée)."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        return np.asarray(payload[0]), np.asarray(payload[1])
    if isinstance(payload, dict) and "mean" in payload and "cov" in payload:
        return np.asarray(payload["mean"]), np.asarray(payload["cov"])
    raise PredictError(f"format PaDiM inconnu dans {path}")


def _run_onnx_subprocess(path: Path, batch: np.ndarray) -> np.ndarray:
    """Inférence dans un process sans torch (évite un AV pytest/Windows)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_file = tmp_dir / "in.npy"
        out_file = tmp_dir / "out.npy"
        np.save(in_file, np.asarray(batch, dtype=np.float32))
        script = (
            "import sys, numpy as np, onnxruntime as ort; "
            "path, inf, outf = sys.argv[1:4]; "
            "batch = np.load(inf); "
            "sess = ort.InferenceSession(path, providers=['CPUExecutionProvider']); "
            "key = sess.get_inputs()[0].name; "
            "np.save(outf, sess.run(None, {key: batch})[0])"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(path), str(in_file), str(out_file)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"onnxruntime a échoué ({result.returncode}): {result.stderr[-2000:]}"
            )
        return np.load(out_file)


def _run_onnx(path: Path, batch: np.ndarray, holder: dict[str, Any]) -> np.ndarray:
    """In-process (Lambda) ou sous-processus si torch est déjà chargé."""
    if "torch" in sys.modules:
        return _run_onnx_subprocess(path, batch)
    import onnxruntime as ort

    if "session" not in holder:
        holder["session"] = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    session = holder["session"]
    name = session.get_inputs()[0].name
    return session.run(None, {name: np.asarray(batch, dtype=np.float32)})[0]


class Runtime:
    """Sessions ONNX + stats PaDiM optionnelles, chargées une fois."""

    def __init__(
        self,
        models: Path | None = None,
        *,
        classifier_onnx: Path | None = None,
        padim_onnx: Path | None = None,
        padim_stats: Path | None = None,
        threshold: float | None = None,
        score_scale: tuple[float, float] | None = None,
        cls_size: int = CLS_SIZE,
        padim_size: int = PADIM_SIZE,
    ) -> None:
        self.models = models_dir() if models is None else Path(models)
        self.classifier_path = classifier_onnx or (self.models / "classifier.onnx")
        self.padim_path = padim_onnx or (self.models / "padim-backbone.onnx")
        stats_candidates = [
            padim_stats,
            self.models / "PADIM.pkl",
            self.models / "padim-best.pkl",
        ]
        self.stats_path = next((p for p in stats_candidates if p is not None and Path(p).is_file()), None)
        self.threshold = _load_threshold(self.models) if threshold is None else float(threshold)
        self.score_scale = _load_score_scale(self.models) if score_scale is None else score_scale
        self.cls_size = int(cls_size)
        self.padim_size = int(padim_size)
        self._clf: dict[str, Any] = {}
        self._padim: dict[str, Any] = {}
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._padim_loaded = False

    def _padim_ready(self) -> bool:
        return (
            self.padim_path.is_file()
            and self.stats_path is not None
            and self.score_scale is not None
        )

    def _ensure_padim(self) -> None:
        if not self._padim_ready() or self._padim_loaded:
            return
        LOGGER.info("chargement stats PaDiM + inversion des covariances")
        mean, cov = _load_padim_stats(Path(self.stats_path))
        self._mean = mean.astype(np.float32)
        self._cov_inv = invert_covariance(cov)
        self._padim_loaded = True

    def predict_pil(self, image: Image.Image, lib: str | None = None) -> dict[str, Any]:
        """Prédit une image PIL (déjà recadrée, ou ``lib`` pour rotate+crop).

        Parameters
        ----------
        image
            RGB.
        lib
            ``Die01``…``Die04`` pour recadrer comme le notebook.

        Returns
        -------
        dict
            ``label_id``, ``label``, ``probs``, ``p_drift``, ``threshold``, ``padim``.
        """
        working = crop_pil(image, lib) if lib else image
        if not self.classifier_path.is_file():
            raise FileNotFoundError(self.classifier_path)
        clf_batch = image_to_nchw(working, self.cls_size)
        probs = _run_onnx(self.classifier_path, clf_batch, self._clf).reshape(-1).astype(
            np.float64
        )
        if probs.size != 6:
            raise RuntimeError(f"sortie classifieur (6,) attendue, reçu {probs.shape}")
        row_sum = float(probs.sum())
        if abs(row_sum - 1.0) > 1e-3:
            probs = np.exp(probs - probs.max())
            probs = probs / probs.sum()

        padim_on = False
        p_drift = 0.0
        if self._padim_ready():
            self._ensure_padim()
            padim_batch = image_to_nchw(working, self.padim_size)
            embeddings = _run_onnx(self.padim_path, padim_batch, self._padim)
            dist = mahalanobis_maps(embeddings.astype(np.float32), self._mean, self._cov_inv)
            score_map = upsample_and_smooth_numpy(dist)[0]
            min_s, max_s = self.score_scale
            p_drift = frozen_image_score(score_map, min_s, max_s)
            padim_on = True

        label_id = int(decide_class(p_drift, probs, threshold=self.threshold))
        return {
            "label_id": label_id,
            "label": ID_TO_LABEL[label_id],
            "probs": {ID_TO_LABEL[i]: float(probs[i]) for i in range(6)},
            "p_drift": float(p_drift),
            "threshold": float(self.threshold),
            "drift": label_id == DRIFT_CLASS,
            "padim": padim_on,
        }

    def predict_bytes(self, raw: bytes, lib: str | None = None) -> dict[str, Any]:
        """Décode des octets d'image puis :meth:`predict_pil`."""
        return self.predict_pil(decode_image_bytes(raw), lib=lib)


_RUNTIME: Runtime | None = None


def get_runtime() -> Runtime:
    """Singleton pour réutiliser les sessions d'un appel Lambda à l'autre."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime()
    return _RUNTIME


def set_runtime(runtime: Runtime | None) -> None:
    """Injecte un runtime (tests) ; ``None`` force un rechargement."""
    global _RUNTIME
    _RUNTIME = runtime


def parse_event(event: dict[str, Any] | str | None) -> dict[str, Any]:
    """Extrait ``image_b64`` / ``lib`` d'un event Lambda ou d'un dict direct.

    Parameters
    ----------
    event
        Event API Gateway (``body`` JSON) ou payload déjà parsé.

    Returns
    -------
    dict
        Au moins ``image_b64``.

    Raises
    ------
    PredictError
        Si le JSON ou l'image manque.
    """
    if event is None:
        raise PredictError("event vide")
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except json.JSONDecodeError as exc:
            raise PredictError("JSON invalide") from exc
    if not isinstance(event, dict):
        raise PredictError("event doit être un objet JSON")
    body = event.get("body", event)
    if isinstance(body, str):
        if event.get("isBase64Encoded"):
            try:
                body = base64.b64decode(body).decode("utf-8")
            except Exception as exc:
                raise PredictError("body base64 invalide") from exc
        try:
            body = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PredictError("body JSON invalide") from exc
    if not isinstance(body, dict):
        raise PredictError("body doit être un objet JSON")
    if not body.get("image_b64"):
        raise PredictError("champ image_b64 manquant")
    return body


def handler(event: dict[str, Any] | None, _context: Any = None) -> dict[str, Any]:
    """Handler Lambda (API Gateway proxy) : image base64 → prédiction JSON.

    Parameters
    ----------
    event
        Event HTTP ou dict ``{image_b64, lib?}``.
    _context
        Contexte Lambda (ignoré).

    Returns
    -------
    dict
        Réponse proxy ``statusCode`` / ``body``.
    """
    headers = {"Content-Type": "application/json"}
    try:
        body = parse_event(event)
        try:
            raw = base64.b64decode(body["image_b64"], validate=False)
        except Exception as exc:
            raise PredictError("image_b64 invalide") from exc
        lib = body.get("lib")
        result = get_runtime().predict_bytes(raw, lib=lib)
        return {"statusCode": 200, "headers": headers, "body": json.dumps(result)}
    except PredictError as exc:
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"error": str(exc)}),
        }
    except FileNotFoundError as exc:
        LOGGER.exception("artefact manquant")
        return {
            "statusCode": 503,
            "headers": headers,
            "body": json.dumps({"error": f"modèle introuvable : {exc}"}),
        }
    except Exception as exc:
        LOGGER.exception("inférence")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": str(exc)}),
        }
