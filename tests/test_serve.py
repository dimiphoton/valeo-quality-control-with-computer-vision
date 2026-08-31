"""Tests du runtime Lambda — ONNX minuscules, pas de PyTorch."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from valeo_qc.anomaly_numpy import frozen_image_score
from valeo_qc.serve import (
    Runtime,
    handler,
    parse_event,
    set_runtime,
)


def _save_onnx(graph: onnx.GraphProto, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _classifier_onnx(path: Path, size: int = 16) -> Path:
    """GAP + Gemm : canal rouge fort → classe Missing (4)."""
    weight = np.zeros((3, 6), dtype=np.float32)
    weight[0, 4] = 8.0
    bias = np.zeros((6,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["gap"]),
            helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
        ],
        "tiny_clf",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, size, size])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", 6])],
        [
            numpy_helper.from_array(weight, name="W"),
            numpy_helper.from_array(bias, name="B"),
        ],
    )
    return _save_onnx(graph, path)


def _padim_onnx(path: Path, size: int = 8, channels: int = 4, spatial: int = 4) -> Path:
    """Conv 1×1 + pool → embeddings ``(N, 4, 4, 4)``."""
    kernel = size // spatial
    weight = np.ones((channels, 3, 1, 1), dtype=np.float32) / 3.0
    bias = np.zeros((channels,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["image", "W", "B"], ["conv"], kernel_shape=[1, 1]
            ),
            helper.make_node(
                "AveragePool",
                ["conv"],
                ["embeddings"],
                kernel_shape=[kernel, kernel],
                strides=[kernel, kernel],
            ),
        ],
        "tiny_padim",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, size, size])],
        [
            helper.make_tensor_value_info(
                "embeddings", TensorProto.FLOAT, ["N", channels, spatial, spatial]
            )
        ],
        [
            numpy_helper.from_array(weight, name="W"),
            numpy_helper.from_array(bias, name="B"),
        ],
    )
    return _save_onnx(graph, path)


def _png_b64(color: tuple[int, int, int] = (255, 0, 0), size: int = 32) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_frozen_image_score_clip() -> None:
    """Sans figé, une carte seule min-maxée vaudrait 1 ; ici on clippe."""
    plane = np.array([[2.0, 8.0], [2.0, 2.0]])
    assert frozen_image_score(plane, min_score=0.0, max_score=10.0) == pytest.approx(0.8)
    assert frozen_image_score(plane, min_score=0.0, max_score=4.0) == pytest.approx(1.0)


def test_parse_event_direct_et_proxy() -> None:
    """Dict direct ou body API Gateway."""
    assert parse_event({"image_b64": "abc"})["image_b64"] == "abc"
    wrapped = {"body": json.dumps({"image_b64": "xyz", "lib": "Die01"})}
    parsed = parse_event(wrapped)
    assert parsed["lib"] == "Die01"
    with pytest.raises(Exception, match="image_b64"):
        parse_event({"body": "{}"})


def test_handler_400_sans_image() -> None:
    """Pas d'image → 400, sans charger de modèle."""
    response = handler({"body": "{}"}, None)
    assert response["statusCode"] == 400
    assert "image_b64" in json.loads(response["body"])["error"]


def test_runtime_classifieur_only_rouge_missing(tmp_path: Path) -> None:
    """Image rouge → Missing ; sans stats PaDiM, p_drift = 0."""
    clf = _classifier_onnx(tmp_path / "clf.onnx")
    runtime = Runtime(
        tmp_path,
        classifier_onnx=clf,
        threshold=0.5,
        score_scale=None,
        cls_size=16,
    )
    image = Image.new("RGB", (40, 40), color=(255, 0, 0))
    result = runtime.predict_pil(image)
    assert result["label"] == "Missing"
    assert result["label_id"] == 4
    assert result["padim"] is False
    assert result["p_drift"] == 0.0
    assert result["drift"] is False


def test_runtime_padim_peut_flagger_drift(tmp_path: Path) -> None:
    """Échelle figée très serrée → score clipé à 1 → drift."""
    import pickle

    clf = _classifier_onnx(tmp_path / "clf.onnx")
    pad = _padim_onnx(tmp_path / "pad.onnx")
    channels, hw = 4, 16
    mean = np.zeros((channels, hw), dtype=np.float32)
    cov = np.stack([np.eye(channels, dtype=np.float32)] * hw, axis=-1)
    stats = tmp_path / "stats.pkl"
    with stats.open("wb") as handle:
        pickle.dump({"mean": mean, "cov": cov, "idx": list(range(channels))}, handle)
    runtime = Runtime(
        tmp_path,
        classifier_onnx=clf,
        padim_onnx=pad,
        padim_stats=stats,
        threshold=0.5,
        score_scale=(0.0, 1e-9),
        cls_size=16,
        padim_size=8,
    )
    image = Image.new("RGB", (40, 40), color=(255, 0, 0))
    result = runtime.predict_pil(image)
    assert result["padim"] is True
    assert result["drift"] is True
    assert result["label"] == "Drift"


def test_handler_200_avec_runtime_injecte(tmp_path: Path) -> None:
    """Event JSON → 200 et label Missing."""
    clf = _classifier_onnx(tmp_path / "clf.onnx")
    runtime = Runtime(tmp_path, classifier_onnx=clf, threshold=0.5, cls_size=16)
    set_runtime(runtime)
    try:
        event = {"image_b64": _png_b64()}
        response = handler(event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["label"] == "Missing"
    finally:
        set_runtime(None)
