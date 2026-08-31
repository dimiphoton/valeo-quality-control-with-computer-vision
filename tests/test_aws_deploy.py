"""Tests du plan de déploiement AWS — pas d'appel réseau, pas d'AWS CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from valeo_qc.aws_deploy import (
    API_TEMPLATE,
    BILLING_REGION,
    BILLING_STACK,
    BILLING_TEMPLATE,
    ECR_TEMPLATE,
    AWS_MISSING,
    DeployError,
    apply,
    deploy,
    format_plan,
    plan,
    require_aws,
)


def test_plan_commence_par_la_facturation() -> None:
    """L'alarme us-east-1 est la première étape, avant ECR et Lambda."""
    steps = plan(email="demo@example.com")
    assert [s["step"] for s in steps] == ["billing", "ecr", "api"]
    assert steps[0]["region"] == BILLING_REGION
    assert steps[0]["region"] == "us-east-1"


def test_templates_alarme_et_lambda() -> None:
    """Les YAML versionnés contiennent budget, alarme billing, Lambda image."""
    billing = BILLING_TEMPLATE.read_text(encoding="utf-8")
    assert "AWS::CloudWatch::Alarm" in billing
    assert "AWS::Budgets::Budget" in billing
    assert "AWS/Billing" in billing
    assert "EstimatedCharges" in billing
    ecr = ECR_TEMPLATE.read_text(encoding="utf-8")
    assert "AWS::ECR::Repository" in ecr
    api = API_TEMPLATE.read_text(encoding="utf-8")
    assert "AWS::Lambda::Function" in api
    assert "PackageType: Image" in api
    assert "ReservedConcurrentExecutions: 1" in api
    assert "AWS::Lambda::Url" in api
    assert "AWS::ECR::Repository" not in api


def test_dry_run_ne_leve_pas_sans_aws() -> None:
    """Sans CLI AWS, le dry-run affiche le plan et s'arrête."""
    text = deploy(email="demo@example.com", dry_run=True)
    assert "EN PREMIER" in text
    assert "valeo-qc-billing" in text
    assert "--apply" in text


def test_apply_sans_aws_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """--apply sans binaire aws explique qu'on n'installe pas le CLI."""
    monkeypatch.setattr("valeo_qc.aws_deploy.aws_cli", lambda: None)
    with pytest.raises(DeployError, match="AWS CLI"):
        deploy(email="demo@example.com", dry_run=False)
    with pytest.raises(DeployError, match="installation automatique"):
        require_aws()
    assert "valider" in AWS_MISSING.lower() or "Dry-run" in AWS_MISSING


def test_apply_n_enchaine_pas_sans_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la stack billing n'est pas là, ECR et Lambda ne partent pas."""
    order: list[str] = []
    monkeypatch.setattr("valeo_qc.aws_deploy.require_aws", lambda: "aws")

    def fake_cfn(*_args: object, **kwargs: object) -> None:
        order.append(str(kwargs["stack"]))

    monkeypatch.setattr("valeo_qc.aws_deploy._cfn_deploy", fake_cfn)
    monkeypatch.setattr("valeo_qc.aws_deploy.billing_stack_exists", lambda _aws: False)
    with pytest.raises(DeployError, match="billing"):
        apply(email="a@b.c", skip_docker=True)
    assert order == [BILLING_STACK]


def test_apply_ordre_billing_ecr_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordre imposé une fois l'alarme en place."""
    order: list[str] = []
    monkeypatch.setattr("valeo_qc.aws_deploy.require_aws", lambda: "aws")

    def fake_cfn(*_args: object, **kwargs: object) -> None:
        order.append(str(kwargs["stack"]))

    monkeypatch.setattr("valeo_qc.aws_deploy._cfn_deploy", fake_cfn)
    monkeypatch.setattr("valeo_qc.aws_deploy.billing_stack_exists", lambda _aws: True)
    monkeypatch.setattr(
        "valeo_qc.aws_deploy._default_image_uri",
        lambda _aws, _region: "123.dkr.ecr.eu-west-3.amazonaws.com/valeo-qc:latest",
    )
    monkeypatch.setattr("valeo_qc.aws_deploy._push_image", lambda *_a, **_k: None)
    result = apply(email="a@b.c")
    assert order == ["valeo-qc-billing", "valeo-qc-ecr", "valeo-qc-api"]
    assert result["billing"] == "valeo-qc-billing"


def test_email_obligatoire() -> None:
    """Pas d'alarme sans destinataire SNS."""
    with pytest.raises(ValueError, match="e-mail"):
        deploy(email="", dry_run=True)


def test_format_plan_ordre_numerote() -> None:
    """Le texte numérote billing=1."""
    text = format_plan(plan(email="a@b.c"))
    assert text.index("[billing]") < text.index("[ecr]") < text.index("[api]")
    assert Path(BILLING_TEMPLATE).name in text
