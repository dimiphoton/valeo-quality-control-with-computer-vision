"""Déploiement AWS : alarme de facturation d'abord, puis ECR + Lambda.

Pas de SAM (non installé, non validé). CloudFormation brut. ``--apply``
exige l'AWS CLI déjà présent — on ne l'installe pas ici.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from valeo_qc.preprocessing import PROJECT_ROOT

INFRA_DIR = PROJECT_ROOT / "deployment" / "infra"
BILLING_TEMPLATE = INFRA_DIR / "billing.yaml"
ECR_TEMPLATE = INFRA_DIR / "ecr.yaml"
API_TEMPLATE = INFRA_DIR / "api.yaml"
BILLING_STACK = "valeo-qc-billing"
ECR_STACK = "valeo-qc-ecr"
API_STACK = "valeo-qc-api"
BILLING_REGION = "us-east-1"
DEFAULT_API_REGION = "eu-west-3"
ECR_REPO = "valeo-qc"
IMAGE_TAG = "latest"
LAMBDA_IMAGE = "valeo-qc-lambda"

# Ordre imposé : la facturation avant toute ressource qui coûte.
STEP_ORDER = ("billing", "ecr", "api")

COST_ESTIMATE = {
    "at_rest_usd_month": 0.0,
    "note": (
        "Free tier : Lambda 1M req + 400k GB-s, ECR 500 MB-month, "
        "Budgets (2). Image ~400 MB, mémoire 2 Go, 1 exécution concurrente. "
        "Au repos ~ 0 USD. Alarme / budget a 1 USD."
    ),
    "alarm_usd": 1.0,
}

AWS_MISSING = (
    "AWS CLI introuvable. Pas d'installation automatique "
    "(brief/objectif.md : CLI/SAM à valider avant install). "
    "Dry-run (défaut) : python -m valeo_qc.cli deploy --email toi@exemple.fr"
)


class DeployError(RuntimeError):
    """Précondition de déploiement non remplie."""


def aws_cli() -> str | None:
    """Chemin de ``aws`` / ``aws.cmd`` si le CLI est déjà installé."""
    return shutil.which("aws") or shutil.which("aws.cmd")


def require_aws() -> str:
    """AWS CLI ou message pour l'installer soi-même."""
    path = aws_cli()
    if path is None:
        raise DeployError(AWS_MISSING)
    return path


def plan(
    *,
    email: str,
    api_region: str = DEFAULT_API_REGION,
    image_uri: str | None = None,
    account: str = "ACCOUNT",
    limit_usd: int = 1,
) -> list[dict[str, Any]]:
    """Étapes dans l'ordre alarme → ECR → Lambda (aucune exécution).

    Parameters
    ----------
    email
        Destinataire SNS / Budget.
    api_region
        Région de l'API (pas us-east-1 obligatoire).
    image_uri
        URI ECR complète. Défaut : ``{account}.dkr.ecr.{region}...``.
    account, limit_usd
        Compte (affichage) et plafond USD.

    Returns
    -------
    list[dict]
        Une entrée par étape, ``billing`` en premier.
    """
    uri = image_uri or (
        f"{account}.dkr.ecr.{api_region}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}"
    )
    return [
        {
            "step": "billing",
            "region": BILLING_REGION,
            "stack": BILLING_STACK,
            "template": str(BILLING_TEMPLATE.as_posix()),
            "reason": "Alarme CloudWatch + budget avant toute ressource payante",
            "params": {"AlertEmail": email, "MonthlyLimitUsd": str(limit_usd)},
        },
        {
            "step": "ecr",
            "region": api_region,
            "stack": ECR_STACK,
            "template": str(ECR_TEMPLATE.as_posix()),
            "reason": "Dépôt ECR après l'alarme, avant le push Docker",
            "params": {},
            "docker": [
                f"docker build -f deployment/Dockerfile -t {LAMBDA_IMAGE} .",
                f"aws ecr get-login-password --region {api_region}",
                f"docker tag {LAMBDA_IMAGE}:latest {uri}",
                f"docker push {uri}",
            ],
        },
        {
            "step": "api",
            "region": api_region,
            "stack": API_STACK,
            "template": str(API_TEMPLATE.as_posix()),
            "reason": "Lambda image + Function URL (après alarme + push)",
            "params": {"ImageUri": uri},
        },
    ]


def format_plan(steps: Sequence[dict[str, Any]]) -> str:
    """Texte dry-run pour le terminal."""
    lines = [
        "Déploiement Valeo QC — alarme de facturation EN PREMIER",
        COST_ESTIMATE["note"],
        "",
    ]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. [{step['step']}] {step['reason']}")
        lines.append(f"   région={step['region']} stack={step['stack']}")
        lines.append(f"   template={step['template']}")
        if step.get("docker"):
            for cmd in step["docker"]:
                lines.append(f"   $ {cmd}")
        lines.append("")
    lines.append("Puis : python -m valeo_qc.cli deploy --apply --email …")
    lines.append("(exige AWS CLI + identifiants + confirmation SNS par e-mail)")
    return "\n".join(lines).rstrip() + "\n"


def _cfn_deploy(
    aws: str,
    *,
    stack: str,
    template: Path,
    region: str,
    params: dict[str, str],
) -> None:
    """``aws cloudformation deploy`` (capabilités IAM si besoin)."""
    args = [
        aws,
        "cloudformation",
        "deploy",
        "--stack-name",
        stack,
        "--template-file",
        str(template),
        "--region",
        region,
        "--no-fail-on-empty-changeset",
    ]
    if params:
        joined = [f"{key}={value}" for key, value in params.items()]
        args.extend(["--parameter-overrides", *joined])
    if stack == API_STACK:
        args.extend(["--capabilities", "CAPABILITY_NAMED_IAM"])
    subprocess.run(args, check=True)


def billing_stack_exists(aws: str) -> bool:
    """True si ``valeo-qc-billing`` est déjà là (us-east-1)."""
    result = subprocess.run(
        [
            aws,
            "cloudformation",
            "describe-stacks",
            "--stack-name",
            BILLING_STACK,
            "--region",
            BILLING_REGION,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def apply(
    *,
    email: str,
    api_region: str = DEFAULT_API_REGION,
    image_uri: str | None = None,
    limit_usd: int = 1,
    skip_docker: bool = False,
) -> dict[str, Any]:
    """Déploie billing puis API. Refuse l'API si l'alarme n'existe pas.

    Parameters
    ----------
    email, api_region, image_uri, limit_usd
        Comme :func:`plan`.
    skip_docker
        Ne pas builder/pusher (image déjà dans ECR).

    Returns
    -------
    dict
        Stacks touchées.

    Raises
    ------
    DeployError
        CLI absent, ou tentative d'API sans stack billing.
    """
    aws = require_aws()
    if not BILLING_TEMPLATE.is_file() or not API_TEMPLATE.is_file() or not ECR_TEMPLATE.is_file():
        raise DeployError("templates CloudFormation introuvables")
    _cfn_deploy(
        aws,
        stack=BILLING_STACK,
        template=BILLING_TEMPLATE,
        region=BILLING_REGION,
        params={"AlertEmail": email, "MonthlyLimitUsd": str(limit_usd)},
    )
    if not billing_stack_exists(aws):
        raise DeployError("stack billing absente après deploy — on n'enchaîne pas l'API")
    _cfn_deploy(
        aws,
        stack=ECR_STACK,
        template=ECR_TEMPLATE,
        region=api_region,
        params={},
    )
    uri = image_uri or _default_image_uri(aws, api_region)
    if not skip_docker:
        _push_image(aws, api_region, uri)
    _cfn_deploy(
        aws,
        stack=API_STACK,
        template=API_TEMPLATE,
        region=api_region,
        params={"ImageUri": uri},
    )
    return {
        "billing": BILLING_STACK,
        "ecr": ECR_STACK,
        "api": API_STACK,
        "image_uri": uri,
    }


def _default_image_uri(aws: str, region: str) -> str:
    """Compte courant + URI ECR."""
    ident = subprocess.run(
        [aws, "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        check=True,
        capture_output=True,
        text=True,
    )
    account = ident.stdout.strip()
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}"


def _push_image(aws: str, region: str, uri: str) -> None:
    """Build + login ECR + push (Docker déjà utilisé en local)."""
    subprocess.run(
        ["docker", "build", "-f", "deployment/Dockerfile", "-t", LAMBDA_IMAGE, "."],
        check=True,
        cwd=PROJECT_ROOT,
    )
    login = subprocess.run(
        [aws, "ecr", "get-login-password", "--region", region],
        check=True,
        capture_output=True,
        text=True,
    )
    host = uri.split("/", 1)[0]
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", host],
        input=login.stdout,
        check=True,
        text=True,
    )
    subprocess.run(["docker", "tag", f"{LAMBDA_IMAGE}:latest", uri], check=True)
    subprocess.run(["docker", "push", uri], check=True)


def deploy(
    *,
    email: str,
    dry_run: bool = True,
    api_region: str = DEFAULT_API_REGION,
    image_uri: str | None = None,
    limit_usd: int = 1,
    skip_docker: bool = False,
) -> str:
    """Dry-run (défaut) ou ``--apply`` réel.

    Parameters
    ----------
    email
        Obligatoire (alarme SNS).
    dry_run
        Si True, n'appelle pas AWS.
    api_region, image_uri, limit_usd, skip_docker
        Voir :func:`apply`.

    Returns
    -------
    str
        Compte-rendu texte.

    Raises
    ------
    ValueError
        E-mail vide.
    DeployError
        ``--apply`` sans AWS CLI.
    """
    if not email or "@" not in email:
        raise ValueError("un e-mail --email est requis pour l'alarme SNS")
    steps = plan(
        email=email,
        api_region=api_region,
        image_uri=image_uri,
        limit_usd=limit_usd,
    )
    if steps[0]["step"] != "billing":
        raise DeployError("invariant cassé : billing doit être la première étape")
    text = format_plan(steps)
    if dry_run:
        return text
    result = apply(
        email=email,
        api_region=api_region,
        image_uri=image_uri,
        limit_usd=limit_usd,
        skip_docker=skip_docker,
    )
    return text + f"\nappliqué : {result}\n"
