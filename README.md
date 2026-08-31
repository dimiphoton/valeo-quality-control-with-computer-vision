# Valeo quality control with computer vision

| | |
|---|---|
| **Role** | Machine learning |
| **Domain** | Industry / quality control |
| **Stack** | ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonwebservices&logoColor=white) |
| **Level** | Intermediate |
| **Status** | v1.0 |

Machine learning · Industry / quality control · Python / PyTorch / ONNX / AWS

## Objective

Classify known defects on Valeo electronic-component images and flag
out-of-distribution frames (`drift`), with the decision rule fitted to the
challenge cost matrix rather than raw accuracy. The trained pipeline will
be served as a serverless inference API.

The dataset comes from Challenge Data ENS #157,
[Improving Industrial Quality Control with Computer Vision](https://challengedata.ens.fr/participants/challenges/157/)
(Valeo). Training set: 8,278 images, six classes, ~91:1 imbalance.
The test set (1,055 images) adds a seventh `drift` class that is absent
from training (open-set recognition).

## Data

Download the files from the challenge page after registering (section
**Files**). Do not commit them: `/data/` is gitignored (licence and size,
including a 1.2 GB PaDiM pickle).

Expected layout after unzipping:

```
data/raw/
  input_train/                 # 8,278 PNG
  input_test_1a4aqAg/input_test/   # 1,055 PNG
  Y_train_eVW9jym.csv          # filename, window, lib, Label (strings)
  Y_random_nKwalR1.csv         # submission format (integer labels 0–6)
  Supp_files/
    Notebook_ENS.ipynb         # official benchmark
    model.py
    Classifier.pt
    PADIM.pkl
    win_and_lib.csv
```

Train labels are strings (`GOOD`, `Missing`, …). Submissions use integers
(`GOOD=0` … `Drift=6`). Test labels are hidden: score via the platform
(2 submissions / 24 h).

After `python -m valeo_qc.cli prepare` (idempotent, skips existing crops):

```
data/processed/
  train/                 # 8,278 cropped PNG (same filenames)
  test/                  # 1,055 cropped PNG
  split.csv              # train/val 80/20, stratified on Label
  class_weights.json     # inverse-frequency weights on the train split only
```

## Result

Images are rotated and cropped into `data/processed/`. The official
classifier is a **resnest50d** (not ResNet50), evaluated on the
stratified val split (1,655 images, no `drift`):

| Model | Val accuracy | Macro F1 |
|---|---|---|
| Official `Classifier.pt` | 99.8 % | 0.973 |
| Ours (15 epochs, class weights) | 99.5 % | 0.960 (best epoch 4) |

The rare class `Boucle plate` is the weak spot (recall 0.86 on both).

PaDiM (WideResNet-50-2, 128 px, 550 dims) scores the val split without
`drift`. Image-level scores are min-max scaled on the scored set
(as in the official notebook). Raw Mahalanobis means:

| Model | Overall | Missing | GOOD | frac > 0.5 |
|---|---|---|---|---|
| Official `PADIM.pkl` | 144 | 102 | 262 | 3.4 % |
| Ours (train split, ridge 0.01) | 148 | 104 | 273 | 2.2 % |

The Gaussian is dominated by **Missing** (~78 % of train), not by GOOD.

Decision rule on val (no `drift` labels — every PaDiM flag is a false
positive). PWA = 1 − penalty / (n × 10 000):

| Pipeline | Threshold | PWA | False drift (of which GOOD) |
|---|---|---|---|
| Official, classifier only | 1.00 | 1.000 | 0 |
| Official, notebook 0.5 | 0.50 | 0.989 | 57 (13 GOOD) |
| Official, **protect GOOD** (exported) | 0.611 | 0.999 | 20 (0 GOOD) |
| Ours, notebook 0.5 | 0.50 | 0.994 | 37 (3 GOOD) |

At 0.5 the official PaDiM pays 13 × 10 000 for GOOD→drift. The exported
threshold is the lowest that never flags a val GOOD.

ONNX export (`python -m valeo_qc.cli export`, opset 18), checked against
PyTorch on a dummy tensor:

| Graph | Size | max abs vs PyTorch |
|---|---|---|
| `models/classifier.onnx` (resnest50d, softmax) | 97 MB | 0 |
| `models/padim-backbone.onnx` (WRN-50-2, 550×32×32) | 95 MB | 2×10⁻⁶ |

The PaDiM Gaussian (mean/cov, ~1.2 GB) stays a pickle — it is not a
network. Single-image inference must freeze the notebook's batch min-max
(otherwise the score is always 1). The local Lambda image runs the
classifier without PyTorch; PaDiM is on only if the pickle and a frozen
`models/score-scale.json` are present.

AWS deploy is CloudFormation (not SAM). Billing alarm in `us-east-1`
**before** ECR/Lambda. Default is dry-run; `--apply` needs an AWS CLI
already on the machine (this repo does not install it).

| Resource | Role | Idle cost |
|---|---|---|
| CloudWatch billing alarm + AWS Budget ($1) | Gate before any paid stack | ~$0 (free-tier budgets) |
| ECR `valeo-qc` (~400 MB image) | Container for Lambda | ~$0 under 500 MB-month |
| Lambda 2 GB, timeout 60 s, reserved concurrency 1 | Inference | ~$0 at rest (1 M req + 400 k GB-s) |
| Function URL (no API Gateway) | Public POST, demo only | $0 extra |

Turn on **Receive Billing Alerts** in the Billing console once; confirm
the SNS email. The Function URL has `AuthType: NONE` — demo, not a
production lock.

## Reproduce

```bash
python -m pip install -e ".[dev]"
pytest
python -m valeo_qc.cli prepare
python -m valeo_qc.cli eval-classifier
python -m valeo_qc.cli train-classifier
python -m valeo_qc.cli eval-padim
python -m valeo_qc.cli train-padim
python -m valeo_qc.cli calibrate
python -m valeo_qc.cli export
python -m valeo_qc.cli predict path/to/image.png
python -m valeo_qc.cli deploy --email you@example.com
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Local Lambda (Docker, after `export`):

```bash
docker build -f deployment/Dockerfile -t valeo-qc-lambda .
docker run --rm -p 9000:8080 valeo-qc-lambda
# POST http://localhost:9000/2015-03-31/functions/function/invocations
# body: {"image_b64": "<base64 PNG>"}
```

AWS (prints the plan; does not call the account unless `--apply`):

```bash
python -m valeo_qc.cli deploy --email you@example.com
# 1. billing stack in us-east-1 (alarm + $1 budget)
# 2. ECR, then docker push
# 3. Lambda image + Function URL (eu-west-3)
# python -m valeo_qc.cli deploy --email you@example.com --apply
```

`--apply` requires AWS CLI + credentials. It deploys billing first and
aborts before ECR/Lambda if that stack is missing. Confirm the SNS
subscription in your inbox.

`prepare` writes cropped PNGs to `data/processed/` (never overwrites
`data/raw/`). `eval-classifier` / `eval-padim` score the official
checkpoints on val. `train-classifier` and `train-padim` log to local
MLflow (SQLite). `calibrate` fuses both models, sweeps the cost
threshold, and writes `models/threshold.json`. `export` writes
`models/classifier.onnx`, `models/padim-backbone.onnx`, and
`models/onnx-manifest.json` (onnxruntime vs PyTorch check). `predict`
runs the same runtime as the Lambda handler. `deploy` prints the
CloudFormation order (billing first). Re-runs of `prepare`
skip existing crops (`--overwrite` to force).

## Repo structure

```
brief/                 # identity, objective, original briefs (French)
src/valeo_qc/          # preprocessing, models, ONNX export, Lambda runtime, AWS plan
deployment/            # Dockerfile, handler, CloudFormation (billing then ECR/API)
tests/
docs/presentations/    # Marp sources (recruiter + technical, FR/EN)
```

`ROADMAP.md` and `JOURNAL.md` are kept in French.

## Presentations

Two audiences × two languages (Marp theme `portfolio`, HTML on GitHub Pages).
The recruiter deck is a ~6-minute pitch; the technical deck is a ~12-minute
deep dive. The recruiter deck asks whether a camera can scrap fewer good parts.
The technical deck asks how to fuse the classifier and PaDiM when
validation has no drift labels. HTML is built by GitHub Actions on push
to `main` (do not run Marp locally).

- [Recruiter overview (EN)](docs/slides/presentation-recruteur-en.html)
- [Technical deep dive (EN)](docs/slides/presentation-technique-en.html)
- [Présentation grand public (FR)](docs/slides/presentation-recruteur-fr.html)
- [Présentation technique (FR)](docs/slides/presentation-technique-fr.html)
