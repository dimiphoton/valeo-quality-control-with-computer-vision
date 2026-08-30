# Valeo quality control with computer vision

| | |
|---|---|
| **Role** | Machine learning |
| **Domain** | Industry / quality control |
| **Stack** | ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonwebservices&logoColor=white) |
| **Level** | Intermediate |
| **Status** | in progress |

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

Images are rotated and cropped into `data/processed/` (never overwriting
`data/raw/`). A stratified 80/20 train/val split and class weights
(computed on the train split only, given the ~91:1 imbalance) are ready
for the classifier baseline. PaDiM, the cost-calibrated decision, ONNX
export, and the Lambda API come next (see `ROADMAP.md`).

## Reproduce

```bash
python -m pip install -e ".[dev]"
pytest
python -m valeo_qc.cli prepare
```

`prepare` writes the stratified split, class weights, and cropped PNGs
to `data/processed/`. It never overwrites `data/raw/`. Re-runs skip
crops that already exist (`--overwrite` to force).

## Repo structure

```
brief/                 # identity, objective, original briefs (French)
src/valeo_qc/          # preprocessing, decision logic, later training
tests/
docs/presentations/    # Marp sources (recruiter + technical, FR/EN)
```

`ROADMAP.md` and `JOURNAL.md` are kept in French.

## Presentations

Two audiences × two languages (Marp theme `portfolio`, HTML on GitHub Pages).
The recruiter deck is a ~6-minute pitch; the technical deck is a ~12-minute
deep dive. They may diverge a lot — the bar is attractive and informative
for each audience, not a mirrored pair of slides.

- [Recruiter overview (EN)](docs/slides/presentation-recruteur-en.html)
- [Technical deep dive (EN)](docs/slides/presentation-technique-en.html)
- [Présentation grand public (FR)](docs/slides/presentation-recruteur-fr.html)
- [Présentation technique (FR)](docs/slides/presentation-technique-fr.html)
