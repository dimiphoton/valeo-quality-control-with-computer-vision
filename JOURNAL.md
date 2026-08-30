# Journal de développement

## 2026-08-30 — Détecteur PaDiM

- Pas d'anomalib : le notebook officiel est un PaDiM maison (WideResNet-50-2,
  128 px, d=550, seed 1024). ``PADIM.pkl`` = moyenne + cov (550, 550, 1024).
- Officiel sur val (1 655, pas de drift) : score brut moyen 144, p95 379.
  **Missing** ~102, **GOOD** ~262 — le Gaussian est calé sur la classe
  majoritaire, pas sur les pièces saines.
- Réentraînement sur le split train (ridge 0,01, GTX 980 Ti) : même
  classement des classes, raw_mean 148. Checkpoint ``models/padim-best.pkl``.
- MLflow : expérience `valeo-qc-padim`.

## 2026-08-30 — Baseline classifieur

- `Classifier.pt` officiel = timm **resnest50d** (pas ResNet50), 15 époques.
  Val (1 655 images, pas de drift) : acc 99,8 %, F1 macro 0,973. Rappel
  `Boucle plate` 0,86 — le point faible.
- Réentraînement local (poids de classes, ImageNet, 15 époques, GTX 980 Ti) :
  meilleur F1 0,960 à l'époque 4. On garde l'officiel comme référence.
- MLflow en SQLite (`mlflow.db`) : expérience `valeo-qc-classifier`.
  PyTorch + timm + MLflow installés.

## 2026-08-30 — Préparation des images

- Split stratifié 80/20 (6 623 train / 1 655 val), toutes les classes
  représentées. Poids inverses à la fréquence, calculés sur le train
  seulement (`Missing` pèse ~0,03, `Boucle plate` ~2,3).
- 8 278 PNG train + 1 055 test recadrés dans `data/processed/` (0 manquant).
  `raw/` intact. Commande : `python -m valeo_qc.cli prepare`.

## 2026-08-30 — Cadrage

- Identité posée : Machine learning · Industrie / contrôle qualité ·
  Python / PyTorch / ONNX / AWS.
- Package `valeo_qc` : matrice de coût et PWA du notebook officiel,
  `rotate_and_crop` vers `processed/` (jamais `raw/`), split stratifié.
- 14 tests pytest verts. Données du challenge hors git (`/data/`).
- PyTorch / MLflow / anomalib non installés (prochaine étape classifieur).

## 2026-08-30 — Initialisation du projet

- Repo créé à partir du template portfolio.
