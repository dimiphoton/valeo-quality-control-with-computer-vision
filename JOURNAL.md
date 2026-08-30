# Journal de développement

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
