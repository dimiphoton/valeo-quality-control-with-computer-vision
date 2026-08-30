# Changelog

## [Non publié]

- PaDiM : reproduction du pickle officiel (WRN-50-2, pas anomalib) et
  réentraînement sur le split train. Val : scores bruts moyens 144
  (officiel) / 148 (nôtre) ; Missing reste la classe la plus « in-distribution ».
- Baseline classifieur : le checkpoint officiel est un **resnest50d**
  (val acc 99,8 %, F1 macro 0,973). Réentraînement 15 époques + poids
  de classes (meilleur F1 0,960 à l'époque 4). Journal MLflow (SQLite).
- Préparation des images : split stratifié 80/20, poids de classes sur
  le split train, `rotate_and_crop` vers `data/processed/` (`valeo-qc prepare`).
- Cadrage : identité Machine learning / industrie, package `valeo_qc`,
  logique de décision (matrice de coût officielle) et prétraitement testés.
- `/data/` entier ignoré par git (licence et volume du challenge).
- Initialisation du projet à partir du template portfolio.
