# Changelog

## [Non publié]

- Decks : slide projet + schéma d'architecture (RH : 4 étapes métier ;
  technique : train vs serve). Moins de graphes de coût côté recruteur.
- GitHub Pages depuis `/docs` : copie des visuels sous `docs/pictures/`,
  index des 4 decks. Les liens README pointent vers github.io, pas le `.md`.
- Présentations Marp recruteur / technique, FR et EN : seuil protège-GOOD,
  graphes de coût, photos Unsplash. HTML via GitHub Actions au push.
- Déploiement AWS (CloudFormation, pas SAM) : stack billing `us-east-1`
  (alarme EstimatedCharges + budget 1 USD) **avant** ECR et Lambda.
  Function URL, 2 Go, 1 exécution concurrente. CLI
  `python -m valeo_qc.cli deploy --email …` (dry-run par défaut ;
  `--apply` exige un AWS CLI déjà présent, sans l'installer).
- API Lambda locale : handler API Gateway + image Docker (Python 3.12,
  onnxruntime, sans PyTorch). Sans pickle PaDiM ni min-max figé, l'API
  classe les 6 défauts connus. SAM n'est pas installé (étape AWS).
- Export ONNX : `classifier.onnx` (resnest50d, 97 Mo, max_abs=0 vs
  PyTorch) et `padim-backbone.onnx` (WRN-50-2, 95 Mo, max_abs 2e-6).
  La gaussienne PaDiM reste un pickle. Inférence onnxruntime dans un
  sous-processus (conflit DLL torch/CUDA sous Windows).
- Calibration du seuil : le val n'a pas de drift, donc le max PWA éteint
  PaDiM. Seuil exporté **protège-GOOD** = 0,611 (0 GOOD→drift). À 0,5
  l'officiel flag 13 GOOD (PWA 0,989 vs 0,999).
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
