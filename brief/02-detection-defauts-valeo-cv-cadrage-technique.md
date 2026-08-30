# Cadrage technique complémentaire — Détection de défauts Valeo (Challenge Data ENS #157)

*Complète `02-detection-defauts-valeo-cv.md`. Ce document reflète l'état réel du challenge (page consultée après connexion) et le scaffold de repo déjà généré. Objectif : donner à Cursor tout le contexte nécessaire pour reprendre l'implémentation sans redécouvrir ce qui est déjà tranché.*

---

## 1. Ce que le brief initial ne pouvait pas savoir

Le brief original a été rédigé avant consultation de la page du challenge (accès restreint). Voici ce qui est maintenant confirmé :

| Point | Brief initial | Réalité constatée |
|---|---|---|
| Volume des données | non précisé | **10 Mo à 1 Go** — pas de la "grosse donnée", pas besoin de stockage distribué complexe |
| Niveau du challenge | non précisé | Intermédiaire |
| Date de début | non précisée | 13 janvier 2026 |
| Saison | non précisée | Saison 2025 exceptionnellement prolongée jusqu'en 2026 ; le leaderboard actuel ne reflète que les résultats 2025 |
| Nombre d'images train | non précisé | 8278 |
| Nombre d'images test | non précisé | 1055 |
| Déséquilibre des classes | mentionné qualitativement | ratio **~91:1** entre la plus grande (Missing, 6472) et la plus petite classe connue (Boucle plate, 71) |
| Fichiers fournis par l'organisateur | non précisé | `x_train`, `y_train`, `x_test`, un exemple de soumission aléatoire, **et des "fichiers supplémentaires" (data-readers, baseline scripts, instructions)** |

Répartition exacte des classes (train) :

| Label | Classe | Effectif |
|---|---|---|
| 0 | GOOD | 1235 |
| 1 | Boucle plate | 71 |
| 2 | Lift-off blanc | 270 |
| 3 | Lift-off noir | 104 |
| 4 | Missing | 6472 |
| 5 | Short circuit MOS | 126 |
| 6 | drift (test uniquement) | — |

Le test contient une 7e classe `drift` : mélange d'anomalies de prise de vue et de défauts rares regroupés, absente du train par construction (open-set recognition, pas un simple oubli de labellisation).

Deux colonnes de métadonnées existent dans `output_train.csv` : `window` (zone d'inspection, 2 valeurs) et `lib` (composant, 4 valeurs) — elles expliquent une partie de la variabilité des prises de vue et sont probablement utiles en feature auxiliaire ou en stratification du split.

## 2. Inconnues à lever après connexion à la plateforme

Avant de coder plus loin que le scaffold actuel :

- **Lire en premier les "fichiers supplémentaires"** (data-readers, baseline scripts, instructions officielles) avant de réimplémenter quoi que ce soit — l'organisateur fournit peut-être déjà un loader d'images ou une partie du benchmark, inutile de dupliquer.
- **Matrice de coût réelle** : fournie sous forme d'image sur la page du challenge, pas de texte exploitable. `COST_MATRIX` dans `src/decision_logic.py` est actuellement un placeholder (coût uniforme hors diagonale) — à remplacer par les vraies valeurs (transcription manuelle depuis l'image).
- **Noms de colonnes réels des CSV** : le scaffold suppose `image_id`, `label`, `window`, `lib` et des fichiers `output_train.csv` / `output_test.csv` — à vérifier et corriger dans `src/preprocessing.py` dès le premier téléchargement.
- **Conditions d'utilisation / droits de republication** : à relire avant tout commit contenant ne serait-ce qu'un échantillon d'image (même pour un notebook d'exploration).
- **Format des labels de test** : probablement caché (soumission via la plateforme pour évaluation), à confirmer — conditionne l'intérêt réel du split de validation interne.

## 3. État du repo déjà scaffoldé

Structure générée (`projet-defauts-valeo-cv/`), à ouvrir directement dans Cursor :

```
projet-defauts-valeo-cv/
├── README.md
├── requirements.txt
├── .env.example
├── data/{raw,processed}/
├── src/
│   ├── preprocessing.py       # implémenté (split stratifié, poids de classes) — rotate_and_crop en TODO
│   ├── train_classifier.py    # scaffoldé (CLI, ResNet50, MLflow) — dataset/dataloader réel en TODO
│   ├── anomaly_detection.py   # scaffoldé (interface fit/predict) — implémentation PaDiM/PatchCore en TODO
│   ├── decision_logic.py      # implémenté ET testé — seuil et matrice de coût à recalibrer avec les vraies valeurs
│   └── export_model.py        # implémenté (export ONNX)
├── deployment/
│   ├── Dockerfile, requirements-lambda.txt
│   ├── lambda_handler.py      # scaffoldé — branchement anomaly_detection réel en TODO
│   └── infra/template.yaml    # template SAM minimal
├── scripts/download_data.sh   # vérifie la présence des fichiers, ne télécharge pas (auth requise)
└── tests/test_decision_logic.py  # 8 tests, passent en l'état
```

Statut par fichier :

| Fichier | Statut |
|---|---|
| `decision_logic.py` + ses tests | ✅ implémenté et validé |
| `preprocessing.py` | 🟡 partiellement implémenté — `rotate_and_crop` à écrire |
| `train_classifier.py` | 🟡 scaffoldé — dataloader + branchement réel à faire |
| `anomaly_detection.py` | 🟡 scaffoldé — méthode à choisir et implémenter |
| `export_model.py` | ✅ implémenté (dépend d'un checkpoint entraîné) |
| `lambda_handler.py` | 🟡 scaffoldé — utilise pour l'instant la confiance softmax comme proxy d'anomalie, à remplacer |

## 4. Décisions d'architecture actées (ne pas reproposer sans raison)

- **Cloud : AWS**, pas d'alternative à comparer — cohérent avec le brief initial et le volume modeste des données (10 Mo à 1 Go, pas besoin de stockage distribué type S3 multi-TB).
  - Entraînement : local si suffisant, sinon EC2 spot allumée seulement pendant l'entraînement.
  - Déploiement : Lambda (image conteneur ECR) + API Gateway, coût quasi nul au repos.
  - Suivi : MLflow, artifact store sur S3.
- **Modèle** : ResNet50 fine-tuné pour la classification (reprend le benchmark), détecteur d'anomalie séparé (PaDiM comme le benchmark, PatchCore ou score softmax calibré comme alternative à comparer).
- **Métrique principale** : coût métier selon la matrice fournie par l'organisateur — jamais l'accuracy globale seule (trompeuse avec un ratio ~91:1).
- **Format d'export** : ONNX (pas TorchScript) — choisi pour la compatibilité avec `onnxruntime` côté Lambda, image plus légère qu'un runtime PyTorch complet.

## 5. Conventions pour la suite (Cursor)

- Ne jamais committer `data/raw/*` ni `.env` — déjà couvert par `.gitignore`, à ne pas modifier sans raison.
- Toute fonction de `decision_logic.py` doit rester couverte par un test avant merge — c'est la seule brique jugée directement sur son comportement (cas limites de seuil notamment).
- `train_classifier.py` et `anomaly_detection.py` : compléter les `TODO` en place plutôt que réécrire les fichiers — la structure CLI/MLflow/docstrings est volontairement déjà posée.
- Toute comparaison chiffrée face au benchmark doit être reproductible via script, jamais un chiffre isolé dans le README.

## 6. Prochaines étapes concrètes, dans l'ordre

1. Se connecter à challengedata.ens.fr, télécharger `x_train`, `y_train`, `x_test`, l'exemple de soumission **et les fichiers supplémentaires**.
2. Lire les fichiers supplémentaires avant d'écrire du code — vérifier s'il existe déjà un data-reader ou un script baseline officiel.
3. Ajuster `src/preprocessing.py` aux noms de colonnes réels des CSV.
4. Transcrire la matrice de coût réelle (capture d'écran de la page du challenge) dans `src/decision_logic.py`.
5. Implémenter `rotate_and_crop` (calibrage visuel sur un échantillon d'images).
6. Brancher un vrai `Dataset`/`DataLoader` PyTorch dans `train_classifier.py`, lancer un premier entraînement baseline.
7. Implémenter `anomaly_detection.py` avec `anomalib` (PaDiM en premier, pour rester comparable au benchmark).
8. Calibrer le seuil final via `find_optimal_threshold` sur le split de validation interne.
9. Exporter en ONNX, tester `lambda_handler.py` en local avant tout déploiement.
10. Configurer l'alarme de facturation CloudWatch, puis déployer via le template SAM (`deployment/infra/template.yaml`).
