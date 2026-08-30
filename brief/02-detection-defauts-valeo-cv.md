# Détection de défauts industriels et reconnaissance d'anomalies

*Projet computer vision — Challenge Data ENS / Valeo, avec déploiement cloud*

## Contexte et problématique

Le contrôle qualité industriel repose de plus en plus sur la vision par ordinateur pour détecter automatiquement les pièces défectueuses en sortie de ligne de production, en complément — voire en remplacement partiel — de l'inspection humaine. Le défi posé par Valeo (Challenge Data, ENS) porte sur des images de composants électroniques, avec deux difficultés combinées : une forte imbalance entre classes de défauts, et la nécessité de détecter des images "hors distribution" (classe *drift*) que le modèle n'a jamais vues à l'entraînement.

Ce projet répond à la question :

> Peut-on construire un système qui classe fiablement les défauts connus tout en détectant les anomalies inédites, avec une logique de décision calibrée sur le coût réel des erreurs plutôt que sur l'accuracy brute ?

## Objectif

Reproduire puis améliorer le pipeline de référence (classification + détection d'anomalie) du challenge, calibrer la logique de décision sur la matrice de coût fournie par l'organisateur, et déployer le modèle final comme une API d'inférence serverless sur AWS, dans les limites du free tier.

## Compétences démontrées

- Deep learning / vision par ordinateur (fine-tuning d'un classifieur CNN)
- Gestion du déséquilibre de classes (pondération de la loss, métriques adaptées : F1 macro, matrice de confusion)
- Détection d'anomalies / reconnaissance en monde ouvert (open-set recognition)
- Évaluation orientée business : calibration d'une logique de décision sur une matrice de coût réelle
- MLOps : suivi d'expériences (MLflow), export de modèle léger, conteneurisation
- Cloud AWS : S3, Lambda, API Gateway, ECR, maîtrise des coûts (free tier, alarme de facturation)
- Rigueur expérimentale : comparaison chiffrée et reproductible face à un benchmark de référence

## Approche et choix techniques

1. **Préparation** : split train/validation interne, indispensable si les labels de test du challenge sont cachés (probable — à vérifier après connexion à la plateforme).
2. **Baseline honnête** : reproduction simplifiée du benchmark fourni (classifieur type ResNet50 + méthode d'anomalie type PaDiM), pour disposer d'un point de comparaison chiffré.
3. **Amélioration ciblée** :
   - Classification : pondération de la loss ou rééquilibrage, évalués sur des métriques par classe (pas l'accuracy globale, trompeuse avec un ratio ~90:1 entre classes).
   - Anomalie : test d'une méthode alternative au benchmark (ex. PatchCore, ou un score de confiance softmax calibré) pour la détection de la classe *drift*.
4. **Logique de décision** : fusion classification + anomalie calibrée explicitement sur la matrice de coût du challenge — documenter pourquoi le seuil choisi est le bon, pas seulement quel seuil a été choisi.
5. **Export** : modèle final exporté en format léger (ONNX ou TorchScript) pour une inférence rapide et peu coûteuse en production.

## Architecture cloud (AWS, budget free tier)

- **S3** : stockage des données prétraitées et des artefacts de modèle (bucket versionné).
- **Entraînement** : local si suffisant, sinon instance EC2 spot activée uniquement pendant l'entraînement — jamais laissée allumée entre deux sessions.
- **Suivi d'expériences** : MLflow, avec artifact store pointant vers S3.
- **Déploiement** : Lambda (image conteneur via ECR) derrière API Gateway — coût quasi nul au repos, compatible free tier, permet une démo vivante réutilisable en entretien.
- **Prérequis impératif avant tout déploiement** : alarme de facturation CloudWatch configurée dès la création du compte.

## Source de données

Dataset du challenge *"Improving Industrial Quality Control with Computer Vision"* (Valeo, Challenge Data ENS, [challengedata.ens.fr/challenges/157](https://challengedata.ens.fr/challenges/157)). **Avant toute chose : vérifier les conditions d'utilisation du challenge.** Les datasets de cette plateforme ne sont généralement pas libres de republication — les images brutes ne doivent probablement pas être commitées sur un repo public. Documenter dans le README comment récupérer les données soi-même plutôt que de les inclure.

## Livrables attendus

1. Scripts de préparation des données et de split de validation interne.
2. Reproduction documentée du benchmark de référence, comme point de comparaison chiffré.
3. Version améliorée (gestion du déséquilibre + méthode d'anomalie alternative), avec comparaison chiffrée face au benchmark.
4. Logique de décision documentée et calibrée sur la matrice de coût du challenge.
5. Export du modèle en format léger (ONNX ou TorchScript).
6. Déploiement Lambda + API Gateway fonctionnel, avec un exemple d'appel (image en entrée, prédiction en sortie).
7. Suivi MLflow de toutes les expériences, baseline comme versions améliorées.
8. README complet : contexte, méthodologie, résultats chiffrés vs benchmark, limites, coût AWS estimé.

## Structure de repo attendue

```
projet-defauts-valeo-cv/
├── README.md
├── data/
│   ├── raw/            # non commité — voir .gitignore
│   └── processed/
├── src/
│   ├── preprocessing.py
│   ├── train_classifier.py
│   ├── anomaly_detection.py
│   ├── decision_logic.py
│   └── export_model.py
├── deployment/
│   ├── Dockerfile
│   ├── lambda_handler.py
│   └── infra/           # template SAM ou équivalent léger
├── notebooks/
├── tests/
│   └── test_decision_logic.py
└── requirements.txt
```

## Règles strictes de professionnalisme

- Environnement figé et reproductible (`requirements.txt` versionné) ; le projet s'installe et tourne via une seule commande documentée.
- Aucune donnée brute du challenge, secret, ni identifiant AWS commité dans le repo — `.gitignore` strict, variables d'environnement pour toute configuration sensible.
- Commits atomiques avec messages conventionnels (`feat:`, `fix:`, `docs:`, `test:`...).
- Toute fonction de la logique de décision documentée par une docstring précisant entrées, sorties et hypothèses.
- Au moins un test unitaire sur la logique de décision (ex. vérifier le comportement aux cas limites du seuil).
- Toute comparaison avec le benchmark chiffrée et reproductible via script — jamais un chiffre isolé dans le README sans le code qui le produit.
- Coût AWS estimé documenté explicitement dans le README, alarme de facturation mentionnée comme prérequis.
- Toute limite méthodologique (ex. incertitude sur la classe *drift*, taille réduite de certaines classes) explicitement mentionnée — pas de survente des résultats.

## Pour aller plus loin (optionnel)

- Comparaison de plusieurs méthodes d'anomalie (PaDiM vs PatchCore vs score softmax calibré).
- Calibration de température sur les probabilités du classifieur.
- Simulation d'un monitoring de dérive du modèle en production.
