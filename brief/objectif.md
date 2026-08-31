# Objectif du projet

- **But** : classer les défauts connus sur des images de composants Valeo
  et détecter la classe `drift` (anomalies inédites), avec une décision
  calée sur la matrice de coût du challenge, puis exposer le modèle en
  API Lambda.
- **Origine** : Challenge Data ENS #157,
  [Improving Industrial Quality Control with Computer Vision](https://challengedata.ens.fr/participants/challenges/157/)
  (Valeo). Brief portfolio `brief/02-detection-defauts-valeo-cv.md` et
  cadrage `brief/02-detection-defauts-valeo-cv-cadrage-technique.md`.
- **Contraintes de départ** :
  - données non republicables — rien sous `data/` dans git ;
  - labels de test cachés (évaluation par soumission, 2 / 24 h) ;
  - AWS free tier, alarme de facturation **avant** tout déploiement ;
  - PyTorch, MLflow, timm et scipy installés (PaDiM maison, pas anomalib).
    AWS CLI / SAM restent à valider avant installation. Le déploiement
    versionné est du CloudFormation brut (alarme billing us-east-1 d'abord).

Métier, domaine et stack se remplissent dans `brief/identite.md`.
