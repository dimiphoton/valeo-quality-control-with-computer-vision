# Roadmap

Classer les défauts Valeo et détecter le `drift`, en calant la décision
sur la matrice de coût du challenge, puis exposer une API d'inférence.

- [x] Cadrage — identité, README, structure `src/valeo_qc`, logique de décision testée
- [x] Préparation des images — split stratifié, poids de classes, `rotate_and_crop` vers `data/processed/`
- [x] Baseline classifieur (reproduire `Classifier.pt`) + journal d'expériences
- [x] Détecteur d'anomalie PaDiM (reproduire le pickle officiel, puis réentraîner)
- [ ] Calibration du seuil sur le coût métier + comparaison chiffrée au benchmark
- [ ] Export ONNX
- [ ] API Lambda locale (handler + image Docker)
- [ ] Déploiement AWS (alarme CloudWatch d'abord)
- [ ] Présentations recruteur / technique FR+EN
