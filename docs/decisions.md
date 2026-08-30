# Décisions

| Date | Décision | Alternative envisagée | Raison |
|---|---|---|---|
| 2026-08-30 | Métier **Machine learning**, domaine **Industrie / contrôle qualité** | Mobilité (Valeo auto) | Le repo montre un classifieur + open-set, pas un sujet transport. « Industrie » est le cas Autre de `brief/identite.md`. |
| 2026-08-30 | Stack bandeau **Python / PyTorch / ONNX / AWS** | TorchScript, GCP, Streamlit | Aligné sur le brief (export ONNX pour Lambda, AWS déjà tranché). |
| 2026-08-30 | Matrice de coût et PWA copiées du notebook officiel | Transcrire l'image du site | Le notebook `Supp_files/Notebook_ENS.ipynb` contient la matrice exploitable. |
| 2026-08-30 | `rotate_and_crop` écrit dans `data/processed/`, jamais dans `raw/` | Script officiel qui écrase la source | Les bruts du challenge ne doivent pas être mutés. |
| 2026-08-30 | Numpy / pandas / Pillow maintenant ; PyTorch, MLflow, anomalib plus tard | Tout installer au cadrage | La règle persona demande un go avant ces outils. |
| 2026-08-30 | Poids de classes calculés sur le **split train** seulement | Poids sur tout `Y_train` | Évite une fuite d'information vers la validation ; le val sert au seuil et à la comparaison. |
| 2026-08-30 | Architecture **resnest50d** (timm), comme le pickle officiel | ResNet50 du brief initial | `Classifier.pt` est un ResNeSt-50d, epoch 15. Recalibrer le brief aurait été un faux benchmark. |
| 2026-08-30 | MLflow 3 en **SQLite locale** (`mlflow.db`) | Store `./mlruns` fichier | MLflow 3 refuse le file store par défaut. SQLite reste local, sans serveur. |
| 2026-08-30 | Pas de normalise ImageNet à l'entrée du classifieur | mean/std ImageNet | Le notebook officiel ne fait que `Resize(224)+ToTensor`. On aligne l'éval. |
