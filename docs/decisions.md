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
| 2026-08-30 | PaDiM **sans anomalib**, copie du notebook (WRN-50-2, seed 1024) | Installer anomalib | Le pickle officiel n'est pas un checkpoint anomalib ; même geste que pour ``resnest50d``. |
| 2026-08-30 | Fit PaDiM sur **toutes** les classes du split train + ridge 0,01 | GOOD only, LedoitWolf | Le pickle officiel a des scores Missing ≪ GOOD : le Gaussian est dominé par la classe majoritaire. LedoitWolf exigerait ~15 Go d'embeddings. |
| 2026-08-31 | Seuil exporté = **protège-GOOD** (0,611 officiel) | Max PWA val (seuil 1) ou 0,5 du notebook | Val sans drift : le max PWA éteint PaDiM. Le 0,5 flag 13 GOOD (×10 000). Protège-GOOD reste comparable au benchmark tout en coupant la case la plus chère. |
| 2026-08-31 | ONNX opset 18, gaussienne PaDiM hors graphe | Un seul graphe classifieur+PaDiM, TorchScript | `col2im` (fold) exige l'opset 18 ; la cov 1,2 Go n'est pas un réseau. Concat PaDiM = upsample nearest (équivalent au unfold/fold officiel). |
| 2026-08-31 | Inférence ONNX en **sous-processus** | onnxruntime in-process | Sous Windows, charger onnxruntime après torch CUDA (pytest) provoque un access violation. |
| 2026-08-31 | Image Lambda **sans PyTorch**, SAM plus tard | SAM local dès cette étape | SAM reste à valider. L'image AWS Lambda + RIE suffit pour tester le handler. PaDiM optionnel (pickle 1,2 Go + min-max figé). |
