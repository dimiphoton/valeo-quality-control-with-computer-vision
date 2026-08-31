---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../pictures/presentations/photos/hero.jpg)

# Du recadrage à Lambda :
# classer le connu,
# détecter le drift,
# au coût métier.

Machine learning · Industrie / contrôle qualité · Python / PyTorch / ONNX / AWS

Valeo · ENS #157 · 8 278 images train

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/motivation.jpg)

# On reproduit le
# pipeline officiel,
# puis on le sert.

Challenge Valeo : six défauts nommés, un drift hors train.
On calibre la fusion sur la matrice de coût, pas sur l'accuracy.

**Livrable : `threshold.json` + API ONNX, sans PyTorch.**

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/hero.jpg)

# GOOD → drift
# coûte 10 000.

Un défaut connu mal nommé : 1.
Un GOOD jeté comme inconnu : 10 000.

**Sans labels drift au val, maximiser la PWA éteint PaDiM.**

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/physique.jpg)

# Qui consomme
# le score.

Le jury : PWA sur un test caché (2 soumissions / 24 h).

Qualité usine : POST une image, recevoir une classe.

**Le notebook n'est pas le produit. L'API l'est.**

---

<!-- _class: full -->

![bg brightness:0.38](../pictures/presentations/photos/physique.jpg)

# Deux modèles,
# une décision.

resnest50d : six défauts connus.
PaDiM (WRN-50-2) : distance au nuage d'entraînement.
Au-dessus du seuil → drift.

---

<!-- _class: chart -->

Train d'un côté, inférence de l'autre. L'alarme billing avant la Lambda.

![w:980](../pictures/presentations/architecture-serve.png)

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/motivation.jpg)

# Grain : une image,
# deux résolutions.

PNG brut → rotate + crop selon `lib`.
224 px classifieur, 128 px PaDiM.

**En prod, le min-max PaDiM est figé.
Un min-max par image donnerait toujours 1.**

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/physique.jpg)

# Ce qu'on isole.

On retire le confondant « max PWA sur un val sans drift ».

Ce qui reste : le plus petit seuil qui ne classe aucun GOOD en drift.

Pas un AUC. Pas un F1 global.

---

<!-- _class: dark -->

# Périmètre.

Benchmark honnête, export ONNX, plan CloudFormation.

Pas de leaderboard, pas d'anomalib, pas de SAM.

---

<!-- _class: chart -->

Missing pèse 6 472 images. Boucle plate, 71. L'accuracy globale ment.

![w:920](../pictures/presentations/class-counts.png)

---

<!-- _class: full -->

![bg brightness:0.38](../pictures/presentations/photos/physique.jpg)

# Le pickle officiel
# fait 99,8 % / F1 0,973.

Notre resnest50d : 99,5 % / 0,960.
Boucle plate reste le point faible (rappel 0,86).

---

<!-- _class: chart -->

PaDiM n'est pas calé sur les pièces saines. Missing est plus « in-distribution » que GOOD.

![w:920](../pictures/presentations/padim-by-class.png)

---

<!-- _class: full -->

![bg brightness:0.38](../pictures/presentations/photos/hero.jpg)

# Protège-GOOD = 0,611.
# PWA 0,999. 0 GOOD flaggé.

Val n = 1 655, zéro label drift.
Vingt faux drift, aucun GOOD.

---

<!-- _class: chart -->

À 0,50 l'officiel paye 13 GOOD. Le seuil retenu coupe cette case. Le nôtre suit.

![w:980](../pictures/presentations/pwa-points.png)

---

<!-- _class: dark -->

# Où ça casse.

Labels test cachés — pas de score local sur le drift.

Gaussienne PaDiM : pickle 1,2 Go, hors Lambda par défaut.

Function URL publique ; `--apply` jamais lancé sur un compte.

---

<!-- _class: cta -->

![bg brightness:0.30](../pictures/presentations/photos/cta.jpg)

# Reproduire.

[github.com/dimiphoton/valeo-quality-control-with-computer-vision](https://github.com/dimiphoton/valeo-quality-control-with-computer-vision)

`python -m valeo_qc.cli predict image.png`

`python -m valeo_qc.cli deploy --email toi@exemple.fr`

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonwebservices&logoColor=white)
