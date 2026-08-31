---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../pictures/presentations/photos/hero.jpg)

# How do you fuse a
# classifier and PaDiM
# when val has no
# drift labels?

Machine learning · Industry / quality control · Python / PyTorch / ONNX / AWS

Valeo · ENS #157 · 8,278 training images

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/motivation.jpg)

# GOOD → drift
# costs 10,000.

A known defect given the wrong name: 1.
A GOOD part thrown as unknown: 10,000.

**With no drift labels on val, maximising PWA turns PaDiM off.**

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/hero.jpg)

# Who consumes
# the score.

The jury: PWA on a hidden test (2 submissions / 24 h).

Plant quality: an API, image in, class out.

**The deliverable is not a notebook. It is an exported threshold.**

---

<!-- _class: full -->

![bg brightness:0.38](../pictures/presentations/photos/physique.jpg)

# Two models,
# one decision.

resnest50d: six known defects.
PaDiM (WRN-50-2): distance to the training cloud.
Above the threshold → drift.

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/motivation.jpg)

# Grain and
# units.

Raw PNG → rotate + crop by `lib`.
224 px classifier, 128 px PaDiM.

**We do not reuse the val-batch min-max
on a single production image.**

---

<!-- _class: split -->

![bg left:46%](../pictures/presentations/photos/physique.jpg)

# What we isolate.

We strip out the confounder “max PWA on a val set with no drift”.

What remains: the lowest threshold that never maps GOOD to drift.

Not an AUC. Not a global F1.

---

<!-- _class: dark -->

# Scope.

An honest benchmark, then an ONNX API.

Not a leaderboard chase, not anomalib, not SAM.

---

<!-- _class: chart -->

Missing is 6,472 images. *Boucle plate* is 71. Headline accuracy lies.

![w:920](../pictures/presentations/class-counts.png)

---

<!-- _class: full -->

![bg brightness:0.38](../pictures/presentations/photos/physique.jpg)

# The official pickle
# is 99.8 % / F1 0.973.

Our resnest50d: 99.5 % / 0.960.
*Boucle plate* stays the weak spot (recall 0.86).

---

<!-- _class: chart -->

PaDiM is not fitted on healthy parts. Missing is more in-distribution than GOOD.

![w:920](../pictures/presentations/padim-by-class.png)

---

<!-- _class: full -->

![bg brightness:0.38](../pictures/presentations/photos/hero.jpg)

# Protect-GOOD = 0.611.
# PWA 0.999. 0 GOOD flagged.

Val n = 1,655, zero drift labels.
Twenty false drifts, none of them GOOD.

---

<!-- _class: chart -->

At 0.50 the official run pays for 13 GOOD. The chosen threshold cuts that cell. Ours follows.

![w:980](../pictures/presentations/pwa-points.png)

---

<!-- _class: split -->

![bg left:40%](../pictures/presentations/photos/hero.jpg)

# Same rule on
# the retrained stack.

Official at 0.50: 13 GOOD.
Ours at 0.50: 3 GOOD.
Protect-GOOD: 0 on both.

---

<!-- _class: dark -->

# Why not max PWA.

It sets the threshold to 1 and kills PaDiM.

Why not a transformer: the official pickle is a ResNeSt-50d.
We benchmark what is there, not a different paper.

---

<!-- _class: dark -->

# Where it breaks.

Hidden test labels — no local score on drift.

PaDiM Gaussian: 1.2 GB pickle, off Lambda by default.

Public Function URL; `--apply` never hit a real account.

---

<!-- _class: cta -->

![bg brightness:0.30](../pictures/presentations/photos/cta.jpg)

# Reproduce.

[github.com/dimiphoton/valeo-quality-control-with-computer-vision](https://github.com/dimiphoton/valeo-quality-control-with-computer-vision)

`python -m valeo_qc.cli predict image.png`

`python -m valeo_qc.cli deploy --email you@example.com`

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonwebservices&logoColor=white)
