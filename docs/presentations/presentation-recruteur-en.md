---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../../pictures/presentations/photos/hero.jpg)

# Can a camera scrap
# fewer good parts
# without letting
# unknowns through?

Machine learning · Industry / quality control

Valeo · Challenge Data ENS #157

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/motivation.jpg)

# A good part labelled
# “unknown” is the
# expensive mistake.

On an electronics line, a false reject is not a rounding error.

**Valeo prices it ten thousand times higher than a mix-up between two known defects.**

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/hero.jpg)

# Who has a call
# to make.

Valeo quality: keep the camera, not halt the line.

The operator: scrap, pass, or open the bin.

**The challenge jury: score that same cost logic.**

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/physique.jpg)

# The camera does not
# see “the board”.
# It sees a window.

We crop. Six defects have names.
The seventh — drift — is absent from training.

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/motivation.jpg)

# How to read
# the pictures.

8,278 training images. Six labels.
One class, “Missing”, drowns the rest.

**Drift shows up only on the hidden test set.**
So we calibrate without ever seeing it.

---

<!-- _class: dark -->

# This project is not.

Not a leaderboard hunt.

Not an HR dashboard.

**A decision rule, then an inference API.**

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/physique.jpg)

# 99.8 % of labels
# are right.
# That is not the point.

At the notebook threshold: 13 good parts
thrown out as unknown.

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# Zero good parts
# labelled unknown.

The threshold is set to protect GOOD.
Twenty false alarms. None on a healthy part.

---

<!-- _class: chart -->

The notebook scraps 13 good parts. The chosen threshold scraps none.

![w:920](../../pictures/presentations/good-flagged.png)

---

<!-- _class: chart -->

On the hold-out set, the cost bill drops from 174,000 to 20,000.

![w:920](../../pictures/presentations/penalty-val.png)

---

<!-- _class: split -->

![bg left:40%](../../pictures/presentations/photos/physique.jpg)

# This is not
# a homemade metric.

The cost grid comes from Valeo’s challenge.

The 13 is the official model, not ours.

---

<!-- _class: actions -->

![bg right:38%](../../pictures/presentations/photos/hero.jpg)

# Monday.

**Quality** — keep the threshold that protects GOOD.

**Ops** — billing alarm before any cloud spend.

Not a magic model. A rule you can defend.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# Your turn.

[Source code](https://github.com/dimiphoton/valeo-quality-control-with-computer-vision)

[Valeo challenge](https://challengedata.ens.fr/participants/challenges/157/)
