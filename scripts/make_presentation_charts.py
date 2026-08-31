"""Graphes des decks — fond crème, pas de titre dans l'image."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pictures" / "presentations"
PAGES_OUT = ROOT / "docs" / "pictures" / "presentations"
OUT.mkdir(parents=True, exist_ok=True)
PAGES_OUT.mkdir(parents=True, exist_ok=True)

BG = "#f3eee6"
INK = "#1c1610"
MUTED = "#5c564c"
ACCENT = "#8a6a12"
GOLD = "#c9a227"


def _style(ax: plt.Axes, fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=11)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    for spine in ax.spines.values():
        spine.set_color(MUTED)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for folder in (OUT, PAGES_OUT):
        path = folder / name
        fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
        print(path)
    plt.close(fig)


def good_flagged() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    _style(ax, fig)
    labels = ["Notebook 0.50", "Protect-GOOD 0.611"]
    values = [13, 0]
    bars = ax.bar(labels, values, color=[GOLD, ACCENT], width=0.52)
    ax.set_ylabel("GOOD → drift (count on val)")
    ax.set_ylim(0, 16)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            str(value),
            ha="center",
            color=INK,
            fontsize=18,
            fontweight="bold",
        )
    _save(fig, "good-flagged.png")


def penalty() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    _style(ax, fig)
    labels = ["Notebook 0.50", "Protect-GOOD 0.611"]
    values = [174_101, 20_101]
    bars = ax.bar(labels, values, color=[GOLD, ACCENT], width=0.52)
    ax.set_ylabel("Penalty on val (cost units)")
    ax.set_ylim(0, 210_000)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4000,
            f"{value:,}".replace(",", " "),
            ha="center",
            color=INK,
            fontsize=14,
            fontweight="bold",
        )
    _save(fig, "penalty-val.png")


def pwa_points() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    _style(ax, fig)
    labels = ["Classifier only", "Notebook 0.50", "Protect-GOOD"]
    official = [1.000, 0.989, 0.999]
    ours = [0.998, 0.994, 0.997]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, official, width, color=ACCENT, label="Official")
    ax.bar(x + width / 2, ours, width, color=GOLD, label="Ours")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PWA on val (no drift labels)")
    ax.set_ylim(0.985, 1.001)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "pwa-points.png")


def padim_class() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _style(ax, fig)
    labels = ["Missing", "Overall", "GOOD"]
    values = [102, 144, 262]
    colors = [ACCENT, MUTED, GOLD]
    bars = ax.bar(labels, values, color=colors, width=0.52)
    ax.set_ylabel("PaDiM raw score (official, val mean)")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 6,
            str(value),
            ha="center",
            color=INK,
            fontsize=16,
            fontweight="bold",
        )
    _save(fig, "padim-by-class.png")


def class_counts() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    _style(ax, fig)
    labels = [
        "Missing",
        "GOOD",
        "Lift-off blanc",
        "Short MOS",
        "Lift-off noir",
        "Boucle plate",
    ]
    values = [6472, 1235, 270, 126, 104, 71]
    y = np.arange(len(labels))
    ax.barh(y, values, color=ACCENT, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Train images (n = 8 278)")
    ax.invert_yaxis()
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 40, yi, f"{value:,}".replace(",", " "), va="center", color=INK, fontsize=11)
    ax.set_xlim(0, 7800)
    _save(fig, "class-counts.png")


def _boxes_row(
    ax: plt.Axes,
    labels: list[str],
    y: float,
    box_w: float = 0.18,
    box_h: float = 0.22,
) -> None:
    """Une rangée de boîtes reliées par des flèches (coordonnées 0–1)."""
    n = len(labels)
    xs = np.linspace(0.08, 0.92, n)
    for i, (x, label) in enumerate(zip(xs, labels, strict=True)):
        rect = plt.Rectangle(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            facecolor="#fff8ee",
            edgecolor=ACCENT,
            linewidth=1.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color=INK,
            fontsize=11,
            fontweight="normal",
            transform=ax.transAxes,
            wrap=True,
        )
        if i < n - 1:
            x0 = x + box_w / 2 + 0.01
            x1 = xs[i + 1] - box_w / 2 - 0.01
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.8},
            )


def architecture_plain(lang: str) -> None:
    """Quatre étapes, langage métier (recruteur)."""
    labels = {
        "fr": ["Recadrer\nla photo", "Nommer\nle défaut connu", "Signaler\nle jamais-vu", "Répondre\npar une API"],
        "en": ["Crop\nthe frame", "Name the\nknown defect", "Flag what\nwas never seen", "Answer\nwith an API"],
    }[lang]
    fig, ax = plt.subplots(figsize=(10.2, 2.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    _boxes_row(ax, labels, y=0.5, box_w=0.19, box_h=0.55)
    _save(fig, f"architecture-plain-{lang}.png")


def architecture_serve() -> None:
    """Train vs serve — une figure, libellés code (FR/EN identiques)."""
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.04, 0.82, "TRAIN", color=ACCENT, fontsize=12, fontweight="bold", transform=ax.transAxes)
    _boxes_row(
        ax,
        ["raw PNG", "rotate+crop", "resnest50d\n+ PaDiM", "threshold.json"],
        y=0.68,
        box_w=0.18,
        box_h=0.22,
    )
    ax.text(0.04, 0.38, "SERVE", color=ACCENT, fontsize=12, fontweight="bold", transform=ax.transAxes)
    _boxes_row(
        ax,
        ["alarm\nbilling", "ONNX\n(+ Gaussian)", "Lambda\nimage", "Function URL"],
        y=0.22,
        box_w=0.18,
        box_h=0.22,
    )
    _save(fig, "architecture-serve.png")


if __name__ == "__main__":
    good_flagged()
    penalty()
    pwa_points()
    padim_class()
    class_counts()
    architecture_plain("fr")
    architecture_plain("en")
    architecture_serve()
