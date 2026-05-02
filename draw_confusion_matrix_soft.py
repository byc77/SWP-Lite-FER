# -*- coding: utf-8 -*-
"""
draw_confusion_matrix_soft.py

?券?
1. 霈??confusion_matrix_test.csv
2. ?舐?? count confusion matrix
3. ?舐 row-normalized confusion matrix
4. 雿輻????剛羲??/ 憟嗥?瘛箄蝟?
雿輻蝭?嚗?python draw_confusion_matrix_soft.py ^
  --input runs\rafdb\laststage_nonla_TTA\confusion_matrix_test.csv ^
  --output cm_laststage_nonla_soft.png ^
  --title "Confusion Matrix - laststage_nonla (flip-TTA)" ^
  --labels Surprise Fear Disgust Happiness Sadness Anger Neutral ^
  --style blue ^
  --normalize row

python draw_confusion_matrix_soft.py ^
  --input runs\rafdb\laststage_nla_TTA\confusion_matrix_test.csv ^
  --output cm_laststage_nla_soft.png ^
  --title "Confusion Matrix - laststage_nla (flip-TTA)" ^
  --labels Surprise Fear Disgust Happiness Sadness Anger Neutral ^
  --style green ^
  --normalize row
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def build_soft_cmap(style: str):
    """
    ???脩嚗?    - blue: ?怨餈芾?
    - green: 憟嗥?
    """
    style = style.lower()

    if style == "blue":
        # ??-> 瘛箇??-> ?怨餈芾? -> 瘛曹?暺???        colors = [
            "#f8fafb",
            "#e8eef1",
            "#d5e0e6",
            "#bccdd6",
            "#9fb6c3",
            "#7f9aa9",
        ]
    elif style == "green":
        # ??-> 憟嗥? -> 曌偏?? -> 瘛曹?暺蝬?        colors = [
            "#fbfcfa",
            "#eef3ed",
            "#dde8df",
            "#c8d7cb",
            "#aebfaf",
            "#8fa58f",
        ]
    elif style == "bluegreen":
        # ??-> 瘛⊿???-> 憟嗥???-> ?啗?蝬?        colors = [
            "#fafcfc",
            "#ebf1f0",
            "#d9e5e2",
            "#bfd1cc",
            "#9fb7b1",
            "#7e9993",
        ]
    else:
        raise ValueError("style must be one of: blue, green, bluegreen")

    return LinearSegmentedColormap.from_list(f"soft_{style}", colors)


def normalize_cm(cm: np.ndarray, mode: str):
    mode = mode.lower()
    if mode == "none":
        return cm.astype(float)
    if mode == "row":
        denom = cm.sum(axis=1, keepdims=True).astype(float)
        denom[denom == 0] = 1.0
        return cm / denom
    raise ValueError("normalize must be 'none' or 'row'")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels,
    title: str,
    output_path: Path,
    style: str = "blue",
    normalize: str = "none",
    figsize=(8.2, 6.8),
    dpi=220,
):
    data = normalize_cm(cm, normalize)
    cmap = build_soft_cmap(style)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalize == "row" else "Count", rotation=90, va="bottom")

    # axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # 憿舐內?詨?    thresh = data.max() * 0.6 if data.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize == "row":
                txt = f"{data[i, j]:.1%}"
                color = "#1f2a30" if data[i, j] < thresh else "white"
            else:
                txt = f"{int(cm[i, j])}"
                color = "#1f2a30" if data[i, j] < thresh else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    # ?質?潛?
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 憭???
    for spine in ax.spines.values():
        spine.set_color("#9aa7ad")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="confusion_matrix_test.csv 頝臬?")
    parser.add_argument("--output", required=True, help="頛詨 png 頝臬?")
    parser.add_argument("--title", default="Confusion Matrix")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"],
        help="憿?迂嚗?摨???????,
    )
    parser.add_argument("--style", choices=["blue", "green", "bluegreen"], default="blue")
    parser.add_argument("--normalize", choices=["none", "row"], default="none")
    args = parser.parse_args()

    cm = np.loadtxt(args.input, delimiter=",", dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("input confusion matrix must be square")

    if len(args.labels) != cm.shape[0]:
        raise ValueError(f"labels count ({len(args.labels)}) != matrix size ({cm.shape[0]})")

    plot_confusion_matrix(
        cm=cm,
        labels=args.labels,
        title=args.title,
        output_path=Path(args.output),
        style=args.style,
        normalize=args.normalize,
    )
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()


