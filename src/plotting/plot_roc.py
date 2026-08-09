#!/usr/bin/env python
"""Generate ROC curves comparing Foldseek and TM-Vec models.

Compares homology detection accuracy at each hierarchy level using
structural classification ground truth (CATH or SCOPe).

Usage:
    python src/plotting/plot_roc.py --dataset cath
    python src/plotting/plot_roc.py --dataset scope40
    python src/plotting/plot_roc.py --dataset cath --suffix _full
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from src.util.clean_ids import clean_seq_id_column


DATASETS = {
    "cath": {
        "levels": ["class", "architecture", "topology", "superfamily"],
        "truth": "src/plotting/cath/truth.tsv",
    },
    "scope40": {
        "levels": ["class", "fold", "superfamily", "family"],
        "truth": "src/plotting/scope/truth.tsv",
    },
}

METHODS = {
    "tmvec1":          {"label": "TM-Vec",    "score_col": "tm_score", "higher_is_similar": True},
    "tmvec2":          {"label": "TM-Vec 2",  "score_col": "tm_score", "higher_is_similar": True},
    "tmvec2_student":  {"label": "TM-Vec 2s", "score_col": "tm_score", "higher_is_similar": True},
    "tmalign":         {"label": "TM-align",  "score_col": "tm_score", "higher_is_similar": True},
    "foldseek":        {"label": "Foldseek",  "score_col": "evalue",   "higher_is_similar": False},
    "plmblast":        {"label": "pLM-BLAST", "score_col": "tm_score", "higher_is_similar": True},
}


def load_scores(dataset, method, suffix=""):
    """Load pairwise scores for a method. Returns DataFrame with seq_pair index."""
    path = Path(f"results/{dataset}{suffix}_{method}_similarities.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path)
    info = METHODS[method]

    for col in ["seq1_id", "seq2_id"]:
        df[col] = clean_seq_id_column(df[col])

    # Create canonical pair key (alphabetically sorted)
    df["seq_pair"] = df.apply(
        lambda row: ",".join(sorted([row["seq1_id"], row["seq2_id"]])), axis=1
    )

    # Extract the score column
    score_col = info["score_col"]
    if score_col not in df.columns:
        score_col = "tm_score"

    scores = df.set_index("seq_pair")[score_col].rename(method)

    # For e-values, convert to a "higher = more similar" score using -log10
    if not info["higher_is_similar"]:
        values = scores.to_numpy(dtype=float).copy()
        values[values == 0] = np.nanmin(values[values > 0]) * 0.1
        scores = pd.Series(-np.log10(values), index=scores.index, name=method)

    return scores


def load_truth(dataset):
    """Load ground truth classification. Returns DataFrame with seq_pair index."""
    path = Path(DATASETS[dataset]["truth"])
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}\n"
            f"Run the get_truth script first:\n"
            f"  python -m src.plotting.get_truth_{dataset}"
        )
    df = pd.read_table(path)
    if "seq_pair" not in df.columns:
        # Legacy format with separate a, b columns
        df["seq_pair"] = df["a"] + "," + df["b"]
        df = df.drop(columns=["a", "b"])
    df = df.set_index("seq_pair")
    return df


def plot_roc(truth, scores_dict, levels, output_path):
    """Plot ROC curves for all methods at each hierarchy level."""
    methods_present = list(scores_dict.keys())

    # Merge all scores with truth on shared pairs
    merged = truth.copy()
    for method, scores in scores_dict.items():
        merged = merged.join(scores, how="inner")
    print(f"Pairs with all methods + truth: {len(merged):,}")

    if merged.empty:
        print("No overlapping pairs found. Skipping.")
        return

    # Set up colors
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(METHODS)}

    # Plot
    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(3.5 * n_levels, 3.5), sharey=True)
    if n_levels == 1:
        axes = [axes]

    for i, level in enumerate(levels):
        ax = axes[i]
        y_true = merged[level].to_numpy()

        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            ax.set_title(f"{level.capitalize()}\n(single class)")
            continue

        aurocs = {}
        for method in methods_present:
            label = METHODS[method]["label"]
            y_score = merged[method].to_numpy()
            auroc = roc_auc_score(y_true, y_score)
            aurocs[method] = auroc
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, color=colors[method], label=f"{label} ({auroc:.3f})")

        # Sort legend by AUROC descending
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(handles)), key=lambda k: -list(aurocs.values())[k])
        ax.legend(
            [handles[k] for k in order],
            [labels[k] for k in order],
            fontsize=8,
            loc="lower right",
        )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        if i == 0:
            ax.set_ylabel("True positive rate")
        ax.set_title(level.capitalize())

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate ROC curves")
    parser.add_argument("--dataset", choices=["cath", "scope40"], required=True)
    parser.add_argument(
        "--suffix", default="",
        help="Results file suffix (e.g. '_full' for scope40_full_tmvec2_similarities.csv)",
    )
    parser.add_argument(
        "--output-dir", default="figures",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    truth = load_truth(args.dataset)
    print(f"Loaded {len(truth):,} ground truth pairs")

    # Load all available method scores
    scores_dict = {}
    for method in METHODS:
        scores = load_scores(args.dataset, method, args.suffix)
        if scores is not None:
            scores_dict[method] = scores
            print(f"  {method}: {len(scores):,} pairs")
        else:
            print(f"  {method}: not found, skipping")

    if len(scores_dict) < 2:
        raise ValueError("Need at least 2 methods to compare.")

    output_path = Path(args.output_dir) / args.dataset / f"roc{args.suffix}.png"
    plot_roc(truth, scores_dict, config["levels"], output_path)


if __name__ == "__main__":
    main()
