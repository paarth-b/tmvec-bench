#!/usr/bin/env python
"""Density scatter plots for benchmarking method comparisons."""

import argparse
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme()


def plot_density_scatter(df, pred_col, truth_col, method_name, output_path=None):
    """Create density scatter plot comparing predicted vs ground truth."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Hexbin density plot with rocket colormap
    hb = ax.hexbin(
        df[truth_col],
        df[pred_col],
        gridsize=100,
        cmap='rocket',
        mincnt=1
    )

    # Add identity line
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2)

    # Determine title based on method
    if 'foldseek' in method_name.lower():
        title = 'SCOPe40 Alignment Results (Foldseek)'
    elif 'tmvec-1' in method_name.lower():
        title = 'SCOPe40 Alignment Results (TMvec-1)'
    elif 'student' in method_name.lower():
        title = 'SCOPe40 Alignment Results (TMvec-Student)'
    else:
        title = f'Alignment Results ({method_name})'

    ax.set_title(title, fontsize=20, fontweight='bold', pad=20, fontfamily='sans-serif')
    ax.set_xlabel('TM-align Ground Truth', fontsize=16, fontweight='bold')
    ax.set_ylabel(method_name, fontsize=16, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Calculate Pearson R
    pearson_r, _ = stats.pearsonr(df[truth_col], df[pred_col])

    # Add total count and Pearson R box
    total_count = len(df)
    stats_text = f'n = {total_count:,}\nPearson R = {pearson_r:.3f}'
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Add colorbar
    plt.colorbar(hb, ax=ax, label='Count')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')

    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate density scatter plots for method comparison')
    parser.add_argument('method', choices=['foldseek', 'tmvec1', 'tmvec2', 'student'],
                        help='Method to compare against TM-align')
    parser.add_argument('--tmalign', default='results/tmalign_similarities.parquet',
                        help='Path to TM-align results')
    parser.add_argument('--method-file', help='Path to method results (auto-detected if not provided)')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum number of pairs to load')
    parser.add_argument('--output-dir', help='Output directory (default: figures/{method})')
    args = parser.parse_args()

    # Auto-detect method file and display name
    method_config = {
        'foldseek': {
            'file': 'results/scope40_foldseek_similarities.parquet',
            'name': 'Foldseek',
            'tmalign_default': 'results/scope40_tmalign_similarities.parquet'
        },
        'tmvec1': {
            'file': 'results/scope40_tmvec1_similarities.parquet',
            'name': 'TMvec-1',
            'tmalign_default': 'results/scope40_tmalign_similarities.parquet'
        },
        'tmvec2': {
            'file': 'results/tmvec2_similarities.parquet',
            'name': 'TMvec-2',
            'tmalign_default': 'results/tmalign_similarities.parquet'
        },
        'student': {
            'file': 'results/scope40_tmvec_student_similarities.parquet',
            'name': 'TMvec-Student',
            'tmalign_default': 'results/scope40_tmalign_similarities.parquet'
        }
    }

    config = method_config[args.method]
    method_file = args.method_file or config['file']
    tmalign_file = args.tmalign if args.tmalign != 'results/tmalign_similarities.parquet' else config['tmalign_default']

    # Load data
    df_tmalign = pq.read_table(tmalign_file).to_pandas()
    if 'tm_score' in df_tmalign.columns:
        df_tmalign['tm_score'] = pd.to_numeric(df_tmalign['tm_score'], errors='coerce')

    df_method = pq.read_table(method_file).to_pandas()
    if args.max_pairs is not None:
        df_method = df_method.head(args.max_pairs)
    if 'tm_score' in df_method.columns:
        df_method['tm_score'] = pd.to_numeric(df_method['tm_score'], errors='coerce')

    # Clean sequence IDs
    df_method['seq1_clean'] = df_method['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
    df_method['seq2_clean'] = df_method['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

    # Merge
    df_merged = pd.merge(
        df_tmalign[['seq1_id', 'seq2_id', 'tm_score']],
        df_method[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_tmalign', '_method')
    )

    print(f"Plotting {len(df_merged)} pairs")

    # Output
    output_dir = Path(args.output_dir or f'figures/{args.method}')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_density_scatter(df_merged, 'tm_score_method', 'tm_score_tmalign', config['name'], output_dir / 'density_scatter')
    print(f"Saved to {output_dir}/density_scatter.png")


if __name__ == "__main__":
    main()
