#!/usr/bin/env python
"""
Visualize KNN Classification Results for CATH Hierarchy

Generates publication-quality figures comparing different methods
at different CATH hierarchy levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Method display names and colors
METHOD_NAMES = {
    'student': 'TM-Vec 2s (Student)',
    'tmvec2': 'TM-Vec 2',
    'tmvec1': 'TM-Vec',
    'foldseek': 'Foldseek',
}

METHOD_COLORS = {
    'student': '#2ecc71',      # Green
    'tmvec1': '#3498db',       # Blue
    'tmvec2': '#9b59b6',       # Purple
    'foldseek': '#e74c3c',     # Red
}

LEVEL_ORDER = ['class', 'architecture', 'topology', 'superfamily']
LEVEL_LABELS = ['Class (C)', 'Architecture (A)', 'Topology (T)', 'Superfamily (H)']


def load_results(csv_path: str) -> pd.DataFrame:
    """Load KNN results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_accuracy_by_level(df: pd.DataFrame, k: int = 1, output_path: str = None):
    """
    Bar chart comparing methods at each CATH hierarchy level for a fixed k.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df['method'].unique()
    x = np.arange(len(LEVEL_ORDER))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        accuracies = []
        for level in LEVEL_ORDER:
            acc = method_data[method_data['level'] == level][f'k={k}'].values
            accuracies.append(acc[0] * 100 if len(acc) > 0 else 0)
        
        label = METHOD_NAMES.get(method, method)
        color = METHOD_COLORS.get(method, f'C{i}')
        offset = (i - len(methods)/2 + 0.5) * width
        
        bars = ax.bar(x + offset, accuracies, width, label=label, color=color, alpha=0.85)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('CATH Hierarchy Level')
    ax.set_title(f'KNN Classification Accuracy (k={k})')
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_accuracy_by_k(df: pd.DataFrame, level: str = 'superfamily', output_path: str = None):
    """
    Line plot showing accuracy vs k for each method at a specific level.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = df['method'].unique()
    k_cols = [col for col in df.columns if col.startswith('k=')]
    k_values = [int(col.split('=')[1]) for col in k_cols]
    
    for method in methods:
        method_data = df[(df['method'] == method) & (df['level'] == level)]
        if len(method_data) == 0:
            continue
            
        accuracies = [method_data[col].values[0] * 100 for col in k_cols]
        label = METHOD_NAMES.get(method, method)
        color = METHOD_COLORS.get(method, None)
        
        ax.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8,
                label=label, color=color)
    
    level_label = LEVEL_LABELS[LEVEL_ORDER.index(level)]
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'KNN Classification at {level_label} Level')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_heatmap(df: pd.DataFrame, output_path: str = None):
    """
    Heatmap showing accuracy for all methods, levels, and k values.
    """
    methods = df['method'].unique()
    k_cols = [col for col in df.columns if col.startswith('k=')]
    
    # Create subplot for each method
    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5), 
                              squeeze=False, sharey=True)
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        method_data = df[df['method'] == method]
        
        # Build matrix: rows = levels, cols = k values
        matrix = np.zeros((len(LEVEL_ORDER), len(k_cols)))
        for i, level in enumerate(LEVEL_ORDER):
            for j, k_col in enumerate(k_cols):
                val = method_data[method_data['level'] == level][k_col].values
                matrix[i, j] = val[0] * 100 if len(val) > 0 else 0
        
        im = axes[idx].imshow(matrix, cmap='RdYlGn', vmin=40, vmax=100, aspect='auto')
        
        # Add text annotations
        for i in range(len(LEVEL_ORDER)):
            for j in range(len(k_cols)):
                text = axes[idx].text(j, i, f'{matrix[i, j]:.1f}',
                                      ha='center', va='center', fontsize=10)
        
        axes[idx].set_xticks(range(len(k_cols)))
        axes[idx].set_xticklabels([col.split('=')[1] for col in k_cols])
        axes[idx].set_xlabel('k')
        
        if idx == 0:
            axes[idx].set_yticks(range(len(LEVEL_ORDER)))
            axes[idx].set_yticklabels(LEVEL_LABELS)
        
        axes[idx].set_title(METHOD_NAMES.get(method, method))
    
    plt.suptitle('KNN Classification Accuracy', y=1.02)
    plt.tight_layout()
    
    # Add colorbar after tight_layout
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='Accuracy (%)', 
                 orientation='vertical', pad=0.02)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot KNN Classification Results')
    parser.add_argument('--input', type=str, default='results/knn_cath_classification.csv',
                       help='Input CSV file with results')
    parser.add_argument('--output-dir', type=str, default='figures/knn',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.input)
    print(f"Loaded results for methods: {df['method'].unique().tolist()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # Accuracy by level for k=1 (most common evaluation)
    plot_accuracy_by_level(df, k=1, output_path=output_dir / 'knn_accuracy_k1.png')
    plot_accuracy_by_level(df, k=1, output_path=output_dir / 'knn_accuracy_k1.svg')
    
    # Accuracy vs k for different levels
    for level in LEVEL_ORDER:
        plot_accuracy_by_k(df, level=level, 
                          output_path=output_dir / f'knn_accuracy_vs_k_{level}.png')
    
    # Heatmap showing all results
    plot_heatmap(df, output_path=output_dir / 'knn_heatmap.png')
    plot_heatmap(df, output_path=output_dir / 'knn_heatmap.svg')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
