#!/usr/bin/env python
"""
KNN Classification Benchmark for CATH Structural Hierarchy

This script evaluates how well protein embeddings from different methods
can distinguish CATH structural classes using K-Nearest Neighbors classification.

CATH Hierarchy (from most general to most specific):
- Class (C): Major structural class (mainly alpha, mainly beta, mixed alpha-beta, etc.)
- Architecture (A): Overall shape/arrangement of secondary structures  
- Topology (T): Connectivity and arrangement of secondary structures (fold level)
- Superfamily (S): Evolutionary relationship inferred from structure similarity

Evaluation approach (following TMvec1 paper methodology - Table S2):
1. Embed all CATH S100 domain sequences using the model
2. L2-normalize embeddings using faiss.normalize_L2() (in-place normalization)
3. Build FAISS IndexFlatIP for inner product search (cosine similarity on normalized vectors)
4. For each query protein, find k nearest neighbors from the database
5. Check if AT LEAST ONE of the top-k neighbors has the same CATH label (hit metric)
6. Report hit rate at each CATH hierarchy level (C, A, T, H)

TMvec1 paper metric (Table S2):
"Top 1 accuracy for Topology is 97.7%, which means that 97.7% of the time, 
the nearest neighbor returned using TM-Vec is in the same fold as the query."
"Top 3 indicates the percentage of time that ONE OF the top 3 neighbors 
shares the same fold as the query."

This is a leave-one-out evaluation: each protein is queried against all others
and we exclude self-matches when computing hit rates.

Reference:
- TM-Vec: template modeling vectors for fast homology detection and alignment
  (Hamamsy et al., 2022)
- CATH database: https://www.cathdb.info/
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

# Add project root to path for imports when running script directly
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import faiss


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fasta(fasta_path: str, max_sequences: Optional[int] = None):
    """Load sequences from FASTA file."""
    seq_ids, sequences = [], []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    seq_ids.append(current_id)
                    sequences.append("".join(current_seq))
                    if max_sequences and len(seq_ids) >= max_sequences:
                        break
                # Extract domain ID from CATH format: >cath|4_4_0|107lA00/1-162
                header = line[1:]
                if "|" in header:
                    current_id = header.split("|")[-1].split("/")[0]
                else:
                    current_id = header.split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and (not max_sequences or len(seq_ids) < max_sequences):
            seq_ids.append(current_id)
            sequences.append("".join(current_seq))

    print(f"Loaded {len(seq_ids)} sequences from {fasta_path}")
    return seq_ids, sequences


def load_cath_classifications(
    cath_list_path: str, 
    domain_ids: list[str]
) -> pd.DataFrame:
    """
    Load CATH domain classifications from cath-domain-list-S100.txt
    
    File format (whitespace-separated):
    domain  class  arch  topo  superfam  S35  S60  S95  S100  count  length  resolution
    
    Returns DataFrame with hierarchical classification labels.
    
    The CATH hierarchy:
    - Class (C): 4 major classes (mainly alpha, mainly beta, mixed, few secondary structures)
    - Architecture (A): Overall shape (e.g., alpha bundle, beta barrel)
    - Topology (T): Fold level - connectivity of secondary structures
    - Superfamily (H): Evolutionary relationship from structural similarity
    
    Labels are stored as hierarchical strings: C, C.A, C.A.T, C.A.T.H
    """
    columns = [
        'class_raw', 'arch_raw', 'topo_raw', 'super_raw',
        'S35', 'S60', 'S95', 'S100', 'count', 'length', 'resolution'
    ]
    
    print(f"Loading CATH classifications from {cath_list_path}")
    df = pd.read_csv(cath_list_path, sep=r'\s+', names=columns, index_col=0)
    
    # Convert to string for proper concatenation
    for col in ['class_raw', 'arch_raw', 'topo_raw', 'super_raw']:
        df[col] = df[col].astype(str)
    
    # Build hierarchical labels (CATH format: C.A.T.H)
    # Each level includes all parent levels for proper classification
    df['class'] = df['class_raw']
    df['architecture'] = df['class_raw'].str.cat(df['arch_raw'], sep='.')
    df['topology'] = df['architecture'].str.cat(df['topo_raw'], sep='.')
    df['superfamily'] = df['topology'].str.cat(df['super_raw'], sep='.')
    
    # Filter to only the domains we need
    missing = set(domain_ids) - set(df.index)
    if missing:
        print(f"Warning: {len(missing)} domains not found in CATH list")
        print(f"  First 10 missing: {list(missing)[:10]}")
    
    # Keep only domains that exist in both
    valid_domains = [d for d in domain_ids if d in df.index]
    df = df.loc[valid_domains]
    
    print(f"Loaded classifications for {len(df)} domains")
    
    # Print class distribution summary
    for level in ['class', 'architecture', 'topology', 'superfamily']:
        n_classes = df[level].nunique()
        print(f"  {level.capitalize()}: {n_classes} unique classes")
    
    return df

def load_foldseek_similarity_matrix(
    results_path: str,
    domain_ids: list[str],
    min_neighbors: int = 10
) -> tuple[np.ndarray, list[int], dict]:
    """
    Load Foldseek pairwise TM-scores and convert to a similarity matrix.
    
    Foldseek results CSV has columns: seq1_id, seq2_id, tm_score, evalue
    
    IMPORTANT: Foldseek only produces scores for ~15% of pairs (those with 
    detectable structural similarity). We identify domains with sufficient 
    coverage for fair KNN evaluation.
    
    Args:
        results_path: Path to Foldseek CSV results
        domain_ids: List of domain IDs
        min_neighbors: Minimum number of non-zero neighbors required for evaluation
    
    Returns:
        similarity_matrix: NxN matrix where entry [i,j] is TM-score 
        valid_indices: Indices of domains with sufficient Foldseek coverage
        coverage_stats: Dictionary with coverage statistics
    """
    print(f"Loading Foldseek results from {results_path}")
    df = pd.read_csv(results_path)
    
    n_total_pairs = len(domain_ids) * (len(domain_ids) - 1) // 2
    print(f"  Loaded {len(df):,} pairwise scores out of {n_total_pairs:,} possible ({100*len(df)/n_total_pairs:.1f}% coverage)")
    
    # Create domain ID to index mapping
    domain_to_idx = {d: i for i, d in enumerate(domain_ids)}
    n = len(domain_ids)
    
    # Initialize similarity matrix with NaN to track coverage (diagonal = 1)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Track neighbor counts for each domain
    neighbor_counts = np.zeros(n, dtype=np.int32)
    
    # Fill in pairwise similarities
    found = 0
    for _, row in df.iterrows():
        seq1, seq2 = row['seq1_id'], row['seq2_id']
        tm_score = row['tm_score']
        
        # Clip TM-score to [0, 1] range (Foldseek sometimes reports > 1)
        tm_score = min(max(tm_score, 0.0), 1.0)
        
        if seq1 in domain_to_idx and seq2 in domain_to_idx:
            i, j = domain_to_idx[seq1], domain_to_idx[seq2]
            similarity_matrix[i, j] = tm_score
            similarity_matrix[j, i] = tm_score  # Symmetric
            neighbor_counts[i] += 1
            neighbor_counts[j] += 1
            found += 1
    
    # Find domains with sufficient coverage for KNN evaluation
    valid_indices = np.where(neighbor_counts >= min_neighbors)[0].tolist()
    
    coverage_stats = {
        'total_pairs': n_total_pairs,
        'foldseek_pairs': len(df),
        'coverage_pct': 100 * len(df) / n_total_pairs,
        'total_domains': n,
        'domains_with_coverage': len(valid_indices),
        'min_neighbors': min_neighbors,
        'mean_neighbors': neighbor_counts.mean(),
        'median_neighbors': np.median(neighbor_counts),
        'max_neighbors': neighbor_counts.max(),
    }
    
    print(f"  Filled {found:,} pairwise entries in {n}x{n} matrix")
    print(f"  Coverage statistics:")
    print(f"    - Mean neighbors per domain: {coverage_stats['mean_neighbors']:.1f}")
    print(f"    - Median neighbors per domain: {coverage_stats['median_neighbors']:.0f}")
    print(f"    - Domains with >= {min_neighbors} neighbors: {len(valid_indices)} ({100*len(valid_indices)/n:.1f}%)")
    
    return similarity_matrix, valid_indices, coverage_stats


def knn_classification_from_similarity(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 3, 5, 10],
    valid_query_indices: list[int] = None
) -> dict:
    """
    Perform KNN search accuracy using a precomputed similarity matrix.
    
    TMvec1 paper metric: for each query, check if AT LEAST ONE of the 
    top-k neighbors has the same label as the query (hit/recall metric).
    
    Args:
        similarity_matrix: NxN pairwise similarity matrix
        labels: Array of class labels for each domain
        k_values: List of k values to evaluate
        valid_query_indices: If provided, only evaluate these query indices
                            (used for Foldseek which has sparse coverage)
    """
    n = len(labels)
    
    # If no valid indices specified, use all
    if valid_query_indices is None:
        valid_query_indices = list(range(n))
    
    results = {}
    
    for k in k_values:
        hits = 0
        
        for i in valid_query_indices:
            query_label = labels[i]
            # Get similarities to all other proteins (exclude self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -np.inf  # Exclude self
            
            # Find k nearest neighbors (highest similarity)
            neighbor_indices = np.argsort(similarities)[-k:][::-1]
            neighbor_labels = labels[neighbor_indices]
            
            # TMvec1 paper metric: hit if AT LEAST ONE neighbor has same label
            if query_label in neighbor_labels:
                hits += 1
        
        accuracy = hits / len(valid_query_indices)
        
        results[k] = {
            'accuracy': accuracy,
            'n_hits': hits,
            'n_evaluated': len(valid_query_indices)
        }
        
        print(f"    Top-{k}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%) [n={len(valid_query_indices)}]")
    
    return results


# =============================================================================
# KNN CLASSIFICATION
# =============================================================================

def knn_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = [1, 3, 5, 10]
) -> dict:
    """
    Perform KNN search accuracy evaluation using FAISS (TMvec1 paper methodology).
    
    TMvec1 paper metric (from Table S2):
    "Top 1 accuracy for Topology in CATH S100 is 97.7%, which means that 97.7% of 
    the time, the nearest neighbor returned using TM-Vec is in the same fold 
    (topology level in CATH) as the query domain's fold."
    
    "Top 3 column for Topology indicates the percentage of the time that ONE OF 
    the top 3 nearest neighbors returned by TM-Vec shares the same fold."
    
    This is a HIT/RECALL metric: for each query, check if AT LEAST ONE of the 
    top-k neighbors has the same label as the query. This is different from 
    majority-voting classification.
    
    Steps:
    1. L2 normalize embeddings using faiss.normalize_L2()
    2. Build IndexFlatIP for inner product search (cosine similarity)
    3. For each query, check if any of the top-k neighbors share the same label
    """
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    # Ensure float32 and C-contiguous for FAISS (required for in-place normalize_L2)
    embeddings_norm = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # L2 normalize using FAISS (in-place) - exactly as TMvec1 paper does
    faiss.normalize_L2(embeddings_norm)
    
    # Build FAISS index (inner product = cosine similarity for normalized vectors)
    d = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_norm)
    
    results = {}
    max_k = max(k_values)
    
    # Query all at once (k+1 to exclude self-match in leave-one-out evaluation)
    print(f"  Building FAISS IndexFlatIP (d={d})...")
    print(f"  Querying {max_k + 1} nearest neighbors...")
    D, I = index.search(embeddings_norm, max_k + 1)
    
    for k in k_values:
        hits = 0
        
        for i in range(len(labels)):
            query_label = labels[i]
            # Get k nearest neighbors (excluding self - first result is always self)
            neighbor_indices = I[i, 1:k+1]  # Skip self (index 0)
            neighbor_labels = labels[neighbor_indices]
            
            # TMvec1 paper metric: hit if AT LEAST ONE neighbor has same label
            if query_label in neighbor_labels:
                hits += 1
        
        accuracy = hits / len(labels)
        
        results[k] = {
            'accuracy': accuracy,
            'n_hits': hits,
            'n_total': len(labels)
        }
        
        print(f"    Top-{k}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings for cosine similarity.
    
    Note: For KNN classification, we use faiss.normalize_L2() directly.
    This function is kept for backward compatibility with other code paths.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    return embeddings / norms


def evaluate_knn_all_levels(
    embeddings: np.ndarray,
    cath_df: pd.DataFrame,
    domain_ids: list[str],
    k_values: list[int] = [1, 3, 5, 10]
) -> dict:
    """
    Evaluate KNN classification at all CATH hierarchy levels using FAISS.
    
    Following TMvec1 paper methodology:
    1. Filter to domains with both embeddings and CATH labels
    2. For each CATH level (Class, Architecture, Topology, Superfamily):
       - Extract labels for that level
       - Run KNN classification with leave-one-out
       - Report accuracy for each k value
    
    The CATH hierarchy (from most general to most specific):
    - Class (C): 4 major classes (mainly alpha, mainly beta, mixed, few SS)
    - Architecture (A): Overall shape/arrangement
    - Topology (T): Fold level - connectivity of secondary structures  
    - Superfamily (H): Evolutionary relationship from structural similarity
    
    Returns results dictionary with accuracy for each level and k value.
    """
    # Filter to domains that have both embeddings and CATH labels
    valid_mask = [d in cath_df.index for d in domain_ids]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_domains = [domain_ids[i] for i in valid_indices]
    
    embeddings_filtered = embeddings[valid_indices]
    
    results = {}
    levels = ['class', 'architecture', 'topology', 'superfamily']
    
    print(f"\n" + "=" * 60)
    print(f"KNN Search Accuracy (TMvec1 paper methodology)")
    print(f"=" * 60)
    print(f"  Database size: {len(valid_domains)} domains")
    print(f"  Embedding dimension: {embeddings_filtered.shape[1]}")
    print(f"  K values: {k_values}")
    print(f"  Metric: Hit rate (at least one Top-k neighbor matches)")
    print("-" * 60)
    
    for level in levels:
        print(f"\n{level.upper()} level:")
        labels = cath_df.loc[valid_domains, level].values
        
        n_classes = len(np.unique(labels))
        print(f"  Number of classes: {n_classes}")
        
        level_results = knn_classification(
            embeddings_filtered, labels, k_values
        )
        
        results[level] = level_results
    
    return results


def evaluate_knn_similarity_all_levels(
    similarity_matrix: np.ndarray,
    cath_df: pd.DataFrame,
    domain_ids: list[str],
    k_values: list[int] = [1, 3, 5, 10],
    foldseek_valid_indices: list[int] = None
) -> dict:
    """
    Evaluate KNN classification using a precomputed similarity matrix.
    
    Used for Foldseek which provides pairwise TM-scores rather than embeddings.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        cath_df: DataFrame with CATH classifications
        domain_ids: List of domain IDs
        k_values: K values to evaluate
        foldseek_valid_indices: Indices of domains with sufficient Foldseek coverage.
                                Only these domains will be used as query points.
    
    Returns results dictionary with accuracy for each level and k value.
    """
    # Filter to domains that have both similarity data and CATH labels
    cath_valid_mask = [d in cath_df.index for d in domain_ids]
    cath_valid_indices = set(i for i, v in enumerate(cath_valid_mask) if v)
    
    # If Foldseek coverage indices provided, intersect with CATH valid indices
    if foldseek_valid_indices is not None:
        # Query indices = domains with both CATH labels AND sufficient Foldseek coverage
        query_indices = sorted(set(foldseek_valid_indices) & cath_valid_indices)
        print(f"\n*** FOLDSEEK SPARSE COVERAGE MODE ***")
        print(f"  - Only {len(foldseek_valid_indices)} domains have sufficient Foldseek neighbors")
        print(f"  - After filtering for CATH labels: {len(query_indices)} query domains")
    else:
        query_indices = sorted(cath_valid_indices)
    
    valid_domains = [domain_ids[i] for i in sorted(cath_valid_indices)]
    
    # Create index mapping from original to filtered
    original_to_filtered = {orig: filt for filt, orig in enumerate(sorted(cath_valid_indices))}
    filtered_query_indices = [original_to_filtered[i] for i in query_indices if i in original_to_filtered]
    
    # Extract submatrix for CATH-valid domains
    similarity_filtered = similarity_matrix[np.ix_(sorted(cath_valid_indices), sorted(cath_valid_indices))]
    
    results = {}
    levels = ['class', 'architecture', 'topology', 'superfamily']
    
    print(f"\nKNN Search Accuracy (TMvec1 paper methodology):")
    print(f"  - Total domains in similarity matrix: {len(sorted(cath_valid_indices))}")
    print(f"  - Query domains (evaluated): {len(filtered_query_indices)}")
    print(f"  - K values: {k_values}")
    print(f"  - Metric: Hit rate (at least one Top-k neighbor matches)")
    print("-" * 60)
    
    for level in levels:
        print(f"\n{level.upper()} level:")
        labels = cath_df.loc[valid_domains, level].values
        
        # Count classes in query set only
        query_labels = labels[filtered_query_indices]
        n_classes = len(np.unique(query_labels))
        print(f"  Number of classes in query set: {n_classes}")
        
        level_results = knn_classification_from_similarity(
            similarity_filtered, labels, k_values,
            valid_query_indices=filtered_query_indices
        )
        
        results[level] = level_results
    
    return results


# =============================================================================
# RESULTS REPORTING
# =============================================================================

def print_results_table(all_results: dict, k_values: list[int]):
    """Print a formatted comparison table of results (TMvec1 paper format)."""
    
    levels = ['class', 'architecture', 'topology', 'superfamily']
    
    print("\n" + "=" * 80)
    print("KNN SEARCH ACCURACY SUMMARY (TMvec1 paper metric)")
    print("Metric: % of queries where at least one Top-k neighbor has same label")
    print("=" * 80)
    
    # Header
    header = f"{'Method':<20} {'Level':<15}"
    for k in k_values:
        header += f"{'Top ' + str(k):<12}"
    print(header)
    print("-" * 80)
    
    for method, results in all_results.items():
        for level in levels:
            row = f"{method:<20} {level:<15}"
            for k in k_values:
                acc = results[level][k]['accuracy']
                row += f"{acc*100:>10.2f}%"
            print(row)
        print()


def save_results_csv(
    all_results: dict, 
    k_values: list[int],
    output_path: str
):
    """Save results to CSV file (TMvec1 paper format)."""
    rows = []
    levels = ['class', 'architecture', 'topology', 'superfamily']
    
    for method, results in all_results.items():
        for level in levels:
            row = {'method': method, 'level': level}
            for k in k_values:
                row[f'Top {k}'] = results[level][k]['accuracy']
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KNN Classification Benchmark for CATH Structural Hierarchy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run student model only
  python -m src.benchmarks.knn_cath_classifier --methods student

  # Run all methods
  python -m src.benchmarks.knn_cath_classifier --methods student tmvec1 tmvec2 foldseek

  # Use custom k values
  python -m src.benchmarks.knn_cath_classifier --k 1 3 5 10 20

  # Specify output file
  python -m src.benchmarks.knn_cath_classifier --output results/knn_results.csv
        """
    )
    
    parser.add_argument(
        '--fasta', type=str, 
        default='data/cath-domain-seqs-S100.fa',
        help='Path to FASTA file with protein sequences'
    )
    parser.add_argument(
        '--cath-list', type=str,
        default='data/cath-domain-list-S100.txt',
        help='Path to CATH domain classification file'
    )
    parser.add_argument(
        '--methods', nargs='+',
        choices=['student', 'tmvec1', 'tmvec2', 'foldseek'],
        default=['student', 'tmvec1', 'tmvec2', 'foldseek'],
        help='Methods to evaluate'
    )
    parser.add_argument(
        '--k', nargs='+', type=int,
        default=[1, 3, 5, 10],
        help='K values for KNN'
    )
    parser.add_argument(
        '--max-sequences', type=int, default=250000,
        help='Maximum number of sequences to process'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for embedding generation'
    )
    parser.add_argument(
        '--output', type=str,
        default='results/knn_cath_classification.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--foldseek-subset', action='store_true',
        help='Also evaluate other methods on Foldseek-covered subset for fair comparison'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load sequences
    seq_ids, sequences = load_fasta(args.fasta, args.max_sequences)
    
    # Check if CATH classification file exists
    cath_path = Path(args.cath_list)
    if not cath_path.exists():
        print(f"Error: CATH classification file not found at {cath_path}")
        sys.exit(1)
    
    # Load CATH classifications
    cath_df = load_cath_classifications(str(cath_path), seq_ids)
    
    # Run evaluation for each method
    all_results = {}
    foldseek_valid_indices = None  # Track for optional subset comparison
    foldseek_coverage_stats = None
    
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'='*60}")
        
        if method == 'foldseek':
            # Check if precomputed results exist, otherwise run Foldseek
            foldseek_results_path = "results/cath_foldseek_similarities.csv"
            if not Path(foldseek_results_path).exists():
                print("Running Foldseek benchmark (precomputed results not found)...")
                from src.benchmarks.foldseek import run_foldseek, parse_results, save_results as save_foldseek
                df = run_foldseek("data/pdb/cath-s100", "binaries/foldseek", 1)
                pairs = parse_results(df)
                save_foldseek(pairs, foldseek_results_path)
            
            # Load with coverage analysis
            max_k = max(args.k)
            similarity_matrix, foldseek_valid_indices, foldseek_coverage_stats = load_foldseek_similarity_matrix(
                foldseek_results_path,
                seq_ids,
                min_neighbors=max_k  # Need at least k neighbors for k-NN
            )
            
            print(f"\n*** Foldseek Coverage Analysis ***")
            print(f"  Total pairwise coverage: {foldseek_coverage_stats['coverage_pct']:.1f}%")
            print(f"  Domains with >= {max_k} neighbors: {len(foldseek_valid_indices)}/{foldseek_coverage_stats['total_domains']} ({100*len(foldseek_valid_indices)/foldseek_coverage_stats['total_domains']:.1f}%)")
            
            # Evaluate using similarity matrix (only on domains with sufficient coverage)
            results = evaluate_knn_similarity_all_levels(
                similarity_matrix,
                cath_df,
                seq_ids,
                k_values=args.k,
                foldseek_valid_indices=foldseek_valid_indices
            )
        else:
            # Embedding-based methods
            if method == 'student':
                from src.benchmarks.tmvec2_student import generate_embeddings
                embeddings = generate_embeddings(sequences=sequences, batch_size=args.batch_size, device=device)
            elif method == 'tmvec1':
                from src.benchmarks.tmvec1 import generate_embeddings
                embeddings = generate_embeddings(sequences=sequences, batch_size=max(8, args.batch_size // 4), device=device)
            elif method == 'tmvec2':
                from src.benchmarks.tmvec2 import generate_embeddings
                embeddings = generate_embeddings(sequences=sequences, batch_size=args.batch_size, device=device)
            
            # Evaluate embeddings on full dataset
            results = evaluate_knn_all_levels(
                embeddings,
                cath_df,
                seq_ids,
                k_values=args.k
            )
        
        all_results[method] = results
    
    # Optional: Also evaluate embedding methods on Foldseek-covered subset for fair comparison
    if args.foldseek_subset and foldseek_valid_indices is not None:
        print(f"\n{'='*60}")
        print("FAIR COMPARISON: Evaluating on Foldseek-covered subset")
        print(f"{'='*60}")
        print(f"Subset size: {len(foldseek_valid_indices)} domains with sufficient Foldseek coverage")
        
        for method in args.methods:
            if method == 'foldseek':
                continue  # Already evaluated on this subset
            
            print(f"\n--- {method.upper()} (Foldseek subset) ---")
            
            # Re-generate embeddings if not cached (in practice, should cache these)
            if method == 'student':
                from src.benchmarks.tmvec2_student import generate_embeddings
                embeddings = generate_embeddings(sequences=sequences, batch_size=args.batch_size, device=device)
            elif method == 'tmvec1':
                from src.benchmarks.tmvec1 import generate_embeddings
                embeddings = generate_embeddings(sequences=sequences, batch_size=max(8, args.batch_size // 4), device=device)
            elif method == 'tmvec2':
                from src.benchmarks.tmvec2 import generate_embeddings
                embeddings = generate_embeddings(sequences=sequences, batch_size=args.batch_size, device=device)
            
            # Evaluate on Foldseek subset using embedding-to-similarity conversion
            # Convert embeddings to similarity matrix for fair comparison
            # Use FAISS normalize_L2 following TMvec1 paper methodology
            embeddings_norm = np.ascontiguousarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_norm)
            embedding_similarity = embeddings_norm @ embeddings_norm.T
            
            results_subset = evaluate_knn_similarity_all_levels(
                embedding_similarity,
                cath_df,
                seq_ids,
                k_values=args.k,
                foldseek_valid_indices=foldseek_valid_indices
            )
            
            all_results[f"{method}_foldseek_subset"] = results_subset
    
    # Print summary table
    print_results_table(all_results, args.k)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(all_results, args.k, str(output_path))
    
    print("\nDone!")

if __name__ == '__main__':
    main()
