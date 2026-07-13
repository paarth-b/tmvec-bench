#!/usr/bin/env python
"""pLM-BLAST benchmark: generate pairwise similarity scores via pLM-BLAST.

Shells out to the pLM-BLAST repository (https://github.com/labstructbioinf/pLM-BLAST)
to (1) compute ProtT5 per-residue embeddings for a FASTA dataset and
(2) run all-vs-all local-alignment search. The resulting alignments are
aggregated to one score per unordered pair and written in the standard
seq1_id,seq2_id,tm_score schema used by the other benchmarks.

Requires a local clone of pLM-BLAST. Point to it via --plmblast-repo or the
PLMBLAST_REPO environment variable. If the repo has a virtualenv at
<repo>/benchmark/, its python is used automatically.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATASETS = {
    "cath": {
        "fasta": "data/fasta/cath-s100-unique-10k.fa",
        "output": "/work/nvme/beut/paarthbatra/data/results/cath_plmblast_similarities.csv",
        "emb_dir": "/work/nvme/beut/paarthbatra/data/plmblast_emb/cath",
    },
    "scope40": {
        "fasta": "data/fasta/scop40.fasta",
        "output": "/work/nvme/beut/paarthbatra/data/results/scope40_plmblast_similarities.csv",
        "emb_dir": "/work/nvme/beut/paarthbatra/data/plmblast_emb/scope40",
    },
}

DEFAULT_REPO = os.environ.get("PLMBLAST_REPO", "/u/paarthbatra/git/pLM-BLAST")


def resolve_python(repo):
    venv_python = Path(repo) / "benchmark" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def embeddings_exist(emb_dir):
    emb_dir = Path(emb_dir)
    index = emb_dir.with_suffix(emb_dir.suffix + ".csv") if emb_dir.suffix else emb_dir.parent / f"{emb_dir.name}.csv"
    return emb_dir.is_dir() and any(emb_dir.iterdir()) and index.exists()


def run_embeddings(python, repo, fasta, emb_dir, embedder, gpu, truncate):
    if embeddings_exist(emb_dir):
        print(f"Using cached embeddings at {emb_dir}")
        return
    Path(emb_dir).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, "embeddings.py", "start",
        str(fasta), str(emb_dir),
        "-embedder", embedder,
        "-bs", "0", "--asdir",
    ]
    if gpu:
        cmd.append("--gpu")
    if truncate:
        cmd.extend(["-t", str(truncate)])
    print(f"Running embeddings: {' '.join(cmd)} (cwd={repo})")
    subprocess.run(cmd, cwd=repo, check=True)


def run_plmblast(python, repo, emb_dir, raw_output, workers, cosine_cutoff, alignment_cutoff):
    Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, "scripts/plmblast.py",
        str(emb_dir), str(emb_dir), str(raw_output),
        "-workers", str(workers),
        "-cosine_percentile_cutoff", str(cosine_cutoff),
        "-alignment_cutoff", str(alignment_cutoff),
    ]
    print(f"Running pLM-BLAST: {' '.join(cmd)} (cwd={repo})")
    subprocess.run(cmd, cwd=repo, check=True)


def parse_results(raw_csv):
    """Aggregate local-alignment hits to one score per unordered pair.

    pLM-BLAST may emit multiple local alignments per (qid, sid). We keep the
    best (max) score per directed pair, then average the two directions to
    produce the final symmetric score.
    """
    df = pd.read_csv(raw_csv, sep=';', low_memory=False)
    df = df[df['qid'] != df['sid']]
    per_dir = df.groupby(['qid', 'sid'], as_index=False)['score'].max()
    per_dir['seq1_id'] = per_dir[['qid', 'sid']].min(axis=1)
    per_dir['seq2_id'] = per_dir[['qid', 'sid']].max(axis=1)
    result = per_dir.groupby(['seq1_id', 'seq2_id'], as_index=False)['score'].mean()
    return result.rename(columns={'score': 'tm_score'})


def main():
    parser = argparse.ArgumentParser(description="pLM-BLAST all-vs-all benchmark")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="cath")
    parser.add_argument("--fasta", help="Override FASTA path")
    parser.add_argument("--output", help="Override final output CSV path")
    parser.add_argument("--emb-dir", help="Override embeddings directory")
    parser.add_argument("--plmblast-repo", default=DEFAULT_REPO,
                        help="Path to pLM-BLAST clone (default: $PLMBLAST_REPO or /u/paarthbatra/git/pLM-BLAST)")
    parser.add_argument("--python", default=None,
                        help="Python interpreter to invoke pLM-BLAST scripts with "
                             "(default: <repo>/benchmark/bin/python if present, else current interpreter)")
    parser.add_argument("--embedder", default="pt", help="pLM-BLAST embedder flag (default: pt for ProtT5)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for embedding generation")
    parser.add_argument("--truncate", type=int, default=None,
                        help="Max residues per sequence (pLM-BLAST -t)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--cosine-cutoff", type=float, default=70.0,
                        help="-cosine_percentile_cutoff (pre-screening)")
    parser.add_argument("--alignment-cutoff", type=float, default=0.3,
                        help="-alignment_cutoff (min pLM-BLAST score to keep)")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    fasta = Path(args.fasta or cfg["fasta"]).absolute()
    output = Path(args.output or cfg["output"]).resolve()
    emb_dir = Path(args.emb_dir or cfg["emb_dir"]).resolve()
    repo = Path(args.plmblast_repo).resolve()
    python = args.python or resolve_python(repo)
    raw_output = output.with_name(output.stem + "_raw.csv")

    print("=" * 80)
    print("pLM-BLAST Benchmark")
    print(f"Dataset:  {args.dataset}")
    print(f"FASTA:    {fasta}")
    print(f"Repo:     {repo}")
    print(f"Python:   {python}")
    print(f"Emb dir:  {emb_dir}")
    print(f"Raw out:  {raw_output}")
    print(f"Output:   {output}")
    print(f"Workers:  {args.workers}")
    print("=" * 80)

    if not repo.exists():
        raise ValueError(f"pLM-BLAST repo not found at {repo}. "
                         f"Set --plmblast-repo or PLMBLAST_REPO.")
    if not fasta.exists():
        raise ValueError(f"FASTA not found: {fasta}")

    index_csv = emb_dir.with_suffix(".csv")
    if index_csv.exists() and emb_dir.exists() and any(emb_dir.glob("*.emb")):
        print(f"Reusing existing embeddings at {emb_dir} ({index_csv} present); skipping embedding step.")
    else:
        run_embeddings(python, repo, fasta, emb_dir,
                       args.embedder, not args.no_gpu, args.truncate)
    run_plmblast(python, repo, emb_dir, raw_output,
                 args.workers, args.cosine_cutoff, args.alignment_cutoff)

    df = parse_results(raw_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved {len(df):,} pairs to {output}")
    if len(df):
        print(f"Mean: {df['tm_score'].mean():.4f}, Std: {df['tm_score'].std():.4f}")


if __name__ == "__main__":
    main()
