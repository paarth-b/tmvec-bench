Analysis of the CATH S100 dataset
------

CATH release 4.4.0 was used in this study. It is available for download at:

- ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/

The CATH domain classification file (`cath-domain-list-S100.txt`) and sequence
file (`cath-domain-seqs-S100.fa`) are included in `data/fasta/`.

The domain list used for benchmarking is `domains.txt` (10,000 domains).

### Workflow (run from repo root)

1. Generate ground truth classifications:
```bash
uv run python -m src.plotting.get_truth_cath
```
This produces `truth.tsv` in this directory.

2. Run all accuracy benchmarks (see main README for commands).

3. Merge results into a combined table:
```bash
uv run python -m src.plotting.merge_results --dataset cath
```
This produces `results.tsv` in this directory.

4. Plot TM-score prediction accuracy:
```bash
uv run python -m src.plotting.plot_accuracy --dataset cath
```
This produces plots in `plots/` and `metrics.tsv`.

5. Calculate and plot homology detection metrics:
```bash
uv run python -m src.plotting.calc_homology --dataset cath
uv run python -m src.plotting.plot_homology --dataset cath
```

For the "all pairs" variant:
```bash
uv run python -m src.plotting.calc_homology_all --dataset cath
uv run python -m src.plotting.plot_homology --dataset cath --suffix .all --metrics-dir src/plotting/cath/metrics_all
```
