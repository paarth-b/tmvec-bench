Analysis of the SCOPe 40 dataset
------

SCOPe release 2.01 was used in this study, in consistency with the Foldseek paper.

SCOPe 40 data were downloaded from the Foldseek repo:

- https://wwwuser.gwdguser.de/~compbiol/foldseek/

The SCOPe 40 sequence file (`scop40.fasta`) is included in `data/fasta/`.

The domain list used for benchmarking is `domains.txt` (11,211 domains).

The SCOPe classification file (`dir.des.scope.2.01-stable.txt`) is auto-downloaded
by `get_truth_scope.py` if not present in `data/fasta/`.

Classification information was retrieved from:

- https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt

### Workflow (run from repo root)

1. Generate ground truth classifications:
```bash
uv run python -m src.plotting.get_truth_scope
```
This produces `truth.tsv` in this directory.

2. Run all accuracy benchmarks (see main README for commands).

3. Merge results into a combined table:
```bash
uv run python -m src.plotting.merge_results --dataset scope40
```
This produces `results.tsv` in this directory.

4. Plot TM-score prediction accuracy:
```bash
uv run python -m src.plotting.plot_accuracy --dataset scope40
```
This produces plots in `plots/` and `metrics.tsv`.

5. Calculate and plot homology detection metrics:
```bash
uv run python -m src.plotting.calc_homology --dataset scope40
uv run python -m src.plotting.plot_homology --dataset scope40
```

For the "all pairs" variant:
```bash
uv run python -m src.plotting.calc_homology_all --dataset scope40
uv run python -m src.plotting.plot_homology --dataset scope40 --suffix .all --metrics-dir src/plotting/scope/metrics_all
```
