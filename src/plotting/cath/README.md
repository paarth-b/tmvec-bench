Analysis of the CATH S100 dataset
------

CATH release 4.4.0 was used in this study. It is available for download at:

- ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/

Download sequence data:

```bash
wget -O cath-domain-seqs-S100.fa ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/sequence-data/cath-domain-seqs-S100-v4_4_0.fa
```

Download classification data:

```bash
wget -O cath-domain-list-S100.txt ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-domain-list-S100-v4_4_0.txt
wget -O cath-names.txt ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-names-v4_4_0.txt
```

Execute `get_truth_cath.py` to obtain a list of ground-truth matches per domain pair per classification unit (1 - same unit; 2 - otherwise). This will generate `truth.tsv`.

Place the raw results in the `results` folder.

Execute `merge_results.py` to combine the results into one table file `results.tsv`.

Execute `plot_accuracy.py` to analyze TM-score prediction accuracy and generate plots.

Execute `plot_homology.py` to analyze homology detection performance using shared protein domain pairs.

Execute `plot_homology_all.py` to analyze homology detection performance using all  protein domain pairs.
