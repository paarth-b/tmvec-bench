Analysis of the SCOPe 40 dataset
------

SCOPe release 2.01 was used in this study, in consistency with the Foldseek paper.

SCOPe 40 data were downloaded from the Foldseek repo:

- https://wwwuser.gwdguser.de/~compbiol/foldseek/

Classification information was retrieved from:

- https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt

Execute `get_truth_scope.py` to obtain a list of ground-truth matches per domain pair per classification unit (1 - same unit; 2 - otherwise). This will generate `truth.tsv`.

Place the raw results in the `results` folder.

Execute `merge_results.py` to combine the results into one table file `results.tsv`.

Execute `plot_accuracy.py` to analyze TM-score prediction accuracy and generate plots.

Edit the following Python scripts to uncomment SCOPe levels and comment CATH levels.

Execute `plot_homology.py` to analyze homology detection performance using shared protein domain pairs.

Execute `plot_homology_all.py` to analyze homology detection performance using all  protein domain pairs.
