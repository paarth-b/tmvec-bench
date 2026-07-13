#!/usr/bin/env python
"""
Compare encoding time speedups of TMVec2 and TMVec2 Student against all other models.

Reads encoding time benchmark CSVs and produces two speedup CSVs.
"""

import pandas as pd
import os

resdir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

methods = {
    'tmvec1': 'TMVec1',
    'tmvec2': 'TMVec2',
    'tmvec2_student': 'TMVec2 Student',
    'foldseek': 'Foldseek',
}


def main():
    times = {}
    for key, name in methods.items():
        df = pd.read_csv(os.path.join(resdir, f'{key}_encoding_times.csv'))
        times[name] = df.set_index('encoding_size')['mean_seconds']

    for ref_model, filename in [('TMVec2', 'tmvec2_encoding_speedups.csv'),
                                ('TMVec2 Student', 'tmvec2_student_encoding_speedups.csv')]:
        rows = []
        for baseline in methods.values():
            if baseline == ref_model:
                continue
            common_sizes = sorted(set(times[ref_model].index).intersection(times[baseline].index))
            for size in common_sizes:
                speedup = times[baseline][size] / times[ref_model][size]
                rows.append({
                    'compared_to': baseline,
                    'encoding_size': size,
                    'speedup': round(speedup, 2),
                })

        result = pd.DataFrame(rows)
        outpath = os.path.join(resdir, filename)
        result.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')


if __name__ == '__main__':
    main()
