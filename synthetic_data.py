"""Generate a small fictitious schema fixture, not a substitute student dataset."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import (BINARY_COLUMNS, CATEGORICAL_COLUMNS, FIRST_SEMESTER_NUMERIC,
                    SECOND_SEMESTER_COLUMNS, CLASSIFICATION_TARGET, REGRESSION_TARGET)


def make_synthetic(rows=180, seed=2026):
    rng = np.random.default_rng(seed)
    data = {}
    for name in FIRST_SEMESTER_NUMERIC + SECOND_SEMESTER_COLUMNS + ['asignaturas_1sem_convalidadas']:
        data[name] = rng.integers(0, 20, size=rows).astype(float)
    for name in BINARY_COLUMNS:
        data[name] = rng.choice(['si', 'no'], size=rows)
    for name in CATEGORICAL_COLUMNS:
        data[name] = rng.choice(['synthetic_A', 'synthetic_B', 'synthetic_C'], size=rows)
    data[CLASSIFICATION_TARGET] = np.resize(['abandono', 'matriculado', 'graduado'], rows)
    data[REGRESSION_TARGET] = np.clip(.7 * data['nota_media_1sem'] + rng.normal(0, 2, rows), 0, 20)
    data[REGRESSION_TARGET][:9] = 0
    return pd.DataFrame(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/synthetic_students.csv'))
    parser.add_argument('--rows', type=int, default=180)
    args = parser.parse_args()
    if args.rows < 60:
        parser.error('Use at least 60 rows for the demonstration CV.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(args.rows).to_csv(args.output, sep=';', index=False)
    print(f'Wrote {args.rows} entirely fictitious rows to {args.output}.')
