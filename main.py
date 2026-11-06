"""Run the student-outcomes experiment on an explicitly supplied local CSV."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version
from threadpoolctl import threadpool_limits
from config import SEED, CV_FOLDS, CLASSIFICATION_TARGET
from preprocessing import (load_data, split_indices, prepare_classification, prepare_regression,
                           prepare_unsupervised, zero_grade_counts)
from eda import run_eda
from classification import run_classification
from regression import run_regression
from unsupervised import run_unsupervised


def run(data_path, output, sample_rows=None, folds=CV_FOLDS, synthetic=False):
    """Execute every task; reduced runs only reduce rows/folds, never model settings."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    total_rows = len(frame)
    if sample_rows is not None and sample_rows < len(frame):
        from sklearn.model_selection import train_test_split
        frame, _ = train_test_split(frame, train_size=sample_rows, stratify=frame[CLASSIFICATION_TARGET], random_state=SEED)
    train, test = split_indices(frame)
    metadata = {'prepared_version_date': '2026-10-17', 'synthetic': synthetic,
                'source_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                'source_rows': total_rows, 'run_rows': len(frame), 'train_rows': len(train), 'holdout_rows': len(test),
                'seed': SEED, 'cv_folds': folds, 'historically_explored_holdout': not synthetic,
                'split': 'One outcome-stratified 80/20 split shared by EDA and supervised tasks',
                'split_index_sha256': hashlib.sha256(train.tobytes() + test.tobytes()).hexdigest(),
                'zero_grade_observations': zero_grade_counts(frame),
                'python': platform.python_version(),
                'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'threadpoolctl']}}
    print(f'Rows: {len(frame)}; train: {len(train)}; holdout: {len(test)}; folds: {folds}', flush=True)
    with threadpool_limits(limits=1):
        run_eda(frame.loc[train], output)
        classification = run_classification(*prepare_classification(frame, train, test), output, folds)[-1]
        regression = run_regression(*prepare_regression(frame, train, test), output, folds)[-1]
        profiles = {}
        for include_second, prefix in [(True, 'with_second_semester'), (False, 'without_second_semester')]:
            scaled, labels, _, _ = prepare_unsupervised(frame, include_second)
            profiles[prefix] = run_unsupervised(scaled, labels, output, prefix)
    metadata['elapsed_seconds'] = time.perf_counter() - started
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    summary = {'classification': classification, 'regression': regression, 'clustering': profiles}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/local_run'))
    parser.add_argument('--sample-rows', type=int, help='Optional stratified subset for a reduced check')
    parser.add_argument('--folds', type=int, default=CV_FOLDS)
    parser.add_argument('--synthetic', action='store_true', help='Label results as a fictitious fixture run')
    args = parser.parse_args()
    if args.folds < 2 or (args.sample_rows is not None and args.sample_rows < 60):
        parser.error('Use at least two folds and 60 sampled rows.')
    run(args.data, args.output, args.sample_rows, args.folds, args.synthetic)
