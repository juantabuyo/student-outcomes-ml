"""One shared holdout, task-specific exclusions, and fold-local encoding.

This corrects the original mismatch between the EDA and regression splits.
It cannot undo historical exploration of the data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from config import (SEED, TEST_SIZE, SEPARATOR, BINARY_COLUMNS, CATEGORICAL_COLUMNS,
                    SECOND_SEMESTER_COLUMNS, CLASSIFICATION_TARGET, REGRESSION_TARGET,
                    FIRST_SEMESTER_NUMERIC, SECOND_SEMESTER_NUMERIC, PASS_THROUGH_COLUMNS)


def load_data(path):
    """Read the supplied semicolon-separated course dataset without changing it."""
    frame = pd.read_csv(Path(path), sep=SEPARATOR)
    required = {CLASSIFICATION_TARGET, REGRESSION_TARGET, *BINARY_COLUMNS,
                *CATEGORICAL_COLUMNS, *FIRST_SEMESTER_NUMERIC, *SECOND_SEMESTER_COLUMNS,
                *PASS_THROUGH_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    # Every remaining column would become a predictor, in both tasks, without any leakage review.
    unexpected = set(frame.columns) - required - set(PASS_THROUGH_COLUMNS)
    if unexpected:
        raise ValueError(f"Columns outside the source contract (data_schema.csv) would enter the "
                         f"models as predictors; remove them first: {sorted(unexpected)}")
    if frame.isna().any().any():
        raise ValueError("Missing values require an explicit policy; this dataset has none.")
    return frame


def encode_binary(frame):
    """Apply a fixed source-data mapping, with no fitted statistics."""
    encoded = frame.copy()
    for column in BINARY_COLUMNS:
        unknown = set(encoded[column].unique()) - {'si', 'no'}
        if unknown:
            raise ValueError(f"Unexpected binary categories in {column}")
        encoded[column] = encoded[column].map({'si': 1, 'no': 0})
    return encoded


def split_indices(frame):
    """Use the original classification split for both tasks and their EDA."""
    if not frame.index.is_unique:
        raise ValueError("Row indices must be unique to preserve split membership.")
    train, test = train_test_split(frame.index.to_numpy(), test_size=TEST_SIZE,
                                  random_state=SEED, shuffle=True,
                                  stratify=frame[CLASSIFICATION_TARGET])
    return train, test


def prepare_classification(frame, train, test):
    encoded = encode_binary(frame)
    features = encoded.drop(columns=[CLASSIFICATION_TARGET])
    labels = LabelEncoder().fit(encoded.loc[train, CLASSIFICATION_TARGET])
    return (features.loc[train], features.loc[test],
            labels.transform(encoded.loc[train, CLASSIFICATION_TARGET]),
            labels.transform(encoded.loc[test, CLASSIFICATION_TARGET]), labels)


def prepare_regression(frame, train, test):
    """Exclude contemporary second-semester fields and the final outcome."""
    encoded = encode_binary(frame)
    features = encoded.drop(columns=SECOND_SEMESTER_COLUMNS + [CLASSIFICATION_TARGET])
    target = encoded[REGRESSION_TARGET]
    return features.loc[train], features.loc[test], target.loc[train], target.loc[test]


def make_pipeline(model, columns, step):
    """Fit the vocabulary and scaling inside each training fold, then the model.

    Scaling includes dummy columns as in the original implementation. Unknown
    categories map to all-zero dummies. All known categories are retained.
    """
    categorical = [column for column in CATEGORICAL_COLUMNS if column in columns]
    encoder = ColumnTransformer(
        [('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)],
        remainder='passthrough', verbose_feature_names_out=False)
    return Pipeline([('encoding', encoder), ('scaler', StandardScaler()), (step, model)])


def prepare_unsupervised(frame, include_second_semester=True):
    """Fit exploratory profiles on all rows; this is not a predictive holdout test."""
    encoded = encode_binary(frame)
    columns = FIRST_SEMESTER_NUMERIC + (SECOND_SEMESTER_NUMERIC if include_second_semester else []) + BINARY_COLUMNS
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(encoded[columns]), columns=columns, index=frame.index)
    return scaled, frame[CLASSIFICATION_TARGET].copy(), columns, scaler


def zero_grade_counts(frame):
    """Report observed zeros without inferring the reason for them."""
    zero = frame[REGRESSION_TARGET].eq(0)
    return {'zero_grades': int(zero.sum()),
            'zero_grades_with_evaluations': int((zero & frame['asignaturas_2sem_evaluadas'].gt(0)).sum())}
