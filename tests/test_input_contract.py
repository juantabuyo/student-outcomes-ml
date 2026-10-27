"""Regression tests: load_data accepts exactly the columns of data_schema.csv.

Any other column used to flow into the predictors of both tasks without review."""
import csv
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import config
from config import (BINARY_COLUMNS, CATEGORICAL_COLUMNS, CLASSIFICATION_TARGET, FIRST_SEMESTER_NUMERIC,
                    REGRESSION_TARGET, SECOND_SEMESTER_COLUMNS, SEPARATOR)
from preprocessing import load_data, prepare_classification, prepare_regression, split_indices

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'data/synthetic_students.csv'


def write(directory, frame):
    path = Path(directory) / 'students.csv'
    frame.to_csv(path, sep=SEPARATOR, index=False)
    return path


class InputContract(unittest.TestCase):
    def test_declared_columns_are_exactly_the_schema_file(self):
        declared = {CLASSIFICATION_TARGET, REGRESSION_TARGET, *BINARY_COLUMNS, *CATEGORICAL_COLUMNS,
                    *FIRST_SEMESTER_NUMERIC, *SECOND_SEMESTER_COLUMNS, *config.PASS_THROUGH_COLUMNS}
        with (ROOT / 'data_schema.csv').open(newline='', encoding='utf-8') as stream:
            schema = {row['column'] for row in csv.DictReader(stream)}
        self.assertEqual(declared, schema)

    def test_the_supplied_fixture_is_accepted_unchanged(self):
        frame = load_data(FIXTURE)
        self.assertEqual(list(frame.columns), list(pd.read_csv(FIXTURE, sep=SEPARATOR).columns))
        train, test = split_indices(frame)
        classification = prepare_classification(frame, train, test)
        regression = prepare_regression(frame, train, test)
        self.assertIn('asignaturas_1sem_convalidadas', classification[0].columns)
        self.assertIn('asignaturas_1sem_convalidadas', regression[0].columns)

    def test_an_extra_column_is_rejected_before_it_can_reach_the_predictors(self):
        base = pd.read_csv(FIXTURE, sep=SEPARATOR)
        for name in ('unexpected_target_proxy', 'Unnamed: 0', 'student_id'):
            with self.subTest(column=name), tempfile.TemporaryDirectory() as directory:
                frame = base.copy()
                frame[name] = range(len(frame))
                with self.assertRaisesRegex(ValueError, name):
                    load_data(write(directory, frame))

    def test_a_missing_column_is_still_reported(self):
        base = pd.read_csv(FIXTURE, sep=SEPARATOR)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing source columns'):
                load_data(write(directory, base.drop(columns=['pib'])))

    def test_a_missing_pass_through_predictor_is_rejected(self):
        base = pd.read_csv(FIXTURE, sep=SEPARATOR)
        for column in config.PASS_THROUGH_COLUMNS:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'Missing source columns:.*' + column):
                    load_data(write(directory, base.drop(columns=[column])))


if __name__ == '__main__':
    unittest.main()
