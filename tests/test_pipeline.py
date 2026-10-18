"""Focused checks for split consistency and preprocessing boundaries."""
import unittest
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from preprocessing import (split_indices, prepare_classification, prepare_regression,
                           make_pipeline, zero_grade_counts, encode_binary)
from classification import GB_CONFIGS, define_models as classification_models
from regression import define_models as regression_models
from config import SECOND_SEMESTER_COLUMNS, CLASSIFICATION_TARGET, REGRESSION_TARGET
from synthetic_data import make_synthetic


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.frame = make_synthetic()
        self.train, self.test = split_indices(self.frame)

    def test_shared_split_and_regression_exclusions(self):
        cls = prepare_classification(self.frame, self.train, self.test)
        reg = prepare_regression(self.frame, self.train, self.test)
        self.assertEqual(set(cls[0].index), set(reg[0].index))
        self.assertEqual(set(cls[1].index), set(reg[1].index))
        self.assertFalse(set(self.train) & set(self.test))
        self.assertEqual(len(self.train) + len(self.test), len(self.frame))
        self.assertFalse(set(SECOND_SEMESTER_COLUMNS + [CLASSIFICATION_TARGET]) & set(reg[0].columns))
        np.testing.assert_array_equal(self.train, split_indices(self.frame)[0])
        self.assertEqual(set(self.frame.loc[self.train].index), set(cls[0].index))

    def test_encoder_and_scaler_fit_only_training_fold(self):
        X_train, _, y_train, _, _ = prepare_classification(self.frame, self.train, self.test)
        train_positions, valid_positions = next(KFold(2, shuffle=True, random_state=42).split(X_train))
        fold_train = X_train.iloc[train_positions].copy()
        fold_valid = X_train.iloc[valid_positions].copy()
        fold_valid['curso'] = 'validation_only_category'
        pipe = make_pipeline(DecisionTreeClassifier(random_state=42), fold_train.columns, 'clf')
        pipe.fit(fold_train, y_train[train_positions])
        encoder = pipe.named_steps['encoding'].named_transformers_['categorical']
        self.assertFalse(any('validation_only_category' in values for values in encoder.categories_))
        transformed = pipe.named_steps['encoding'].transform(fold_train)
        np.testing.assert_allclose(pipe.named_steps['scaler'].mean_, transformed.mean(axis=0))
        self.assertEqual(len(pipe.predict(fold_valid)), len(valid_positions))

    def test_heldout_categories_do_not_change_training_vocabulary(self):
        X_train, X_test, y_train, _, _ = prepare_classification(self.frame, self.train, self.test)
        X_test = X_test.copy()
        X_test['curso'] = 'heldout_only'
        pipe = make_pipeline(DecisionTreeClassifier(random_state=42), X_train.columns, 'clf').fit(X_train, y_train)
        self.assertNotIn('heldout_only', ' '.join(pipe.named_steps['encoding'].get_feature_names_out()))
        self.assertEqual(len(pipe.predict(X_test)), len(X_test))

    def test_zero_grade_does_not_imply_no_evaluation(self):
        self.frame.loc[0, REGRESSION_TARGET] = 0
        self.frame.loc[0, 'asignaturas_2sem_evaluadas'] = 3
        counts = zero_grade_counts(self.frame)
        self.assertGreater(counts['zero_grades_with_evaluations'], 0)
        self.assertGreaterEqual(counts['zero_grades'], counts['zero_grades_with_evaluations'])

    def test_original_model_settings_preserved(self):
        self.assertEqual(regression_models()['GradientBoosting'].max_depth, 3)
        self.assertEqual(classification_models()['GradientBoosting'].max_depth, 4)
        self.assertEqual(len(GB_CONFIGS), 5)
        self.assertEqual([c['max_depth'] for c in GB_CONFIGS], [3, 5, 3, 5, 3])

    def test_bad_binary_category_is_explicit(self):
        self.frame.loc[0, 'deudor'] = 'unknown'
        with self.assertRaises(ValueError):
            encode_binary(self.frame)

    def test_original_frame_unchanged(self):
        before = self.frame.copy(deep=True)
        prepare_classification(self.frame, self.train, self.test)
        prepare_regression(self.frame, self.train, self.test)
        self.assertTrue(before.equals(self.frame))


if __name__ == '__main__':
    unittest.main()
