"""Compare the original classifiers and tune the original five GB settings."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
from config import SEED, CV_FOLDS, N_JOBS, CLASS_LABELS
from preprocessing import make_pipeline
from plots import save_figure, compare_models, feature_importance

# These are the five configurations in the selected university submission.
GB_CONFIGS = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
    {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
]


def define_models():
    return {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=SEED, solver='lbfgs'),
        'LogReg_balanced': LogisticRegression(max_iter=2000, random_state=SEED, solver='lbfgs', class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'DecisionTree': DecisionTreeClassifier(random_state=SEED, max_depth=10),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=N_JOBS, max_depth=10),
        'RF_balanced': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=N_JOBS, max_depth=10, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED, max_depth=4, learning_rate=0.1),
    }


def compare_models_cv(X_train, y_train, folds=CV_FOLDS):
    """Macro-F1 is primary; accuracy adds context for unequal class frequencies."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, model in define_models().items():
        print(f'Classification CV: {name}', flush=True)
        scores = cross_validate(make_pipeline(model, X_train.columns, 'clf'), X_train, y_train,
                                cv=cv, scoring=['accuracy', 'f1_macro'], n_jobs=N_JOBS, error_score='raise')
        row = {'model': name}
        for metric in ['accuracy', 'f1_macro']:
            values = scores['test_' + metric]
            row.update({metric + '_mean': values.mean(), metric + '_std': values.std()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values('f1_macro_mean', ascending=False)


def tune_gradient_boosting(X_train, y_train, output, folds=CV_FOLDS):
    """Choose within the original restricted grid, using training CV only.

    The comparison baseline uses depth 4, which is absent from this grid.
    Therefore this is not a claim to select the best of all compared models.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    best_score, best_config = -np.inf, None
    for config in GB_CONFIGS:
        print(f'Classification grid: {config}', flush=True)
        pipe = make_pipeline(GradientBoostingClassifier(random_state=SEED, **config), X_train.columns, 'clf')
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=N_JOBS, error_score='raise')
        rows.append({**config, 'f1_macro_mean': scores.mean(), 'f1_macro_std': scores.std()})
        if scores.mean() > best_score:
            best_score, best_config = float(scores.mean()), config
    pd.DataFrame(rows).sort_values('f1_macro_mean', ascending=False).to_csv(output / 'classification_grid.csv', index=False)
    pipe = make_pipeline(GradientBoostingClassifier(random_state=SEED, **best_config), X_train.columns, 'clf')
    pipe.fit(X_train, y_train)
    return pipe, best_config, best_score


def evaluate_model(pipe, X_test, y_test, encoder, output):
    predicted = pipe.predict(X_test)
    metrics = {'accuracy': accuracy_score(y_test, predicted), 'f1_macro': f1_score(y_test, predicted, average='macro')}
    labels = [CLASS_LABELS.get(label, label) for label in encoder.classes_]
    report = classification_report(y_test, predicted, target_names=labels, zero_division=0)
    (output / 'classification_report.txt').write_text(report)
    for normalise in [False, True]:
        matrix = confusion_matrix(y_test, predicted, normalize='true' if normalise else None)
        figure, axis = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, cmap='Blues', values_format='.2f' if normalise else 'd')
        axis.set_title('Normalised holdout confusion matrix' if normalise else 'Holdout confusion matrix')
        save_figure(figure, output, 'classification_confusion_normalised.png' if normalise else 'classification_confusion.png')
    return predicted, metrics


def run_classification(X_train, X_test, y_train, y_test, encoder, output, folds=CV_FOLDS):
    comparison = compare_models_cv(X_train, y_train, folds)
    comparison.to_csv(output / 'classification_comparison.csv', index=False)
    compare_models(comparison, [('accuracy', 'Accuracy'), ('f1_macro', 'Macro-F1')], output, 'classification_comparison.png', folds)
    pipe, config, score = tune_gradient_boosting(X_train, y_train, output, folds)
    predictions, metrics = evaluate_model(pipe, X_test, y_test, encoder, output)
    metrics.update({'selected_config': config, 'selection_cv_f1_macro': score,
                    'comparison_baseline_f1_macro': float(comparison.loc[comparison.model.eq('GradientBoosting'), 'f1_macro_mean'].iloc[0]),
                    'selection_scope': 'Best within the original five-configuration GB grid, not all compared models'})
    (output / 'classification_metrics.json').write_text(json.dumps(metrics, indent=2))
    importance = feature_importance(pipe, X_train, X_test, y_test, 'clf', 'f1_macro', output, 'classification')
    return pipe, comparison, predictions, importance, metrics
