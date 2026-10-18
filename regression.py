"""Predict the second-semester grade using the original model settings."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import SEED, CV_FOLDS, N_JOBS
from preprocessing import make_pipeline
from plots import save_figure, compare_models, feature_importance
from classification import GB_CONFIGS


def define_models():
    """Preserve regularised linear models, neighbours, trees and ensembles."""
    return {
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.1, random_state=SEED, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=5000),
        'KNN': KNeighborsRegressor(n_neighbors=7),
        'DecisionTree': DecisionTreeRegressor(random_state=SEED, max_depth=10),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=N_JOBS, max_depth=10),
        # Depth 3 is the default in the selected final source, not the old depth-4 baseline.
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=SEED, max_depth=3),
    }


def compare_models_cv(X_train, y_train, folds=CV_FOLDS):
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    scoring = {'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    for name, model in define_models().items():
        print(f'Regression CV: {name}', flush=True)
        scores = cross_validate(make_pipeline(model, X_train.columns, 'reg'), X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=N_JOBS, error_score='raise')
        row = {'model': name}
        for metric in scoring:
            values = scores['test_' + metric] * (1 if metric == 'r2' else -1)
            row.update({metric + '_mean': values.mean(), metric + '_std': values.std()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values('r2_mean', ascending=False)


def tune_gradient_boosting(X_train, y_train, output, folds=CV_FOLDS):
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    best_score, best_config = -np.inf, None
    for config in GB_CONFIGS:
        print(f'Regression grid: {config}', flush=True)
        pipe = make_pipeline(GradientBoostingRegressor(random_state=SEED, **config), X_train.columns, 'reg')
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2', n_jobs=N_JOBS, error_score='raise')
        rows.append({**config, 'r2_mean': scores.mean(), 'r2_std': scores.std()})
        if scores.mean() > best_score:
            best_score, best_config = float(scores.mean()), config
    pd.DataFrame(rows).sort_values('r2_mean', ascending=False).to_csv(output / 'regression_grid.csv', index=False)
    pipe = make_pipeline(GradientBoostingRegressor(random_state=SEED, **best_config), X_train.columns, 'reg')
    pipe.fit(X_train, y_train)
    return pipe, best_config, best_score


def evaluate_model(pipe, X_test, y_test, output):
    predicted = pipe.predict(X_test)
    metrics = {'r2': r2_score(y_test, predicted), 'rmse': float(np.sqrt(mean_squared_error(y_test, predicted))),
               'mae': mean_absolute_error(y_test, predicted)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, predicted, alpha=0.3, s=10)
    limits = [min(y_test.min(), predicted.min()) - .5, max(y_test.max(), predicted.max()) + .5]
    axes[0].plot(limits, limits, 'r--', label='y = x')
    axes[0].set(xlabel='Observed grade', ylabel='Predicted grade', title=f'Holdout predictions (R²={metrics["r2"]:.3f})')
    axes[0].legend()
    axes[1].hist(y_test - predicted, bins=40, color='#2ecc71', edgecolor='black', alpha=.7)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set(xlabel='Residual (observed − predicted)', ylabel='Frequency', title='Holdout residuals')
    save_figure(figure, output, 'regression_predictions.png')
    return predicted, metrics


def run_regression(X_train, X_test, y_train, y_test, output, folds=CV_FOLDS):
    comparison = compare_models_cv(X_train, y_train, folds)
    comparison.to_csv(output / 'regression_comparison.csv', index=False)
    compare_models(comparison, [('r2', 'R²'), ('rmse', 'RMSE'), ('mae', 'MAE')], output, 'regression_comparison.png', folds)
    pipe, config, score = tune_gradient_boosting(X_train, y_train, output, folds)
    predictions, metrics = evaluate_model(pipe, X_test, y_test, output)
    metrics.update({'selected_config': config, 'selection_cv_r2': score, 'baseline_max_depth': 3})
    (output / 'regression_metrics.json').write_text(json.dumps(metrics, indent=2))
    importance = feature_importance(pipe, X_train, X_test, y_test, 'reg', 'r2', output, 'regression')
    return pipe, comparison, predictions, importance, metrics
