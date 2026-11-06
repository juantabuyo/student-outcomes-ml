"""Small plotting helpers shared by the two supervised tasks."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from config import SEED, N_JOBS, feature_label


def save_figure(figure, output, name):
    figure.tight_layout()
    figure.savefig(output / name, dpi=140, bbox_inches='tight')
    plt.close(figure)


def compare_models(table, metrics, output, name, folds):
    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)
    for axis, (column, label) in zip(axes[0], metrics):
        axis.barh(table['model'], table[column + '_mean'], xerr=table[column + '_std'], color='#3498db')
        axis.set_title(f'{label} (CV, {folds} folds)')
    save_figure(figure, output, name)


def feature_importance(pipe, X_train, X_test, y_test, step, scoring, output, prefix):
    """Interpret an already fitted model; these results never select a model.

    Impurity importance is per encoded feature. Permutation importance is now
    per original column, since the pipeline also owns one-hot encoding.
    Figures use English field labels; CSVs retain the source identifiers.
    """
    features = pipe.named_steps['encoding'].get_feature_names_out()
    impurity = pd.Series(pipe.named_steps[step].feature_importances_, index=features).sort_values(ascending=False)
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=5,
                                  random_state=SEED, n_jobs=N_JOBS, scoring=scoring)
    permutation = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    for values, suffix, title in [(impurity, 'impurity', 'Impurity importance (encoded features)'),
                                  (permutation, 'permutation', f'Permutation importance ({scoring}, holdout)')]:
        values.rename('importance').to_csv(output / f'{prefix}_{suffix}_importance.csv', index_label='source_feature')
        figure, axis = plt.subplots(figsize=(10, 8))
        values.head(20).rename(index=feature_label).plot.barh(ax=axis, color='#3498db')
        axis.invert_yaxis()
        axis.set(title=title, xlabel='Importance', ylabel='Feature')
        save_figure(figure, output, f'{prefix}_{suffix}_importance.png')
    return impurity
