"""Descriptive plots from the common training rows only."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLASSIFICATION_TARGET, REGRESSION_TARGET, CLASS_LABELS, FEATURE_LABELS, feature_label
from plots import save_figure


def run_eda(frame_train, output):
    """Retain the original outcome, distribution, correlation and category plots.

    These plots cover the training rows shared by both supervised tasks. Source
    column names stay unchanged; figures use English field labels. Source
    categorical values remain unchanged for traceability.
    """
    counts = frame_train[CLASSIFICATION_TARGET].value_counts()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([CLASS_LABELS.get(x, x) for x in counts.index], counts.values)
    axis.set(title='Outcome distribution (training rows)', ylabel='Students')
    save_figure(figure, output, 'eda_outcomes.png')
    columns = ['nota_cualificacion_previa', 'nota_admision', 'edad_al_matricularse',
               'asignaturas_1sem_aprobadas', 'nota_media_1sem', 'asignaturas_2sem_aprobadas', 'nota_media_2sem']
    figure, axes = plt.subplots(len(columns), 1, figsize=(10, 3 * len(columns)))
    for axis, column in zip(axes, columns):
        for label, subset in frame_train.groupby(CLASSIFICATION_TARGET):
            axis.hist(subset[column], bins=30, alpha=.5, label=CLASS_LABELS.get(label, label))
        axis.set_title(f'{feature_label(column)} (training rows)')
        axis.legend()
    save_figure(figure, output, 'eda_histograms.png')
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, column in zip(axes.flat, columns):
        groups = [frame_train.loc[frame_train[CLASSIFICATION_TARGET].eq(label), column] for label in CLASS_LABELS]
        axis.boxplot(groups, tick_labels=list(CLASS_LABELS.values()))
        axis.set_title(feature_label(column), fontsize=9)
    axes.flat[-1].set_visible(False)
    figure.suptitle('Grade and entry variables by outcome (training rows)')
    save_figure(figure, output, 'eda_boxplots.png')
    correlations = frame_train.select_dtypes(include=[np.number]).corr()
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlations.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS), mask=np.triu(np.ones_like(correlations, dtype=bool)), cmap='RdBu_r', center=0, ax=axis)
    axis.set_title('Numeric correlations (training rows)')
    save_figure(figure, output, 'eda_correlations.png')
    correlations[REGRESSION_TARGET].drop(REGRESSION_TARGET).sort_values().to_csv(output / 'eda_grade_correlations.csv', index_label='source_feature')
    categorical = ['genero', 'becado', 'deudor', 'matricula_al_dia', 'asistencia_diurna_vespertina', 'desplazado', 'internacional']
    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    for i, (axis, column) in enumerate(zip(axes.flat, categorical)):
        table = pd.crosstab(frame_train[column], frame_train[CLASSIFICATION_TARGET], normalize='index') * 100
        table.rename(columns=CLASS_LABELS).plot.bar(stacked=True, ax=axis, legend=i == 0)
        axis.set(title=feature_label(column), ylabel='Percentage', xlabel='Source category')
    axes.flat[-1].set_visible(False)
    figure.suptitle('Categorical associations (training rows)')
    save_figure(figure, output, 'eda_categories.png')
