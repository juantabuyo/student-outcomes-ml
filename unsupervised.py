"""PCA, K-means and Ward profiles: exploration of the full supplied dataset."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from config import SEED, CLASS_LABELS, FEATURE_LABELS
from plots import save_figure


def analyse_pca(scaled, labels, output, prefix):
    pca = PCA(random_state=SEED).fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    count90 = int(np.argmax(cumulative >= .90) + 1)
    count95 = int(np.argmax(cumulative >= .95) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(cumulative) + 1), pca.explained_variance_ratio_)
    axes[0].set(title='Explained variance by component', xlabel='Component', ylabel='Explained variance')
    axes[1].plot(range(1, len(cumulative) + 1), cumulative, 'o-')
    for threshold in [.9, .95]:
        axes[1].axhline(threshold, linestyle='--', label=f'{threshold:.0%}')
    axes[1].set(title='Cumulative explained variance', xlabel='Components', ylabel='Cumulative variance')
    axes[1].legend()
    save_figure(figure, output, prefix + '_pca_variance.png')
    pca2 = PCA(n_components=2, random_state=SEED)
    projected = pca2.fit_transform(scaled)
    figure, axis = plt.subplots(figsize=(10, 7))
    for label in labels.unique():
        mask = labels == label
        axis.scatter(projected[mask, 0], projected[mask, 1], alpha=.4, s=15, label=CLASS_LABELS.get(label, label))
    axis.set(xlabel=f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})', ylabel=f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})',
             title='PCA coloured by outcome (outcome excluded from fit)')
    axis.legend()
    save_figure(figure, output, prefix + '_pca_outcomes.png')
    pd.DataFrame(pca2.components_.T, index=scaled.columns, columns=['PC1', 'PC2']).to_csv(output / (prefix + '_loadings.csv'), index_label='source_feature')
    return projected, count90, count95


def elbow_and_silhouette(scaled, output, prefix):
    rows = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(scaled)
        rows.append({'k': k, 'inertia': model.inertia_, 'silhouette': silhouette_score(scaled, model.labels_)})
    table = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric in zip(axes, ['inertia', 'silhouette']):
        axis.plot(table.k, table[metric], 'o-')
        axis.set(xlabel='Clusters (k)', ylabel=metric.capitalize(), title=metric.capitalize())
    save_figure(figure, output, prefix + '_elbow_silhouette.png')
    table.to_csv(output / (prefix + '_cluster_search.csv'), index=False)
    return int(table.loc[table.silhouette.idxmax(), 'k'])


def run_unsupervised(scaled, labels, output, prefix):
    """Keep k=3 for interpretation; report rather than optimise away disagreement.

    The original choice of three clusters was motivated by the three known
    outcome categories. Outcomes do not enter the numerical clustering fit,
    but calling this choice entirely label-independent would be misleading.
    """
    projected, count90, count95 = analyse_pca(scaled, labels, output, prefix)
    best_k = elbow_and_silhouette(scaled, output, prefix)
    summary = {'components_90_percent': count90, 'components_95_percent': count95,
               'best_k_by_silhouette': best_k, 'reported_k': 3,
               'scope': 'Exploratory full-data profiles; no out-of-sample clustering claim'}
    for name, model in [('kmeans', KMeans(n_clusters=3, random_state=SEED, n_init=10)),
                        ('ward', AgglomerativeClustering(n_clusters=3, linkage='ward'))]:
        clusters = model.fit_predict(scaled)
        summary[name] = {'silhouette': silhouette_score(scaled, clusters), 'ari': adjusted_rand_score(labels, clusters)}
        figure, axis = plt.subplots(figsize=(10, 7))
        scatter = axis.scatter(projected[:, 0], projected[:, 1], c=clusters, cmap='Set1', alpha=.4, s=15)
        axis.set(title=f'{name.capitalize()} clusters (k=3)', xlabel='PC1', ylabel='PC2')
        figure.colorbar(scatter, ax=axis, label='Cluster')
        save_figure(figure, output, prefix + '_' + name + '.png')
        if name == 'kmeans':
            profiles = scaled.assign(cluster=clusters).groupby('cluster').mean()
            top = profiles.var().sort_values(ascending=False).head(10).index
            figure, axis = plt.subplots(figsize=(12, 6))
            sns.heatmap(profiles[top].T.rename(index=FEATURE_LABELS), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axis)
            axis.set(title='Standardised cluster means (most variable fields)', xlabel='Cluster', ylabel='Feature')
            save_figure(figure, output, prefix + '_profiles.png')
            profiles.to_csv(output / (prefix + '_profile_means.csv'))
            pd.crosstab(clusters, labels.map(CLASS_LABELS), margins=True).to_csv(output / (prefix + '_outcome_crosstab.csv'))
    (output / (prefix + '_metrics.json')).write_text(json.dumps(summary, indent=2))
    return summary
