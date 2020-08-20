import os
from typing import List, Set

import numpy as np
from pyclam import Manifold, Cluster, criterion, Graph

from src import datasets as chaoda_datasets
from src.datasets import METRICS
from src.meta_ml import auc_scores, METHOD_NAMES
from src.utils import TRAIN_PATH

CONSTANTS = os.path.join(os.path.dirname(__file__), '..', 'constants.npy')
NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 20


def ema(current, previous, smoothing=2.0, period=10):
    """ Calculates the Exponential moving average.
    """
    alpha = smoothing / (1 + period)
    return alpha * current + previous * (1 - alpha)


def calculate_ratios(cluster: Cluster):
    """ Calculates the relevant ratios and ema of ratios for a given cluster.
    """
    # Add features here as they are needed.
    cluster.ratios = np.array([  # Child/Parent Ratios
        cluster.local_fractal_dimension / cluster.parent.local_fractal_dimension,  # local fractal dimension
        cluster.cardinality / cluster.parent.cardinality,  # cardinality
        max(cluster.radius, 1e-16) / max(cluster.parent.radius, 1e-16)  # radius
    ])
    # noinspection PyUnresolvedReferences
    cluster.ema_ratios = np.array([  # Exponential Moving Averages
        ema(c, p) for c, p in zip(cluster.ratios, cluster.parent.ratios)
    ])
    return


def predict_auc(manifold: Manifold, constants: np.ndarray) -> Manifold:
    """ Predicts the auc contribution from each cluster in the manifold using the constants provided by a linear regression.
    """
    manifold.root.ratios = np.ones(shape=(len(constants) // 2,), dtype=float)
    for layer in manifold.layers[1:]:
        clusters: List[Cluster] = [c for c in layer.clusters if c.depth == layer.depth]
        for cluster in clusters:
            calculate_ratios(cluster)
            # noinspection PyUnresolvedReferences
            chaoda_features = np.concatenate([cluster.ratios, cluster.ema_ratios])
            cluster.chaoda_auc = np.dot(constants, chaoda_features)

    return manifold


def subtree_auc(cluster: Cluster) -> np.ndarray:
    """ Returns the distribution of predicted auc for the subtree of the given cluster.
    Assumes that cluster has children.
    """
    # noinspection PyUnresolvedReferences
    return np.asarray([
        child.chaoda_auc
        for layer in cluster.manifold.layers[cluster.depth + 1:]
        for child in layer.clusters
        if cluster.name == child.name[:len(cluster.name)]
    ])


def select_graph(manifold: Manifold) -> Manifold:
    graph: Set[Cluster] = set()
    clusters: List[Cluster] = [manifold.root]
    while clusters:
        new_clusters: List[Cluster] = list()
        for cluster in clusters:
            if cluster.children:
                # noinspection PyUnresolvedReferences
                if cluster.chaoda_auc >= np.percentile(subtree_auc(cluster), q=60):
                    graph.add(cluster)
                else:
                    new_clusters.extend(cluster.children)
            else:
                graph.add(cluster)
        clusters = new_clusters

    manifold.graph = Graph(*graph)
    manifold.graph.build_edges()

    return manifold


def evaluate_auc(datasets: List[str], metrics: List[str], constants: np.array, filename: str):
    """ Evaluate auc performance from linear regression constants.

    :param datasets: the dataset to work on.
    :param metrics: the distance metric to use.
    :param constants: the constants learned by linear regression.
    :param filename: file in which to store auc performance.
    """
    for dataset in datasets:
        for metric in metrics:
            data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
            min_points: int
            if len(data) < 1_000:
                min_points = 1
                # continue
            elif len(data) < 4_000:
                min_points = 2
            elif len(data) < 16_000:
                min_points = 4
            elif len(data) < 64_000:
                min_points = 8
                # continue
            else:
                min_points = 16
                # continue

            manifold = Manifold(data, METRICS[metric]).build_tree(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.MinPoints(min_points),
            )

            manifold = predict_auc(manifold, constants)
            manifold = select_graph(manifold)

            with open(filename, 'a') as fp:
                scores = auc_scores(manifold.graph, labels)
                line = ','.join([f'{score:.3f}' for score in scores])
                fp.write(f'{dataset},{metric},{line}\n')

    return


if __name__ == "__main__":
    os.makedirs(TRAIN_PATH, exist_ok=True)

    _datasets = ['musk']
    _metrics = ['euclidean', 'manhattan']
    _constants = np.array([1, 2, 3, 4, 5, 6])  # np.fromfile(CONSTANTS, dtype=np.float)
    _filename = os.path.join(TRAIN_PATH, 'lr_predictions.csv')
    with open(_filename, 'w') as _fp:
        _methods = ','.join([f'{_method}' for _method in METHOD_NAMES.values()])
        _fp.write(f'dataset,metric,{_methods}\n')

    evaluate_auc(_datasets, _metrics, _constants, _filename)
