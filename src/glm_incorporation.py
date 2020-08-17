import os
from typing import List

import numpy as np
from pyclam import Manifold, Cluster, criterion

from src import datasets

DATASETS = [
    'musk',
]

CONSTANTS = os.path.join(os.path.dirname(__file__), '..', 'constants.npy')

assert all(d in datasets.DATASETS.keys() for d in DATASETS)


def ema(current, previous, smoothing=2.0, period=10):
    """Exponential moving average."""
    alpha = smoothing / (1 + period)
    return alpha * current + previous * (1 - alpha)


def main(dataset: str, constants: np.array):
    """Do things.

    :param dataset: the dataset to work on, duh
    :param constants: the things Najib gives me.
    """
    dataset, labels = datasets.read(dataset)
    manifold = Manifold(dataset, 'euclidean').build(
        criterion.MaxDepth(20),
        criterion.Layer(10),
    )

    manifold.root.ratios = np.ones(shape=(len(constants) // 2,), dtype=float)
    for layer in manifold.layers[1:]:
        clusters: List[Cluster] = [c for c in layer.clusters if c.depth == layer.depth]
        for cluster in clusters:
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
            # Here we compute the AUC
            chaoda_features = np.concatenate([cluster.ratios, cluster.ema_ratios])
            cluster.chaoda_auc = np.dot(constants, chaoda_features)

    return


if __name__ == "__main__":
    _constants = np.array([1, 2, 3, 4, 5, 6])  # np.fromfile(CONSTANTS, dtype=np.float)
    for _dataset in DATASETS:
        main(_dataset, _constants)
