import os
import numpy as np

from pyclam import Manifold, Graph, Cluster

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
    :oaram constants: the things Najib gives me.
    """
    dataset, labels = datasets.read(dataset)
    manifold = Manifold(dataset, 'euclidean')
    manifold.build()
    for layer in manifold.layers[1:]:
        clusters: List[Cluster] = [c for c in layer.clusters if c.depth == layer.depth]
        for c in clusters:
            # Add features here as they are needed.
            c._chaoda_features = np.array([
                # Child/Parent Ratio: Fractal Dimension
                c.local_fractal_dimension / c.parent.local_fractal_dimension,
                # Exponential Moving Average: Fractal Dimension
                ema(c.local_fractal_dimension, c.parent.local_fractal_dimension),
                # Child/Parent Ratio: 
            ])
            # Here we compute the AUC
            c._chaoda_auc = np.dot(constants, c._chaoda_features)


if __name__ == "__main__":
    constants = np.array([1, 2, 3]) # np.fromfile(CONSTANTS, dtype=np.float)
    for dataset in DATASETS:
        main(dataset, constants)
