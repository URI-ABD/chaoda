import os
from typing import List, Dict, Tuple

from matplotlib import pyplot as plt
from pyclam import Manifold, criterion, Cluster

from src import datasets as chaoda_datasets
from src.datasets import METRICS
from src.utils import PLOTS_PATH

NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 50

EXPLORATION_DATASETS = [
    'cardio',
    'musk',
    'thyroid',
    'vowels',
]


def draw_histogram(values: List[float], filename: str):
    plt.close('all')
    plt.hist(values)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
    plt.close('all')
    return


def explore_features():
    for dataset in EXPLORATION_DATASETS:
        # make folder to store histograms
        histograms_directory = os.path.join(PLOTS_PATH, f'meta_ml', f'{dataset}')
        os.makedirs(histograms_directory, exist_ok=True)

        data, labels = chaoda_datasets.read(dataset, NORMALIZE, SUB_SAMPLE)
        min_points: int
        if len(data) < 1_000:
            min_points = 1
        elif len(data) < 4_000:
            min_points = 2
        elif len(data) < 16_000:
            min_points = 4
        elif len(data) < 64_000:
            min_points = 8
        else:
            min_points = 16

        for metric in ['euclidean', 'manhattan']:
            manifold: Manifold = Manifold(data, METRICS[metric]).build(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.MinPoints(min_points),
                criterion.Layer(MAX_DEPTH),
            )

            # extract features for every cluster in the tree
            clusters: Dict[Cluster, Tuple[float, float, float]] = {manifold.root: (0, 0, 0)}
            for layer in manifold.layers:
                for cluster in layer.clusters:
                    for child in cluster.children:
                        clusters[child] = (
                            child.local_fractal_dimension / cluster.local_fractal_dimension,
                            child.cardinality / cluster.cardinality,
                            max(child.radius, 1e-16) / max(cluster.radius, 1e-16)
                        )

            ratio_names = ['lfd', 'cardinality', 'radius']
            for i, name in enumerate(ratio_names):
                ratios = list(v[i] for v in clusters.values())
                filename = os.path.join(histograms_directory, f'{metric}-{name}.png')
                draw_histogram(ratios, filename)
    return


if __name__ == '__main__':
    explore_features()
