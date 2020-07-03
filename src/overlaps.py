import logging
import os
from typing import List

import numpy as np
from pyclam import Manifold, Graph, criterion, Cluster

from src.datasets import get, read, METRICS
from src.reproduce import BUILD_PATH, PLOTS_PATH

NORMALIZE = False
SUB_SAMPLE = 50_000
MAX_DEPTH = 20
FRACTION = 0.05
STEP = 10


# noinspection DuplicatedCode
def subsumed_fractions(datasets: List[str], metrics: List[str]):
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    for dataset in datasets:
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        labels = np.squeeze(labels)

        for metric in metrics:
            logging.info(f'dataset: {dataset}, metric: {metric}, shape: {data.shape}, outliers: {labels.sum()}')
            manifold = Manifold(data, METRICS[metric]).build(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.LFDRange(80, 20),
                criterion.MinimizeSubsumed(FRACTION),
            )

            fraction = len(manifold.graph.subsumed_clusters) / manifold.graph.cardinality
            logging.info(f'optimal_graph, fraction_subsumed: {fraction:.4f}')


if __name__ == '__main__':
    _datasets = [
        'vowels',
        # 'cardio',
        # 'thyroid',
        # 'musk',
        # 'satimage-2',
        # 'satellite',
        # 'optdigits',
    ]
    _metrics = ['euclidean']
    subsumed_fractions(_datasets, _metrics)
