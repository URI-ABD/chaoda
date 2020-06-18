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
            manifold = Manifold(data, METRICS[metric])
            manifold.layers = [Graph(manifold.root)]
            manifold.build_tree(criterion.MaxDepth(MAX_DEPTH))

            max_lfd, min_lfd, grace_depth = manifold.lfd_range(percentiles=(80, 60))
            [cluster.mark(max_lfd, min_lfd) for cluster in manifold.layers[grace_depth].clusters]
            manifold.build_graph()

            num_subsumed = len(manifold.graph.clusters) - len(manifold.graph.transition_clusters)
            logging.info(f'optimal_graph, clusters: {len(manifold.graph.clusters)}, num_subsumed: {num_subsumed}, '
                         f'fraction_subsumed: {num_subsumed / len(manifold.graph.clusters):.4f}')

            for depth, layer in enumerate(manifold.layers[1:]):
                num_subsumed = len(layer.clusters) - len(layer.transition_clusters)
                logging.info(f'depth: {depth + 1}, clusters: {len(layer.clusters)}, num_subsumed: {num_subsumed}, '
                             f'fraction_subsumed: {num_subsumed / len(layer.clusters):.4f}')


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
