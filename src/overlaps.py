import logging
import os
from typing import List

import numpy as np
from pyclam import Manifold, Graph, criterion, Cluster

from src.datasets import get, read, METRICS
from src.reproduce import BUILD_PATH, PLOTS_PATH

NORMALIZE = False
SUB_SAMPLE = 50_000
MAX_DEPTH = 30
STEP = 10


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
            manifold.graphs = [Graph(manifold.root)]
            manifold.build_tree(criterion.MaxDepth(MAX_DEPTH))

            for depth, graph in enumerate(manifold.graphs[1:]):
                clusters: List[Cluster] = list(graph.clusters.keys())
                clusters.sort(key=lambda c: c.radius)

                for i in range(len(clusters) - 1, -1, -1):
                    if clusters[i].cardinality > 1:
                        for j in range(i):
                            if not clusters[j].absorbable:
                                c1, c2 = clusters[i], clusters[j]
                                if c1.distance_from([c2.argmedoid]) < c1.radius - c2.radius:
                                    c2.absorbable = True

                num_absorbable = len([cluster for cluster in clusters if cluster.absorbable])
                logging.info(f'depth: {depth + 1}, clusters: {len(clusters)}, num_absorbable: {num_absorbable}, '
                             f'fraction_ absorbable: {num_absorbable / len(clusters):.4f}')


if __name__ == '__main__':
    _datasets = [
        # 'vowels',
        'cardio',
        # 'thyroid',
        # 'musk',
        # 'satimage-2',
        # 'satellite',
        # 'optdigits',
    ]
    _metrics = ['euclidean']
    subsumed_fractions(_datasets, _metrics)
