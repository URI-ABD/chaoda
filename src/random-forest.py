import logging
import os
from typing import List, Dict

import numpy as np
from pyclam import Manifold, criterion, Graph
from scipy.stats import gmean, hmean
from sklearn.metrics import roc_auc_score

from src import datasets as chaoda_datasets
from src.datasets import DATASETS
from src.methods import METHODS

TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train'))

NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 20
TRAIN_DATASETS = [
    'annthyroid',
    'cardio',
    'pima',
    'shuttle',
    'thyroid',
    'vowels',
]
MEANS = {
    'gmean': gmean,  # uses log. getting log of zero error.
    'hmean': hmean,
    'mean': np.mean,
}

# TODO: Normalize features to account for different ranges of lfd, radii, size of dataset, etc


def lfd_features(graph: Graph) -> List[float]:
    lfds = [cluster.local_fractal_dimension / cluster.parent.local_fractal_dimension for cluster in graph.clusters]
    means = [mean(lfds) for mean in MEANS.values()]
    return list(map(float, means))


def cardinality_features(graph: Graph) -> List[float]:
    cardinalities = [cluster.cardinality / cluster.parent.cardinality for cluster in graph.clusters]
    means = [mean(cardinalities) for mean in MEANS.values()]
    return list(map(float, means))


def radii_features(graph: Graph) -> List[float]:
    radii = [
        (cluster.radius if cluster.radius != 0 else 1e-4)
        / (cluster.parent.radius if cluster.parent.radius != 0 else 1e-4)
        for cluster in graph.clusters
    ]
    means = [mean(radii) for mean in MEANS.values()]
    return list(map(float, means))


def subgraph_cardinality_features(graph: Graph) -> List[float]:
    cardinalities = [subgraph.cardinality for subgraph in graph.subgraphs]
    factor = sum(cardinalities)
    cardinalities = [c / factor for c in cardinalities]
    means = [mean(cardinalities) for mean in MEANS.values()]
    return list(map(float, means))


def subgraph_population_features(graph: Graph) -> List[float]:
    factor = graph.manifold.root.cardinality
    populations = [subgraph.population / factor for subgraph in graph.subgraphs]
    means = [mean(populations) for mean in MEANS.values()]
    return list(map(float, means))


# TODO: Subgraph diameter
# TODO: Subgraph centrality
# TODO: Think of other features to extract


FEATURE_EXTRACTORS = {
    'lfd': lfd_features,
    'cardinality': cardinality_features,
    'radii': radii_features,
    # 'subgraph-cardinalities': subgraph_cardinality_features,
    # 'subgraph-populations': subgraph_population_features,
}


def create_features(graph: Graph) -> List[float]:
    features: List[float] = list()
    [features.extend(extractor(graph)) for extractor in FEATURE_EXTRACTORS.values()]
    return features


def auc_scores(graph: Graph, labels: List[int]) -> List[float]:
    scores: List[float] = list()
    for method in METHODS:
        anomalies: Dict[int, float] = METHODS[method](graph)
        y_true, y_score = list(), list()
        [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
        scores.append(roc_auc_score(y_true, y_score))

    return scores


def create_training_data(filename: str, datasets: List[str]):
    feature_names = ','.join([f'{extractor}-{mean}' for extractor in FEATURE_EXTRACTORS for mean in MEANS])
    labels = ','.join([f'auc-{method}' for method in METHODS])
    header = f'dataset,depth,{labels},{feature_names}\n'

    with open(filename, 'w') as fp:
        fp.write(header)
        for dataset in datasets:
            data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
            manifold = Manifold(data, metric='euclidean').build(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.Depth(MAX_DEPTH),
            )
            for layer in manifold.layers:
                if layer.cardinality >= 32:
                    logging.info(f'writing layer {layer.depth}...')
                    features = create_features(layer)
                    scores = auc_scores(layer, labels)
                    features_line = ','.join([f'{f:.12f}' for f in features])
                    scores_line = ','.join([f'{f:.12f}' for f in scores])
                    fp.write(f'{dataset},{layer.depth},{scores_line},{features_line}\n')
            # break
    return


if __name__ == '__main__':
    os.makedirs(TRAIN_PATH, exist_ok=True)
    _train_filename = os.path.join(TRAIN_PATH, 'train.csv')
    create_training_data(_train_filename, TRAIN_DATASETS)
    _test_filename = os.path.join(TRAIN_PATH, 'test.csv')
    _test_datasets = [_d for _d in DATASETS if _d not in TRAIN_DATASETS]
    create_training_data(_test_filename, _test_datasets)
