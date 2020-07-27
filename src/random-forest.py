import logging
from typing import List, Dict, Union

import numpy as np
from pyclam import Manifold, criterion, Graph
from scipy.stats import gmean, hmean
from sklearn.metrics import roc_auc_score

from src import datasets as chaoda_datasets
from src.methods import METHODS

TRAIN_DATASETS = ['annthyroid', 'cardio', 'pima', 'shuttle', 'thyroid', 'vowels']
NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 20
MEANS = {
    # 'gmean': gmean,  # uses log. getting log of zero error.
    'hmean': hmean,
    'mean': np.mean,
}


def lfd_features(graph: Graph) -> List[float]:
    lfds = [cluster.local_fractal_dimension for cluster in graph.clusters]
    means = [mean(lfds) for mean in MEANS.values()]
    return list(map(float, means))


def cardinality_features(graph: Graph) -> List[float]:
    cardinalities = [cluster.cardinality for cluster in graph.clusters]
    means = [mean(cardinalities) for mean in MEANS.values()]
    return list(map(float, means))


def radii_features(graph: Graph) -> List[float]:
    radii = [cluster.radius for cluster in graph.clusters]
    means = [mean(radii) for mean in MEANS.values()]
    return list(map(float, means))


def parent_child_features(graph: Graph) -> List[float]:
    ratios = [cluster.parent.cardinality / cluster.cardinality for cluster in graph.clusters]
    means = [mean(ratios) for mean in MEANS.values()]
    return list(map(float, means))


FEATURE_EXTRACTORS = {
        'lfd': lfd_features,
        'cardinality': cardinality_features,
        'radii': radii_features,
        'pc_ratio': parent_child_features,
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

    return list(map(float, [mean(scores) for mean in MEANS.values()]))


def create_training_data(filename: str, datasets: Union[List[str], None] = None):
    if datasets is None:
        datasets = TRAIN_DATASETS

    header = ','.join([f'{extractor}-{mean}' for extractor in FEATURE_EXTRACTORS for mean in MEANS])
    header = header + ',' + ','.join([f'auc-{mean}' for mean in MEANS]) + '\n'

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
                    features_line = ','.join([f'{f:.3f}' for f in features])
                    scores_line = ','.join([f'{f:.3f}' for f in scores])
                    fp.write(f'{features_line},{scores_line}\n')
            break
    return


if __name__ == '__main__':
    _filename = '../train.csv'
    create_training_data(_filename)
