import itertools
from typing import List, Dict, Tuple, Union

import numpy as np
from pyclam import Manifold, criterion, Graph
from sklearn.metrics import roc_auc_score

from src import datasets as chaoda_datasets
from src.methods import METHODS

TRAIN_DATASETS = ['annthyroid', 'cardio', 'pima', 'shuttle', 'thyroid', 'vowels']
NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 30


def lfd_features(graph: Graph) -> List[float]:
    # TODO
    pass


def cardinality_features(graph: Graph) -> List[float]:
    # TODO
    pass


def radii_features(graph: Graph) -> List[float]:
    # TODO
    pass


def parent_child_features(graph: Graph) -> List[float]:
    # TODO
    pass


def create_features(graph: Graph) -> List[float]:
    feature_extractors = [
        lfd_features,
        cardinality_features,
        radii_features,
        parent_child_features,
    ]
    features: List[float] = list()
    [features.extend(extractor(graph)) for extractor in feature_extractors]
    return features


def auc_scores(graph: Graph, labels: List[int]) -> List[float]:
    # TODO: get auc scores for graph
    pass


def create_training_data(filename: str, datasets: Union[List[str], None] = None):
    if datasets is None:
        datasets = TRAIN_DATASETS

    with open(filename, 'w') as fp:
        for dataset in datasets:
            data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
            manifold = Manifold(data, metric='euclidean').build(criterion.MaxDepth(MAX_DEPTH))
            for layer in manifold.layers:
                if layer.cardinality >= 32:
                    features = create_features(layer)
                    scores = auc_scores(layer, labels)
                    features_line = ','.join([f'{f:.3f}' for f in features])
                    scores_line = ','.join([f'{f:.3f}' for f in scores])
                    fp.write(f'{features_line},{scores_line}\n')
                    pass
    return


if __name__ == '__main__':
    pass
