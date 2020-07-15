from typing import List, Dict, Tuple

import numpy as np
from pyclam import Manifold, criterion
from sklearn.metrics import roc_auc_score

from src import datasets
from src.methods import METHODS

TRAIN_DATASETS = ['annthyroid', 'cardio', 'pima', 'shuttle', 'thyroid', 'vowels']
NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 30


def create_features():
    space = (100, 100)
    features = (np.indices(space).reshape(len(space), -1).T + 1)
    features = np.asarray([row for row in features if row[0] >= row[1]])
    return features


def score(upper: float, lower: float) -> Tuple[float, float]:
    auc_scores: List[float] = list()

    for dataset in TRAIN_DATASETS:
        data, labels = datasets.read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        manifold = Manifold(data, metric='euclidean').build(
            criterion.MaxDepth(MAX_DEPTH),
            criterion.LFDRange(upper, lower),
        )

        for method in METHODS:
            anomalies: Dict[int, float] = METHODS[method](manifold)
            y_true, y_score = list(anomalies.keys()), list(anomalies.values())
            auc_scores.append(roc_auc_score(y_true, y_score))

    return float(np.mean(auc_scores)), float(np.prod(auc_scores))


if __name__ == '__main__':
    print(create_features())
    print(create_features().shape)
