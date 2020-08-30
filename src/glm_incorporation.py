import os
from typing import List, Dict

import numpy as np
from pyclam import Manifold, criterion
from pyclam.criterion import SELECTION_MODES
from sklearn.metrics import roc_auc_score

from src import datasets as chaoda_datasets
from src.datasets import METRICS, DATASETS
from src.methods import METHODS, ensemble, ENSEMBLE_MODES, METHOD_NAMES
from src.utils import TRAIN_PATH

CONSTANTS = os.path.join(os.path.dirname(__file__), '..', 'train', 'linear_regression.csv')
NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 50


def evaluate_auc(
        datasets: List[str],
        metrics: List[str],
        selections: List[str],
        methods: List[str],
        modes: List[str],
        filename: str,
):
    """ Evaluate auc performance from ensemble of linear regression models.

    :param datasets: the dataset to work on.
    :param metrics: the distance metric to use.
    :param selections: ways in which to select Clusters for criteria.
    :param methods: methods from which to build the ensemble.
    :param modes: modes to use for each ensemble.
    :param filename: file in which to store auc performance.
    """
    assert all((dataset in DATASETS.keys() for dataset in datasets))
    assert all((metric in METRICS.keys() for metric in metrics))
    assert all((selection in SELECTION_MODES.keys() for selection in selections))
    assert all((method in METHODS.keys() for method in methods))
    assert all((mode in ENSEMBLE_MODES for mode in modes))

    for dataset in datasets:
        for metric in metrics:
            data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
            min_points: int
            if len(data) < 1_000:
                min_points = 1
                # continue
            elif len(data) < 4_000:
                min_points = 2
            elif len(data) < 16_000:
                min_points = 4
            elif len(data) < 64_000:
                min_points = 8
                # continue
            else:
                min_points = 16
                # continue

            for selection in selections:
                selection_criteria = [
                    # amean
                    criterion.LinearRegressionConstants([-0.14319379, 0.93448878, 0.06358055, 1.06531136, -0.87266898, 0.74385150], mode=selection),  # CC
                    criterion.LinearRegressionConstants([-0.30966183, 1.05683555, -0.34853912, 0.24952643, -0.80869111, 0.55849258], mode=selection),  # PC
                    criterion.LinearRegressionConstants([-0.44543049, 0.65531289, -0.68583568, 0.13843472, -0.29488228, 0.17655670], mode=selection),  # KN
                    criterion.LinearRegressionConstants([-0.19476319, 0.37860120, -0.38484169, 0.01405269, -0.15549294, 0.06742359], mode=selection),  # SC
                    # gmean
                    criterion.LinearRegressionConstants([0.92364676, 1.48489494, -0.13286499, 0.80007102, -0.28516064, 0.54944094], mode=selection),  # CC
                    criterion.LinearRegressionConstants([0.50948156, 1.04453613, -0.16324464, -0.20873248, 0.21199436, -0.31026809], mode=selection),  # PC
                    criterion.LinearRegressionConstants([0.04960365, 0.19498710, -0.06669465, -0.60589915, 0.51893140, -0.97682891], mode=selection),  # KN
                    criterion.LinearRegressionConstants([-0.02220604, 0.07368668, -0.01657270, -0.31052205, 0.27907633, -0.55491548], mode=selection),  # SC
                    # hmean
                    criterion.LinearRegressionConstants([0.14722934, -0.43870436, 0.08297998, -1.49424028, -0.35540001, -0.16619623], mode=selection),  # CC
                    criterion.LinearRegressionConstants([0.06298725, -0.09990198, -0.00211826, -1.72168161, 0.05267363, -0.72643832], mode=selection),  # PC
                    criterion.LinearRegressionConstants([0.31918357, 0.35341069, -0.02277890, -0.70065582, 0.29002082, -0.88020777], mode=selection),  # KN
                    criterion.LinearRegressionConstants([0.17663877, 0.15718700, 0.00676820, -0.42367319, 0.15126572, -0.49953437], mode=selection),  # SC
                ]

                manifold = Manifold(data, METRICS[metric]).build(
                    criterion.MaxDepth(MAX_DEPTH),
                    criterion.MinPoints(min_points),
                    *selection_criteria,
                )

                for i, graph in enumerate(manifold.graphs):
                    graph.method = methods[i % len(methods)]

                ensemble_scores: List[float] = list()
                for mode in modes:
                    anomalies: Dict[int, float] = ensemble(manifold, mode)
                    y_true, y_score = list(), list()
                    [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                    ensemble_scores.append(roc_auc_score(y_true, y_score))

                method_scores: List[float] = list()
                for graph in manifold.graphs:
                    anomalies: Dict[int, float] = METHODS[graph.method](graph)
                    y_true, y_score = list(), list()
                    [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                    method_scores.append(roc_auc_score(y_true, y_score))

                ensemble_scores: str = ','.join([f'{score:.3f}' for score in ensemble_scores])
                method_scores: str = ','.join([f'{score:.3f}' for score in method_scores])
                with open(filename, 'a') as fp:
                    fp.write(f'{dataset},{metric},{selection},{ensemble_scores},{method_scores}\n')
    return


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs(TRAIN_PATH, exist_ok=True)

    # _datasets = list(DATASETS.keys())
    _datasets = ['cardio']
    _metrics = ['euclidean', 'manhattan']
    _selections = ['percentile', 'ranked']
    _methods = ['cluster_cardinality', 'hierarchical', 'k_neighborhood', 'subgraph_cardinality']

    _modes = ['mean', 'product', 'max', 'min', 'max25', 'min25']
    _ensemble_labels = ','.join([f'ensemble_{_m}' for _m in _modes])

    _means = ['amean', 'gmean', 'hmean']
    _method_labels = ','.join([f'{METHOD_NAMES[_n]}_{_m}' for _m in _means for _n in _methods])

    _filename = os.path.join(TRAIN_PATH, 'ensemble_predictions.csv')
    with open(_filename, 'w') as _fp:
        _fp.write(f'dataset,metric,selection,{_ensemble_labels},{_method_labels}\n')

    evaluate_auc(_datasets, _metrics, _selections, _methods, _modes, _filename)
