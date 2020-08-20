import logging
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import pydotplus
from pyclam import Manifold, criterion, Graph
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from src import datasets as chaoda_datasets
from src.datasets import DATASETS, METRICS
from src.glm_incorporation import calculate_ratios
from src.methods import METHODS
from src.utils import TRAIN_PATH

NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 20
TRAIN_DATASETS = [
    'annthyroid',
    'cardio',
    'cover',
    'http',
    'mammography',
    'mnist',
    'musk',
    'optdigits',
    'pendigits',
    'satellite',
    'satimage-2',
    'shuttle',
    'smtp',
    'thyroid',
    'vowels',
]
FEATURE_NAMES = [
    'lfd',
    'cardinality',
    'radius',
]
METHOD_NAMES = {
    'cluster_cardinality': 'CC',
    'hierarchical': 'PC',
    'k_neighborhood': 'KN',
    'subgraph_cardinality': 'SC',
}


def auc_scores(graph: Graph, labels: List[int]) -> List[float]:
    scores: List[float] = list()
    for method in METHODS:
        anomalies: Dict[int, float] = METHODS[method](graph)
        y_true, y_score = list(), list()
        [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
        scores.append(roc_auc_score(y_true, y_score))

    return scores


def create_training_data(filename: str, datasets: List[str]):
    feature_names = ','.join(FEATURE_NAMES)
    ema_names = ','.join([f'{name}_ema' for name in FEATURE_NAMES])
    feature_names = f'{feature_names},{ema_names}'

    labels = ','.join([f'{METHOD_NAMES[method]}' for method in METHODS])
    header = f'dataset,metric,depth,{labels},{feature_names}\n'

    if not os.path.exists(filename):
        with open(filename, 'w') as fp:
            fp.write(header)

    for dataset in datasets:
        data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE)
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

        for metric in ['euclidean', 'manhattan']:
            manifold = Manifold(data, metric=METRICS[metric]).build(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.MinPoints(min_points),
                criterion.Layer(MAX_DEPTH),
            )

            logging.info(f'extracting features for {dataset}-{metric}')
            manifold.root.ratios = np.ones(shape=(3,), dtype=float)
            for layer in manifold.layers[1:]:
                [calculate_ratios(cluster) for cluster in layer.clusters]

                logging.info(f'writing layer {layer.depth}...')
                # noinspection PyUnresolvedReferences
                features = np.stack([
                    np.concatenate([cluster.ratios, cluster.ema_ratios])
                    for cluster in layer.clusters
                ])
                features = list(np.mean(features, axis=0))
                scores = auc_scores(layer, labels)
                features_line = ','.join([f'{f:.8f}' for f in features])
                scores_line = ','.join([f'{f:.8f}' for f in scores])

                with open(filename, 'a') as fp:
                    fp.write(f'{dataset},{metric},{layer.depth},{scores_line},{features_line}\n')
            # break
    return


def create_train_test_data(train_file: str):
    for seed in range(10):
        np.random.seed(seed)
        create_training_data(train_file, list(DATASETS.keys()))
    return


def train_trees(train_file: str):
    features = FEATURE_NAMES + [f'{name}_ema' for name in FEATURE_NAMES]
    targets = list(METHOD_NAMES.values())
    train_datasets = list(sorted(np.random.choice(TRAIN_DATASETS, 8, replace=False)))
    print(train_datasets)

    df = pd.read_csv(train_file)

    train_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in train_datasets
    ])
    train_x = train_df[features]

    test_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in DATASETS
        if dataset not in train_datasets
    ])
    test_x = test_df[features]

    for target in targets:
        train_y = train_df[target]
        test_y = test_df[target]

        model = linear_regression(train_x, train_y, export='text', feature_names=features, target=target)

        pred_y = model.predict(test_x)
        mse = mean_squared_error(test_y, pred_y)
        print(f'{target} MSE: {mse:.3f}')
        # print(f'{target} RMSE: {np.sqrt(mse):.3f}')
    return


def regression_tree(train_x: np.ndarray, train_y: np.ndarray, *, export: str = None, feature_names: List[str] = None, target: str = None):
    decision_tree = DecisionTreeRegressor(max_depth=3)
    decision_tree = decision_tree.fit(train_x, train_y)
    if export:
        if export == 'graphviz':
            export = export_graphviz(decision_tree, out_file=None, feature_names=feature_names)
            graph = pydotplus.graph_from_dot_data(export)
            graph.write_png(os.path.join(TRAIN_PATH, f'{target}_tree.png'))
        else:
            pass
    return decision_tree


def linear_regression(train_x: np.ndarray, train_y: np.ndarray, *, export: str = None, feature_names: List[str] = None, target: str = None):
    model = LinearRegression()
    model = model.fit(train_x, train_y)
    if export:
        filename = os.path.join(TRAIN_PATH, f'linear_regression.csv')
        if not os.path.exists(filename):
            with open(filename, 'w') as fp:
                header = ','.join(feature_names)
                fp.write(f'method,{header}\n')
        with open(filename, 'a') as fp:
            line = ','.join([f'{coefficient:.3f}' for coefficient in model.coef_])
            fp.write(f'{target},{line}\n')
    return model


def auc_from_clause():
    filename = os.path.join(TRAIN_PATH, f'auc_clauses.csv')
    with open(filename, 'w') as fp:
        header = ','.join(['dataset', 'metric', 'method'] + list(METHODS.keys()))
        fp.write(f'{header}\n')

    for dataset in DATASETS.keys():
        data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE)
        min_points: int
        if len(data) < 1_000:
            min_points = 1
        elif len(data) < 4_000:
            min_points = 2
        elif len(data) < 16_000:
            min_points = 4
        # elif len(data) < 64_000:
        #     min_points = 8
        else:
            # min_points = 16
            continue

        clauses: List[criterion.Clause] = [
            criterion.Clause((0.87, np.inf), (0.229, 0.372), (0, np.inf)),  # CC
            # criterion.Clause((0, np.inf), (0, 0.37), (0, 0.784)),  # PC
            # criterion.Clause((0, np.inf), (0.371, np.inf), (0, 0.691)),  # KN, SC
        ]

        for metric in ['euclidean', 'manhattan']:
            logging.info(f'testing clause for {dataset}-{metric}')
            manifold: Manifold = Manifold(data, METRICS[metric]).build(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.MinPoints(min_points),
                criterion.SelectionClauses(clauses),
            )
            scores = auc_scores(manifold.graph, labels)
            scores = ','.join([f'{score:.3f}' for score in scores])
            with open(filename, 'a') as fp:
                fp.write(f'{dataset},{metric},{scores}\n')
    return


if __name__ == '__main__':
    np.random.seed(42)
    os.makedirs(TRAIN_PATH, exist_ok=True)
    _train_filename = os.path.join(TRAIN_PATH, 'train.csv')
    create_train_test_data(_train_filename)
    # train_trees(_train_filename)
    # auc_from_clause()
