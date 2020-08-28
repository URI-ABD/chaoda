import logging
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import pydotplus
from pyclam import Manifold, criterion, Graph
from scipy.stats import gmean, hmean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text

from src import datasets as chaoda_datasets
from src.datasets import DATASETS, METRICS
from src.methods import METHODS, METHOD_NAMES
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
    feature_names = list(f'{feature_names},{ema_names}'.split(','))
    feature_names = [f'{name}_amean' for name in feature_names] + [f'{name}_gmean' for name in feature_names] + [f'{name}_hmean' for name in feature_names]
    feature_names = ','.join(feature_names)

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
            for layer in manifold.layers[1:]:
                logging.info(f'writing layer {layer.depth}...')
                # noinspection PyUnresolvedReferences
                features = np.stack([
                    np.concatenate([cluster.ratios, cluster.ema_ratios])
                    for cluster in layer.clusters
                ])
                features = list(np.mean(features, axis=0)) + list(gmean(features, axis=0)) + list(hmean(features, axis=0))
                features_line = ','.join([f'{f:.8f}' for f in features])

                scores = auc_scores(layer, labels)
                scores_line = ','.join([f'{f:.8f}' for f in scores])

                with open(filename, 'a') as fp:
                    fp.write(f'{dataset},{metric},{layer.depth},{scores_line},{features_line}\n')
    return


def create_train_test_data(train_file: str):
    for seed in [42, 503, 4138]:
        np.random.seed(seed)
        create_training_data(train_file, list(DATASETS.keys()))
    return


def train_meta_model(train_file: str, model: str, export: str):
    train_datasets = list(sorted(np.random.choice(TRAIN_DATASETS, 6, replace=False)))

    df = pd.read_csv(train_file)

    # apply sigmoid to all columns with lfd ratios
    df = df.apply(lambda x: 1 / (1 + np.exp(-x)) if x.name[:3] == 'lfd' else x, axis=0)
    # df = df.apply(lambda x: 1 / (1 + np.exp(-x)) if x.name[-4:] == 'mean' else x, axis=0)

    targets = list(METHOD_NAMES.values())
    meta_columns = [target for target in targets] + ['dataset', 'metric', 'depth']
    features = [name for name in df.columns if name not in meta_columns]

    train_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in train_datasets
    ])
    test_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in DATASETS
        if dataset not in train_datasets
    ])

    for mean in ['amean', 'gmean', 'hmean']:
        feature_names = [name for name in features if name[-len(mean):] == mean]
        train_x = train_df[feature_names]
        test_x = test_df[feature_names]

        feature_names = [name[:-6] for name in feature_names]
        for target in targets:
            train_y = train_df[target]
            test_y = test_df[target]

            if model == 'linear_regression':
                linear_regression(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    export=export,
                    mean=mean,
                    feature_names=feature_names,
                    target=target,
                )
            elif model == 'regression_tree':
                regression_tree(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    export=export,
                    mean=mean,
                    feature_names=feature_names,
                    target=target,
                )
            else:
                raise ValueError(f'{model} not implemented')
    return


def regression_tree(
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x,
        test_y,
        *,
        export: str = None,
        mean: str = None,
        feature_names: List[str] = None,
        target: str = None,
):
    decision_tree = DecisionTreeRegressor(max_depth=3)
    decision_tree = decision_tree.fit(train_x, train_y)

    pred_y = decision_tree.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)
    print(f'{mean} {target} MSE: {mse:.3f}')
    # print(f'{target} RMSE: {np.sqrt(mse):.3f}')

    if export:
        filename = os.path.join(TRAIN_PATH, f'{target}_tree_{mean}')
        if export == 'png':
            export = export_graphviz(
                decision_tree,
                out_file=None,
                feature_names=feature_names,
                proportion=True,
                precision=3
            )
            graph = pydotplus.graph_from_dot_data(export)
            graph.write_png(f'{filename}.png')
        elif export == 'txt':
            export = export_text(
                decision_tree,
                feature_names=feature_names,
                decimals=8,
            )
            with open(f'{filename}.txt', 'w') as fp:
                fp.write(export)
        else:
            raise ValueError(f'Regression tree cannot be exported as {export}. Must be one of \'png\' or \'txt\'')
    return decision_tree


def linear_regression(
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x,
        test_y,
        *,
        export: str = None,
        mean: str = None,
        feature_names: List[str] = None,
        target: str = None,
):
    model = LinearRegression()
    model = model.fit(train_x, train_y)

    pred_y = model.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)

    if export:
        if export == 'csv':
            filename = os.path.join(TRAIN_PATH, f'linear_regression.csv')
            if not os.path.exists(filename):
                with open(filename, 'w') as fp:
                    header = ','.join(feature_names)
                    fp.write(f'method,mean,MSE,{header}\n')
            with open(filename, 'a') as fp:
                line = ','.join([f'{coefficient:.8f}' for coefficient in model.coef_])
                fp.write(f'{target},{mean},{mse:.3f},{line}\n')
        else:
            raise ValueError(f'Linear Regression model cannot be exported as {export}. Must be \'csv\'.')
    return model


if __name__ == '__main__':
    np.random.seed(42)
    os.makedirs(TRAIN_PATH, exist_ok=True)
    _train_filename = os.path.join(TRAIN_PATH, 'train.csv')
    # create_train_test_data(_train_filename)
    train_meta_model(_train_filename, model='regression_tree', export='txt')
    # auc_from_clause()
