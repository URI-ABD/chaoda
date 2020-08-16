import logging
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import pydotplus
from pyclam import Manifold, criterion, Graph
from scipy.stats import gmean, hmean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from src import datasets as chaoda_datasets
from src.datasets import DATASETS, METRICS
from src.methods import METHODS
from src.utils import TRAIN_PATH

NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 20
TRAIN_DATASETS = [
    # 'cover',
    'mnist',
    'musk',
    'optdigits',
    'satimage-2',
    # 'shuttle',
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
        (cluster.radius if cluster.radius > 0 else 1e-4)
        / (cluster.parent.radius if cluster.parent.radius > 0 else 1e-4)
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
    header = f'dataset,metric,depth,{labels},{feature_names}\n'

    if not os.path.exists(filename):
        with open(filename, 'w') as fp:
            fp.write(header)

    for dataset in datasets:
        data, labels = chaoda_datasets.read(dataset, normalize=NORMALIZE)
        min_points: int
        if len(data) < 1_000:
            min_points = 1
        elif len(data) < 4_000:
            min_points = 2
        elif len(data) < 16_000:
            min_points = 4
        elif len(data) < 64_000:
            min_points = 8
        else:
            min_points = 16

        for metric in METRICS.keys():
            logging.info(f'extracting features for {dataset}-{metric}')
            manifold = Manifold(data, metric=METRICS[metric]).build(
                criterion.MaxDepth(MAX_DEPTH),
                criterion.MinPoints(min_points),
                criterion.Layer(MAX_DEPTH),
            )
            for layer in manifold.layers:
                if layer.cardinality >= 32:
                    logging.info(f'writing layer {layer.depth}...')
                    features = create_features(layer)
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
    features = ['lfd-gmean', 'lfd-hmean', 'lfd-mean',
                'cardinality-gmean', 'cardinality-hmean', 'cardinality-mean',
                'radii-gmean', 'radii-hmean', 'radii-mean']
    targets = [
        'auc-cluster_cardinality',
        'auc-hierarchical',
        'auc-k_neighborhood',
        'auc-subgraph_cardinality',
    ]

    df = pd.read_csv(train_file)

    train_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in TRAIN_DATASETS
    ])
    train_x = train_df[features]

    test_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in DATASETS
        if dataset not in TRAIN_DATASETS
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
        filename = os.path.join(TRAIN_PATH, f'{target}_linear_regression.csv')
        with open(filename, 'w') as fp:
            header = ','.join(feature_names)
            line = ','.join([f'{coefficient:.3f}' for coefficient in model.coef_])
            fp.write(f'{header}\n')
            fp.write(f'{line}\n')
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
    os.makedirs(TRAIN_PATH, exist_ok=True)
    _train_filename = os.path.join(TRAIN_PATH, 'train.csv')
    # create_train_test_data(_train_filename)
    train_trees(_train_filename)
    # auc_from_clause()
