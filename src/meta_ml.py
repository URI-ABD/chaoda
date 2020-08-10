import logging
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import pydotplus
from pyclam import Manifold, criterion, Graph
from scipy.stats import gmean, hmean
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from src import datasets as chaoda_datasets
from src.datasets import DATASETS, METRICS
from src.methods import METHODS
from src.utils import TRAIN_PATH

NORMALIZE = True
SUB_SAMPLE = 100_000
MAX_DEPTH = 50
TRAIN_DATASETS = [
    'cover',
    'mnist',
    'musk',
    'optdigits',
    'satimage-2',
    'shuttle',
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
                criterion.Depth(MAX_DEPTH),
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


def train_tree(train_file: str, target: str):
    features = ['lfd-gmean', 'lfd-hmean', 'lfd-mean',
                'cardinality-gmean', 'cardinality-hmean', 'cardinality-mean',
                'radii-gmean', 'radii-hmean', 'radii-mean']

    df = pd.read_csv(train_file)

    train_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in TRAIN_DATASETS
    ])
    train_x = train_df[features]
    train_y = train_df[target]

    test_df = pd.concat([
        df[df['dataset'].str.contains(dataset)]
        for dataset in DATASETS
        if dataset not in TRAIN_DATASETS
    ])
    test_x = test_df[features]
    test_y = test_df[target]

    decision_tree = DecisionTreeRegressor(max_depth=3)
    decision_tree = decision_tree.fit(train_x, train_y)
    pred_y = decision_tree.predict(test_x)

    mse = mean_squared_error(test_y, pred_y)
    print(f'{target} MSE: {mse:.3f}')
    # print(f'{target} RMSE: {np.sqrt(mse):.3f}')

    export = export_graphviz(decision_tree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(export)
    graph.write_png(os.path.join(TRAIN_PATH, f'{target}_tree.png'))

    return


def print_trees(train_file: str):
    targets = [
        'auc-cluster_cardinality',
        'auc-hierarchical',
        'auc-k_neighborhood',
        'auc-subgraph_cardinality',
    ]
    [train_tree(train_file, target) for target in targets]
    return


if __name__ == '__main__':
    os.makedirs(TRAIN_PATH, exist_ok=True)
    _train_filename = os.path.join(TRAIN_PATH, 'train.csv')
    # create_train_test_data(_train_filename)
    print_trees(_train_filename)
