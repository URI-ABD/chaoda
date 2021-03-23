import logging
import random
from typing import Dict
from typing import List

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from datasets import DATASET_LINKS
from datasets import read
from pyclam import CHAODA
from pyclam import Manifold
from utils import *

SAMPLING_DATASETS = [
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
    'cardinality',
    'radius',
    'lfd',
]
FEATURE_NAMES.extend([f'{_name}_ema' for _name in FEATURE_NAMES])
METHODS = {
    'cc': 'cluster_cardinality',
    'sc': 'component_cardinality',
    'gn': 'graph_neighborhood',
    'pc': 'parent_cardinality',
    'rw': 'parent_cardinality',
    'sp': 'stationary_probabilities',
}


def create_data(filepath: str, datasets: List[str]):
    """ Create training data for the meta-ml models.

    :param filepath: path to .csv file where the data will be written.
    :param datasets: List of datasets for which to generate data.
    """
    with open(filepath, 'w') as fp:
        fp.write(f'dataset,metric,depth,mean,{",".join(METHODS)},{",".join(FEATURE_NAMES)}\n')

    for dataset in datasets:
        logging.info(f'extracting features for {dataset}.')

        data, labels = read(dataset, NORMALIZE, SUB_SAMPLE)
        chaoda: CHAODA = CHAODA(
            metrics=METRICS,
            max_depth=MAX_DEPTH,
            min_points=assign_min_points(data.shape[0]),
            normalization_mode=None,
        )

        manifolds: List[Manifold] = chaoda.build_manifolds(
            data=data,
        )
        for manifold in manifolds:
            print('-' * 160)
            print('-' * 160)
            logging.info(f'extracting features for {dataset}-{manifold.metric}.')
            for layer in manifold.layers[1:]:
                layer.build_edges()
                logging.info(f'writing layer {layer.depth} with {layer.cardinality} clusters.')
                features: np.array = np.stack([manifold.cluster_ratios(cluster) for cluster in layer.clusters])

                # noinspection PyProtectedMember
                def score(method):
                    if method in chaoda.slow_methods and layer.pruned_graph[0].cardinality > chaoda.speed_threshold:
                        return -1
                    else:
                        scores = chaoda._score_points(chaoda._names[method](layer))
                        scores = [scores[j] for j in range(data.shape[0])]
                        return float(roc_auc_score(labels, scores))

                scores_string = ','.join([f'{score(method):.4f}' for method in METHODS.values()])
                for mean in MEANS:
                    features_mean = MEANS[mean](features, axis=0)
                    features_string = ','.join([f'{float(feature):.4f}' for feature in features_mean])
                    with open(filepath, 'a') as fp:
                        fp.write(f'{dataset},{manifold.metric},{layer.depth},{mean},{scores_string},{features_string}\n')
            print('-' * 160)
            print('-' * 160)
    return


def extract_dt(tree: DecisionTreeRegressor, method: str, mean: str) -> str:
    """ just a bit of meta-programming. Don't question the black magic wielded here. """
    # noinspection PyProtectedMember
    from sklearn.tree import _tree

    feature_name = [
        FEATURE_NAMES[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.tree_.feature
    ]
    tree_code: List[str] = [
        f'def from_dt_{method}_{mean}(ratios: np.array) -> float:',
        f'    {", ".join(FEATURE_NAMES)} = tuple(ratios)',
    ]

    def extract_lines(node, depth):
        indent = "    " * depth
        if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.tree_.threshold[node]
            tree_code.append(f'{indent}if {name} <= {threshold:.5f}:')

            extract_lines(tree.tree_.children_left[node], depth + 1)
            tree_code.append(f'{indent}else:')

            extract_lines(tree.tree_.children_right[node], depth + 1)
        else:
            tree_code.append(f'{indent}return {tree.tree_.value[node][0][0]:.5f}')

    extract_lines(0, 1)
    return '\n'.join(tree_code)


def extract_lr(model: LinearRegression, method: str, mean: str):
    return '\n'.join([
        f'def from_lr_{method}_{mean}(ratios: np.array) -> float:',
        f'    return float(np.dot(np.asarray(',
        f'        a=[{", ".join([str(float(round(c, 5))) for c in model.coef_])}],',
        f'        dtype=float,',
        f'    ), ratios))'
    ])


def train_models(
        train_path: str,
        train_datasets: List[str],
):
    """ Train all meta-ml models and export results for the auto-parser.

    :param train_path: path to the training data for meta-ml as generated by create_data.
    :param train_datasets: list of chaoda datasets on which to train.
    """
    raw_df = pd.read_csv(train_path)

    full_train_df = pd.concat([
        raw_df[raw_df['dataset'].str.contains(dataset)]
        for dataset in train_datasets
    ])

    models_codes: Dict[str, str] = dict()
    for mean in MEANS:
        train_df = full_train_df[full_train_df['mean'].str.contains(mean)]
        full_train_x = train_df[FEATURE_NAMES]

        for method in METHODS:
            full_train_y = train_df[method]

            rows = train_df[method] > 0
            train_x, train_y = full_train_x[rows], full_train_y[rows]

            lr_model = LinearRegression()
            lr_model.fit(train_x, train_y)
            models_codes[f'lr_{method.lower()}_{mean}'] = extract_lr(lr_model, method.lower(), mean)

            dt_model = DecisionTreeRegressor(max_depth=3)
            dt_model.fit(train_x, train_y)
            models_codes[f'dt_{method.lower()}_{mean}'] = extract_dt(dt_model, method.lower(), mean)

    return models_codes


def write_meta_models(model_codes: Dict[str, str], out_path: str):
    with open(out_path, 'w') as fp:
        fp.write('import numpy as np\n\n')

        for code in model_codes.values():
            fp.write(f'\n{code}\n\n')

        fp.write('\nMETA_MODELS = {\n')
        for name in model_codes.keys():
            fp.write(f'    \'from_{name}\': from_{name},\n')
        fp.write('}\n')
    return


if __name__ == '__main__':
    os.makedirs(TRAIN_DIR, exist_ok=True)
    _train_path = os.path.join(TRAIN_DIR, 'train.csv')
    _datasets = list(DATASET_LINKS.keys())
    # _datasets = ['vertebral']  # for testing
    create_data(_train_path, _datasets)

    np.random.seed(42), random.seed(42)
    _train_datasets = list(sorted(np.random.choice(SAMPLING_DATASETS, 6, replace=False)))
    # _train_datasets = ['vertebral']  # for testing
    print(_train_datasets)

    _out_path = os.path.join(SRC_DIR, 'meta_models.py')
    _models_codes = train_models(_train_path, _datasets)
    write_meta_models(_models_codes, _out_path)
