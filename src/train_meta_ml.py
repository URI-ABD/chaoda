import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy
from pyclam import CHAODA
from pyclam import Cluster
from pyclam import Graph
from pyclam import Manifold
from pyclam.criterion import MetaMLSelect
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import utils
from datasets import read

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
NOT_NORMALIZED = {
    'shuttle',
}
FEATURE_NAMES = ['cardinality', 'radius', 'lfd', 'cardinality_ema', 'radius_ema', 'lfd_ema']


def extract_dt(tree: DecisionTreeRegressor, metric: str, method: str) -> str:
    """ Just a bit of meta-programming. Do not question the black magic wielded herein.
    """
    # noinspection PyProtectedMember
    from sklearn.tree import _tree

    feature_name = [
        FEATURE_NAMES[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.tree_.feature
    ]

    # start function definition
    tree_code: List[str] = [
        f'def dt_{metric}_{method}(ratios: numpy.ndarray) -> float:',
        f'    {", ".join(FEATURE_NAMES)} = tuple(ratios)',
    ]

    def extract_lines(node, depth):
        indent = "    " * depth
        if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree.tree_.threshold[node]

            # if block
            tree_code.append(f'{indent}if {name} <= {threshold:.6e}:')
            extract_lines(tree.tree_.children_left[node], depth + 1)

            # else block
            tree_code.append(f'{indent}else:')
            extract_lines(tree.tree_.children_right[node], depth + 1)
        else:
            # return
            tree_code.append(f'{indent}return {tree.tree_.value[node][0][0]:.6e}')

    extract_lines(0, 1)
    return '\n'.join(tree_code)


def extract_lr(model: LinearRegression, metric: str, method: str):
    coefficients = [f'{float(c):.6e}' for c in model.coef_]
    return '\n'.join([
        f'def lr_{metric}_{method}(ratios: numpy.ndarray) -> float:',
        f'    return float(numpy.dot(numpy.asarray(',
        f'        a=[{", ".join(coefficients)}],',
        f'        dtype=float,',
        f'    ), ratios))'
    ])


def _build_data(
        method_function: Callable[[Graph], Dict[Cluster, float]],
        graph: Graph,
        manifold: Manifold,
        labels: numpy.ndarray,
):
    cluster_scores = list((cluster, score) for cluster, score in method_function(graph).items())

    train_x = numpy.zeros(shape=(len(cluster_scores), 6), dtype=numpy.float32)
    train_y = numpy.zeros(shape=(len(cluster_scores),))

    for i, (cluster, score) in enumerate(cluster_scores):
        train_x[i] = manifold.cluster_ratios(cluster)

        y_true = numpy.asarray(labels[cluster.argpoints], dtype=numpy.float32)

        # TODO: Why was this numpy.sum instead of float?
        loss = float(numpy.mean(numpy.square(score - y_true))) / cluster.cardinality

        train_y[i] = 1. - loss

    return train_x, train_y


def train_models(train_datasets: List[str], num_epochs: int) -> Dict[str, str]:
    meta_models: Dict[str, Dict[str, Tuple[LinearRegression, DecisionTreeRegressor]]] = {
        metric_name: {
            method_name: (LinearRegression(), DecisionTreeRegressor(max_depth=3))
            for method_name in CHAODA.method_names
        }
        for metric_name in utils.METRICS
    }
    train_x_dict: Dict[str, Dict[str, numpy.ndarray]] = {
        metric_name: {method_name: None for method_name in CHAODA.method_names}
        for metric_name in utils.METRICS
    }
    train_y_dict: Dict[str, Dict[str, numpy.ndarray]] = {
        metric_name: {method_name: None for method_name in CHAODA.method_names}
        for metric_name in utils.METRICS
    }

    for dataset in train_datasets:
        print('-' * 200)
        print(f'Dataset: {dataset}')
        print('-' * 200)

        normalization_mode = 'gaussian' if dataset in NOT_NORMALIZED else None
        data, labels = read(dataset, normalization_mode)

        chaoda = CHAODA(
            metrics=utils.METRICS,
            max_depth=utils.MAX_DEPTH,
            min_points=utils.assign_min_points(data.shape[0]),
        )
        manifolds = chaoda.build_manifolds(data)
        # noinspection PyProtectedMember
        inner_methods = list(
            (inner_name, inner_method) for inner_name, inner_method in chaoda._names.items()
        )

        for epoch in range(num_epochs):
            print('-' * 160)
            print(f'\tEpoch: {epoch + 1} of {num_epochs}')
            print('-' * 160)

            for manifold in manifolds:
                metric_name: str = manifold.metric

                # build the graphs
                if epoch == 0:
                    graphs = [layer.build_edges() for layer in manifold.layers[5::5]]
                else:
                    criteria = [
                        MetaMLSelect(lambda ratios: meta_model.predict([ratios]))
                        for models in meta_models[metric_name].values()
                        for meta_model in models
                    ]
                    graphs = [Graph(*criterion(manifold.root)).build_edges() for criterion in criteria]

                # filter large graphs for slow methods
                methods_functions_graphs: List[Tuple[str, Callable[[Graph], Dict[Cluster, float]], Graph]] = list()
                for method_name, method_function in inner_methods:
                    for graph in graphs:
                        if graph.cardinality > 128 and method_name in chaoda.slow_methods:
                            continue
                        methods_functions_graphs.append((method_name, method_function, graph))

                # build the data
                for method_name, method_function, graph in methods_functions_graphs:
                    print('-' * 120)
                    print(f'\t\tDataset: {dataset}, Epoch: {epoch + 1} of {num_epochs}, Metric: {metric_name}, Method: {method_name}')
                    print('-' * 120)
                    train_x, train_y = _build_data(method_function, graph, manifold, labels)
                    if train_x_dict[metric_name][method_name] is None:
                        train_x_dict[metric_name][method_name] = train_x
                        train_y_dict[metric_name][method_name] = train_y
                    else:
                        train_x_dict[metric_name][method_name] = numpy.concatenate((train_x_dict[metric_name][method_name], train_x), axis=0)
                        train_y_dict[metric_name][method_name] = numpy.concatenate((train_y_dict[metric_name][method_name], train_y), axis=0)

                # train the meta-models
                for method_name, _, _ in methods_functions_graphs:
                    train_x = train_x_dict[metric_name][method_name]
                    train_y = train_y_dict[metric_name][method_name]

                    lr_model, dt_model = meta_models[metric_name][method_name]
                    lr_model.fit(train_x, train_y)
                    dt_model.fit(train_x, train_y)

    model_codes: Dict[str, str] = dict()
    for metric_name in utils.METRICS:
        for method_name in CHAODA.method_names:
            lr_model, dt_model = meta_models[metric_name][method_name]
            model_codes[f'lr_{metric_name}_{method_name}'] = extract_lr(lr_model, metric_name, method_name)
            model_codes[f'dt_{metric_name}_{method_name}'] = extract_dt(dt_model, metric_name, method_name)

    return model_codes


def write_meta_models(model_codes: Dict[str, str], out_path: str):
    with open(out_path, 'w') as fp:
        fp.write('import numpy\n\n')

        for code in model_codes.values():
            fp.write(f'\n{code}\n\n')

        fp.write('\nMETA_MODELS = {\n')
        for name in model_codes.keys():
            fp.write(f'    \'{name}\': {name},\n')
        fp.write('}\n')
    return


if __name__ == '__main__':
    numpy.random.seed(42), random.seed(42)
    _train_datasets = ['vertebral', 'wine']  # for testing
    # _train_datasets = list(sorted(numpy.random.choice(SAMPLING_DATASETS, 6, replace=False)))
    # print(_train_datasets)
    # exit(1)
    write_meta_models(
        train_models(_train_datasets, num_epochs=10),
        utils.ROOT_DIR.joinpath('src').joinpath('custom_meta_models.py')
    )
