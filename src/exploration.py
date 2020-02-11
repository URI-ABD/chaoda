import logging
import os
from typing import Dict, Tuple, List

import numpy as np
from pyclam import Manifold, criterion
from src.methods import METHODS
from src.plot import roc_curve

np.random.seed(42)


BASE_PATH = '/data/nishaq/'

APOGEE2_PATH = BASE_PATH + 'APOGEE2/'
APOGEE2 = {
    # 'data': APOGEE2_PATH + 'apo25m_data.memmap',
    'data': APOGEE2_PATH + 'apo100k_data.memmap',
    'metadata': APOGEE2_PATH + 'apo25m_data_metadata.csv',
    'queries': APOGEE2_PATH + 'apo25m_queries.memmap',
    # 'rows': 264_160 - 10_000,
    'rows': 100_000,
    'dims': 8_575,
    'dtype': np.float32,
}

GREENGENES_PATH = BASE_PATH + 'GreenGenes/'
GREENGENES = {
    # 'data': GREENGENES_PATH + 'gg_data.memmap',
    'data': GREENGENES_PATH + 'gg100k_data.memmap',
    'metadata': GREENGENES_PATH + 'gg_data_metadata.csv',
    'queries': GREENGENES_PATH + 'gg_queries.memmap',
    # 'rows': 1_027_383 - 10_000,
    'rows': 100_000,
    'dims': 7_682,
    'dtype': np.int8,
}

# GREENGENES2_PATH = BASE_PATH + 'GreenGenes2/'
# GREENGENES2 = {
#     'data': GREENGENES2_PATH + 'gg2_data.memmap',
#     'metadata': GREENGENES2_PATH + 'gg2_data_metadata.csv',
#     'queries': GREENGENES2_PATH + 'gg2_queries.memmap',
#     'rows': 134_512 - 10_000,
#     'dims': 8_575,
#     'dtype': np.int8,
# }

DATASETS = {
    'apogee2': APOGEE2,
    'greegenes': GREENGENES,
}


def get_data(data_dict: Dict, subsample: bool) -> Tuple[np.memmap, List[int]]:
    data: np.memmap = np.memmap(
        filename=data_dict['data'],
        mode='r',
        dtype=data_dict['dtype'],
        shape=(data_dict['rows'], data_dict['dims']),
    )

    if subsample and data.shape[0] > 100_000:
        argpoints = sorted(list(np.random.choice(data.shape[0], 100_000, replace=False)))
    else:
        argpoints = list(range(data.shape[0]))

    return data, argpoints


def mutate_greengenes(mutation_rates: List[float] = None, mutations_per_rate: int = 1000):
    if mutation_rates is None:
        mutation_rates = [0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.]
        mutation_rates = [r / 100 for r in mutation_rates]

    GREENGENES['mutated'] = GREENGENES_PATH + 'gg100k_mutated.memmap'
    if os.path.exists(GREENGENES['mutated']):
        mutated: np.memmap = np.memmap(
            filename=GREENGENES['mutated'],
            dtype=GREENGENES['dtype'],
            mode='r',
            shape=(GREENGENES['rows'], GREENGENES['dims']),
        )
    else:
        data, _ = get_data(GREENGENES, subsample=False)
        mutated: np.memmap = np.memmap(
            filename=GREENGENES['mutated'],
            dtype=GREENGENES['dtype'],
            mode='w+',
            shape=data.shape
        )

        for i, rate in enumerate(mutation_rates):
            for j in range(mutations_per_rate):
                idx = j + i * mutations_per_rate
                sequence = data[idx].copy()
                not_gaps = np.argwhere(sequence != 0).flatten()
                num_mutations = int(rate * not_gaps.shape[0])
                mutations = np.random.choice(
                    not_gaps,
                    size=num_mutations,
                    replace=False,
                )
                sequence[mutations] = 0
                mutated[idx] = sequence
        else:
            for idx in range(len(mutation_rates) * mutations_per_rate, data.shape[0]):
                mutated[idx] = data[idx]
        mutated.flush()

    labels = [1] * len(mutation_rates) * mutations_per_rate
    labels.extend([0 for _ in range(len(mutation_rates) * mutations_per_rate, mutated.shape[0])])
    assert mutated.shape[0] == len(labels)
    return mutated, labels


def build_manifold(dataset: str, metric: str, subsample: bool = True) -> Manifold:
    data, argpoints = get_data(DATASETS[dataset], subsample)
    manifold: Manifold = Manifold(data, metric, argpoints)

    filename = BASE_PATH + 'build/' + dataset + '.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            manifold = manifold.load(fp, data)
    else:
        manifold.build(
            criterion.MaxDepth(100),
            criterion.MinPoints(3),
        )
        os.makedirs(filename)
        with open(filename, 'wb') as fp:
            manifold.dump(fp)
    return manifold


def explore_greengenes():
    dataset, metric, max_depth = 'gg_mutated', 'hamming', 20
    data, labels = mutate_greengenes()
    manifold: Manifold = Manifold(data, metric=metric)

    build_dir = BASE_PATH + 'build/'
    filename = build_dir + dataset + '.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            manifold = manifold.load(fp, data)
    if manifold.depth < max_depth:
        manifold.build_tree(
            criterion.MaxDepth(20),
            criterion.MinPoints(10),
        ).build_graphs()
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    with open(filename, 'wb') as fp:
        manifold.dump(fp)

    for method in METHODS:
        for depth in range(manifold.depth + 1):
            logging.info(f'{dataset}, {metric}, {depth}/{manifold.depth}, {method}')
            roc_curve(
                true_labels=labels,
                anomalies=METHODS[method](manifold.graphs[depth]),
                dataset=dataset,
                metric='hamming',
                method=method,
                depth=depth,
                save=True,
            )
    return


if __name__ == '__main__':
    # build_manifold('apogee2', 'euclidean')
    # build_manifold('greegenes', 'hamming')
    explore_greengenes()
