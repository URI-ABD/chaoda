import os
from typing import Dict, Tuple, List

import numpy as np
from pyclam import Manifold, criterion

np.random.seed(42)


BASE_PATH = '/scratch/nishaq/'

APOGEE2_PATH = BASE_PATH + 'APOGEE2/'
APOGEE2 = {
    'data': APOGEE2_PATH + 'apo25m_data.memmap',
    'metadata': APOGEE2_PATH + 'apo25m_data_metadata.csv',
    'queries': APOGEE2_PATH + 'apo25m_queries.memmap',
    'rows': 264_160 - 10_000,
    'dims': 8_575,
    'dtype': np.float32,
}

GREENGENES_PATH = BASE_PATH + 'GreenGenes/'
GREENGENES = {
    'data': GREENGENES_PATH + 'gg_data.memmap',
    'metadata': GREENGENES_PATH + 'gg_data_metadata.csv',
    'queries': GREENGENES_PATH + 'gg_queries.memmap',
    'rows': 1_027_383 - 10_000,
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


def build_manifold(dataset: str, metric: str, subsample: bool = True) -> Manifold:
    data, argpoints = get_data(DATASETS[dataset], subsample)
    manifold: Manifold = Manifold(data, metric, argpoints)

    filename = BASE_PATH + 'build/' + dataset + '.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            manifold = manifold.load(fp, data)
    else:
        manifold.build(
            criterion.MaxDepth(50),
            criterion.MinPoints(10),
        )
        os.makedirs(filename)
        with open(filename, 'wb') as fp:
            manifold.dump(fp)
    return manifold


if __name__ == '__main__':
    build_manifold('apogee2', 'euclidean')
