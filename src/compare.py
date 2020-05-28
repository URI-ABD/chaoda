import logging
import os

import numpy as np
from pyclam import Manifold, criterion
from sklearn.metrics import roc_auc_score

from src.datasets import DATASETS, METRICS, get, read
from src.methods import METHODS
from src.reproduce import METRIC_NAMES, METHOD_NAMES, BUILD_PATH, PLOTS_PATH

NORMALIZE = False
MAX_DEPTH = 20
SUB_SAMPLE = None


def _manifold_path(dataset, metric) -> str:
    return os.path.join(
        BUILD_PATH,
        ':'.join(map(str, [dataset, METRIC_NAMES[metric], f'.pickle']))
    )


def depth_auc_tables():
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    for dataset in DATASETS:
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        labels = np.squeeze(labels)
        results = {depth: [] for depth in range(1, MAX_DEPTH + 1)}

        for metric in ['euclidean', 'manhattan', 'cosine']:
            logging.info(', '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}',
            ]))
            manifold = Manifold(data, METRICS[metric])

            filepath = _manifold_path(dataset, metric)
            if os.path.exists(filepath):
                # load from memory and continue build if needed
                with open(filepath, 'rb') as fp:
                    logging.info(f'loading manifold {filepath}')
                    manifold = manifold.load(fp, data)
                temp_depth = manifold.depth
                manifold.build_tree(criterion.MaxDepth(MAX_DEPTH))
                if manifold.depth > temp_depth:
                    manifold.build_graphs()
                    # save manifold
                    with open(filepath, 'wb') as fp:
                        logging.info(f'dumping manifold {filepath}')
                        manifold.dump(fp)
            else:
                # build manifold from scratch
                manifold.build(criterion.MaxDepth(MAX_DEPTH))
                # save manifold
                with open(filepath, 'wb') as fp:
                    logging.info(f'dumping manifold {filepath}')
                    manifold.dump(fp)

            for method in ['cluster_cardinality', 'hierarchical', 'k_neighborhood', 'subgraph_cardinality']:
                logging.info(f'{dataset}:{METRIC_NAMES[metric]}:({METHOD_NAMES[method]})')

                for depth in range(1, manifold.depth + 1):
                    anomalies = METHODS[method](manifold.graphs[depth])
                    y_true, y_score = list(), list()
                    [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                    results[depth].append(f'{roc_auc_score(y_true, y_score):.3f}')

        table_filepath = os.path.join(PLOTS_PATH, f'{dataset}-table.txt')
        with open(table_filepath, 'w') as fp:
            for depth in range(1, manifold.depth + 1):
                line = ' & '.join(results[depth])
                fp.write(f'\\bfseries {depth} & {line} \\\\ \n\\hline')
    return


if __name__ == '__main__':
    depth_auc_tables()
