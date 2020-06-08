import logging
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyclam import Manifold, criterion, Graph
from sklearn.metrics import roc_auc_score

from src.datasets import DATASETS, get, read, METRICS
from src.methods import METHODS
from src.reproduce import BUILD_PATH, METRIC_NAMES, METHOD_NAMES, PLOTS_PATH

HEAT_MAP_PATH = os.path.join(PLOTS_PATH, 'heat_maps')
sns.set(color_codes=True, font_scale=1.2)

NORMALIZE = False
SUB_SAMPLE = 50_000
MAX_DEPTH = 40
STEP = 1


def _manifold_path(dataset, metric) -> str:
    return os.path.join(
        BUILD_PATH,
        ':'.join(map(str, [dataset, METRIC_NAMES[metric], f'.pickle']))
    )


# noinspection DuplicatedCode
def main():
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    results = {dataset: list() for dataset in DATASETS}

    for dataset in DATASETS:
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        labels = np.squeeze(labels)

        for metric in ['euclidean', 'manhattan', 'cosine']:
            logging.info(', '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}',
            ]))
            manifold = Manifold(data, METRICS[metric])
            manifold.build(criterion.MaxDepth(MAX_DEPTH))

            # filepath = _manifold_path(dataset, metric)
            # if os.path.exists(filepath):
            #     # load from memory and continue build if needed
            #     with open(filepath, 'rb') as fp:
            #         logging.info(f'loading manifold {filepath}')
            #         manifold = manifold.load(fp, data)
            # else:
            #     # build manifold from scratch
            #     manifold.build(criterion.MaxDepth(MAX_DEPTH))
            #     # save manifold
            #     with open(filepath, 'wb') as fp:
            #         logging.info(f'dumping manifold {filepath}')
            #         manifold.dump(fp)

            for method in ['cluster_cardinality', 'hierarchical', 'k_neighborhood', 'subgraph_cardinality']:
                logging.info(f'{dataset}:{METRIC_NAMES[metric]}:({METHOD_NAMES[method]})')

                anomalies = METHODS[method](manifold)
                y_true, y_score = list(), list()
                [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                results[dataset].append(f'{roc_auc_score(y_true, y_score):.3f}')

    table_filepath = os.path.join(PLOTS_PATH, f'table.txt')
    with open(table_filepath, 'a') as fp:
        for dataset in results:
            line = ' & '.join(results[dataset])
            fp.write(f'\\bfseries {dataset} & {line} \\\\\n')
            fp.write(f'\\hline\n')

    return


# noinspection DuplicatedCode
def create_heat_maps(datasets: List[str], metrics: List[str]):
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(HEAT_MAP_PATH, exist_ok=True)

    for dataset in datasets:
        dataset_path = os.path.join(HEAT_MAP_PATH, f'{dataset}')
        os.makedirs(dataset_path, exist_ok=True)

        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        labels = np.squeeze(labels)

        for metric in metrics:
            logging.info(', '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}',
            ]))
            manifold = Manifold(data, METRICS[metric])
            manifold.graphs = [Graph(manifold.root)]
            manifold.build_tree(criterion.MaxDepth(10))

            results = {method: np.zeros(shape=(100 // STEP, 100 // STEP), dtype=float)
                       for method in METHODS}
            thresholds = [i for i in range(1, 101, STEP)]

            for upper in thresholds:
                for lower in thresholds:
                    max_lfd, min_lfd = manifold.lfd_range(percentiles=(upper, lower))
                    manifold.root.mark(max_lfd, min_lfd)
                    manifold.build_graph()

                    for method in METHODS:
                        logging.info(f'{dataset}:{METRIC_NAMES[metric]}:({METHOD_NAMES[method]})')

                        anomalies = METHODS[method](manifold)
                        y_true, y_score = list(), list()
                        [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                        i, j = (upper - 1) // STEP, (lower - 1) // STEP
                        results[method][i][j] = roc_auc_score(y_true, y_score)

                    for cluster in manifold.optimal_graph:
                        cluster.__dict__['_optimal'] = False

            for method, table in results.items():
                filename = os.path.join(dataset_path, f'{metric}-{method}.csv')
                table = np.maximum(table, 0.5)
                with open(filename, 'w') as fp:
                    header = ','.join([f'{i}' for i in thresholds]) + '\n'
                    fp.write(header)

                    for i, row in zip(thresholds, table):
                        line = f'{i},' + ','.join([f'{score:.6f}' for score in row]) + '\n'
                        fp.write(line)
    return


def plot_heat_maps(datasets: List[str], metrics: List[str]):
    for dataset in datasets:
        dataset_path = os.path.join(HEAT_MAP_PATH, f'{dataset}')
        for metric in metrics:
            for method in METHODS:
                filename = os.path.join(dataset_path, f'{metric}-{method}.csv')
                table = pd.read_csv(filename)
                plt.figure(figsize=(8, 8))
                sns.heatmap(
                    data=table,
                    vmin=0.5,
                    vmax=1,
                    square=True,
                    cmap='coolwarm',
                    annot=True,
                    fmt='.4f',
                    linewidths=0.5,
                )
                plt.show()
    return


if __name__ == '__main__':
    _datasets = ['cardio']
    _metrics = ['euclidean']
    create_heat_maps(_datasets, _metrics)
    plot_heat_maps(_datasets, _metrics)
