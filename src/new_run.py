import logging
import os
from collections import Counter
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
SUB_SAMPLE = 20_000
MAX_DEPTH = 30
FRACTION = 0.2
STEP = 10


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
def depth_distributions(datasets: List[str], metrics: List[str]):
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(HEAT_MAP_PATH, exist_ok=True)

    for dataset in datasets:
        dataset_path = os.path.join(HEAT_MAP_PATH, f'{dataset}')
        os.makedirs(dataset_path, exist_ok=True)

        depths_path = os.path.join(PLOTS_PATH, f'depth-distributions', f'{dataset}')
        os.makedirs(depths_path, exist_ok=True)

        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        labels = np.squeeze(labels)
        # y_max = 1 + int(np.floor(np.log2(len(labels))))

        for metric in metrics:
            logging.info(', '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}',
            ]))
            manifold = Manifold(data, METRICS[metric])
            manifold.build_tree(criterion.MaxDepth(MAX_DEPTH))

            results = {method: np.zeros(shape=(100 // STEP, 100 // STEP), dtype=float)
                       for method in METHODS}

            # freq_filename = os.path.join(depths_path, f'{metric}.csv')
            # with open(freq_filename, 'w') as freq_fp:
            #     header = ','.join([f'{d}' for d in range(manifold.depth + 1)])
            #     freq_fp.write(f'upper,lower,clusters/points,{header}\n')

            thresholds = [i for i in range(STEP, 101, STEP)]
            for upper in thresholds:
                for lower in thresholds:
                    if lower > upper:
                        continue
                    logging.info(f'upper {upper}, lower {lower}')
                    [cluster.clear_cache() for layer in manifold.layers for cluster in layer.clusters]
                    manifold.graph = Graph(*criterion.LFDRange(upper, lower)(manifold.root))
                    manifold.build_graph(criterion.MinimizeSubsumed(FRACTION))

                    for method in METHODS:
                        logging.info(f'{dataset}:{METRIC_NAMES[metric]}:({METHOD_NAMES[method]})')

                        anomalies = METHODS[method](manifold)
                        y_true, y_score = list(), list()
                        [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                        i, j = (upper - 1) // STEP, (lower - 1) // STEP
                        results[method][j][i] = roc_auc_score(y_true, y_score)

                            # cluster_frequencies = dict(Counter((cluster.depth for cluster in manifold.graph)))
                            # cluster_frequencies = [np.log2(cluster_frequencies[d] + 1) if d in cluster_frequencies else 0
                            #                        for d in range(manifold.depth + 1)]
                            # line = ','.join([f'{freq:.6f}' for freq in cluster_frequencies])
                            # freq_fp.write(f'{upper},{lower},clusters,{line}\n')
                            #
                            # point_frequencies = [0 for _ in range(manifold.depth + 1)]
                            # for cluster in manifold.graph:
                            #     point_frequencies[cluster.depth] += cluster.cardinality
                            # point_frequencies = [np.log2(freq + 1) for freq in point_frequencies]
                            # line = ','.join([f'{freq:.6f}' for freq in point_frequencies])
                            # freq_fp.write(f'{upper},{lower},points,{line}\n')
                            #
                            # plt.clf()
                            # title = f'{upper}-{lower}-clusters'
                            # plt.figure(figsize=(9, 9))
                            # plt.bar(range(manifold.depth + 1), cluster_frequencies)
                            # plt.title(title)
                            # plt.xlabel('depth')
                            # plt.ylabel('log(frequency + 1)')
                            # plt.xticks(range(manifold.depth + 1))
                            # plt.yticks(range(y_max + 1))
                            # plotname = os.path.join(depths_path, f'{title}.png')
                            # plt.savefig(fname=plotname)
                            # plt.close('all')
                            #
                            # plt.clf()
                            # title = f'{upper}-{lower}-points'
                            # plt.figure(figsize=(9, 9))
                            # plt.bar(range(manifold.depth + 1), point_frequencies)
                            # plt.title(title)
                            # plt.xlabel('depth')
                            # plt.ylabel('log(frequency + 1)')
                            # plt.xticks(range(manifold.depth + 1))
                            # plt.yticks(range(y_max + 1))
                            # plotname = os.path.join(depths_path, f'{title}.png')
                            # plt.savefig(fname=plotname)
                            # plt.close('all')

            for method, table in results.items():
                auc_filename = os.path.join(depths_path, f'{metric}-{METHOD_NAMES[method]}.csv')
                table = np.maximum(table, 0.5)
                with open(auc_filename, 'w') as auc_fp:
                    header = ','.join([f'{i}' for i in thresholds]) + '\n'
                    auc_fp.write(header)

                    for i, row in reversed(list(zip(thresholds, table))):
                        line = f'{i},' + ','.join([f'{score:.6f}' for score in row]) + '\n'
                        auc_fp.write(line)

                table = pd.read_csv(auc_filename)
                plotname = os.path.join(dataset_path, f'{metric}-{method}.png')
                plt.clf()
                plt.figure(figsize=(12, 12))
                sns.heatmap(
                    data=table,
                    vmin=0.5,
                    vmax=1,
                    square=True,
                    cmap='coolwarm',
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5,
                )
                plt.xlabel('Upper Threshold')
                plt.ylabel('Lower Threshold')
                plt.title(f'{dataset}-{metric}-{method}')
                plt.savefig(fname=plotname)
                plt.close('all')

    return


if __name__ == '__main__':
    _datasets = [
        'vowels',
        'cardio',
        'thyroid',
        # 'musk',
        # 'satimage-2',
        # 'satellite',
        # 'optdigits',
    ]
    _metrics = ['euclidean']
    depth_distributions(_datasets, _metrics)
