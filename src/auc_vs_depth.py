from glob import glob

import numpy as np
import os
from typing import Dict, List
from matplotlib import pyplot as plt
from pyclam import Manifold

from src.__main__ import BUILD_DIR, METRICS, NORMALIZE, SUB_SAMPLE
from src.datasets import DATASETS, read

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
AUC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kdd', 'static', 'auc_vs_depth'))
LFD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kdd', 'static', 'lfd_vs_depth'))

PAPERS = {
    'SVM sklearn': {
        'lympho': 0.70,
        'wbc': 0.78,
        'glass': 0.58,
        'vowels': 0.69,
        'cardio': 0.78,
        'thyroid': 0.78,
        'musk': 0.74,
        'satimage-2': 0.76,
        'pima': 0.61,
        'satellite': 0.60,
        'shuttle': 0.77,
        'breastw': 0.85,
        'arrhythmia': 0.68,
        'ionosphere': 0.73,
        'mnist': 0.73,
        'optdigits': 0.50,
        'http': 0.76,
        'cover': 0.66,
        'smtp': 0.69,
        'mammography': 0.65,
        'annthyroid': 0.55,
        'pendigits': 0.75,
        'wine': 0.72,
        'vertebral': 0.54,
    },
    'LOF Lazarevic': {
        'satimage-2': '',
        'lympho': '',
        'mammography': '',
        'http': '',
        'annthyroid': 0.869,
        'letter': 0.820,
        'shuttle': 0.839,
    },
    'HiCS Keller': {
        'annthyroid': 0.951,
        'arrhythmia': 0.623,
        'wbc': 0.593,
        'breastw': 0.942,
        'pima': 0.725,
        'glass': 0.801,
        'ionosphere': 0.823,
        'pendigits': 0.950,
    },
    'LODES Sathe': {
        'glass': 0.873,
        'pendigits': 0.944,
        'ecoli': 0.893,
        'vowels': 0.911,
        'cardio': 0.721,
        'wine': 0.966,
        'thyroid': 0.684,
        'vertebral': 0.582,
        'yeast': 0.814,
        'seismic': 0.634,
        'heart': 0.591
    },
    'iForest Liu': {
        'http': 1.00,
        'cover': 0.88,
        'mulcross': 0.97,
        'smtp': 0.88,
        'shuttle': 1.00,
        'mammography': 0.86,
        'annthyroid': 0.82,
        'satellite': 0.71,
        'pima': 0.67,
        'breastw': 0.99,
        'arrhythmia': 0.80,
        'ionosphere': 0.85,
    },  # iForest -> Mass -> Mass Estimation
    'Mass Ting': {
        'http': 1.00,
        'cover': 0.89,
        'mulcross': 0.99,
        'smtp': 0.90,
        'shuttle': 1.00,
        'mammography': 0.86,
        'annthyroid': 0.73,
        'satellite': 0.74,
        'pima': 0.69,
        'breastw': 0.99,
        'arrhythmia': 0.84,
        'ionosphere': 0.80,
    },
    'MassE Ting': {
        'http': 1.00,
        'cover': 0.92,
        'mulcross': 0.99,
        'smtp': 0.91,
        'shuttle': 1.00,
        'mammography': 0.86,
        'annthyroid': 0.75,
        'satellite': 0.77,
    },
    'AOD Abe': {
        'mammography': 0.81,
        'http': 0.935,
        'shuttle': 0.999,
        'annthyroid': 0.97,
    },
    'HST Tan': {
        'http': 0.982,
        'smtp': 0.740,
        'cover': 0.854,
        'mulcross': 0.998,
        'shuttle': 0.999,
    },
    'iNNE Bandaragoda': {
        'http': 1.00,
        'cover': 0.97,
        'mulcross': 1.00,
        'smtp': 0.95,
        'shuttle': 0.99,
        'mnist': 0.87,
        'p53mutants': 0.73,
    },
    # 'Online OD DLA. Yamanishi': dict(),
    # 'RNN for OD. Williams': {
    #     'breastw': '',
    #     'http': '',
    #     'smtp': '',
    # },
    # 'Subsampling. Zimek': {
    #     'lympho': '',
    #     'wbc': '',
    #     'satimage-2': '',
    # },
    # 'Theoretical. Aggarwal': {
    #     'glass': '',
    #     'lympho': '',
    #     'wbc': '',
    #     'vowels': '',
    #     'satimage-2': '',
    #     'cardio': '',
    #     'optdigits': '',
    #     'musk': '',
    # },
}
METHODS = {
    'cluster_cardinality': 'CC',
    'hierarchical': 'PC',
    'k_neighborhood': 'KN',
    'random_walk': 'RW',
    'subgraph_cardinality': 'SC',
}
_METRICS = {
    'cosine': 'Cos',
    'euclidean': 'L2',
    'manhattan': 'L1',
}


def plot_auc_vs_depth(dataset: str):
    metrics = list(_METRICS.keys())
    methods = list(METHODS.keys())
    plot_lines = get_plot_lines(dataset, plot=True)
    for metric in metrics:
        if metric not in plot_lines:
            continue
        method_lines = plot_lines[metric]

        plt.close('all')
        fig = plt.figure(figsize=(8, 4), dpi=300)
        fig.add_subplot(111)

        x = list(range(max(map(len, method_lines.values()))))
        x_min, x_max = -1, ((x[-1] // 5) * 5)
        if x[-1] % 5 > 0:
            x_max += 6
        else:
            x_max += 1
        plt.xlim([x_min, x_max])
        plt.xticks(list(range(0, x_max, 5)))

        plt.ylim([-0.05, 1.05])
        plt.yticks([0., 0.25, 0.5, 0.75, 1.])

        labels = []
        for method in methods:
            if method not in method_lines:
                continue
            auc_scores = method_lines[method]
            labels.append(method)
            if len(x) > len(auc_scores):
                x = x[:len(auc_scores)]
            elif len(auc_scores) > len(x):
                auc_scores = auc_scores[:len(x)]
            plt.plot(x, auc_scores)

        plt.legend(labels, loc='lower left')
        title = f'{dataset}-{metric}'
        plt.title(title)
        filename = os.path.join(AUC_PATH, f'{title}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
        # plt.show()

    return


def get_plot_lines(dataset: str, plot: bool) -> Dict[str, Dict[str, List[float]]]:
    """ Returns dict of method -> metric -> scores if plot else metric -> method -> scores """
    log_path = os.path.join(BASE_PATH, dataset, 'roc_scores.log')
    with open(log_path, 'r') as fp:
        lines = list(filter(lambda line: 'roc_curve' in line, fp.readlines()))
    lines = [line.strip('\n').split(', ')[1:] for line in lines]
    metrics = {line[0] for line in lines}
    methods = {line[2] for line in lines}
    plot_lines: Dict[str, Dict[str, List[float]]]

    if plot:
        plot_lines = {
            metric: {method: [] for method in methods}
            for metric in metrics
        }
        [plot_lines[line[0]][line[2]].append(float(line[3].split(':-:')[1]))
         for line in lines]
    else:
        plot_lines = {
            method: {metric: [] for metric in metrics}
            for method in methods
        }

        [plot_lines[line[2]][line[0]].append(float(line[3].split(':-:')[1]))
         for line in lines]

    return plot_lines


def make_table(dataset: str):
    for paper in PAPERS:
        if dataset not in PAPERS[paper]:
            PAPERS[paper][dataset] = ''

    plot_lines = get_plot_lines(dataset, plot=False)
    best_scores = {method: ('', '') for method in METHODS}
    best_best_metric, best_best_score = '', 0.
    for method in METHODS:
        if method not in plot_lines:
            continue
        method_lines = plot_lines[method]
        best_score = 0.
        best_metric = ''
        for metric in _METRICS:
            if metric not in method_lines:
                continue
            auc_scores = method_lines[metric]

            depth = int(np.argmax(auc_scores))
            score = auc_scores[depth]
            if score > best_score:
                best_metric, best_score = _METRICS[metric], score
                if score > best_best_score:
                    best_best_metric = _METRICS[metric]
        best_scores[method] = (best_metric, best_score)

    chaoda_scores = ['-' if type(score) is str else f'{score:.2f}'
                     for metric, score in best_scores.values()]
    chaoda_line = ' & '.join(chaoda_scores)
    comparisons = ['-' if type(PAPERS[paper][dataset]) is str else f'{PAPERS[paper][dataset]:.2f}'
                   for paper in PAPERS]
    comparisons_line = ' & '.join(comparisons)
    line = f'\\bfseries {dataset} & {best_best_metric} & {chaoda_line} & {comparisons_line} \\\\ \n\\hline'
    print(line)
    return


def _plot_lfd_helper(
        percentiles: List[float],
        lfds: np.ndarray,
        dataset: str,
        metric: str,
):
    percentiles = [f'{p / 100:.1f}' for p in percentiles]
    plt.close('all')
    fig = plt.figure(figsize=(8, 4), dpi=300)
    fig.add_subplot(111)

    x = list(range(lfds.shape[1]))
    x_min, x_max = -1, ((x[-1] // 5) * 5)
    if x[-1] % 5 > 0:
        x_max += 6
    else:
        x_max += 1
    plt.xlim([x_min, x_max])
    plt.xticks(list(range(0, x_max, 5)))

    for row in lfds:
        plt.plot(x, row)

    plt.legend(percentiles, loc='lower right')
    title = f'{dataset}-{metric}'
    plt.title(title)
    filename = os.path.join(LFD_PATH, f'{title}.png')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
    return


def plot_lfd_vs_depth(dataset: str):
    builds = glob(BUILD_DIR + '/*')
    percentiles = [i for i in range(0, 101, 10)]
    for b in builds:
        for metric in METRICS:
            if dataset in b and metric in b:
                data, _ = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
                manifold = Manifold(data, metric)
                with open(b, 'rb') as fp:
                    manifold = manifold.load(fp, data)

                lfds = np.zeros(shape=(manifold.depth + 1, len(percentiles)))
                for graph in manifold.graphs:
                    lfds[graph.depth] = np.percentile([cluster.local_fractal_dimension for cluster in graph.clusters], percentiles)

                _plot_lfd_helper(percentiles, lfds.T, dataset, metric)
    return


if __name__ == '__main__':
    os.makedirs(AUC_PATH, exist_ok=True)
    # print(list(DATASETS.keys()))
    # [plot_auc_vs_depth(dataset) for dataset in DATASETS]
    method_names = ' & \\bfseries '.join(METHODS.values())
    paper_names = ' & \\bfseries '.join([paper.split()[0] for paper in PAPERS.keys()])
    header = f'\\bfseries Dataset & \\bfseries Metric & \\bfseries {method_names} & \\bfseries {paper_names} \\\\ \n\\hline'
    print(header)
    [make_table(dataset) for dataset in DATASETS]
    # [plot_lfd_vs_depth(dataset) for dataset in DATASETS]
