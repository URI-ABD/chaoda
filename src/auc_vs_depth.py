import numpy as np
import os
from typing import Dict, List
from matplotlib import pyplot as plt

from src.datasets import DATASETS

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
STATIC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kdd', 'static', 'auc_vs_depth'))


def plot_auc_vs_depth(
        dataset: str,
):
    metrics = ['cosine', 'euclidean', 'manhattan', 'hamming']
    methods = ['cluster_cardinality', 'hierarchical', 'k_neighborhood', 'random_walk', 'subgraph_cardinality']
    plot_lines = get_plot_lines(dataset)
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
            plt.plot(x, auc_scores)

        plt.legend(labels, loc='lower left')
        title = f'{dataset}-{metric}'
        plt.title(title)
        filename = os.path.join(STATIC_PATH, f'{title}.png')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
        # plt.show()

    return


def get_plot_lines(dataset: str) -> Dict[str, Dict[str, List[float]]]:
    log_path = os.path.join(BASE_PATH, dataset, 'roc_scores.log')
    with open(log_path, 'r') as fp:
        lines = list(filter(lambda line: 'roc_curve' in line, fp.readlines()))

    lines = [line.strip('\n').split(', ')[1:] for line in lines]
    metrics = {line[0] for line in lines}
    methods = {line[2] for line in lines}
    plot_lines: Dict[str, Dict[str, List[float]]] = {
        metric: {method: [] for method in methods}
        for metric in metrics
    }

    [plot_lines[line[0]][line[2]].append(float(line[3].split(':-:')[1]))
     for line in lines]

    return plot_lines


def make_table(dataset: str):
    metrics = ['cosine', 'euclidean', 'manhattan', 'hamming']
    methods = ['cluster_cardinality', 'hierarchical', 'k_neighborhood', 'random_walk', 'subgraph_cardinality']
    plot_lines = get_plot_lines(dataset)
    for metric in metrics:
        if metric not in plot_lines:
            continue
        method_lines = plot_lines[metric]
        for method in methods:
            if method not in method_lines:
                continue
            auc_scores = method_lines[method]
            best_depth = int(np.argmax(auc_scores))
            prev_depth = best_depth - 2 if best_depth > 1 else 0
            next_depth = best_depth + 2 if best_depth < len(auc_scores) - 2 else len(auc_scores) - 1

            best_score = auc_scores[best_depth]
            prev_score = auc_scores[prev_depth]
            next_score = auc_scores[next_depth]
            line = f'{dataset} & {metric} & {method} & {prev_score:.3f} & {best_score:.3f} & {next_score:.3f} \\\\'
            print(line)
    return


if __name__ == '__main__':
    os.makedirs(STATIC_PATH, exist_ok=True)
    # print(list(DATASETS.keys()))
    # [plot_auc_vs_depth(dataset) for dataset in DATASETS.keys()]
    [make_table(dataset) for dataset in ['lympho']]
