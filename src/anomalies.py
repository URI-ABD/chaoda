import os
import random
from collections import deque
from typing import Dict, Set, List

import matplotlib.pyplot as plt
import numpy as np
from chess import criterion
from chess.manifold import Manifold, Graph, Cluster
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from visualizations import DATASETS, read_data, make_dirs


def normalize(anomalies: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v = np.min(list(anomalies.values())), np.max(list(anomalies.values()))
    min_v, max_v, = float(min_v), float(max_v)
    if min_v == max_v:
        max_v += 1.
    return {k: (v - min_v) / (max_v - min_v) for k, v in anomalies.items()}


def n_points_in_ball(graph: Graph) -> Dict[int, float]:
    manifold = graph.manifold
    data = manifold.data

    radius = 0.5
    results = {i: 0 for i in range(data.shape[0])}
    while len(results) > 5:
        results = {i: len(manifold.find_points(data[i], radius)) for i, _ in results.items()}
        results = dict(sorted(results.items(), key=lambda e: e[1])[:len(results) // 2 + 1])
        radius /= 2
    return normalize(results)


def k_nearest_neighbors_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by considering the kNearestNeighbors
    """
    manifold = graph.manifold
    data = manifold.data
    sample = list(map(int, np.random.choice(data.shape[0], 100)))
    sample.extend([i for i in range(int(data.shape[0] * 0.99), int(data.shape[0]))])
    sample = sorted(list(set(sample)))
    knn = {s: list(manifold.find_knn(manifold.data[s], 100).items()) for s in sample}
    scores = {i: sum([distances[k][1] for k in range(0, 100, 10)]) for i, distances in knn.items()}
    return normalize(scores)


def hierarchical_anomalies(graph: Graph) -> Dict[int, float]:
    manifold = graph.manifold
    data = manifold.data

    results = {i: list() for i in range(data.shape[0])}
    for graph in manifold.graphs[1:]:
        for cluster in graph.clusters.keys():
            if cluster.name[-1] == '1':
                parent = manifold.select(cluster.name[:-1])
                f = float(len(cluster.argpoints)) / len(parent.argpoints)
                if f < 0.25:
                    [results[p].append(1) for p in cluster.argpoints]
            elif cluster.name[-1] == '0':
                [results[p].append(1) for p in cluster.argpoints]

    results = {k: len(v) for k, v in results.items()}
    return normalize(results) if results else {}


def outrank_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the Outrank algorithm.

    :param graph: manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    subgraphs: Set[Graph] = graph.subgraphs
    anomalies: Dict[int, float] = dict()
    for subgraph in subgraphs:
        results: Dict[Cluster, int] = subgraph.random_walk(
            steps=max(len(subgraph.clusters.keys()) // 10, 10),
            walks=max(len(subgraph.clusters.keys()) * 10, 10),
        )
        anomalies.update({p: v for c, v in results.items() for p in c.argpoints})

    anomalies = normalize(anomalies)
    return {k: 1 - v for k, v in anomalies.items()}


def k_neighborhood_anomalies(graph: Graph, k: int = 10) -> Dict[int, float]:
    """ Determines anomalies by the considering the graph-neighborhood of clusters.

    :param graph: manifold in which to find anomalies.
    :param k: size of neighborhood to consider.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    def bft(start: Cluster) -> int:
        visited = set()
        queue = deque([start])
        for _ in range(k):
            if queue:
                c = queue.popleft()
                if c not in visited:
                    visited.add(c)
                    [queue.append(neighbor) for neighbor in c.neighbors.keys()]
            else:
                break
        return len(visited)

    results = {c: bft(c) for c in graph.clusters}
    anomalies: Dict[int, float] = {p: v for c, v in results.items() for p in c.argpoints}
    anomalies = normalize(anomalies)
    return {k: 1. - v for k, v in anomalies.items()}


def cluster_cardinality_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of clusters in the graph.

    :param graph: Manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: len(c.argpoints)
        for c in graph.clusters.keys()
        for p in c.argpoints
    }
    anomalies = normalize(anomalies)
    return {p: 1. - v for p, v in anomalies.items()}


def subgraph_cardinality_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of connected components in the graph.

    :param graph: Manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: len(subgraph.clusters.keys())
        for subgraph in graph.subgraphs
        for c in subgraph.clusters.keys()
        for p in c.argpoints
    }
    anomalies = normalize(anomalies)
    return {p: 1. - v for p, v in anomalies.items()}


def plot_histogram(
        x: List[float],
        dataset: str,
        metric: str,
        method: str,
        depth: int,
        save: bool,
):
    plt.clf()
    fig = plt.figure()
    n, bins, patches = plt.hist(x=x, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Anomalousness')
    plt.ylabel('Counts')
    max_freq = n.max()
    plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    if save is True:
        filepath = f'../data/{dataset}/plots/{method}/{metric}-{depth}-histogram.png'
        make_folders(dataset, method)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def plot_confusion_matrix(
        true_labels: List[int],
        anomalies: Dict[int, float],
        dataset: str,
        metric: str,
        method: str,
        depth: int,
        save: bool,
):
    p = float(sum([1 for k in anomalies.keys() if true_labels[k] == 1]))
    n = float(sum([1 for k in anomalies.keys() if true_labels[k] == 0]))

    threshold = float(np.percentile(list(anomalies.values()), 0.95))
    # threshold = 0.8
    tp = float(sum([1 if v > threshold else 0 for k, v in anomalies.items() if true_labels[k] == 1]))
    tn = float(sum([1 if v > threshold else 0 for k, v in anomalies.items() if true_labels[k] == 0]))

    tpr, tnr = tp / p, tn / n
    fpr, fnr = 1 - tnr, 1 - tpr

    confusion_matrix = [[tpr, fpr], [fnr, tnr]]

    plt.clf()
    fig = plt.figure()
    # noinspection PyUnresolvedReferences
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Wistia)
    class_names = ['Normal', 'Anomaly']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    s = [['TPR', 'FPR'], ['FNR', 'TNR']]
    [plt.text(j, i, f'{s[i][j]} = {confusion_matrix[i][j]:.3f}', position=(-0.2 + j, 0.03 + i))
     for i in range(2)
     for j in range(2)]

    if save is True:
        filepath = f'../data/{dataset}/plots/{method}/{metric}-{depth}-confusion_matrix.png'
        make_folders(dataset, method)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def make_folders(dataset, method):
    dir_paths = [f'../data',
                 f'../data/{dataset}',
                 f'../data/{dataset}/plots',
                 f'../data/{dataset}/plots/{method}']
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    return


def plot_data(data: np.ndarray, labels: List[int], name: str):
    plt.clf()
    fig = plt.figure(figsize=(6, 6), dpi=300)
    if data.shape[1] == 2:
        ax = fig.add_subplot(111)
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='Dark2', s=5.)
        ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    elif data.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=5., cmap='Dark2')
        ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])
        ax.set_zlim([np.min(data[:, 2]), np.max(data[:, 2])])
        ax.view_init(elev=20, azim=60)

    plt.savefig(name, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    plt.close('all')
    return


def main():
    make_dirs(list(DATASETS.keys()))
    methods = {
        # 'n_points_in_ball': n_points_in_ball,
        'k_nearest': k_nearest_neighbors_anomalies,
        'hierarchical': hierarchical_anomalies,
        'outrank': outrank_anomalies,
        # 'k_neighborhood': k_neighborhood_anomalies,
        # 'cluster_cardinality': cluster_cardinality_anomalies,
        # 'subgraph_cardinality': subgraph_cardinality_anomalies,
    }
    for dataset in DATASETS.keys():
        # if dataset not in ['mnist']:
        #     continue
        for metric in ['euclidean', 'manhattan', 'cosine']:
            np.random.seed(42)
            random.seed(42)
            data, labels = read_data(dataset)

            manifold: Manifold = Manifold(data, metric)
            if not os.path.exists(f'../logs'):
                os.mkdir(f'../logs')

            max_depth, min_points = 100, 1
            filepath = f'../logs/{dataset}_{metric}_{max_depth}_{min_points}.pickle'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as infile:
                    manifold = manifold.load(infile, data)
            else:
                manifold.build(
                    criterion.MaxDepth(max_depth),
                    criterion.MinPoints(min_points),
                )
                with open(filepath, 'wb') as infile:
                    manifold.dump(infile)

            print(f'\ndataset: {dataset}, metric: {metric}')
            for depth in range(0, manifold.depth + 1):
                print(f'depth: {depth},'
                      f' num_subgraphs: {len(manifold.graphs[depth].subgraphs)},'
                      f' num_clusters: {len(manifold.graphs[depth].clusters.keys())}')
                for method in methods.keys():
                    if method in ['n_points_in_ball', 'k_nearest'] and depth < manifold.depth:
                        continue
                    anomalies = methods[method](manifold.graphs[depth])
                    plot_histogram(
                        x=[v for _, v in anomalies.items()],
                        dataset=dataset,
                        metric=metric,
                        method=method,
                        depth=depth,
                        save=True,
                    )
                    plot_confusion_matrix(
                        true_labels=labels,
                        anomalies=anomalies,
                        dataset=dataset,
                        metric=metric,
                        method=method,
                        depth=depth,
                        save=True,
                    )
                    plt.close('all')
    return


if __name__ == '__main__':
    main()
