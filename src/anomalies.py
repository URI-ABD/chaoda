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
from sklearn.metrics import roc_curve, auc
# noinspection PyUnresolvedReferences
from visualizations import DATASETS, read_data, make_dirs


def normalize(anomalies: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v = np.min(list(anomalies.values())), np.max(list(anomalies.values()))
    min_v, max_v, = float(min_v), float(max_v)
    if min_v == max_v:
        max_v += 1.
    return {k: (v - min_v) / (max_v - min_v) for k, v in anomalies.items()}


def n_points_in_ball(graph: Graph) -> Dict[int, float]:
    # TODO: Fix
    manifold = graph.manifold
    data = manifold.data

    starting_radius = float(manifold.select('').radius) * (10 ** -2)
    sample_size = int(data.shape[0] * (0.05 if data.shape[0] > 10_000 else 1.0))
    print(sample_size)
    sample = sorted(list(map(int, np.random.choice(data.shape[0], sample_size, replace=False))))
    scores = {i: 0. for i in range(sample_size)}
    for point in sample:
        radius = starting_radius
        num_results = len(manifold.find_points(data[point], radius))
        while num_results > 10:
            radius /= 2.
            num_results = len(manifold.find_points(data[point], radius))
        scores[point] = 0. - radius

    [print(k, v) for k, v in sorted(scores.items())]
    scores = normalize(scores)
    return scores


def k_nearest_neighbors_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by considering the kNearestNeighbors
    """
    manifold = graph.manifold
    data = manifold.data

    sample_size = min(10_000, int(data.shape[0] * 0.05))
    if sample_size < data.shape[0]:
        sample = sorted(list(map(int, np.random.choice(data.shape[0], sample_size, replace=False))))
    else:
        sample = list(range(data.shape[0]))
    knn = {s: list(manifold.find_knn(manifold.data[s], 10).items()) for s in sample}
    scores = {i: sum([distances[k][1] for k in range(0, 10)]) for i, distances in knn.items()}
    return normalize(scores)


def hierarchical_anomalies(graph: Graph) -> Dict[int, float]:
    manifold = graph.manifold
    depth = list(graph.clusters.keys())[0].depth
    data = manifold.data

    results = {i: list() for i in range(data.shape[0])}
    for g in manifold.graphs[1: depth]:
        for cluster in g.clusters.keys():
            parent = manifold.select(cluster.name[:-1])
            f = 1. / (float(len(cluster.argpoints)) / len(parent.argpoints))
            [results[p].append(f) for p in cluster.argpoints]
            
    results = {k: sum(v) for k, v in results.items()}
    return normalize(results)


def outrank_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the Outrank algorithm.

    :param graph: manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    subgraphs: Set[Graph] = graph.subgraphs
    anomalies: Dict[int, float] = dict()
    for subgraph in subgraphs:
        results: Dict[Cluster, int] = subgraph.random_walk(
            steps=1000,  # max(len(subgraph.clusters.keys()) // 10, 10),
            walks=max(len(subgraph.clusters.keys()) // 100, 10),
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
        filepath = f'../data/{dataset}/plots/{metric}/{method}/{depth}-histogram.png'
        make_folders(dataset, metric, method)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def plot_roc_curve(true_labels, anomalies, dataset, metric, method, depth, save):
    y_true, y_score = [], []
    [(y_true.append(true_labels[k]), y_score.append(v)) for k, v in anomalies.items()]
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.clf()
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.6f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset}-{metric}-{method}-{depth}')
    plt.legend(loc="lower right")

    if save is True:
        filepath = f'../data/{dataset}/plots-98/{metric}/{method}/{depth}-roc_curve.png'
        make_folders(dataset, metric, method)
        fig.savefig(filepath)
    else:
        plt.show()
    
    csv_filepath = f'../data/{dataset}/plots-98/{metric}/{method}/roc_curves.csv'
    if not os.path.exists(csv_filepath):
        with open(csv_filepath, 'w') as outfile:
            outfile.write('depth,scores\n')
    with open(csv_filepath, 'a') as outfile:
        line = '_'.join([f'{s:.16f}' for i, s in sorted(list(anomalies.items()))])
        outfile.write(f'{depth},{line}\n')
    
    return


def plot_confusion_matrix(
        true_labels: List[bool],
        anomalies: Dict[int, float],
        dataset: str,
        metric: str,
        method: str,
        depth: int,
        save: bool,
):
    p = float(len([k for k in anomalies.keys() if true_labels[k] == 1]))
    n = float(len([k for k in anomalies.keys() if true_labels[k] == 0]))

    threshold = float(np.percentile(list(anomalies.values()), 95))
    values = [float(v) for v in anomalies.values()]
    print(threshold, min(values), max(values), np.mean(values), np.std(np.asarray(values)))
    tp = float(sum([v > threshold for k, v in anomalies.items() if true_labels[k] == 1]))
    tn = float(sum([v < threshold for k, v in anomalies.items() if true_labels[k] == 0]))

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
        filepath = f'../data/{dataset}/plots/{metric}/{method}/{depth}-confusion_matrix.png'
        make_folders(dataset, metric, method)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def make_folders(dataset, metric, method):
    dir_paths = [f'../data',
                 f'../data/{dataset}',
                 f'../data/{dataset}/plots-98',
                 f'../data/{dataset}/plots-98/{metric}',
                 f'../data/{dataset}/plots-98/{metric}/{method}']
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
        # 'k_nearest': k_nearest_neighbors_anomalies,
        'hierarchical': hierarchical_anomalies,
        'random_walk': outrank_anomalies,
        'k_neighborhood': k_neighborhood_anomalies,
        'cluster_cardinality': cluster_cardinality_anomalies,
        'subgraph_cardinality': subgraph_cardinality_anomalies,
    }

    for dataset in DATASETS.keys():
        # if dataset not in ['optdigits']:
        #     continue
        
        np.random.seed(42)
        random.seed(42)
        data, labels = read_data(dataset)

        if data.shape[0] > 10_000:
            continue
        
        for metric in ['cosine', 'euclidean', 'manhattan', ]:  # 
            print(f'\ndataset: {dataset}, metric: {metric}, data_shape: {data.shape}, outliers: {len([l for l in labels if l > 0.9]) / len(labels)}')

            if metric == 'manhattan':
                manifold = Manifold(data, 'cityblock')
            # elif metric == 'cosine':
            #     pass
            else:
                manifold = Manifold(data, metric)
            if not os.path.exists(f'../logs'):
                os.mkdir(f'../logs')

            max_depth, min_points = 50, 1
            graph_ratio = 100
            # manifold.build(
            #     criterion.MaxDepth(max_depth),
            #     criterion.MinPoints(min_points),
            # )

            filepath = f'../logs/{dataset}-{metric}-{max_depth}-{min_points}-{graph_ratio}.pickle'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as infile:
                    manifold = manifold.load(infile, data)
            else:
                for d in range(max_depth - 1, 0, -1):
                    oldfile = f'../logs/{dataset}-{metric}-{d}-{min_points}-{graph_ratio}.pickle'
                    if os.path.exists(oldfile):
                        with open(oldfile, 'rb') as infile:
                            manifold = manifold.load(infile, data)
                        manifold.build_tree(
                            criterion.MaxDepth(max_depth),
                            criterion.MinPoints(min_points),
                        )
                        [graph.build_edges() for graph in manifold.graphs[d + 1:]]
                        break
                else:
                    manifold.build(
                        criterion.MaxDepth(max_depth),
                        criterion.MinPoints(min_points),
                    )
                filepath = f'../logs/{dataset}-{metric}-{len(manifold.graphs) - 1}-{min_points}-{graph_ratio}.pickle'
                with open(filepath, 'wb') as infile:
                    # print(f'\n\n SAVING!!!!! \n{filepath} \n\n')
                    manifold.dump(infile)

            for depth in range(0, manifold.depth + 1, 1):
                print(f'depth: {depth},'
                      f' num_subgraphs: {len(manifold.graphs[depth].subgraphs)},'
                      f' num_clusters: {len(manifold.graphs[depth].clusters.keys())}')
                for method in methods.keys():
                    print(f'method: {method}')
                    if method in ['n_points_in_ball', 'k_nearest'] and depth < manifold.depth:
                        continue
                    anomalies = methods[method](manifold.graphs[depth])

                    # plot_histogram(
                    #     x=[v for _, v in anomalies.items()],
                    #     dataset=dataset,
                    #     metric=metric,
                    #     method=method,
                    #     depth=depth,
                    #     save=True,
                    # )
                    plot_roc_curve(
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
