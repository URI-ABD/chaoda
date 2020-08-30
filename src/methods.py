from collections import deque
from typing import Dict, List

import numpy as np
from pyclam.manifold import Cluster, Graph, Manifold
from scipy.special import erf

# TODO: Parent-Child radii ratios as proxy to anomalousness.
# TODO: Print graphs in dot format and visualize them
# TODO: 1 / depth as proxy of anomalousness

METHOD_NAMES = {
    'cluster_cardinality': 'CC',
    'hierarchical': 'PC',
    'k_neighborhood': 'KN',
    'subgraph_cardinality': 'SC',
}
ENSEMBLE_MODES = ['mean', 'product', 'max', 'min', 'max25', 'min25']
SCALING_MODES = ['linear', 'gaussian']


def normalize(anomalies: Dict[int, float], mode: str = 'gaussian') -> Dict[int, float]:
    if mode == 'linear':
        min_v, max_v = np.min(list(anomalies.values())), np.max(list(anomalies.values()))
        min_v, max_v, = float(min_v), float(max_v)
        if min_v == max_v:
            max_v += 1.
        return {k: (v - min_v) / (max_v - min_v) for k, v in anomalies.items()}
    elif mode == 'gaussian':
        indices: List[int] = list(anomalies.keys())
        scores: np.array = np.asarray(list(anomalies.values()), dtype=float)

        mu, sigma = np.mean(scores), np.std(scores)
        if sigma < 1e-3:
            sigma = 1

        scores = erf((scores - mu) / (sigma * np.sqrt(2))).ravel().clip(0, 1)

        return {i: s for i, s in zip(indices, scores)}
    else:
        raise ValueError(f'scaling mode {mode} is undefined. Choose one of {SCALING_MODES}.')


# def n_points_in_ball(graph: Graph) -> Dict[int, float]:
#     # TODO: Fix
#     manifold = graph.manifold
#     data = manifold.data
#
#     starting_radius = float(manifold.select('').radius) * (10 ** -2)
#     sample_size = int(data.shape[0] * (0.05 if data.shape[0] > 10_000 else 1.0))
#     print(sample_size)
#     sample = sorted(list(map(int, np.random.choice(data.shape[0], sample_size, replace=False))))
#     scores = {i: 0. for i in range(sample_size)}
#     for point in sample:
#         radius = starting_radius
#         num_results = len(manifold.find_points(data[point], radius))
#         while num_results > 10:
#             radius /= 2.
#             num_results = len(manifold.find_points(data[point], radius))
#         scores[point] = 0. - radius
#
#     [print(k, v) for k, v in sorted(scores.items())]
#     scores = normalize(scores)
#     return scores


# def k_nearest_neighbors_anomalies(graph: Graph) -> Dict[int, float]:
#     """ Determines anomalies by considering the kNearestNeighbors
#     """
#     # TODO: fix subsampling
#
#     manifold = graph.manifold
#     data = manifold.data
#
#     sample_size = min(2_000, int(data.shape[0] * 0.05))
#     # sample_size = int(data.shape[0])
#     sample = sorted(list(map(int, np.random.choice(data.shape[0], sample_size, replace=False))))
#     knn = {s: list(manifold.find_knn(manifold.data[s], 10)) for s in sample}
#     scores = {i: sum([distances[k][1] for k in range(0, 10)]) for i, distances in knn.items()}
#     return normalize(scores)


def hierarchical_anomalies(graph: Graph) -> Dict[int, float]:
    manifold = graph.manifold
    data = manifold.data

    results = {i: list() for i in range(data.shape[0])}
    for cluster in graph:
        ancestry = manifold.ancestry(cluster)
        for i in range(1, len(ancestry)):
            score = float(ancestry[i - 1].cardinality) / (ancestry[i].cardinality * np.sqrt(i))
            [results[p].append(score) for p in cluster.argpoints]

    results = {k: sum(v) for k, v in results.items()}
    return normalize(results)


def outrank_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the Outrank-Algorithm.

    :param graph: Graph in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    print(f'clusters: {graph.cardinality}, components: {len(graph.subgraphs)}')
    num_samples = 500
    if graph.cardinality < num_samples:
        sample_clusters = list(graph.clusters)
    else:
        sample_clusters = list(np.random.choice(list(graph.clusters), num_samples, False))
    steps = 10 * int(graph.population / np.sqrt(len(sample_clusters)))
    print(len(sample_clusters), steps)

    scores: Dict[Cluster, float] = graph.random_walks(
        starts=sample_clusters,
        steps=min(graph.cardinality * 5, steps),
    )

    anomalies: Dict[int, float] = {point: 0 for cluster in scores for point in cluster.argpoints}
    for cluster, v in scores.items():
        for p in cluster.argpoints:
            anomalies[p] += v
    anomalies = normalize(anomalies)
    return {k: 1. - v for k, v in anomalies.items()}


def k_neighborhood_anomalies(graph: Graph, k: int = 10) -> Dict[int, float]:
    """ Determines anomalies by the considering the graph-neighborhood of clusters.

    :param graph: Graph in which to find anomalies.
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
                    [queue.append(neighbor) for neighbor in graph.neighbors(c)]
            else:
                break
        return len(visited)

    results = {c: bft(c) for c in graph.clusters}
    anomalies: Dict[int, float] = {p: v for c, v in results.items() for p in c.argpoints}
    anomalies = normalize(anomalies)
    return {k: 1. - v for k, v in anomalies.items()}


def cluster_cardinality_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of clusters in the graph.

    :param graph: Graph in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: c.cardinality
        for c in graph.clusters
        for p in c.argpoints
    }
    anomalies = normalize(anomalies)
    return {p: 1. - v for p, v in anomalies.items()}


def subgraph_cardinality_anomalies(graph: Graph) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of connected components in the graph.

    :param graph: Graph in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: subgraph.cardinality
        for subgraph in graph.subgraphs
        for c in subgraph.clusters
        for p in c.argpoints
    }
    anomalies = normalize(anomalies)
    return {p: 1. - v for p, v in anomalies.items()}


METHODS = {
    'cluster_cardinality': cluster_cardinality_anomalies,
    'hierarchical': hierarchical_anomalies,
    # 'k_nearest': k_nearest_neighbors_anomalies,
    'k_neighborhood': k_neighborhood_anomalies,
    # 'n_points_in_ball': n_points_in_ball,
    # 'random_walk': outrank_anomalies,
    'subgraph_cardinality': subgraph_cardinality_anomalies,
}


def ensemble(manifold: Manifold, mode: str) -> Dict[int, float]:
    """ Builds ensemble model from given methods, using given option of combining scores.

    :param manifold: manifold from which to calculate outlier scores
    :param mode: how to combine scores. One of 'mean', 'product', 'max' or 'min'.
    """
    assert mode in ENSEMBLE_MODES

    scores: List[np.ndarray] = list()
    for graph in manifold.graphs:
        score: Dict[int, float] = METHODS[graph.method](graph)
        scores.append(np.asarray([score[i] for i in range(len(score.keys()))], dtype=float))

    if mode == 'mean':
        means: np.ndarray = np.sum(scores, axis=0) / len(scores)
        return {i: float(s) for i, s in enumerate(means)}
    elif mode == 'product':
        products = np.ones_like(scores[0])
        for score in scores:
            products = np.multiply(products, score)
        return {i: float(p) for i, p in enumerate(products)}
    elif mode == 'max':
        return {i: float(max(s[i] for s in scores)) for i in range(len(scores[0]))}
    elif mode == 'min':
        return {i: float(min(s[i] for s in scores)) for i in range(len(scores[0]))}
    elif mode == 'max25':
        quarter = len(scores[0]) // 4
        return {i: float(sum(list(sorted([s[i] for s in scores], reverse=True))[:quarter]) / quarter) for i in range(len((scores[0])))}
    elif mode == 'min25':
        quarter = len(scores[0]) // 4
        return {i: float(sum(list(sorted([s[i] for s in scores]))[:quarter]) / quarter) for i in range(len((scores[0])))}
