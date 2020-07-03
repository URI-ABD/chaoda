from collections import deque
from typing import Dict

import numpy as np
from pyclam.manifold import Cluster, Manifold


# TODO: Parent-Child radii ratios as proxy to anomalousness.


def normalize(anomalies: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v = np.min(list(anomalies.values())), np.max(list(anomalies.values()))
    min_v, max_v, = float(min_v), float(max_v)
    if min_v == max_v:
        max_v += 1.
    return {k: (v - min_v) / (max_v - min_v) for k, v in anomalies.items()}


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


def hierarchical_anomalies(manifold: Manifold) -> Dict[int, float]:
    data = manifold.data

    results = {i: list() for i in range(data.shape[0])}
    for cluster in manifold.graph:
        ancestry = manifold.ancestry(cluster)
        for i in range(1, len(ancestry)):
            score = float(ancestry[i - 1].cardinality) / (ancestry[i].cardinality * np.sqrt(i))
            [results[p].append(score) for p in cluster.argpoints]

    results = {k: sum(v) for k, v in results.items()}
    return normalize(results)


def outrank_anomalies(manifold: Manifold) -> Dict[int, float]:
    """ Determines anomalies by the Outrank-Algorithm.

    :param manifold: manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    graph = manifold.graph
    print(f'clusters: {graph.cardinality}, components: {len(graph.subgraphs)}')
    num_samples = 500
    if graph.cardinality < num_samples:
        sample_clusters = list(graph.clusters)
    else:
        sample_clusters = list(np.random.choice(list(graph.clusters), num_samples, False))
    steps = 10 * int(graph.population / np.sqrt(len(sample_clusters)))
    print(len(sample_clusters), steps)

    scores: Dict[Cluster, float] = graph.random_walks(
        clusters=sample_clusters,
        steps=min(graph.cardinality * 5, steps),
    )

    anomalies: Dict[int, float] = {point: 0 for cluster in scores for point in cluster.argpoints}
    for cluster, v in scores.items():
        for p in cluster.argpoints:
            anomalies[p] += v
    anomalies = normalize(anomalies)
    return {k: 1. - v for k, v in anomalies.items()}


def k_neighborhood_anomalies(manifold: Manifold, k: int = 10) -> Dict[int, float]:
    """ Determines anomalies by the considering the graph-neighborhood of clusters.

    :param manifold: manifold in which to find anomalies.
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
                    [queue.append(neighbor) for neighbor in manifold.graph.neighbors(c)]
            else:
                break
        return len(visited)

    results = {c: bft(c) for c in manifold.graph.clusters}
    anomalies: Dict[int, float] = {p: v for c, v in results.items() for p in c.argpoints}
    anomalies = normalize(anomalies)
    return {k: 1. - v for k, v in anomalies.items()}


def cluster_cardinality_anomalies(manifold: Manifold) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of clusters in the graph.

    :param manifold: Manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: c.cardinality
        for c in manifold.graph.clusters
        for p in c.argpoints
    }
    anomalies = normalize(anomalies)
    return {p: 1. - v for p, v in anomalies.items()}


def subgraph_cardinality_anomalies(manifold: Manifold) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of connected components in the graph.

    :param manifold: Manifold in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: subgraph.cardinality
        for subgraph in manifold.graph.subgraphs
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

# TODO: Print graphs in dot format and visualize them
