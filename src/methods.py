from collections import deque
from typing import Dict, Set

import numpy as np
from pyclam.manifold import Graph, Cluster


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

    sample_size = min(2_000, int(data.shape[0] * 0.05))
    # sample_size = int(data.shape[0])
    sample = sorted(list(map(int, np.random.choice(data.shape[0], sample_size, replace=False))))
    knn = {s: list(manifold.find_knn(manifold.data[s], 10)) for s in sample}
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
        results: Dict[Cluster, int] = subgraph.random_walks(
            clusters=list(np.random.choice(list(subgraph.clusters.keys()), int(np.sqrt(subgraph.cardinality)))),
            steps=max(int(np.sqrt(subgraph.cardinality)), 100),
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


METHODS = {
    'cluster_cardinality': cluster_cardinality_anomalies,
    'hierarchical': hierarchical_anomalies,
    # 'k_nearest': k_nearest_neighbors_anomalies,
    'k_neighborhood': k_neighborhood_anomalies,
    # 'n_points_in_ball': n_points_in_ball,
    'random_walk': outrank_anomalies,
    'subgraph_cardinality': subgraph_cardinality_anomalies,
}
