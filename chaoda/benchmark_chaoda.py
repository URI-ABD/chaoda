from pathlib import Path
from time import time
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy
from pyclam import CHAODA
from sklearn.metrics import roc_auc_score

from utils import constants
from utils import datasets
from utils import helpers
from utils import paths

try:
    from . import custom_meta_models as meta_models
except ImportError:
    from . import meta_models

__all__ = ['META_MODELS', 'bench_chaoda']

META_MODELS: List[Tuple[str, str, Callable[[numpy.array], float]]] = [
    # tuple of (metric, method, function)
    (name.split('_')[1], '_'.join(name.split('_')[2:]), method)
    for name, method in meta_models.META_MODELS.items()
]


def bench_dataset(dataset: str, fast: bool, individuals_csv_path: Optional[Path]) -> Tuple[float, float]:
    """ Runs CHAODA on the dataset and returns the tuple of (auc-score, time-taken).

    :param dataset: Name of the dataset to benchmark
    :param fast: Whether to use the speed heuristic
    :param individuals_csv_path:
    """
    data, labels = datasets.read(dataset, None, None)
    chaoda_name = 'CHAODA-Fast' if fast else 'CHAODA'
    helpers.print_blurb(chaoda_name, dataset, data.shape)
    speed_threshold = max(128, int(numpy.sqrt(len(labels)))) if fast else None
    print(f'speed threshold set to {speed_threshold}')

    start = time()
    detector: CHAODA = CHAODA(
        metrics=constants.METRICS,
        max_depth=constants.MAX_DEPTH,
        min_points=helpers.assign_min_points(data.shape[0]),
        meta_ml_functions=META_MODELS,
        speed_threshold=speed_threshold,
    ).fit(data=data)

    if individuals_csv_path is not None:
        # Print individual method scores.

        if not individuals_csv_path.exists():
            with open(individuals_csv_path, 'w') as individuals_csv:
                columns = ','.join([
                    'dataset',
                    'cardinality',
                    'dimensionality',
                    'num_components',
                    'num_clusters',
                    'num_edges',
                    'min_depth',
                    'max_depth',
                    'method',
                    'auc_roc',
                ])
                individuals_csv.write(f'{columns}\n')

        index = 0
        # noinspection PyProtectedMember
        for method_name, graph in detector._graphs:
            # noinspection PyProtectedMember
            scores = detector._individual_scores[index]
            auc = roc_auc_score(labels, scores)
            index += 1

            with open(individuals_csv_path, 'a') as individuals_csv:
                features = ','.join([
                    dataset,
                    f'{data.shape[0]}',
                    f'{data.shape[1]}',
                    f'{len(graph.components)}',
                    f'{graph.cardinality}',
                    f'{len(graph.edges)}',
                    f'{graph.depth_range[0]}',
                    f'{graph.depth_range[1]}',
                    method_name,
                    f'{auc:.2f}',
                ])
                individuals_csv.write(f'{features}\n')

    # Carry on with the ensemble.
    return float(roc_auc_score(labels, detector.scores)), float(time() - start)


def bench_chaoda(dataset_names: List[str], fast: bool, individuals_csv_path: Optional[Path] = None):
    """ Calculates anomaly scores for all datasets using the CHAODA class.
    """
    scores_df, times_df = helpers.get_dataframes()
    model_name = 'CHAODA_FAST' if fast else 'CHAODA'

    for i, dataset in enumerate(dataset_names):
        roc_score, time_taken = bench_dataset(dataset, fast, individuals_csv_path)

        scores_df.at[model_name, dataset] = roc_score
        times_df.at[model_name, dataset] = time_taken

        scores_df.to_csv(paths.SCORES_PATH, float_format='%.2f')
        times_df.to_csv(paths.TIMES_PATH, float_format='%.2e')
    return
