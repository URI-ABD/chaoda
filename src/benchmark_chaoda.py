import random
from time import time
from typing import List

from sklearn.metrics import roc_auc_score

import datasets
from datasets import DATASETS
# you can replace this next import with your own generated file to verify results
from meta_models_trained import META_MODELS
from pyclam import CHAODA
from utils import *

BASE_METHODS = {
    'cc': 'cluster_cardinality',
    'sc': 'component_cardinality',
    'gn': 'graph_neighborhood',
    'pc': 'parent_cardinality',
    'rw': 'random_walks',
    'sp': 'stationary_probabilities',
}
# DATASETS = ['vertebral']  # for testing
_META_MODELS = [
    (BASE_METHODS[name.split('_')[2]], decider)
    for name, decider in META_MODELS.items()
    if (name.split('_')[2] in BASE_METHODS)
]
_NORMALIZATIONS = [
    None,
    'linear',
    'gaussian',
    'sigmoid',
]


def _score_dataset(dataset: str, renormalize: bool, fast: bool) -> Tuple[float, float]:
    """ Runs CHAODA on the dataset and returns the tuple of (auc-score, time-taken). """
    data, labels = datasets.read(dataset, None, None)
    name = 'CHAODA-Fast' if fast else 'CHAODA'
    print_blurb(name, dataset, data.shape)
    speed_threshold = max(128, int(np.sqrt(data.shape[0]))) if fast else None
    if fast:
        print(f"Speed threshold set to {speed_threshold}.")

    start = time()
    detector: CHAODA = CHAODA(
        metrics=METRICS,
        max_depth=MAX_DEPTH,
        min_points=assign_min_points(data.shape[0]),
        meta_ml_functions=_META_MODELS,
        speed_threshold=speed_threshold,
    ).fit(
        data=data,
        renormalize=renormalize,
    )
    return float(roc_auc_score(labels, detector.scores)), float(time() - start)


def run_chaoda(fast: bool):
    """ Calculates anomaly scores for all datasets using the CHAODA class.
    """
    scores_file = CHAODA_FAST_SCORES_PATH if fast else CHAODA_SCORES_PATH
    if not os.path.exists(scores_file):
        labels = ','.join(DATASETS)
        with open(scores_file, 'w') as fp:
            fp.write(f'model,{labels}\n')

    times_file = CHAODA_FAST_TIMES_PATH if fast else CHAODA_TIMES_PATH
    if not os.path.exists(times_file):
        labels = ','.join(DATASETS)
        with open(times_file, 'w') as fp:
            fp.write(f'model,{labels}\n')

    model_name = 'CHAODA_FAST' if fast else 'CHAODA'

    # run model on all datasets
    performances: List[Tuple[float, float]] = [
        _score_dataset(dataset, False, fast)
        for dataset in DATASETS
    ]

    # write scores and times to file
    scores: str = ','.join([f'{s:.2f}' for s, _ in performances])
    times: str = ','.join([f'{t:.2f}' for _, t in performances])

    with open(scores_file, 'a') as fp:
        fp.write(f'{model_name},{scores}\n')
    with open(times_file, 'a') as fp:
        fp.write(f'{model_name},{times}\n')
    return


if __name__ == '__main__':
    np.random.seed(42), random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_chaoda(fast=True)
    # run_chaoda(fast=False)  # uncomment this to run CHAODA without the speed heuristic
