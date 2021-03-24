import random
from time import time
from typing import List

from pyclam import CHAODA
from sklearn.metrics import roc_auc_score

import datasets
from datasets import DATASET_LINKS
from datasets import OTHER_DATASETS
# you can replace this next import with your own generated file to verify results
from meta_models_trained import META_MODELS
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


def run_chaoda(dataset_names: List[str], fast: bool):
    """ Calculates anomaly scores for all datasets using the CHAODA class.
    """

    scores_file = CHAODA_FAST_SCORES_PATH if fast else CHAODA_SCORES_PATH
    times_file = CHAODA_FAST_TIMES_PATH if fast else CHAODA_TIMES_PATH
    model_name = 'CHAODA_FAST' if fast else 'CHAODA'

    header_line = f'model,{",".join(dataset_names)}\n'

    scores_list = ['_' for _ in dataset_names]
    times_list = ['_' for _ in dataset_names]

    for i, dataset in enumerate(dataset_names):
        s, t = _score_dataset(dataset, False, fast)
        scores_list[i] = f'{s:.2f}'
        times_list[i] = f'{t:.2f}'

        # write scores and times to file
        scores, times = ','.join(scores_list), ','.join(times_list)

        with open(scores_file, 'w') as fp:
            fp.writelines([header_line, f'{model_name},{scores}\n'])

        with open(times_file, 'w') as fp:
            fp.writelines([header_line, f'{model_name},{times}\n'])

    return


if __name__ == '__main__':
    np.random.seed(42), random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _dataset_names = list(DATASET_LINKS.keys())
    # run_chaoda(_dataset_names, fast=True)
    run_chaoda(OTHER_DATASETS, fast=True)
    # run_chaoda(fast=False)  # uncomment this to run CHAODA without the speed heuristic
