import errno
import os
import signal
from functools import wraps
from typing import Tuple

import numpy as np
from scipy.stats import gmean
from scipy.stats import hmean

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ABD_DATA_DIR = '/data/abd/chaoda_data'
DATA_DIR = os.path.join(SRC_DIR, 'data')
CLAM_DIR = os.path.join(SRC_DIR, 'clam')
TRAIN_DIR = os.path.join(SRC_DIR, 'train')
RESULTS_DIR = os.path.join(SRC_DIR, 'results')
PLOTS_DIR = os.path.join(SRC_DIR, 'plots')
UMAPS_DIR = os.path.join(SRC_DIR, 'umaps')

PYOD_SCORES_PATH = os.path.join(RESULTS_DIR, 'pyod_scores.csv')
PYOD_TIMES_PATH = os.path.join(RESULTS_DIR, 'pyod_times.csv')
CHAODA_SCORES_PATH = os.path.join(RESULTS_DIR, 'chaoda_scores.csv')
CHAODA_TIMES_PATH = os.path.join(RESULTS_DIR, 'chaoda_times.csv')
CHAODA_FAST_SCORES_PATH = os.path.join(RESULTS_DIR, 'chaoda_fast_scores.csv')
CHAODA_FAST_TIMES_PATH = os.path.join(RESULTS_DIR, 'chaoda_fast_times.csv')
SCORES_PATH = os.path.join(RESULTS_DIR, 'scores.csv')
TIMES_PATH = os.path.join(RESULTS_DIR, 'times.csv')

METRICS = ['cityblock', 'euclidean']

MEANS = {
    'amean': np.mean,
    'gmean': gmean,
    'hmean': hmean,
}


NORMALIZE = 'gaussian'
SUB_SAMPLE = 64_000  # for testing the implementation
MAX_DEPTH = 25  # even though no dataset reaches this far


def print_blurb(model: str, dataset: str, shape: Tuple[int, int]):
    print()
    print('-' * 80)
    print()
    print(f'Running model {model} on dataset \'{dataset}\' with shape {shape}.')
    print()
    print('-' * 80)
    return


def manifold_path(dataset: str, metric: str) -> str:
    """ Generate proper path to manifold. """
    os.makedirs(CLAM_DIR, exist_ok=True)
    return os.path.join(CLAM_DIR, f'{dataset}-{metric}.clam')


def assign_min_points(num_points: int) -> int:
    """ Improves runtime speed. set to 1 if willing to wait. """
    return max((num_points // 1000), 1)


def timeout(seconds, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        # noinspection PyUnusedLocal
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)
    return decorator
