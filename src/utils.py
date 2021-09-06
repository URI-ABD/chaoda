from pathlib import Path
from typing import Tuple

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR.joinpath('data')
RESULTS_DIR = ROOT_DIR.joinpath('results')
PLOTS_DIR = ROOT_DIR.joinpath('plots')
UMAPS_DIR = ROOT_DIR.joinpath('umaps')

SCORES_PATH = RESULTS_DIR.joinpath('scores.csv')
TIMES_PATH = RESULTS_DIR.joinpath('times.csv')

METRICS = ['cityblock', 'euclidean']
NORMALIZE = 'gaussian'
SUB_SAMPLE = 64_000  # for testing the implementation
MAX_DEPTH = 50  # even though no dataset reaches this far


def get_dataframes():
    import datasets

    if not SCORES_PATH.exists():
        scores_df = pd.DataFrame(columns=datasets.DATASET_NAMES)
        scores_df.index.name = 'model'
        times_df = pd.DataFrame(columns=datasets.DATASET_NAMES)
        times_df.index.name = 'model'
    else:
        scores_df = pd.read_csv(SCORES_PATH, index_col='model')
        times_df = pd.read_csv(TIMES_PATH, index_col='model')
    return scores_df, times_df


def print_blurb(model: str, dataset: str, shape: Tuple[int, int]):
    print()
    print('-' * 80)
    print(f'Running model {model} on dataset \'{dataset}\' with shape {shape}.')
    print('-' * 80)
    print()
    return


def assign_min_points(num_points: int) -> int:
    """ Improves runtime speed without impacting scores. set to 1 if willing to wait. """
    return max((num_points // 1000), 1)
