from typing import Tuple

import pandas

from . import paths


def get_dataframes():
    import datasets

    if not paths.SCORES_PATH.exists():
        scores_df = pandas.DataFrame(columns=datasets.DATASET_NAMES)
        scores_df.index.name = 'model'
        times_df = pandas.DataFrame(columns=datasets.DATASET_NAMES)
        times_df.index.name = 'model'
    else:
        scores_df = pandas.read_csv(paths.SCORES_PATH, index_col='model')
        times_df = pandas.read_csv(paths.TIMES_PATH, index_col='model')
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
