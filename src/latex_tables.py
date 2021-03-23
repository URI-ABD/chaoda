from typing import List

import pandas as pd

from src.datasets import DATASET_LINKS
from src.utils import *

TRAIN_DATASETS: List[str] = [
    'annthyroid',
    'mnist',
    'pendigits',
    'satellite',
    'shuttle',
    'thyroid',
]


def bold_best(values: List[str], *, margin: float = 0.02, high: bool = True) -> List[str]:
    """ Highlight the best value in the given values. """
    to_indices, ex_indices = list(), list()
    for i, v in enumerate(values):
        if v == 'TO':
            to_indices.append(i)
        elif v == 'EX':
            ex_indices.append(i)
        else:
            continue

    for i in to_indices:
        values[i] = '-1'
    for i in ex_indices:
        values[i] = '-1'

    values: List[float] = [float(v) for v in values]
    if high:
        threshold: float = max(values) - margin
        values: List[str] = [''.join(['\\textbf{', f'{v:.2f}', '}']) if v >= threshold else f'{v:.2f}' for v in values]
    else:
        threshold: float = max(min(values), 0.1) * (1 + margin)
        values: List[str] = [''.join(['\\textbf{', f'{v:.2f}', '}']) if v <= threshold else f'{v:.2f}' for v in values]

    for i in range(len(values)):
        if i in to_indices:
            values[i] = '\\textit{TO}'
        elif i in ex_indices:
            values[i] = '\\textit{EX}'
        else:
            continue

    return values


def get_path(mode: str, pyod: bool):
    if mode == 'scores':
        path: str = PYOD_SCORES_PATH if pyod else CHAODA_SCORES_PATH
        high: bool = True
    elif mode == 'times':
        path: str = PYOD_TIMES_PATH if pyod else CHAODA_TIMES_PATH
        high: bool = False
    else:
        raise ValueError(f'mode must be \'scores\' or \'times\'. Got {mode} instead.')

    return path, high


def bold_column(column: List[str]) -> List[str]:
    return [''.join(['\\textbf{' + c + '}']) for c in column]


# noinspection DuplicatedCode
def parse_csv(mode: str, datasets: List[str]):
    if mode not in ['scores', 'times']:
        raise ValueError(f'mode must be \'scores\' or \'times\'. Got {mode} instead.')

    high = mode == 'scores'
    path = SCORES_PATH if high else TIMES_PATH
    raw_df: pd.DataFrame = pd.read_csv(path, dtype=str)

    new_df: pd.DataFrame = pd.DataFrame()
    new_df['model'] = list(raw_df['model'].values)
    for dataset in datasets:
        dataset = 'mammo' if dataset == 'mammography' else dataset
        new_df[dataset] = bold_best(list(raw_df[dataset].values), high=high)

    return new_df, bold_column(list(new_df.columns))


# noinspection DuplicatedCode
def get_latex(mode: str, datasets: List[str]):
    df, columns = parse_csv(mode, datasets)

    latex_string: str = df.to_latex(
        header=columns,
        index=False,
        column_format='|' + 'c|' * len(columns),
        escape=False,
    )
    latex_list = latex_string.split('\n')
    latex_list.pop(1)
    latex_list.pop(2)
    latex_list.pop(-3)

    for i in range(len(latex_list[:-2]), 0, -1):
        latex_list.insert(i, '\\hline')

    return '\n'.join(latex_list)


def write_tables():
    out_path = os.path.join(RESULTS_DIR, 'latex')

    def _write_tables(name: str, datasets: List[str]):
        path = f'{out_path}_scores_{name}.txt'
        with open(path, 'w') as fp:
            fp.write(get_latex('scores', datasets))

        path = f'{out_path}_times_{name}.txt'
        with open(path, 'w') as fp:
            fp.write(get_latex('times', datasets))

        return

    test_datasets = list(sorted([
        dataset for dataset in DATASET_LINKS.keys()
        if dataset not in TRAIN_DATASETS
    ]))
    half_num = len(test_datasets) // 2

    _write_tables('train', TRAIN_DATASETS)
    _write_tables('test_1', test_datasets[:half_num])
    _write_tables('test_2', test_datasets[half_num:])
    return


def parse_chaoda(mode: str, datasets: List[str]):
    path, high = get_path(mode, False)
    chaoda_df: pd.DataFrame = pd.read_csv(path, dtype=str)

    new_df: pd.DataFrame = pd.DataFrame()
    new_df['voting'] = chaoda_df['voting']
    new_df['normed'] = chaoda_df['normed']
    for dataset in datasets:
        name = '\\textbf{' + dataset + '}'
        new_df[name] = bold_best(list(chaoda_df[dataset].values), high=high)

    return new_df, bold_column(list(new_df.columns))


# noinspection DuplicatedCode
def parse_pyod(mode: str, datasets: List[str]):
    path, high = get_path(mode, True)
    pyod_df: pd.DataFrame = pd.read_csv(path, dtype=str)

    new_df: pd.DataFrame = pd.DataFrame()
    new_df['\\textbf{model}'] = bold_column(list(pyod_df['model'].values))
    for dataset in datasets:
        name = '\\textbf{' + dataset + '}'
        new_df[name] = bold_best(list(pyod_df[dataset].values), high=high)

    return new_df, bold_column(list(new_df.columns))


# noinspection DuplicatedCode
def get_latex_old(mode: str, pyod: bool, datasets: List[str]):
    """ Produces latex tables. """
    # TODO: Do train-test1-test2 splits

    df, columns = parse_pyod(mode, datasets) if pyod else parse_chaoda(mode, datasets)

    latex_string: str = df.to_latex(
        header=columns,
        index=False,
        column_format='|' + 'c|' * len(columns),
        escape=False,
    )
    latex_list = latex_string.split('\n')
    latex_list.pop(1)
    latex_list.pop(2)
    latex_list.pop(-3)

    for i in range(len(latex_list[:-2]), 0, -1):
        latex_list.insert(i, '\\hline')

    return '\n'.join(latex_list)


def write_tables_old():
    num_datasets = len(DATASET_LINKS.keys())
    step = num_datasets // 3
    chaoda_path = os.path.join(RESULTS_DIR, 'latex_chaoda')
    # pyod_path = os.path.join(RESULTS_DIR, 'latex_pyod')

    for i, j in enumerate(range(0, num_datasets, step)):
        datasets = list(DATASET_LINKS.keys())[j: j + step]
        path = f'{chaoda_path}_scores_{i + 1}.txt'
        with open(path, 'w') as fp:
            fp.write(get_latex_old('scores', False, datasets))

        path = f'{chaoda_path}_times_{i + 1}.txt'
        with open(path, 'w') as fp:
            fp.write(get_latex_old('times', False, datasets))

        # path = f'{pyod_path}_scores_{i + 1}.txt'
        # with open(path, 'w') as fp:
        #     fp.write(get_latex('scores', True, datasets))
        #
        # path = f'{pyod_path}_times_{i + 1}.txt'
        # with open(path, 'w') as fp:
        #     fp.write(get_latex('times', True, datasets))
    return


if __name__ == '__main__':
    write_tables()
