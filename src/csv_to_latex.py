import os
from pprint import pprint
from typing import List, Dict

from src.datasets import DATASETS
from src.utils import RESULTS_PATH

TRAIN_DATASETS = ['annthyroid', 'mnist', 'pendigits', 'satellite', 'shuttle', 'thyroid']
METRIC_NAMES = ['euclidean', 'manhattan']
METHOD_NAMES = ['CC', 'PC', 'KN', 'SC']
ENSEMBLE_NAMES = ['mean', 'product', 'max', 'min', 'max25', 'min25']
MEANS = ['amean', 'gmean', 'hmean']
SELECTION_MODES = [
    'percentile',
    'ranked',
]


def row_to_latex(row: List[float], margin: float = 0.02) -> List[str]:
    threshold = max(row) - margin
    return ['\\bfseries ' + f'{v:.2f}' if v >= threshold else f'{v:.2f}' for v in row]


def pyod_to_latex(filename: str):
    print(filename)
    return


def ensemble_to_latex(filename: str, ensemble_out: str):
    ensemble_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        dataset: {
            metric: {
                selection: {
                    method: -1 for method in ENSEMBLE_NAMES
                } for selection in SELECTION_MODES
            } for metric in METRIC_NAMES
        } for dataset in DATASETS.keys()
    }

    with open(filename, 'r') as fp:
        _ = fp.readline()
        for line in fp.readlines():
            row = line.split(',')
            [dataset, metric, selection] = row[:3]
            values = row[3:3 + 6]

            for method, value in zip(ENSEMBLE_NAMES, values):
                ensemble_dict[dataset][metric][selection][method] = float(value)

    column_names = ['dataset', 'metric'] + [name for _ in SELECTION_MODES for name in ENSEMBLE_NAMES]
    column_names = ['\\textbf{' + name + '}' for name in column_names]
    headers = [
        '\\begin{tabular}{|' + 'c|' * (2 + len(SELECTION_MODES) * len(ENSEMBLE_NAMES)) + '}',
        '\\hline',
        ' &  & ' + ' & '.join(['\\multicolumn{6}{c|}{' + f'{selection}' + '}' for selection in SELECTION_MODES]) + ' \\\\',
        '\\hline',
        ' & '.join(column_names) + ' \\\\',
        '\\hline',
    ]
    ensemble_rows: List[str] = list()
    for dataset in DATASETS:
        dataset_name = '\\textbf{' + dataset + '}' if dataset in TRAIN_DATASETS else dataset
        for metric in METRIC_NAMES:
            ensemble_row: List[float] = [float(ensemble_dict[dataset][metric][selection][method])
                                         for selection in SELECTION_MODES
                                         for method in ENSEMBLE_NAMES]
            ensemble_row: List[str] = [dataset_name, metric] + row_to_latex(ensemble_row)
            ensemble_rows.extend([' & '.join(ensemble_row) + ' \\\\', '\\hline'])
    footer = '\\end{tabular}'

    with open(ensemble_out, 'w') as fp:
            [fp.write(f'{header}\n') for header in headers]
            [fp.write(f'{row}\n') for row in ensemble_rows]
            fp.write(footer)
    return


def individual_to_latex(filename: str, individual_out: str):
    individual_dict: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {
        dataset: {
            metric: {
                selection: {
                    mean: {
                        method: -1 for method in METHOD_NAMES
                    } for mean in MEANS
                } for selection in SELECTION_MODES
            } for metric in METRIC_NAMES
        } for dataset in DATASETS.keys()
    }

    with open(filename, 'r') as fp:
        _ = fp.readline()
        for line in fp.readlines():
            row = line.split(',')
            [dataset, metric, selection] = row[:3]
            values = row[-12:]

            for i, mean in enumerate(MEANS):
                for j, method in enumerate(METHOD_NAMES):
                    index = j + i * len(MEANS)
                    individual_dict[dataset][metric][selection][mean][method] = float(values[index])

    column_names = ['dataset', 'metric'] + [name for _ in range(3) for name in METHOD_NAMES]
    column_names = ['\\textbf{' + name + '}' for name in column_names]
    headers = [
        '\\begin{tabular}{|' + 'c|' * (2 + len(MEANS) * len(METHOD_NAMES)) + '}',
        '\\hline',
        ' &  & ' + ' & '.join(['\\multicolumn{4}{c|}{' + f'{mean}' + '}' for mean in MEANS]) + ' \\\\',
        '\\hline',
        ' & '.join(column_names) + ' \\\\',
        '\\hline',
    ]
    individual_rows: Dict[str: List[str]] = {selection: list() for selection in SELECTION_MODES}
    for dataset in DATASETS:
        dataset_name = '\\textbf{' + dataset + '}' if dataset in TRAIN_DATASETS else dataset
        for metric in METRIC_NAMES:
            for selection in SELECTION_MODES:
                individual_row: List[float] = [float(individual_dict[dataset][metric][selection][mean][method])
                                               for method in METHOD_NAMES
                                               for mean in MEANS]
                individual_row: List[str] = [dataset_name, metric] + row_to_latex(individual_row)
                individual_rows[selection].extend([' & '.join(individual_row) + ' \\\\', '\\hline'])
    footer = '\\end{tabular}'

    for selection in SELECTION_MODES:
        outfile = individual_out[:-4] + f'_{selection}.txt'
        with open(outfile, 'w') as fp:
            [fp.write(f'{header}\n') for header in headers]
            [fp.write(f'{row}\n') for row in individual_rows[selection]]
            fp.write(footer)

    return


if __name__ == '__main__':
    os.makedirs(RESULTS_PATH, exist_ok=True)

    _chaoda_results = os.path.join(RESULTS_PATH, 'chaoda_predictions.csv')

    _ensemble_out = os.path.join(RESULTS_PATH, 'ensemble_table.txt')
    ensemble_to_latex(_chaoda_results, _ensemble_out)

    # _individual_out = os.path.join(RESULTS_PATH, 'individual_table.txt')
    # individual_to_latex(_chaoda_results, _individual_out)

    # _pyod_comparisons = os.path.join(RESULTS_PATH, 'pyod_comparisons.csv')
    # pyod_to_latex(_pyod_comparisons)
