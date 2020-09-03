import os
from pprint import pprint
from typing import List, Dict, Tuple

from src.datasets import DATASETS
from src.utils import RESULTS_PATH

TRAIN_DATASETS = ['annthyroid', 'mnist', 'pendigits', 'satellite', 'shuttle', 'thyroid']
METRIC_NAMES = ['euclidean', 'manhattan']
METRIC_SHORTS = {'euclidean': 'L2', 'manhattan': 'L1'}
METHOD_NAMES = ['CC', 'PC', 'KN', 'SC']
ENSEMBLE_NAMES = ['mean', 'product', 'max', 'min', 'max25', 'min25']
MEANS = ['amean', 'gmean', 'hmean']
SELECTION_MODES = ['percentile', 'ranked']
PYOD_EXCEPTIONS = ['LSCP', 'XGBOD']


def row_to_latex(row: List[float], margin: float = 0.02) -> List[str]:
    threshold = max(row) - margin
    return ['\\bfseries ' + f'{v:.2f}' if v >= threshold else f'{v:.2f}' for v in row]


def pyod_to_dict(filename: str) -> Tuple[Dict[str, List[float]], List[str]]:
    pyod_dict: Dict[str, List[float]] = dict()

    with open(filename, 'r') as fp:
        header = fp.readline().strip().split()[0]
        method_names: List[str] = [name for name in header.split(',')[1:] if name not in PYOD_EXCEPTIONS]
        for line in fp.readlines():
            row = line.strip().split(',')
            row = [v for v in row if 'Exception' != v]
            for i in range(1, len(row)):
                if 'Timeout' in row[i]:
                    row[i] = '-1'
            pyod_dict[row[0]] = [float(v) for v in row[1:]]

    return pyod_dict, method_names


def pyod_to_latex(filename: str, pyod_out: str):
    pyod_dict, method_names = pyod_to_dict(filename)
    column_names: List[str] = ['\\textbf{dataset}'] + ['\\textbf{' + name + '}' for name in method_names if name not in PYOD_EXCEPTIONS]

    pyod_row_dict: Dict[str, str] = dict()
    for dataset in pyod_dict.keys():
        values = row_to_latex(pyod_dict[dataset])
        for i in range(len(values)):
            if '-1' in values[i]:
                values[i] = '\\textit{time}'
        pyod_row_dict[dataset] = ' & '.join(values)

    pyod_rows: List[str] = [
        '\\begin{tabular}{|' + 'c|' * len(column_names) + '}',
        '\\hline',
        ' & '.join(column_names) + ' \\\\',
        '\\hline',
    ]
    for dataset in sorted(pyod_row_dict.keys()):
        if dataset in PYOD_EXCEPTIONS:
            continue
        dataset_name = '\\textbf{' + dataset + '}' if dataset in TRAIN_DATASETS else dataset
        pyod_rows.extend([f'{dataset_name} & {pyod_row_dict[dataset]}' + ' \\\\', '\\hline'])

    with open(pyod_out, 'w') as fp:
        [fp.write(f'{line}\n') for line in pyod_rows]
        fp.write('\\end{tabular}')
    return


def ensemble_to_dict(filename: str) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
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
            row = line.strip().split(',')
            [dataset, metric, selection] = row[:3]
            values = row[3:3 + 6]

            for method, value in zip(ENSEMBLE_NAMES, values):
                ensemble_dict[dataset][metric][selection][method] = float(value)
    return ensemble_dict


def ensemble_to_latex(filename: str, ensemble_out: str):
    ensemble_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = ensemble_to_dict(filename)

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
            row = line.strip().split(',')
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


def comparison_table(chaoda_results: str, pyod_results: str, comparisons_out: str):
    ensemble, selection = 'mean', 'ranked'

    ensemble_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = ensemble_to_dict(chaoda_results)
    pyod_dict, pyod_methods = pyod_to_dict(pyod_results)

    def _helper(datasets: List[str], filename: str):
        comparisons_dict: Dict[str, Dict[str, str]] = dict()
        for dataset in datasets:
            column_values: List[float] = [ensemble_dict[dataset][metric][selection][ensemble] for metric in METRIC_NAMES]
            column_values.extend(pyod_dict[dataset])
            column_values: List[str] = row_to_latex(column_values)

            for i, metric in enumerate(METRIC_NAMES):
                metric_short = f'ensemble-{METRIC_SHORTS[metric]}'
                if metric_short not in comparisons_dict:
                    comparisons_dict[metric_short] = dict()
                comparisons_dict[metric_short][dataset] = column_values[i]
            for method, value in zip(pyod_methods, column_values[len(METRIC_NAMES):]):
                if method not in comparisons_dict:
                    comparisons_dict[method] = dict()
                comparisons_dict[method][dataset] = value

        column_names = ['dataset'] + datasets.copy()
        column_names = ['\\textbf{' + name + '}' for name in column_names]
        headers = [
            '\\begin{tabular}{|' + 'c|' * len(column_names) + '}',
            '\\hline',
            ' & '.join(column_names) + ' \\\\',
            '\\hline',
        ]

        methods = [f'ensemble-{METRIC_SHORTS[metric]}' for metric in METRIC_NAMES] + pyod_methods.copy()
        comparison_rows: List[str] = list()
        for method in methods:
            row: List[str] = [method] + [comparisons_dict[method][dataset] for dataset in datasets]
            for i in range(len(row)):
                if '-1' in row[i]:
                    row[i] = '\\textit{time}'
            comparison_rows.extend([' & '.join(row) + ' \\\\', '\\hline'])
        footer = '\\end{tabular}'

        with open(filename, 'w') as fp:
            [fp.write(f'{header}\n') for header in headers]
            [fp.write(f'{row}\n') for row in comparison_rows]
            fp.write(footer)
        return

    first_datasets = TRAIN_DATASETS.copy()
    first_filename = comparisons_out[:-4] + '_train.txt'
    _helper(first_datasets, first_filename)

    test_datasets = [dataset for dataset in DATASETS if dataset not in TRAIN_DATASETS]
    split_dataset = 'mammography'

    second_datasets = [dataset for dataset in test_datasets if dataset <= split_dataset]
    second_filename = comparisons_out[:-4] + '_test_1.txt'
    _helper(second_datasets, second_filename)

    third_datasets = [dataset for dataset in test_datasets if dataset > split_dataset]
    third_filename = comparisons_out[:-4] + '_test_2.txt'
    _helper(third_datasets, third_filename)
    return


if __name__ == '__main__':
    os.makedirs(RESULTS_PATH, exist_ok=True)

    _chaoda_results = os.path.join(RESULTS_PATH, 'chaoda_predictions.csv')
    _pyod_comparisons = os.path.join(RESULTS_PATH, 'pyod_comparisons.csv')

    # _ensemble_out = os.path.join(RESULTS_PATH, 'ensemble_table.txt')
    # ensemble_to_latex(_chaoda_results, _ensemble_out)

    # _individual_out = os.path.join(RESULTS_PATH, 'individual_table.txt')
    # individual_to_latex(_chaoda_results, _individual_out)

    # _pyod_out = os.path.join(RESULTS_PATH, 'pyod_table.txt')
    # pyod_to_latex(_pyod_comparisons, _pyod_out)

    _comparisons_out = os.path.join(RESULTS_PATH, 'comparisons_table.txt')
    comparison_table(_chaoda_results, _pyod_comparisons, _comparisons_out)
