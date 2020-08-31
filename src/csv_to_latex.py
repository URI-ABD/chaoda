import os
from typing import List, Dict

from src.datasets import DATASETS
from src.utils import RESULTS_PATH

TRAIN_DATASETS = ['annthyroid', 'mnist', 'pendigits', 'satellite', 'shuttle', 'thyroid']
METRIC_NAMES = ['euclidean', 'manhattan']
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


# noinspection DuplicatedCode
def chaoda_to_latex(filename: str, ensemble_out: str, individual_out: str):
    ensemble_table: List[str] = list()
    individual_tables: Dict[str: List[str]] = {selection: list() for selection in SELECTION_MODES}
    with open(filename, 'r') as fp:
        header = fp.readline()
        column_names = header.split(',')
        column_names = [' '.join(name.split('_')) if '_' in name else name
                        for name in column_names]
        meta_names = [name for name in column_names[:3]]

        ensemble_names = ['E ' + name.split()[1] for name in column_names if 'ensemble' in name]
        selection_names = ['\\multicolumn{' + str(len(ensemble_names)) + '}{c|}{' + f'{selection}' + '}' for selection in SELECTION_MODES]
        ensemble_table.extend([
            '\\begin{tabular}{|' + 'c|' * (len(meta_names) - 1 + len(ensemble_names) * len(selection_names)) + '}',
            '\\hline',
            ' &  & ' + ' & '.join(selection_names) + ' \\\\',
            '\\hline',
            ' & '.join(meta_names[:2]) + (' & ' + ' & '.join(ensemble_names)) * len(selection_names) + ' \\\\',
            '\\hline',
            ])

        individual_names = [name for name in column_names[3:] if 'ensemble' not in name]
        individual_mean_names = [name.split()[0] for name in individual_names[:len(individual_names) // 3]]
        mean_names = ['\\multicolumn{' + str(len(individual_mean_names)) + '}{c|}{' + f'{mean}' + '}' for mean in MEANS]
        selection_names = ['\\multicolumn{' + str(len(individual_names)) + '}{c|}{' + f'{selection}' + '}' for selection in SELECTION_MODES]
        for selection in SELECTION_MODES:
            individual_tables[selection].extend([
                '\\begin{tabular}{|' + 'c|' * (len(meta_names) - 1 + len(individual_names) * len(selection_names)) + '}',
                '\\hline',
                ' &  & ' + ' & '.join(selection_names) + ' \\\\',
                '\\hline',
                ' &  & ' + ' & '.join(mean_names) + ' \\\\',
                '\\hline',
                ' & '.join(meta_names[:2]) + (' & ' + ' & '.join(individual_mean_names * 3)) * len(selection_names) + ' \\\\',
                '\\hline',
            ])

        ensemble_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
            dataset: {
                metric: {
                    selection: {
                        method: -1 for method in ensemble_names
                    } for selection in SELECTION_MODES
                } for metric in METRIC_NAMES
            } for dataset in DATASETS.keys()
        }

        individual_dict: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {
            dataset: {
                metric: {
                    selection: {
                        mean: {
                            method: -1 for method in individual_mean_names
                        } for mean in MEANS
                    } for selection in SELECTION_MODES
                } for metric in METRIC_NAMES
            } for dataset in DATASETS.keys()
        }

        for line in fp.readlines():
            row: List[str] = line.split(',')
            meta = row[:len(meta_names)]
            [dataset, metric, selection] = meta
            if selection not in SELECTION_MODES:
                continue

            ensemble_values = row[len(meta_names):len(meta_names) + len(ensemble_names)]
            for i, value in enumerate(ensemble_values):
                ensemble_dict[dataset][metric][selection][ensemble_names[i]] = float(value)

            individual_values = row[len(meta_names) + len(ensemble_names):]
            for value in individual_values:
                for mean in MEANS:
                    for individual in individual_mean_names:
                        individual_dict[dataset][metric][selection][mean][individual] = float(value)

    for dataset in DATASETS:
        for metric in METRIC_NAMES:
            ensemble_row: List[float] = list()
            for selection in SELECTION_MODES:
                for method in ensemble_names:
                    ensemble_row.append(float(ensemble_dict[dataset][metric][selection][method]))
            ensemble_row: List[str] = [dataset, metric] + row_to_latex(ensemble_row)
            if dataset in TRAIN_DATASETS:
                ensemble_row[0] = '\\textbf{' + dataset + '}'
            ensemble_table.extend([' & '.join(ensemble_row) + ' \\\\', '\\hline'])

            for selection in SELECTION_MODES:
                individual_row: List[float] = list()
                for mean in MEANS:
                    for method in individual_mean_names:
                        score = float(individual_dict[dataset][metric][selection][mean][method])
                        individual_row.append(score)

                individual_row: List[str] = [dataset, metric] + row_to_latex(individual_row)
                if dataset in TRAIN_DATASETS:
                    individual_row[0] = '\\textbf{' + dataset + '}'
                individual_tables[selection].extend([' & '.join(individual_row) + ' \\\\', '\\hline'])

    ensemble_table.append('\\end{tabular}')
    with open(ensemble_out, 'w') as fp:
        [fp.write(line + '\n') for line in ensemble_table]

    for selection in SELECTION_MODES:
        outfile = individual_out[:-4] + f'_{selection}.txt'
        individual_table = individual_tables[selection]
        individual_table.append('\\end{tabular}')
        with open(outfile, 'w') as fp:
            [fp.write(line + '\n') for line in individual_table]

    return


if __name__ == '__main__':
    os.makedirs(RESULTS_PATH, exist_ok=True)

    _chaoda_results = os.path.join(RESULTS_PATH, 'chaoda_predictions.csv')
    _ensemble_out = os.path.join(RESULTS_PATH, 'ensemble_table.txt')
    _individual_out = os.path.join(RESULTS_PATH, 'individual_table.txt')
    chaoda_to_latex(_chaoda_results, _ensemble_out, _individual_out)

    # _pyod_comparisons = os.path.join(RESULTS_PATH, 'pyod_comparisons.csv')
    # pyod_to_latex(_pyod_comparisons)
