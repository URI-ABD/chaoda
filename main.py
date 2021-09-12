import argparse
import random
import warnings

import numpy

import chaoda
import comparisons
import sdss
import utils

MODES = [
    'download-datasets',
    'train-meta-ml',
    'bench-chaoda',
    'bench-pyod',
    'preparse-apogee',
    'score-apogee',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main',
        description='Run CHAODA in various modes',
    )

    parser.add_argument(
        '--mode',
        dest='mode',
        type=str,
        help=f'Your options are: {MODES}.',
        required=True,
    )

    parser.add_argument(
        '--meta-ml-epochs',
        dest='meta_ml_epochs',
        type=int,
        help=f'How many epochs to train the meta-ml models. Defaults to 10.',
        required=False,
        default=10,
    )

    parser.add_argument(
        '--fast',
        dest='fast',
        type=bool,
        help=f'Whether to use CHAODA\'s speed heuristic.',
        required=False,
        default=True,
    )



    parser.add_argument(
        '--datasets',
        dest='datasets',
        type=str,
        help=f'Which datasets to run benchmarks on.',
        required=False,
    )

    parser.add_argument(
        '--report-individual-methods',
        dest='report_individual_methods',
        type=bool,
        help=f'Whether to report the performance of each of CHAODA\'s individual algorithms. Defaults to False.',
        required=False,
        default=False,
    )

    parser.add_argument(
        '--pyod-time-limit',
        dest='pyod_time_limit',
        type=int,
        help=f'How much time, in seconds, to allow each pyod model to run. Defaults to 10 hours.',
        required=False,
        default=36_000,
    )

    args = parser.parse_args()

    _mode = args.mode
    if _mode not in MODES:
        raise ValueError(f'--mode {_mode} is not one of {MODES}.')

    utils.paths.create_required_folders()

    if _mode == 'download-datasets':
        utils.datasets.download_datasets()

    elif _mode == 'train-meta-ml':
        numpy.random.seed(42), random.seed(42)

        _sampling_datasets = chaoda.train_meta_ml.SAMPLING_DATASETS
        _train_datasets = list(sorted(numpy.random.choice(_sampling_datasets, 6, replace=False)))
        print(f'training meta-ml models on {_train_datasets} datasets...')

        chaoda.create_models(_train_datasets, args.meta_ml_epochs)

    elif _mode == 'bench-chaoda':
        numpy.random.seed(42), random.seed(42)

        _dataset_names = args.datasets
        if _dataset_names is None:
            _dataset_names = utils.datasets.DATASET_NAMES
        else:
            _dataset_names = str(_dataset_names).split(',')
            for _dataset_name in _dataset_names:
                assert _dataset_name in utils.datasets.DATASET_NAMES, f'--dataset {_dataset_name} not found. Must be one of {utils.datasets.DATASET_NAMES}.'

        if args.report_individual_methods:
            _individual_scores_path = utils.paths.RESULTS_DIR.joinpath('individual_scores.csv')
        else:
            _individual_scores_path = None

        chaoda.bench_chaoda(
            dataset_names=_dataset_names,
            fast=args.fast,
            individuals_csv_path=_individual_scores_path,
        )

    elif _mode == 'bench-pyod':
        warnings.filterwarnings('ignore')
        numpy.random.seed(42), random.seed(42)

        _dataset_names = utils.datasets.DATASET_NAMES

        comparisons.against_pyod.bench_pyod(_dataset_names, args.pyod_time_limit)

    elif _mode == 'preparse-apogee':
        sdss.extract_apogee()

    else:  # _mode == 'score-apogee'
        sdss.score_apogee()
