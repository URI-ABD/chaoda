import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from pyclam import Manifold, criterion
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

# noinspection PyUnresolvedReferences
from datasets import get, read, DATASETS, METRICS
# noinspection PyUnresolvedReferences
from methods import METHODS

PAPERS = {
    'SVM sklearn': {
        'lympho': '',
        'wbc': '',
        'glass': '',
        'vowels': '',
        'cardio': '',
        'thyroid': '',
        'musk': '',
        'satimage-2': '',
        'pima': '',
        'satellite': '',
        'shuttle': '',
        'breastw': '',
        'arrhythmia': '',
        'ionosphere': '',
        'mnist': '',
        'optdigits': '',
        'http': '',
        'cover': '',
        'smtp': '',
        'mammography': '',
        'annthyroid': '',
        'pendigits': '',
        'wine': '',
        'vertebral': '',
    },
    'LOF Lazarevic': {
        'satimage-2': '',
        'lympho': '',
        'mammography': '',
        'http': '',
        'annthyroid': 0.869,
        'letter': 0.820,
        'shuttle': 0.839,
    },
    'HiCS Keller': {
        'annthyroid': 0.951,
        'arrhythmia': 0.623,
        'wbc': 0.593,
        'breastw': 0.942,
        'pima': 0.725,
        'glass': 0.801,
        'ionosphere': 0.823,
        'pendigits': 0.950,
    },
    'LODES Sathe': {
        'glass': 0.873,
        'pendigits': 0.944,
        'ecoli': 0.893,
        'vowels': 0.911,
        'cardio': 0.721,
        'wine': 0.966,
        'thyroid': 0.684,
        'vertebral': 0.582,
        'yeast': 0.814,
        'seismic': 0.634,
        'heart': 0.591
    },
    'iForest Liu': {
        'http': 1.00,
        'cover': 0.88,
        'mulcross': 0.97,
        'smtp': 0.88,
        'shuttle': 1.00,
        'mammography': 0.86,
        'annthyroid': 0.82,
        'satellite': 0.71,
        'pima': 0.67,
        'breastw': 0.99,
        'arrhythmia': 0.80,
        'ionosphere': 0.85,
    },  # iForest -> Mass -> Mass Estimation
    'Mass Ting': {
        'http': 1.00,
        'cover': 0.89,
        'mulcross': 0.99,
        'smtp': 0.90,
        'shuttle': 1.00,
        'mammography': 0.86,
        'annthyroid': 0.73,
        'satellite': 0.74,
        'pima': 0.69,
        'breastw': 0.99,
        'arrhythmia': 0.84,
        'ionosphere': 0.80,
    },
    'MassE Ting': {
        'http': 1.00,
        'cover': 0.92,
        'mulcross': 0.99,
        'smtp': 0.91,
        'shuttle': 1.00,
        'mammography': 0.86,
        'annthyroid': 0.75,
        'satellite': 0.77,
    },
    'AOD Abe': {
        'mammography': 0.81,
        'http': 0.935,
        'shuttle': 0.999,
        'annthyroid': 0.97,
    },
    'HST Tan': {
        'http': 0.982,
        'smtp': 0.740,
        'cover': 0.854,
        'mulcross': 0.998,
        'shuttle': 0.999,
    },
    'iNNE Bandaragoda': {
        'http': 1.00,
        'cover': 0.97,
        'mulcross': 1.00,
        'smtp': 0.95,
        'shuttle': 0.99,
        'mnist': 0.87,
        'p53mutants': 0.73,
    },
    # 'Online OD DLA. Yamanishi': dict(),
    # 'RNN for OD. Williams': {
    #     'breastw': '',
    #     'http': '',
    #     'smtp': '',
    # },
    # 'Subsampling. Zimek': {
    #     'lympho': '',
    #     'wbc': '',
    #     'satimage-2': '',
    # },
    # 'Theoretical. Aggarwal': {
    #     'glass': '',
    #     'lympho': '',
    #     'wbc': '',
    #     'vowels': '',
    #     'satimage-2': '',
    #     'cardio': '',
    #     'optdigits': '',
    #     'musk': '',
    # },
}
METHOD_NAMES = {
    'cluster_cardinality': 'CC',
    'hierarchical': 'PC',
    # 'k_nearest': 'KNN',
    'k_neighborhood': 'KN',
    # 'points_in_ball': 'NPB',
    # 'random_walk': 'RW',
    'subgraph_cardinality': 'SC',
}
METRIC_NAMES = {
    'cosine': 'Cos',
    'euclidean': 'L2',
    'manhattan': 'L1',
}
NORMALIZE = False
SUB_SAMPLE = 100_000  # Set this to None to not subsample large datasets
MAX_DEPTH = 50
PERCENTILES = list(range(0, 101, 10))
BUILD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
PLOTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
AUC_PATH = os.path.join(PLOTS_PATH, 'auc_vs_depth')
LFD_PATH = os.path.join(PLOTS_PATH, 'lfd_vs_depth')


def _manifold_path(dataset, metric, min_points) -> str:
    return os.path.join(
        BUILD_PATH,
        ':'.join(map(str, [dataset, METRIC_NAMES[metric], f'{min_points}.pickle']))
    )


def build_manifolds():
    # Initialize table for results
    results = {dataset: {metric: {method: [] for method in METHODS}
                         for metric in METRICS}
               for dataset in DATASETS}
    lfds_dict = {dataset: {metric: None for metric in METRICS} for dataset in DATASETS}
    os.makedirs(BUILD_PATH, exist_ok=True)
    os.makedirs(AUC_PATH, exist_ok=True)
    os.makedirs(LFD_PATH, exist_ok=True)

    for dataset in DATASETS:
        # download the dataset and read it in
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        labels = np.squeeze(labels)

        # set this to 1 to cluster to singletons
        min_points = min(1 + data.shape[0] // 1000, 25)

        for metric in METRICS:
            logging.info('; '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}',
            ]))
            manifold = Manifold(data, METRICS[metric])

            filepath = _manifold_path(dataset, metric, min_points)
            if os.path.exists(filepath):
                # load from memory and continue build if needed
                with open(filepath, 'rb') as fp:
                    logging.info(f'loading manifold {filepath}')
                    manifold = manifold.load(fp, data)
                temp_depth = manifold.depth
                manifold.build_tree(
                    criterion.MaxDepth(MAX_DEPTH),
                    criterion.MinPoints(min_points),
                )
                if manifold.depth > temp_depth:
                    manifold.build_graphs()
                    # save manifold
                    with open(filepath, 'wb') as fp:
                        logging.info(f'dumping manifold {filepath}')
                        manifold.dump(fp)
            else:
                # build manifold from scratch
                manifold.build(
                    criterion.MaxDepth(MAX_DEPTH),
                    criterion.MinPoints(min_points),
                )
                # save manifold
                with open(filepath, 'wb') as fp:
                    logging.info(f'dumping manifold {filepath}')
                    manifold.dump(fp)

            # get auc for each method for each depth in the manifold
            for method in METHODS:
                logging.info(f'{dataset}:{METRIC_NAMES[metric]}:({METHOD_NAMES[method]})')
                for depth in range(manifold.depth + 1):
                    if method in {'n_points_in_ball', 'k_nearest'} and depth < manifold.depth:
                        continue

                    anomalies = METHODS[method](manifold.graphs[depth])
                    y_true, y_score = list(), list()
                    [(y_true.append(labels[k]), y_score.append(v)) for k, v in anomalies.items()]
                    auc = roc_auc_score(y_true, y_score)
                    results[dataset][metric][method].append(auc)

            # get LFDs
            percentiles = list(range(0, 101, 10))
            lfds = np.zeros(shape=(manifold.depth + 1, len(percentiles)))
            for graph in manifold.graphs:
                candidates = [cluster.local_fractal_dimension for cluster in graph.clusters if cluster.cardinality > 2]
                if len(candidates) == 0:
                    candidates = [0.]
                lfds[graph.depth] = np.percentile(candidates, percentiles)
            lfds_dict[dataset][metric] = lfds.T

        # Run OneClassSVM on each dataset
        inliers, outliers = np.argwhere(labels == 0).flatten(), np.argwhere(labels == 1).flatten()
        if outliers.shape[0] > 500:
            outliers = np.random.choice(outliers, 500, replace=False)
        size = min(outliers.shape[0] * 18, inliers.shape[0])
        indices = np.concatenate([
            np.random.choice(inliers, size=size, replace=False),
            outliers,
        ])
        train, test = train_test_split(indices, stratify=labels[indices])

        model = OneClassSVM()
        model.fit(data[train], y=labels[train])
        predicted = model.predict(data[test])
        predicted = np.clip(predicted, a_min=0, a_max=1)
        predicted = [1 - p for p in predicted]
        PAPERS['SVM sklearn'][dataset] = roc_auc_score(labels[test], predicted)

    return results, lfds_dict


def plot_results(results, lfds_dict):
    for dataset in DATASETS:
        for metric in METRICS:
            # plot auc vs depth for each method
            logging.info(f'plotting {dataset} {METRIC_NAMES[metric]}')
            plt.close('all')
            fig = plt.figure(figsize=(8, 4), dpi=300)
            fig.add_subplot(111)

            x = list(range(max(map(len, results[dataset][metric].values()))))
            x_min, x_max = -1, ((x[-1] // 5) * 5)
            if x[-1] % 5 > 0:
                x_max += 6
            else:
                x_max += 1

            plt.xlim([x_min, x_max])
            plt.xticks(list(range(0, x_max, 5)))

            plt.ylim([-0.05, 1.05])
            plt.yticks([0., 0.25, 0.5, 0.75, 1.])

            legend = []
            for method in METHODS:
                auc_scores = results[dataset][metric][method]
                legend.append(f'{METHOD_NAMES[method]}: {max(auc_scores):.3f}')
                if len(x) > len(auc_scores):
                    x = x[:len(auc_scores)]
                elif len(auc_scores) > len(x):
                    auc_scores = auc_scores[:len(x)]
                plt.plot(x, auc_scores)

            plt.legend(legend, loc='lower left')
            title = f'{dataset}-{METRIC_NAMES[metric]}'
            plt.title(title)
            filename = os.path.join(AUC_PATH, f'{title}.png')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)

            # plot lfd vs depth
            lfds = lfds_dict[dataset][metric]

            plt.close('all')
            fig = plt.figure(figsize=(8, 4), dpi=300)
            fig.add_subplot(111)

            plt.xlim([x_min, x_max])
            plt.xticks(list(range(0, x_max, 5)))

            [plt.plot(x, row) for row in lfds]

            legend = [f'{p / 100:.1f}' for p in PERCENTILES]
            plt.legend(legend, loc='lower right')
            title = f'{dataset}-{METRIC_NAMES[metric]}'
            plt.title(title)
            filename = os.path.join(LFD_PATH, f'{title}.png')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)

    # create table
    logging.info(f'building table')
    os.makedirs(PLOTS_PATH, exist_ok=True)
    table_filepath = os.path.join(PLOTS_PATH, 'table.txt')
    
    method_names = ' & \\bfseries '.join(METHOD_NAMES.values())
    papers_names = ' & \\bfseries '.join([paper.split()[0] for paper in PAPERS])
    header = f'\\bfseries Dataset & \\bfseries {method_names} & \\bfseries {papers_names} \\\\ \n\\hline\n'
    table_dict = {dataset: {method: ('', 0.) for method in METHODS} for dataset in DATASETS}

    if not os.path.exists(table_filepath):
        with open(table_filepath, 'w') as fp:
            fp.write(header)

    with open(table_filepath, 'a') as fp:
        for dataset in DATASETS:
            for paper in PAPERS:
                if dataset not in PAPERS[paper]:
                    PAPERS[paper][dataset] = ''

                metric_to_methods = results[dataset]
                for metric in METRICS:
                    if metric not in metric_to_methods:
                        continue
                    method_to_auc = metric_to_methods[metric]
                    for method in METHODS:
                        if method not in method_to_auc:
                            continue
                        score = max(method_to_auc[method])
                        if score > table_dict[dataset][method][1]:
                            table_dict[dataset][method] = METRIC_NAMES[metric], score
            chaoda_line = ' & '.join(['-' if type(score) is str else f'{metric} {score:.3f}'
                                      for metric, score in table_dict[dataset].values()])
            others_line = ' & '.join(['-' if type(PAPERS[paper][dataset]) is str else f'{PAPERS[paper][dataset]:.3f}'
                                      for paper in PAPERS])
            line = f'\\bfseries {dataset} & {chaoda_line} & {others_line} \\\\ \n\\hline'
            fp.write(line)

    return


if __name__ == '__main__':
    _results, _lfds_dict = build_manifolds()
    plot_results(_results, _lfds_dict)
