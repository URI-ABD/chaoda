import logging
import os
import random
from glob import glob

import click
import numpy as np
from pyclam import criterion
from pyclam.manifold import Manifold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from .datasets import DATASETS, get, read
from .methods import METHODS
from .plot import RESULT_PLOTS, embed_umap, plot_2d, PLOT_DIR

np.random.seed(42)
random.seed(42)

SUB_SAMPLE = 100_000
NORMALIZE = False

METRICS = {
    'cosine': 'cosine',
    'euclidean': 'euclidean',
    'manhattan': 'cityblock',
}

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
UMAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'umaps'))


def _manifold_path(dataset, metric, min_points, graph_ratio) -> str:
    """ Generate proper path to manifold. """
    return os.path.join(
        BUILD_DIR,
        ':'.join(map(str, [dataset, metric, min_points, f'{graph_ratio}.pickle']))
    )


def _dataset_from_path(path):
    return os.path.basename(path).split(':')[0]


def _metric_from_path(path):
    return os.path.basename(path).split(':')[1]


def _meta_from_path(path):
    bundle = os.path.basename(path).split(':')
    # noinspection PyTypeChecker
    return {
        'dataset': bundle[0],
        'metric': bundle[1],
        'min_points': bundle[2],
        'graph_ratio': bundle[3].split('.pickle')[0],
    }


class State:
    """ This gets passed between chained commands. """

    def __init__(self, dataset=None, metric=None, manifold=None):
        self.dataset = dataset
        self.metric = metric
        self.manifold = manifold

    def __getattr__(self, name):
        return None


@click.group(chain=True)
@click.option('--dataset', type=click.Choice(DATASETS.keys()))
@click.option('--metric', type=click.Choice(METRICS.keys()))
@click.pass_context
def cli(ctx, dataset, metric):
    ctx.obj = State(dataset, metric)


@cli.command()
@click.pass_obj
def svm(state):
    datasets = [state.dataset] if state.dataset else DATASETS.keys()

    # Build.
    for dataset in datasets:
        np.random.seed(42)
        get(dataset)
        data, labels = read(dataset, normalize=False)
        labels = np.squeeze(labels)
        normal, anomalies = np.argwhere(labels == 0).flatten(), np.argwhere(labels == 1).flatten()
        if anomalies.shape[0] > 500:
            anomalies = np.random.choice(anomalies, 500, replace=False)
        size = min(anomalies.shape[0] * 18, normal.shape[0])
        indices = np.concatenate([
            np.random.choice(normal, size=size, replace=False),
            anomalies
        ])
        train, test = train_test_split(indices, stratify=labels[indices])
        filepath = os.path.join(BUILD_DIR, 'svm', ':'.join([dataset]))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model = OneClassSVM()
        model.fit(data[train], y=labels[train])

        # Test.
        predicted = model.predict(data[test])
        predicted = np.clip(predicted, a_min=0, a_max=1)
        predicted = [1 - p for p in predicted]
        score = roc_auc_score(labels[test], predicted)
        RESULT_PLOTS['roc_curve'](labels[test], {i: s for i, s in enumerate(np.clip(predicted, a_min=0, a_max=1))},
                                  dataset, '', 'SVM', 0, True)
        print(f'{dataset}: {score:.6f},')
    return


@cli.command()
@click.option('--dataset', type=click.Choice(DATASETS.keys()))
@click.option('--metric', type=click.Choice(METRICS.keys()))
@click.option('--max-depth', type=int, default=50)
@click.option('--min-points', type=int, default=1)
@click.option('--graph-ratio', type=int, default=100)
def build(dataset, metric, max_depth, min_points, graph_ratio):
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(module)s.%(funcName)s:%(message)s",
        force=True,
    )
    datasets = [dataset] if dataset else DATASETS.keys()
    metrics = [metric] if metric else METRICS.keys()

    for dataset in datasets:
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)

        for metric in metrics:
            logging.info('; '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}'
            ]))
            manifold = Manifold(data, METRICS[metric])

            min_points = max(10, min_points) if data.shape[0] > 10_000 else 3
            filepath = _manifold_path(dataset, metric, min_points, graph_ratio)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as fp:
                    logging.info(f'loading manifold {filepath}')
                    manifold = manifold.load(fp, data)
                manifold.build_tree(
                    criterion.MaxDepth(max_depth),
                    criterion.MinPoints(min_points),
                )
                manifold.build_graphs()
                with open(filepath, 'wb') as fp:
                    logging.info(f'dumping manifold {filepath}')
                    manifold.dump(fp)
            else:
                manifold.build(
                    criterion.MaxDepth(max_depth),
                    criterion.MinPoints(min_points),
                )
                with open(filepath, 'wb') as fp:
                    logging.info(f'dumping manifold {filepath}')
                    manifold.dump(fp)
    return


@cli.command()
@click.option('--plot', type=click.Choice(RESULT_PLOTS.keys()))
@click.option('--method', type=click.Choice(METHODS.keys()))
@click.option('--dataset', type=str, default='*')
@click.option('--metric', type=str, default='*')
@click.option('--starting-depth', type=int, default=0)
@click.option('--min-points', type=str, default='*')
@click.option('--graph-ratio', type=str, default='*')
def plot_results(plot, method, dataset, metric, starting_depth, min_points, graph_ratio):
    methods = [method] if method else METHODS.keys()
    plots = [plot] if plot else RESULT_PLOTS.keys()
    for manifold in glob(_manifold_path(dataset, metric, min_points, graph_ratio)):
        meta = _meta_from_path(manifold)
        dataset, metric = str(meta['dataset']), meta['metric']
        if dataset not in DATASETS:
            continue
        log_file = os.path.join(PLOT_DIR, dataset)
        os.makedirs(log_file, exist_ok=True)
        log_file = os.path.join(log_file, 'roc_scores.log')
        # noinspection PyArgumentList
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(name)s:%(module)s.%(funcName)s:%(message)s",
            force=True,
        )
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        with open(manifold, 'rb') as fp:
            logging.info(f'loading manifold {manifold}')
            manifold = Manifold.load(fp, data)
        for method in methods:
            for depth in range(starting_depth, manifold.depth + 1):
                if method in {'n_points_in_ball', 'k_nearest'} and depth < manifold.depth:
                    continue
                for plot in plots:
                    auc = RESULT_PLOTS[plot](
                        labels,
                        METHODS[method](manifold.graphs[depth]),
                        dataset,
                        metric,
                        method,
                        depth,
                        save=True
                    )
                    logging.info(f'{dataset}, {metric}, {depth}/{manifold.depth}, {method}, {plot}:-:{auc:.6f}')
    return


@cli.command()
@click.option('--dataset', type=click.Choice(DATASETS.keys()))
@click.option('--metric', type=click.Choice(METRICS.keys()))
@click.option('--neighbors', type=int, default=8)
@click.option('--components', type=click.Choice([2, 3]), default=3)
def plot_data(dataset, metric, neighbors, components):
    datasets = [dataset] if dataset else DATASETS.keys()
    metrics = [metric] if metric else METRICS.keys()
    for dataset in datasets:
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        for metric in metrics:
            logging.info('; '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
            ]))
            filename = f'{UMAP_DIR}/{dataset}/'
            if not os.path.exists(filename):
                os.makedirs(filename)
                os.makedirs(filename + 'umaps/')
            if data.shape[1] <= components:
                embedding = data
            else:
                suffix = f'umap{components}d-{neighbors}-{metric}.pickle'
                embedding = embed_umap(data, neighbors, components, metric, filename + 'umaps/' + suffix)
            title = f'{dataset}-{metric}-{neighbors}_{components}'
            if components == 3:
                # folder = f'../data/{dataset}/frames/{metric}-'
                # plot_3d(embedding, labels, title, folder)

                pass
            elif components == 2:
                suffix = f'umap{components}d-{neighbors}-{metric}.png'
                plot_2d(embedding, labels, title, filename + suffix)
    return


@cli.command()
@click.option('--dataset', type=str, default='*')
@click.option('--metric', type=str, default='*')
def animate(dataset, metric):
    # TODO
    raise NotImplementedError
    run([
        'ffmpeg',
        '-framerate', '30',
        '-i', os.path.join(dataset, 'frames', f'{metric}-%03d.png'),
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        os.path.join(dataset, f'{metric}-30fps.mp4'),
    ])


if __name__ == "__main__":
    os.makedirs(BUILD_DIR, exist_ok=True)
    cli(prog_name='src')
