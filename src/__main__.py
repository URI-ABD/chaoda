import logging
import os
import random
from glob import glob

import click
import numpy as np
from pyclam import criterion
from pyclam.manifold import Manifold

from .datasets import DATASETS, METRICS, get, read
from .methods import METHODS
from .plot import RESULT_PLOTS, embed_umap, plot_2d, PLOT_DIR

np.random.seed(42)
random.seed(42)

NORMALIZE = False
SUB_SAMPLE = 10_000

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
UMAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots', 'umaps'))


def _manifold_path(dataset, metric, min_points) -> str:
    """ Generate proper path to manifold. """
    return os.path.join(
        BUILD_DIR,
        ':'.join(map(str, [dataset, metric, f'{min_points}.pickle']))
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
    """ State is passed between chained commands. """

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
@click.option('--dataset', type=click.Choice(DATASETS.keys()))
@click.option('--metric', type=click.Choice(METRICS.keys()))
@click.option('--max-depth', type=int, default=50)
@click.option('--min-points', type=int, default=0)
def build(dataset, metric, max_depth, min_points):
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

            min_points = min_points if min_points else min(1 + data.shape[0] // 1000, 25)
            filepath = _manifold_path(dataset, metric, min_points)
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
@click.option('--dataset', type=str, default='*')
@click.option('--metric', type=str, default='*')
@click.option('--min-points', type=str, default='*')
def inspect(dataset, metric, min_points):
    percentiles = [0, 40, 50, 60, 100]
    for manifold in glob(_manifold_path(dataset, metric, min_points)):
        meta = _meta_from_path(manifold)
        dataset, metric = str(meta['dataset']), meta['metric']
        if dataset not in DATASETS or metric not in METRICS:
            continue
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        with open(manifold, 'rb') as fp:
            logging.info(f'loading manifold {manifold}')
            manifold = Manifold.load(fp, data)
        lfd_range = [], []
        for graph in manifold.graphs:
            clusters = [cluster.local_fractal_dimension
                        for cluster in graph.clusters
                        if cluster.cardinality > 2]
            if len(clusters) > 0:
                lfds = np.percentile(
                    a=clusters,
                    q=percentiles,
                )
                lfd_range[0].append(lfds[1]), lfd_range[1].append(lfds[3])
                lfds = [f'{d:.3f}' for d in lfds]
                print(f'depth: {graph.depth:2d}, '
                      f'clusters: {graph.cardinality:5d}, '
                      f'non-singletons: {len(clusters):5d}, '
                      f'lfds:  {"  ".join(lfds)}')
        lfd_range = float(np.median(lfd_range[0])), float(np.median(lfd_range[1]))
        print(f'medians: {lfd_range[0]:.3f}, {lfd_range[1]:.3f}')
    return


@cli.command()
@click.option('--plot', type=click.Choice(RESULT_PLOTS.keys()))
@click.option('--method', type=click.Choice(METHODS.keys()))
@click.option('--dataset', type=str, default='*')
@click.option('--metric', type=str, default='*')
@click.option('--starting-depth', type=int, default=0)
@click.option('--min-points', type=str, default='*')
def plot_results(plot, method, dataset, metric, starting_depth, min_points):
    methods = [method] if method else METHODS.keys()
    plots = [plot] if plot else RESULT_PLOTS.keys()
    for manifold in glob(_manifold_path(dataset, metric, min_points)):
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
                if method in {'n_points_in_ball', 'k_nearest', 'random_walk'} and depth < manifold.depth:
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
@click.option('--components', type=click.Choice([2, 3]), default=2)
def plot_data(dataset, metric, neighbors, components):
    # TODO: Fix
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
            filename = f'{UMAP_DIR}/'
            os.makedirs(filename, exist_ok=True)

            if data.shape[1] <= components:
                embedding = data
            else:
                suffix = f'{dataset}-{metric}-umap{components}d.pickle'
                embedding = embed_umap(data, neighbors, components, metric, filename + suffix)
            title = f'{dataset}-{metric}'
            if components == 3:
                # folder = f'data/{dataset}/frames/{metric}-'
                # plot_3d(embedding, labels, title, folder)

                pass
            elif components == 2:
                suffix = f'{dataset}-{metric}-umap{components}d.png'
                plot_2d(embedding, labels, title, filename + suffix)
    return


# noinspection PyUnreachableCode,PyUnusedLocal
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
    return


if __name__ == "__main__":
    os.makedirs(BUILD_DIR, exist_ok=True)
    cli(prog_name='src')
