from pathlib import Path
from typing import List

import numpy
import umap
from matplotlib import pyplot
from sklearn import metrics

import paths


def _directory(plot: str, metric: str, method: str) -> Path:
    dir_path = paths.PLOTS_DIR.joinpath(plot).joinpath(method).joinpath(metric)
    dir_path.mkdir(exist_ok=True)
    return dir_path


def histogram(
        x: numpy.ndarray,
        dataset: str,
        metric: str,
        method: str,
):
    """ Plot a histogram of the given outlier scores.

    :param x: 1-d array of anomaly scores to plot in a histogram.
    :param dataset: The dataset which was scored.
    :param metric: The metric used with the dataset.
    :param method: The method used for generating the scores.
    """
    fig = pyplot.figure(figsize=(16, 10), dpi=200)
    fig.add_subplots(111)
    pyplot.hist(x=x, color='#0504aa', rwidth=0.85)
    pyplot.xlabel('Outlier Score')
    pyplot.ylabel('Counts')
    plot_path = _directory('histograms', metric, method).joinpath(f'{dataset}.png')
    pyplot.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    pyplot.close(fig)
    return


def roc_curve(
        y_true: numpy.ndarray,
        y_pred: numpy.ndarray,
        dataset: str,
        metric: str,
        method: str,
):
    """ Plot the Roc-Curve, given the labels and predictions.

    :param y_true: True Labels, 1-d array.
    :param y_pred: Predicted scores, 1-d array.
    :param dataset: The dataset which was scored.
    :param metric: The metric used with the dataset.
    :param method: The method used for generating the scores.
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    fig = pyplot.figure(figsize=(16, 10), dpi=200)
    fig.add_subplots(111)
    pyplot.plot(fpr, tpr, color='darkorange', lw=2., label=f'area: {auc:.6f}')
    pyplot.plot([0, 1], [0, 1], color='navy', lw=2., linestyle='--')
    pyplot.xlim([0, 1.05]), pyplot.ylim([0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title(f'{dataset}-{metric}-{method}')
    pyplot.legend(loc='lower right')
    plot_path = _directory('roc_curves', metric, method).joinpath(f'{dataset}.png')
    pyplot.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    pyplot.close(fig)
    return


def scatter_2d(
        data: numpy.array,
        labels: List[int],
        plot_path: Path,
):
    """ Make a 2d scatter plot of the data

    :param data: Numpy array with shape (2, n) with rows representing the x and y axes respectively.
    :param labels: A list of labels for each point.
    :param plot_path: The path where to save the plot.
    """
    if data.shape[0] != 2:
        raise ValueError(f'expected an array of shape (2, n). Got {data.shape} instead.')

    fig = pyplot.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplots(111)
    pyplot.scatter(data[0, :], data[1, :], c=labels, cmap='Dark2', s=5.)
    ax.set_xlim([numpy.min(data[0, :]), numpy.max(data[0, :])])
    ax.set_ylim([numpy.min(data[1, :]), numpy.max(data[1, :])])
    pyplot.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    pyplot.close(fig)
    return


def scatter_3d(
        data: numpy.array,
        labels: List[str],
        plot_path: Path,
):
    """ Make a 3d scatter plot of the data

    :param data: Numpy array with shape (3, n) with rows representing the x and y axes respectively.
    :param labels: A list of labels for each point.
    :param plot_path: The path where to save the plot.
    """
    if data.shape[0] != 3:
        raise ValueError(f'expected an array of shape (2, n). Got {data.shape} instead.')

    fig = pyplot.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplots(111)
    pyplot.scatter(data[0, :], data[1, :], data[2, :], c=labels, cmap='Dark2', s=5.)
    ax.set_xlim([numpy.min(data[0, :]), numpy.max(data[0, :])])
    ax.set_ylim([numpy.min(data[1, :]), numpy.max(data[1, :])])
    ax.set_zlim([numpy.min(data[2, :]), numpy.max(data[2, :])])
    pyplot.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    pyplot.close(fig)
    return


def embed_umap(
        data: numpy.array,
        n_components: int,
        n_neighbors: int,
        dataset: str,
        metric: str,
):
    """ Create and save a UMAP embedding of the data.

    :param data: raw data to be embedded.
    :param n_components: will create an n-dimensional embedding.
    :param n_neighbors: number of neighbors for umap reduction. Low numbers emphasize local structure.
    :param dataset: name of the dataset being reduced.
    :param metric: metric to use for the reduction.
    """
    umap_path = paths.UMAPS_DIR.joinpath(f'{dataset}_{metric}.npy')
    if not umap_path.exists():
        embedding: numpy.ndarray = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
        ).fit_transform(data)

        paths.UMAPS_DIR.mkdir(exist_ok=True)
        numpy.save(
            file=umap_path,
            arr=embedding,
            allow_pickle=True,
            fix_imports=True,
        )
    else:
        embedding = numpy.load(
            file=umap_path,
            mmap_mode='r',
        )

    return embedding
