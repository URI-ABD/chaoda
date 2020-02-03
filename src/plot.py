import os
import pickle
from subprocess import run
from typing import Dict, List
from sklearn import metrics

import numpy as np
import umap
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D


PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))


def _directory(dataset, metric, method):
    return os.path.join(PLOT_DIR, dataset, metric, method)


def histogram(
        x: List[float],
        dataset: str,
        metric: str,
        method: str,
        depth: int,
        save: bool,
):
    plt.clf()
    fig = plt.figure()
    n, bins, patches = plt.hist(x=x, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Anomalousness')
    plt.ylabel('Counts')
    max_freq = n.max()
    plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    if save is True:
        filepath = f'../data/{dataset}/plots/{metric}/{method}/{depth}-histogram.png'
        make_folders(dataset, metric, method)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def roc_curve(true_labels, anomalies, dataset, metric, method, depth, save):
    y_true, y_score = [], []
    [(y_true.append(true_labels[k]), y_score.append(v)) for k, v in anomalies.items()]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)

    plt.clf()
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.6f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset}-{metric}-{method}-{depth}')
    plt.legend(loc="lower right")

    directory = _directory(dataset, metric, method)
    os.makedirs(directory, exist_ok=True)
    if save is True:
        filepath = os.path.join(directory, f'{depth}-roc.png')
        fig.savefig(filepath)
    else:
        plt.show()
    
    csv_filepath = os.path.join(directory, f'roc.csv')
    if not os.path.exists(csv_filepath):
        with open(csv_filepath, 'w') as outfile:
            outfile.write('depth,scores\n')
    with open(csv_filepath, 'a') as outfile:
        line = '_'.join([f'{s:.16f}' for i, s in sorted(list(anomalies.items()))])
        outfile.write(f'{depth},{line}\n')
    plt.close('all')


def confusion_matrix(
        true_labels: List[bool],
        anomalies: Dict[int, float],
        dataset: str,
        metric: str,
        method: str,
        depth: int,
        save: bool,
):
    p = float(len([k for k in anomalies.keys() if true_labels[k] == 1]))
    n = float(len([k for k in anomalies.keys() if true_labels[k] == 0]))

    threshold = float(np.percentile(list(anomalies.values()), 95))
    values = [float(v) for v in anomalies.values()]
    tp = float(sum([v > threshold for k, v in anomalies.items() if true_labels[k] == 1]))
    tn = float(sum([v < threshold for k, v in anomalies.items() if true_labels[k] == 0]))

    tpr, tnr = tp / p, tn / n
    fpr, fnr = 1 - tnr, 1 - tpr

    confusion_matrix = [[tpr, fpr], [fnr, tnr]]

    plt.clf()
    fig = plt.figure()
    # noinspection PyUnresolvedReferences
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Wistia)
    class_names = ['Normal', 'Anomaly']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    s = [['TPR', 'FPR'], ['FNR', 'TNR']]
    [plt.text(j, i, f'{s[i][j]} = {confusion_matrix[i][j]:.3f}', position=(-0.2 + j, 0.03 + i))
     for i in range(2)
     for j in range(2)]

    directory = _directory(dataset, metric, method)
    os.makedirs(directory, exist_ok=True)
    if save is True:
        filepath = os.path.join(directory, f'{depth}-confusion_matrix.png')
        fig.savefig(filepath)
    else:
        plt.show()
    plt.close('all')


def scatter(data: np.ndarray, labels: List[int], name: str):
    plt.clf()
    fig = plt.figure(figsize=(6, 6), dpi=300)
    if data.shape[1] == 2:
        ax = fig.add_subplot(111)
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='Dark2', s=5.)
        ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    elif data.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=5., cmap='Dark2')
        ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])
        ax.set_zlim([np.min(data[:, 2]), np.max(data[:, 2])])
        ax.view_init(elev=20, azim=60)

    plt.savefig(name, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    plt.close('all')


def umap(
        data: np.ndarray,
        n_neighbors: int,
        n_components: int,
        metric: str,
        filename: str,
) -> np.ndarray:
    if os.path.exists(filename):
        with open(filename, 'rb') as infile:
            embedding = pickle.load(infile)
    else:
        embedding = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
        ).fit_transform(data)

        with open(filename, 'wb') as outfile:
            pickle.dump(embedding, outfile)

    return embedding


def plot_2d(
        data: np.ndarray,
        labels: np.ndarray,
        title: str,
        figsize=(8, 8),
        dpi=128,
):
    x, y = data[:, 0], data[:, 1]
    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=[float(d) for d in labels], s=10. * labels + 1., cmap='Dark2')
    plt.title(title)
    plt.show()
    plt.close('all')
    return


def plot_3d(
        data: np.ndarray,
        labels: np.ndarray,
        title: str,
        folder: str,
        figsize=(8, 8),
        dpi=128,
):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    plt.close('all')

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=[float(d) for d in labels], s=(10. * labels + .1), cmap='Dark2')
    plt.title(title)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)

    for azimuth in range(0, 360):
        ax.view_init(elev=10, azim=azimuth)
        plt.savefig(folder + f'{azimuth:03d}.png', bbox_inches='tight', pad_inches=0)

    plt.close('all')
    return


RESULT_PLOTS = {
    'roc_curve': roc_curve,
    'confusion_matrix': confusion_matrix,
}

DATA_PLOTS = {
    'histogram': histogram,
    'scatter': scatter,
    'umap': umap
}