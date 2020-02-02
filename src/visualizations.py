import os
import pickle
from subprocess import run
from typing import Dict, List

import numpy as np
import umap
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

DATASETS: Dict = {
    'mnist': 'https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=0',
    'cover': 'https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=0',
    'letter': 'https://www.dropbox.com/s/rt9i95h9jywrtiy/letter.mat?dl=0',
    'http': 'https://www.dropbox.com/s/iy9ucsifal754tp/http.mat?dl=0',
    'smtp': 'https://www.dropbox.com/s/dbv2u4830xri7og/smtp.mat?dl=0',
    'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=0',
    'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=0',
    'mammography': 'https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=0',
    'annthyroid': 'https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=0',
    'breastw': 'https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=0',
    'vowels': 'https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=0',
    'musk': 'https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=0',
    'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=0',
    'wine': 'https://www.dropbox.com/s/uvjaudt2uto7zal/wine.mat?dl=0',
    'pendigits': 'https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=0',
    'optdigits': 'https://www.dropbox.com/s/w52ndgz5k75s514/optdigits.mat?dl=0',
    'p53mutants': 'https://archive.ics.uci.edu/ml/machine-learning-databases/p53/p53_new_2012.zip',
    'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=0',
    'ionosphere': 'https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=0',
    'lympho': 'https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=0',
    'wbc': 'https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=0',
    'glass': 'https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=0',
    'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=0',
    'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=0',
    'speech': 'https://www.dropbox.com/s/w6xv51ctea6uauc/speech.mat?dl=0',
}


def min_max_normalization(data):
    for i in range(data.shape[1]):
        min_x, max_x = np.min(data[:, i]), np.max(data[:, i])
        if min_x == max_x:
            data[:, i] = 0.5
        else:
            data[:, i] = (data[:, i] - min_x) / (max_x - min_x)
    return data


def read_mutants(normalize: bool = False):
    filename = '../data/p53mutants/K9.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as infile:
            data, labels = pickle.load(infile)
    else:
        shape = (31_420, 5_408)
        data, labels = np.zeros(shape=shape, dtype=np.float64), np.zeros(shape=shape[0], dtype=np.int8)
        with open('../data/p53mutants/K9.data', 'r') as infile:
            for i, line in enumerate(infile):
                line = line.split(',')
                label = np.int8(line[-2] == 'active')
                datum = np.asarray([float(s) if s != '?' else np.nan for s in line[:-2]], dtype=np.float64)
                data[i, :], labels[i] = datum, label

        # replace nans with column mean
        col_mean = np.nanmean(data, axis=0)
        indexes = np.where(np.isnan(data))
        data[indexes] = np.take(col_mean, indexes[1])

        with open(filename, 'wb') as outfile:
            pickle.dump((data, labels), outfile)

    if normalize is True:
        data = min_max_normalization(data)

    return data, labels


def read_data(dataset: str, normalize: bool = False):
    if dataset == 'p53mutants':
        return read_mutants(normalize=normalize)

    filename = f'../data/{dataset}/{dataset}.mat'
    data_dict: Dict = {}
    try:
        import scipy.io
        data_dict = scipy.io.loadmat(filename)
    except NotImplementedError:
        import h5py
        with h5py.File(filename, 'r') as infile:
            for k, v in infile.items():
                if k in ['X', 'y']:
                    data_dict[k] = np.asarray(v, dtype=np.float64).T

    data = np.asarray(data_dict['X'], dtype=np.float64)
    labels = np.asarray(data_dict['y'], dtype=np.int8)

    if data.shape[0] > 100_000:
        samples = sorted(list(np.random.choice(data.shape[0], 100_000, replace=False)))
        data = np.asarray([data[s] for s in samples], dtype=np.float64)
        labels = np.asarray([labels[s] for s in samples], dtype=np.int8)

    if normalize is True:
        data = min_max_normalization(data)

    return data, labels


def make_umap(
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


def make_dirs(datasets: List[str]):
    if not os.path.exists('../data'):
        os.mkdir('../data')

    for dataset in datasets:
        if dataset == 'p53mutants':
            continue
        if not os.path.exists(f'../data/{dataset}'):
            os.mkdir(f'../data/{dataset}')
        for folder in ['umap', 'frames']:
            if not os.path.exists(f'../data/{dataset}/{folder}'):
                os.mkdir(f'../data/{dataset}/{folder}')

        filename = f'../data/{dataset}/{dataset}.mat'
        if not os.path.exists(filename):
            run(['wget', DATASETS[dataset], '-O', filename])
    return


def main():
    datasets = list(DATASETS.keys())
    make_dirs(datasets)

    metrics = [
        'euclidean',
        # 'manhattan',
        # 'cosine',
    ]

    for dataset in list(DATASETS.keys()):
        if dataset not in ['p53mutants']:
            continue
        normalize = dataset not in ['mnist']
        data, labels = read_data(dataset, normalize)
        print(f'data_shape: {data.shape}, num_outliers: {len([l for l in labels if l == 1])}')
        # data, labels = data[: 10_000, :], labels[: 10_000]
        for metric in metrics:
            for n_neighbors in [32]:
                for n_components in [3]:
                    filename = f'../data/{dataset}/umap/{n_neighbors}-{n_components}d-{metric}.pickle'
                    if data.shape[1] > n_components:
                        embedding = data
                        # embedding = make_umap(data, n_neighbors, n_components, metric, filename)
                    else:
                        embedding = data
                    title = f'{dataset}-{metric}-{n_neighbors}'
                    if n_components == 3:
                        # folder = f'../data/{dataset}/frames/{metric}-'
                        # plot_3d(embedding, labels, title, folder)

                        # run([
                        #     'ffmpeg',
                        #     '-framerate', '30',
                        #     '-i', f'{dataset}/frames/{metric}-%03d.png',
                        #     '-c:v', 'libx264',
                        #     '-profile:v', 'high',
                        #     '-crf', '20',
                        #     '-pix_fmt', 'yuv420p',
                        #     f'{dataset}/{metric}-30fps.mp4'
                        # ])

                        pass
                    if n_components == 2:
                        # plot_2d(embedding, labels, title)
                        pass

    return


if __name__ == '__main__':
    main()
