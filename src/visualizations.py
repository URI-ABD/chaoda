import os
import pickle

import umap
import numpy as np
import scipy.io

from matplotlib import pyplot as plt


def min_max_normalization(data):
    for i in range(data.shape[1]):
        min_x, max_x = np.min(data[:, i]), np.max(data[:, i])
        data[:, i] = (data[:, i] - min_x) / (max_x - min_x)
    return data


def read_data(dataset: str, normalize: bool = False):
    data_dict = scipy.io.loadmat(f'../data/{dataset}/{dataset}.mat')

    data = np.asarray(data_dict['X'], dtype=np.float64)
    labels = np.asarray(data_dict['y'], dtype=np.int8)

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
        figsize=(6, 6),
        dpi=150,
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
        figsize=(6, 6),
        dpi=150,
):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=[float(d) for d in labels], s=10. * labels + 1., cmap='Dark2')
    plt.title(title)
    for azimuth in range(0, 360):
        ax.view_init(elev=10, azim=azimuth)
        plt.savefig(folder + f'{azimuth:03d}.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

    """ from the data directory:
    ffmpeg -framerate 30 -i mnist/frames/euclidean-%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p mnist-euclidean-30fps.mp4
    """
    return


def main():
    datasets = [
        'mnist',
        'cover',
        'letter',
        # 'http',
    ]
    metrics = [
        'euclidean',
        'manhattan',
        'cosine',
    ]
    if not os.path.exists('../data'):
        os.mkdir('../data')

    for dataset in datasets:
        if not os.path.exists(f'../data/{dataset}'):
            os.mkdir(f'../data/{dataset}')
        for folder in ['umap', 'frames', 'videos']:
            if not os.path.exists(f'../data/{dataset}/{folder}'):
                os.mkdir(f'../data/{dataset}/{folder}')

        normalize = dataset not in ['mnist']
        data, labels = read_data(dataset, normalize)
        for metric in metrics:
            for n_neighbors in [32]:
                for n_components in [3]:
                    filename = f'../data/{dataset}/umap/{n_neighbors}-{n_components}d-{metric}.pickle'
                    embedding = make_umap(data, n_neighbors, n_components, metric, filename)
                    if n_components == 3:
                        folder = f'../data/{dataset}/frames/{metric}-'
                        title = f'{dataset}-{metric}-{n_neighbors}'
                        plot_3d(embedding, labels, title, folder)

    return


if __name__ == '__main__':
    main()
