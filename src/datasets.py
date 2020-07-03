import os
from io import TextIOWrapper
from subprocess import run
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import scipy.io
from scipy.io.matlab.miobase import MatReadError

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

METRICS = {
    'cosine': 'cosine',
    'euclidean': 'euclidean',
    'manhattan': 'cityblock',
}
DATASETS: Dict = {
    'lympho': 'https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=0',
    'wbc': 'https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=0',
    'glass': 'https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=0',
    'vowels': 'https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=0',
    'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=0',
    'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=0',
    'musk': 'https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=0',
    'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=0',
    'letter': 'https://www.dropbox.com/s/rt9i95h9jywrtiy/letter.mat?dl=0',
    'speech': 'https://www.dropbox.com/s/w6xv51ctea6uauc/speech.mat?dl=0',
    'pima': 'https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=0',
    'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=0',
    'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=0',
    'breastw': 'https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=0',
    'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=0',
    'ionosphere': 'https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=0',
    'mnist': 'https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=0',
    'optdigits': 'https://www.dropbox.com/s/w52ndgz5k75s514/optdigits.mat?dl=0',
    'http': 'https://www.dropbox.com/s/iy9ucsifal754tp/http.mat?dl=0',
    'cover': 'https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=0',
    # 'mulcross': '',
    'smtp': 'https://www.dropbox.com/s/dbv2u4830xri7og/smtp.mat?dl=0',
    'mammography': 'https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=0',
    'annthyroid': 'https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=0',
    'pendigits': 'https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=0',
    # 'ecoli': '',
    'wine': 'https://www.dropbox.com/s/uvjaudt2uto7zal/wine.mat?dl=0',
    'vertebral': 'https://www.dropbox.com/s/5kuqb387sgvwmrb/vertebral.mat?dl=0',
    # 'yeast': '',
    # 'seismic': '',
    # 'heart': '',
    # 'p53mutants': 'https://archive.ics.uci.edu/ml/machine-learning-databases/p53/p53_new_2012.zip',
    # 'santander': 'None',
}


def min_max_normalization(data):
    for i in range(data.shape[1]):
        min_x, max_x = np.percentile(a=data[:, i], q=[5, 95])
        if min_x == max_x:
            data[:, i] = 0.5
        else:
            data[:, i] = (data[:, i] - min_x) / (max_x - min_x)
    return data


def get(dataset: str) -> None:
    """ Download a dataset if it does not exist. """
    if dataset not in DATASETS:
        raise ValueError('dataset given is not in datasets')
    
    filename = os.path.join(DATA_DIR, f'{dataset}.mat')
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        run(['wget', DATASETS[dataset], '-O', filename])

    if dataset == 'p53mutants':
        _get_mutants(filename)
    if dataset == 'santander':
        _get_santander(filename)
    return


def read(dataset: str, normalize: bool = True, subsample: int = None):
    filename = os.path.join(DATA_DIR, f'{dataset}.mat')
    if not os.path.exists(filename):
        get(dataset)
        if not os.path.exists(filename):
            raise ValueError(f'dataset does not exist: {dataset}')

    data_dict: Dict = {}
    try:
        data_dict = scipy.io.loadmat(filename)
    except (NotImplementedError, MatReadError):
        import h5py
        with h5py.File(filename, 'r') as fp:
            for k, v in fp.items():
                if k in ['X', 'y']:
                    data_dict[k] = np.asarray(v, dtype=np.float64).T

    data = np.asarray(data_dict['X'], dtype=np.float64)
    labels = np.asarray(data_dict['y'], dtype=np.int8)

    if subsample and subsample < data.shape[0]:
        np.random.seed(42)
        negatives: List[int] = list(map(int, np.argwhere(labels < 0.9).flatten()))

        samples: List[int] = list(map(int, np.argwhere(labels > 0.9).flatten()))
        samples.extend(np.random.choice(negatives, subsample - len(samples), replace=False))
        data = np.asarray([data[s] for s in samples], dtype=np.float64)
        labels = np.asarray([labels[s] for s in samples], dtype=np.int8)

    if normalize is True:
        data = min_max_normalization(data)

    return data, np.squeeze(labels)


def _save_as_mat(filename, data, labels):
    # replace nans with column mean
    col_mean = np.nanmean(data, axis=0)
    indexes = np.where(np.isnan(data))
    data[indexes] = np.take(col_mean, indexes[1])

    data_dict = {'X': data, 'y': labels}
    scipy.io.savemat(filename, data_dict)
    return


def _get_mutants(filename):
    try:
        scipy.io.loadmat(filename)
    except ValueError:
        pass
    else:
        return

    shape = (31_420, 5_408)
    data, labels = np.zeros(shape=shape, dtype=np.float64), np.zeros(shape=shape[0], dtype=np.int8)
    with ZipFile(filename) as zp:
        with zp.open(os.path.join('Data Sets', 'K9.data')) as fp:
            for i, line in enumerate(TextIOWrapper(fp)):
                line = line.split(',')
                label = np.int8(line[-2] == 'active')
                datum = np.asarray([float(s) if s != '?' else np.nan for s in line[:-2]], dtype=np.float64)
                data[i, :], labels[i] = datum, label
    os.remove(filename)
    return _save_as_mat(filename, data, labels)


def _get_santander(filename):
    try:
        scipy.io.loadmat(filename)
    except (ValueError, MatReadError):
        pass
    else:
        return

    shape = (200_000, 200)
    data, labels = np.zeros(shape=shape, dtype=np.float64), np.zeros(shape=shape[0], dtype=np.int8)

    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines[1:]):
            line = line.split(',')
            labels[i] = int(line[1])
            data[i] = np.asarray([float(v) for v in line[2:]], dtype=np.float64)

    return _save_as_mat(filename, data, labels)
