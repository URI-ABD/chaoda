import subprocess
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import h5py
import numpy
from pyclam.utils import normalize
from scipy.io import loadmat
from scipy.io.matlab.miobase import MatReadError

from utils import DATA_DIR

__all__ = [
    'DATASET_LINKS',
    'DATASET_NAMES',
    'SHORT_NAMES',
    'read',
]

DATASET_LINKS: Dict[str, str] = {
    'annthyroid': 'https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=0',
    'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=0',
    'breastw': 'https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=0',
    'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=0',
    'cover': 'https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=0',
    'glass': 'https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=0',
    'http': 'https://www.dropbox.com/s/iy9ucsifal754tp/http.mat?dl=0',
    'ionosphere': 'https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=0',
    'lympho': 'https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=0',
    'mammography': 'https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=0',
    'mnist': 'https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=0',
    'musk': 'https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=0',
    'optdigits': 'https://www.dropbox.com/s/w52ndgz5k75s514/optdigits.mat?dl=0',
    'pendigits': 'https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=0',
    'pima': 'https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=0',
    'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=0',
    'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=0',
    'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=0',
    'smtp': 'https://www.dropbox.com/s/dbv2u4830xri7og/smtp.mat?dl=0',
    'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=0',
    'vertebral': 'https://www.dropbox.com/s/5kuqb387sgvwmrb/vertebral.mat?dl=0',
    'vowels': 'https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=0',
    'wbc': 'https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=0',
    'wine': 'https://www.dropbox.com/s/uvjaudt2uto7zal/wine.mat?dl=0',
}

SHORT_NAMES = {
    'annthyroid': 'ANNTH.',
    'arrhythmia': 'ARRH.',
    'breastw': 'BRE.W',
    'cardio': 'CARD.',
    'cover': 'COVER',
    'glass': 'GLASS',
    'http': 'HTTP',
    'ionosphere': 'IONO.',
    'lympho': 'LYMP.',
    'mammography': 'MAMMO.',
    'mnist': 'MNIST',
    'musk': 'MUSK',
    'optdigits': 'O.DIG.',
    'pendigits': 'P.DIG.',
    'pima': 'PIMA',
    'satellite': 'SATL.',
    'satimage-2': 'SAT.I-2',
    'shuttle': 'SHUTTLE',
    'smtp': 'SMTP.',
    'thyroid': 'THYR.',
    'vertebral': 'VERT.',
    'vowels': 'VOWELS',
    'wbc': 'WBC',
    'wine': 'WINE',
}

DATASET_NAMES = list(DATASET_LINKS.keys())


def get(dataset: str):
    """ Download the dataset if needed, and returns the filename used to store it. """
    link = DATASET_LINKS[dataset]
    data_path = DATA_DIR.joinpath(f'{dataset}.npy')
    labels_path = DATA_DIR.joinpath(f'{dataset}_labels.npy')

    if not DATA_DIR.exists():
        DATA_DIR.mkdir()

    filename = DATA_DIR.joinpath(f'{dataset}.mat')
    if not filename.exists():
        subprocess.run(['wget', link, '-O', filename])

    if not filename.exists():
        raise ValueError(f'Could not get dataset {dataset}.')

    data_dict: Dict = dict()
    try:
        data_dict = loadmat(filename)
    except (NotImplementedError, MatReadError):
        # noinspection PyUnresolvedReferences
        with h5py.File(filename, 'r') as fp:
            for k, v in fp.items():
                if k in ['X', 'y']:
                    data_dict[k] = numpy.asarray(v, dtype=float).T

    numpy.save(data_path, numpy.asarray(data_dict['X'], dtype=numpy.float32))
    numpy.save(labels_path, numpy.asarray(data_dict['y'], dtype=numpy.uint8))
    return


def read(
        dataset: str,
        normalization_mode: Optional[str] = None,
        subsample: Optional[int] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """ Read and preparse the dataset.
    Returns the data and the labels.
    In the data, rows are instances, and columns are attributes.
    """
    data_path = DATA_DIR.joinpath(f'{dataset}.npy')
    labels_path = DATA_DIR.joinpath(f'{dataset}_labels.npy')

    if not data_path.exists():
        get(dataset)

    data: numpy.array = numpy.load(data_path)
    labels: numpy.array = numpy.load(labels_path)

    if subsample is not None and subsample < data.shape[0]:
        outliers: List[int] = [i for i, j in enumerate(labels) if j == 1]
        inliers: List[int] = [i for i, j in enumerate(labels) if j == 0]

        samples: List[int] = list(numpy.random.choice(outliers, int(subsample * (len(outliers) / data.shape[0])), replace=False))
        samples.extend(list(numpy.random.choice(inliers, 1 + int(subsample * (len(inliers) / data.shape[0])), replace=False)))

        data = numpy.asarray(data[samples], dtype=float)
        labels = numpy.asarray(labels[samples], dtype=int)

    if normalization_mode is not None:
        data = normalize(data, normalization_mode)

    return numpy.asarray(data, dtype=numpy.float32), numpy.asarray(numpy.squeeze(labels), dtype=numpy.uint8)
