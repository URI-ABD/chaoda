import random
import time
import warnings
from collections import Counter
from typing import List

from pyod.models import abod
from pyod.models import auto_encoder
from pyod.models import cblof
from pyod.models import cof
from pyod.models import hbos
from pyod.models import iforest
from pyod.models import knn
from pyod.models import lmdd
from pyod.models import loci
from pyod.models import loda
from pyod.models import lof
from pyod.models import mcd
from pyod.models import mo_gaal
from pyod.models import ocsvm
from pyod.models import so_gaal
from pyod.models import sod
from pyod.models import sos
from pyod.models import vae
from sklearn.metrics import roc_auc_score

import datasets as chaoda_datasets
from utils import *

# TODO: Break out deep-learning based methods in a separate comparisons table.
# TODO: Add the following deep-learning based methods to comparisons


def _neurons(dataset):
    """ This sets up default shapes for neural-network based methods
    that would crash without this as input.
    We allow for deeper networks for larger datasets.
    However, we have not investigated any other model architectures for these methods.
    """
    return [
        dataset.shape[1],
        dataset.shape[1] // 2,
        dataset.shape[1] // 4,
        dataset.shape[1] // 8,
        dataset.shape[1] // 4,
        dataset.shape[1] // 2,
        dataset.shape[1],
    ] if dataset.shape[1] > 32 else [
        dataset.shape[1],
        dataset.shape[1] // 2,
        dataset.shape[1] // 4,
        dataset.shape[1] // 2,
        dataset.shape[1],
    ] if dataset.shape[1] > 8 else [
        dataset.shape[1],
        dataset.shape[1] // 2,
        dataset.shape[1],
    ] if dataset.shape[1] > 4 else [
        dataset.shape[1],
        dataset.shape[1],
    ]


MODELS = {
    'ABOD': lambda _, c: abod.ABOD(contamination=c),
    'AutoEncoder': lambda d, c: auto_encoder.AutoEncoder(contamination=c, hidden_neurons=_neurons(d)),
    'CBLOF': lambda _, c: cblof.CBLOF(contamination=c),
    'COF': lambda _, c: cof.COF(contamination=c),
    'HBOS': lambda _, c: hbos.HBOS(contamination=c),
    'IFOREST': lambda _, c: iforest.IForest(contamination=c),
    'KNN': lambda _, c: knn.KNN(contamination=c),
    'LMDD': lambda _, c: lmdd.LMDD(contamination=c),
    'LOCI': lambda _, c: loci.LOCI(contamination=c),
    'LODA': lambda _, c: loda.LODA(contamination=c),
    'LOF': lambda _, c: lof.LOF(contamination=c),
    'MCD': lambda _, c: mcd.MCD(contamination=c),
    'MOGAAL': lambda _, c: mo_gaal.MO_GAAL(contamination=c),
    'OCSVM': lambda _, c: ocsvm.OCSVM(contamination=c),
    'SOD': lambda _, c: sod.SOD(contamination=c),
    'SOGAAL': lambda _, c: so_gaal.SO_GAAL(contamination=c),
    'SOS': lambda _, c: sos.SOS(contamination=c),
    'VAE': lambda d, c: vae.VAE(contamination=c, encoder_neurons=_neurons(d), decoder_neurons=reversed(_neurons(d))),
}


@timeout(36_000)  # 10 hours
def train_model(model, data):
    start = time.time()
    model.fit(data)
    predictions = model.predict(data)
    time_taken = float(time.time() - start)
    return predictions, time_taken


# noinspection PyBroadException
def run_model(model_name: str, datasets: List[str], scores_path: str, times_path: str):
    scores: List[str] = list()
    times: List[str] = list()
    for dataset in datasets:
        data, labels = chaoda_datasets.read(dataset, NORMALIZE, SUB_SAMPLE)
        print_blurb(model_name, dataset, data.shape)

        contamination: float = 0.1  # this is the default set by the authors of pyOD.
        # this is supposed to be the points that are outliers. We feel that
        # giving this information to an algorithm constitutes a form of cheating.
        # if you are feeling generous towards our competitors, feel free to
        # use the next line.
        # contamination: float = dict(Counter(labels))[1] / len(labels)
        try:
            rankings, time_taken = train_model(MODELS[model_name](data, contamination), data)
            score, time_taken = f'{roc_auc_score(labels, rankings):.3f}', f'{time_taken:.2f}'
        except TimeoutError:
            score, time_taken = 'TO', 'TO'
        except Exception as _:
            score, time_taken = 'EX', 'EX'

        scores.append(score), times.append(time_taken)

    scores: str = ','.join(scores)
    with open(scores_path, 'a') as fp:
        fp.write(f'{model_name},{scores}\n')

    times: str = ','.join(times)
    with open(times_path, 'a') as fp:
        fp.write(f'{model_name},{times}\n')
    return


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(42), random.seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _datasets = list(sorted(list(chaoda_datasets.DATASET_LINKS.keys())))
    # _datasets = ['vertebral']  # for testing

    for _filename in [PYOD_SCORES_PATH, PYOD_TIMES_PATH]:
        with open(_filename, 'w') as _fp:
            _header = ','.join(_datasets)
            _fp.write(f'model,{_header}\n')

    for _name in MODELS:
        run_model(_name, _datasets, PYOD_SCORES_PATH, PYOD_TIMES_PATH)
