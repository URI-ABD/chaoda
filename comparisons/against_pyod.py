import random
import signal
import time
import warnings
from typing import List

import numpy
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

from utils import constants
from utils import datasets
from utils import helpers
from utils import paths


# TODO: Break out deep-learning based methods in a separate comparisons table.
# TODO: Try to add the following deep-learning based methods to comparisons
#  REPEN: Code bugs out because authors did not provide information on recreating their environment and version numbers for keras/tensorflow
#  DAGMM: Deep Autoencoder with Gaussian Mixture Model. Can't find source code from authors. Also, model is weakly supervised.
#  RDP: Random Distance Predicting. https://github.com/billhhh/RDP/
#  AE-1SVM: https://github.com/minh-nghia/AE-1SVM  This would be a lot of work to get running because
#                                                  the authors have a separate jupyter notebook for each dataset.
#  DEC: Deep Embedded Clustering. https://github.com/piiswrong/dec  Implementation uses a custom build of Caffe from berkeley...
#  APE: can't find source code


def _neurons(dataset: numpy.ndarray):
    """ This sets up default shapes for neural-network based methods
    that would crash without this as input.
    We allow for deeper networks for larger datasets.
    However, we have not investigated any other model architectures for these methods.
    """
    return [
        dataset.shape[1],
        dataset.shape[1] // 4,
        dataset.shape[1] // 8,
        dataset.shape[1] // 4,
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
    ]


PYOD_MODELS = {
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


class TimeoutException(Exception):
    pass


# noinspection PyUnusedLocal
def timeout_handler(signum, frame):
    raise TimeoutException


def run_model(model_name: str, dataset_names: List[str], max_time: int = 36_000):
    """ Runs a PyOD model on all datasets.

    Args:
        model_name: name of pyod-model
        dataset_names: list of dataset names fom ODDS.
        max_time: max number of seconds to allow the model.
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    for dataset in dataset_names:
        data, labels = datasets.read(dataset, constants.NORMALIZE, constants.SUB_SAMPLE)
        helpers.print_blurb(model_name, dataset, data.shape)

        contamination: float = 0.1  # This is the default set by the authors of pyOD.
        # This is supposed to be the fraction of points that are outliers. We feel that
        # giving this information to an algorithm constitutes a form of cheating.
        # If you are feeling generous towards our competitors, feel free to use the next line.
        # contamination: float = dict(Counter(labels))[1] / len(labels)

        signal.alarm(max_time)
        # noinspection PyBroadException
        try:
            pyod_model = PYOD_MODELS[model_name](data, contamination)

            start = time.time()

            pyod_model.fit(data)
            rankings = pyod_model.predict(data)

            time_taken = float(time.time() - start)

            score = roc_auc_score(labels, rankings)

        except TimeoutException as _:
            score, time_taken = 'TO', 'TO'

        except Exception as _:
            score, time_taken = 'EX', 'EX'

        scores_df, times_df = helpers.get_dataframes()

        scores_df.at[model_name, dataset] = score
        times_df.at[model_name, dataset] = time_taken

        scores_df.to_csv(paths.SCORES_PATH, float_format='%.2f')
        times_df.to_csv(paths.TIMES_PATH, float_format='%.2e')

    return


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    numpy.random.seed(42), random.seed(42)
    paths.RESULTS_DIR.mkdir(exist_ok=True)
    _dataset_names = datasets.DATASET_NAMES
    # _datasets = ['lympho']  # for testing

    for _name in PYOD_MODELS:
        run_model(_name, _dataset_names, 600)
