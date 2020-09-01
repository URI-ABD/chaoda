import os
import warnings
from typing import List

import numpy as np
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
from pyod.models import lscp
from pyod.models import mcd
from pyod.models import mo_gaal
from pyod.models import ocsvm
from pyod.models import so_gaal
from pyod.models import sod
from pyod.models import sos
from pyod.models import vae
from pyod.models import xgbod
from sklearn.metrics import roc_auc_score

import src.datasets as chaoda_datasets
from src.glm_incorporation import NORMALIZE, SUB_SAMPLE
from src.utils import RESULTS_PATH, timeout

MODELS = {
    # 'ABOD': abod.ABOD, # Neuron issue
    # 'AUTOENCODER': auto_encoder.AutoEncoder, # Neuron issue
    'CBLOF': cblof.CBLOF,
    'COF': cof.COF,
    'HBOS': hbos.HBOS,
    'IFOREST': iforest.IForest,
    'KNN': knn.KNN,
    'LMDD': lmdd.LMDD,
    'LOCI': loci.LOCI,
    'LODA': loda.LODA,
    'LOF': lof.LOF,
    # 'LSCP': lscp.LSCP,  # Exception
    'MCD': mcd.MCD,
    # 'MOGAAL': mo_gaal.MO_GAAL,  # need gpu
    'OCSVM': ocsvm.OCSVM,
    'SOD': sod.SOD,  # takes too long
    # 'SOGAAL': so_gaal.SO_GAAL,  # need gpu
    'SOS': sos.SOS,
    # 'VAE': vae.VAE, # Disabled due to: "ValueError: The number of neurons should not exceed the number of features"
    # 'XGBOD': xgbod.XGBOD,  # Exception
}


@timeout(600)
def train_model(model, data):
    model.fit(data)
    return model.predict(data)


def run_pyod_models(filename: str, dataset: str) -> None:
    data, labels = chaoda_datasets.read(dataset, NORMALIZE, SUB_SAMPLE)

    scores: List[str] = list()
    print(f'{dataset}:', end=' ')
    for name, model in MODELS.items():
        print(f'{name}', end=', ')
        # noinspection PyBroadException
        try:
            outlier_scores = train_model(model(), data)
            score = roc_auc_score(labels, outlier_scores)
            score = f'{score:.3f}'
        except TimeoutError:
            score = 'Timeout'
        except Exception as _:
            score = 'Exception'
        scores.append(score)
    print('')

    scores: str = ','.join(scores)
    with open(filename, 'a') as fp:
        fp.write(f'{dataset},{scores}\n')
    return


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(42)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    _filename = os.path.join(RESULTS_PATH, 'pyod_comparisons.csv')
    with open(_filename, 'w') as _fp:
        _header = ','.join(MODELS.keys())
        _fp.write(f'dataset,{_header}\n')

    _datasets = list(chaoda_datasets.DATASETS.keys())
    for _dataset in _datasets:
        run_pyod_models(_filename, _dataset)
