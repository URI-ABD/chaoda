import os
import warnings
import time
import csv

import numpy as np

from pyod.models import cblof
from pyod.models import cof
from pyod.models import hbos
from pyod.models import iforest
from pyod.models import knn
from pyod.models import lmdd
from pyod.models import loda
from pyod.models import lof
from pyod.models import lscp
from pyod.models import mcd
from pyod.models import mo_gaal
from pyod.models import ocsvm
from pyod.models import sod
from pyod.models import so_gaal
from pyod.models import sos
from pyod.models import xgbod

import src.datasets as chaoda_datasets
from src.utils import RESULTS_PATH

TRAIN = 0.80

# TODO: Explore Neuron issues and see if they can be fixed.
MODELS = {
    # 'ABOD': abod.ABOD, # Neuron issue
    # 'AUTOENCODER': auto_encoder.AutoEncoder, # Neuron issue
    'CBLOF': cblof.CBLOF,
    'COF': cof.COF,
    'HBOS': hbos.HBOS,
    'IFOREST': iforest.IForest,
    'KNN': knn.KNN,
    'LMDD': lmdd.LMDD,
    'LODA': loda.LODA,
    'LOF': lof.LOF,
    # 'LOCI': loci.LOCI, # training takes too long
    'LSCP': lscp.LSCP,
    'MCD': mcd.MCD,
    'MOGAAL': mo_gaal.MO_GAAL,
    'OCSVM': ocsvm.OCSVM,
    'SOD': sod.SOD,
    'SOGAAL': so_gaal.SO_GAAL,
    'SOS': sos.SOS,
    # 'VAE': vae.VAE, # Disabled due to: "ValueError: The number of neurons should not exceed the number of features"
    'XGBOD': xgbod.XGBOD,
    'ALL': None
}


def train_test(model, data, labels):
    # TODO: Unsupervised training, generate outlier scores and use auc for comparison

    # Split the data into train/test.
    n_train = round(len(data) * TRAIN)
    train_data, test_data = data[:n_train], data[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # Train the model.
    start = time.time()
    model.fit(train_data, train_labels)
    train_time = time.time() - start

    # Evaluate.
    start = time.time()
    predicted = model.predict(test_data)
    predict_time = time.time() - start

    # TODO: Use auc_roc for score
    score = np.sum(predicted == test_labels) / len(test_labels)
    print('\n'.join([
        f'{"#" * 15} BEGIN SUMMARY {"#" * 15}',
        f'method={str(model)}',
        f'train=({train_data.shape}, {train_labels.shape})',
        f'test=({test_data.shape}, {test_labels.shape})',
        f'predicted={predicted.shape}',
        f'score={score:0.3f}',
        f'{"#" * 15} END SUMMARY  {"#" * 15}'
    ]))
    return score, train_time, predict_time


# @click.command()
# @click.argument('dataset', type=click.Choice(DATASETS.keys(), case_sensitive=False))
# @click.argument('method', type=click.Choice(METHODS.keys(), case_sensitive=False))
def main(filename: str, dataset: str, method: str) -> None:
    if not os.path.exists(filename):
        with open(filename, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['dataset', 'model', 'train-time', 'predict-time', 'score'])

    data, labels = chaoda_datasets.read(dataset)
    if method.upper() == 'ALL':
        for name, model in MODELS.items():
            print(f'Train/test: {name}')
            # noinspection PyBroadException
            try:
                score, train_time, predict_time = train_test(model(), data, labels)
            except Exception as _:
                train_time = 'NA'
                predict_time = 'NA'
                score = 'EXCEPTION'

            with open(filename, 'a') as fp:
                writer = csv.writer(fp)
                writer.writerow([dataset, name, train_time, predict_time, score])

    else:
        model = MODELS[method.upper()]()
        train_test(model, data, labels)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(42)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    _filename = os.path.join(RESULTS_PATH, 'comparisons.csv')

    # _datasets = list(chaoda_datasets.DATASETS.keys())
    _datasets = ['cardio']
    for _dataset in _datasets:
        main(_filename, _dataset, 'All')
