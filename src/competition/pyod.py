import os
import warnings
import time
import csv

import numpy as np
import click

from pyod.models import abod
from pyod.models import auto_encoder
from pyod.models import cblof
from pyod.models import cof
from pyod.models import hbos
from pyod.models import iforest
from pyod.models import knn
from pyod.models import lmdd
from pyod.models import loda
from pyod.models import lof
from pyod.models import loci
from pyod.models import lscp
from pyod.models import mcd
from pyod.models import mo_gaal
from pyod.models import ocsvm
from pyod.models import sod
from pyod.models import so_gaal
from pyod.models import sos
from pyod.models import vae
from pyod.models import xgbod

from ..datasets import read
from ..datasets import DATASETS

TRAIN = 0.10

METHODS = {
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
    'LOCI': loci.LOCI,
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
    # Split the data into train/test.
    n_train = round(len(data) * TRAIN)
    train_data, test_data = data[:n_train], data[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]
    
    # Train the model.
    start = time.time()
    model.fit(train_data, train_labels)
    traintime = time.time() - start

    # Evaluate.
    start = time.time()
    predicted = model.predict(test_data)
    predicttime = time.time() - start

    score = np.sum(predicted == test_labels)/len(test_labels)
    print('\n'.join([
        f'{"#" * 15} BEGIN SUMMARY {"#" * 15}',
        f'method={str(model)}',
        f'train=({train_data.shape}, {train_labels.shape})', 
        f'test=({test_data.shape}, {test_labels.shape})', 
        f'predicted={predicted.shape}',
        f'score={score:0.3f}',
        f'{"#" * 15} END SUMMARY  {"#" * 15}'
    ]))
    return score, traintime, predicttime


@click.command()
@click.argument('dataset', type=click.Choice(DATASETS.keys(), case_sensitive=False))
@click.argument('method', type=click.Choice(METHODS.keys(), case_sensitive=False))
def main(dataset: str, method: str) -> None:
    data, labels = read(dataset)
    if method.upper() == 'ALL':
        with open('results.csv', 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['model', 'dataset', 'train-time', 'predict-time', 'score'])

        for name, model in METHODS.items():
            print(f'Train/test: {name}')
            try:
                score, traintime, predicttime = train_test(model(), data, labels)
            except Exception as err:
                score = 'FAILURE'
                traintime = 'NA'
                predicttime = 'NA'
            
            with open('results.csv', 'a') as fp:
                writer = csv.writer(fp)
                writer.writerow([name, dataset, traintime, predicttime, score])

    else:
        model = METHODS[method.upper()]()
        train_test(model, data, labels)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
