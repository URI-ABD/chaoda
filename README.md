# CLAM: CHAODA

This readme contains the information and specific commands you would need to set up a python virtual environment and reproduce our results for CHAODA and the comparisons against other algorithms.

## Setting up Python and a Virtual Environment:

We developed and tested our implementation on a machine with:

* **OS**: Manjaro Linux x86_64
* **Kernel**: 5.13.13-1-MANJARO
* **Python**: 3.9.6

If you have a Mac, you made a terrible life decision some time ago and will have to adapt some commands for yourself.

For the paper, we benchmarked using a machine with:

* **CPU**: Intel Xeon E5-2690 v4 (28 cores) @ 3.500G
* **Memory**: 512GiB
* **OS**: CentOS Linux 7 (Core) x86_64
* **Kernel**: 3.10.0-1127.13.1.el7.x86_64
* **Python**: 3.6.8

First, make sure to install the required packages to create and manage Python virtual-environments.

**Arch:**
```zsh
$ sudo pacman -S python-pip
```

**Ubuntu:**
```bash
$ sudo apt install python3-dev python3-pip python3-venv
```

Next, create a new virtual environment.

**Arch:**
```zsh
$ python -m venv venv
```

**Ubuntu:**
```bash
$ python3 -m venv venv
```

Update `pip` and install the required Python packages.
```bash
$ source venv/bin/activate
$ pip install --upgrade pip setuptools wheel
```

Now `pip install` the following packages for:

* CHAODA benchmarks:
  * `pyclam`
  * `numpy`
  * `scipy`
  * `scikit-learn`
  * `h5py`
  * `pandas`
* competing algorithms:
  * `pyod`
  * `tensorflow`
  * `keras`
* (my) sanity:
  * `tqdm`
* SDSS-APOGEE2 benchmarks:
  * `astropy`
* generating some neat plots:
  * `matplotlib`
  * `umap-learn`

We include a `requirements.txt` file with the exact version numbers we used for each package.
If you're using Linux and have an old enough kernel, these versions will not be available.
In this case, you're on your own.
If you have a Mac or a Windows computer, re-read the previous sentence. 

## Running CHAODA:

We provide an easy and helpful CLI through `main.py`.
For help, run:
```bash
$ python main.py --help
```

We assume write access for the local folder of this repository.

### Downloading ODDS datasets:

You will need these datasets to reproduce the benchmarks for CHAODA and competitors.

```bash
$ python main.py --mode download-datasets
```

### Training the Meta-ML Models:

We provide the meta-ml models we trained and used for the reported benchmarks.
These are in `chaoda/meta_models.py`.
If you're curious as to how these are generated (and how they strictly adhere to PEP8), feel free to dig around in `chaoda/train_meta_ml.py`.

If you want to retrain the meta-ml models for yourself, run:
```bash
$ python main.py --mode train-meta-ml --meta-ml-epochs 10
```
`5` epochs are likely enough.
We went with `10` by default.

Running this command will produce a new file `chaoda/custom_meta_models.py` which will override the meta-ml models we provide.
If you wish to revert to the models we provide, just delete this file.

### Benchmarking CHAODA:

You can reproduce CHAODA benchmarks with:
```bash
$ python main.py --mode bench-chaoda
```

This will create two files:

* `results/scores.csv`, and
* `results/times.csv`

If you want to use our heuristic for what we call CHAODA_FAST, simply add the `--fast` flag:
```bash
$ python main.py --mode bench-chaoda --fast
```

If you want to see detailed statistics on each individual algorithm in the ensemble, add the `--report-individual-methods` flag:
```bash
$ python main.py --mode bench-chaoda --report-individual-methods
```
This will create a giant file called `results/individual_scores.csv`.

### Benchmarking Competitors:

We benchmark many competing algorithms from the PyOD suite.
You can reproduce these with:
```bash
$ python main.py --mode bench-pyod
```
We default to allowing each of these algorithms up to `10` hours to finish running.
You can override the default by passing in the number of seconds with the `--pyod-time-limit` argument:
```bash
$ python main.py --mode bench-pyod --pyod-time-limit 600
```

This will add benchmarks to `results/scores.csv` and `results/times.csv`.

### Scoring the APOGEE Data:

This part is a bit more involved than the rest.

First, you need to download the `APOGEE2` data from the `SDSS` archives.
Go to [this link](https://www.sdss.org/dr16/data_access/bulk/) and follow their instructions.
You will need approximately `1.5TB` at the time of this writing (likely more in the future).
You may also need to read up on their [data model](https://data.sdss.org/datamodel/) if you run into problems here.

Edit `APO_TELESCOPE` and `APOGEE_PATH` in `sdss/preparse.py` as needed to point the script to the apogee2 data.

Run the preparse script to extract the APOGEE spectra that we used:
```bash
$ python main.py --mode preparse-apogee
```

For us, this extracted `528,319` spectra.
There are probably more in this dataset, but you will need to come up with some regex magic to extract them.

The preparse script will need another approx `20GB` of space to store a numpy array.
CHAODA **does not** load the whole thing into RAM.
From here on, you should be able to reproduce APOGEE scores on a machine with `12GB` of RAM.

To score the APOGEE dataset, run:
```bash
$ python main.py --mode score-apogee
```

This will produce a `json` file in the `results` directory with the amount of time taken and the anomaly score for each spectrum we extracted.
These scores correspond with the fits files named in the `filenames.csv` in the `data` directory.
If you're an expert on the APOGEE data, have fun!
