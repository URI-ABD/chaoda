# CLAM: CHAODA

This readme contains the information and specific commands you would need to set up a python virtual environment and reproduce our results for CHAODA and the comparisons against other algorithms.

## Setting up Python and a Virtual Environment:

We developed, tested and benchmarked our implementation with:

* OS: Manjaro Linux x86_64
* Kernel: 5.13.13-1-MANJARO
* Python: 3.9.6

If you have a Mac, you made a terrible life decision some time ago and will have to adapt some commands for yourself.

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
$ pyhton main.py --mode download-datasets
```

### Training the Meta-ML Models:

We provide the meta-ml models we trained and used for the reported benchmarks.
These are in `chaoda/meta_models.py`.
If you're curious as to how these are generated (and how they strictly adhere to PEP8), feel free to dig around in `chaoda/train_meta_ml.py`.

If you want to retrain the meta-ml models for yourself, run:
```bash
$ pyhton main.py --mode train-meta-ml --meta-ml-epochs 10
```
`5` epochs are likely enough.
We went with `10` by default.

Running this command will produce a new file `chaoda/custom_meta_models.py` which will override the meta-ml models we provide.
If you wish to revert to the models we provide, just delete this file.

### Benchmarking CHAODA:



You should see a new folder called ```data``` in this directory.
This folder contains all downloaded datasets in ```.mat``` format.

If you are happy to use the pre-trained meta-ml models we provide, you can skip this next step.
However, should you wish to re-train the meta-ml models for CHAODA, run the following command:
```bash
$ python train_meta_ml.py
```

This should have created a new Python file named ```meta_models.py```.
You will have to change line 10 in ```benchmark_chaoda.py``` to use the newly trained meta-ml models.

Next, you can run the CHAODA benchmarks.
```bash
$ python benchmark_chaoda.py
```

If you wish to benchmark the methods we compared against in the paper, you can run:
```bash
$ python comparisons.py
```

These last three commands each take a very long time to run.
Grab a coffee, cook a turkey, bake a cake, watch a movie, and repeat as necessary.

You should see a new folder called ```results``` in this directory.
This folder contains the ```.csv``` files containing the AUC performance of CHAODA and competitors.
