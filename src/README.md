# CLAM: CHAODA

This readme contains the information and specific commands you would need to set up a python virtual environment and reproduce our results for CHAODA and the comparisons against pyOD and other algorithms.

## Setting up Python and a virtual environment

We developed, tested and benchmarked our implementation with Ubuntu 20.04 and Python 3.6.
If you have a Mac, you made a terrible life decision some time ago and will have to adapt some commands for yourself.

First, make sure to install the required packages to create and manage Python virtual-environments.
```bash
$ sudo apt install python3-dev python3-pip python3-venv
```

Next, create a new virtual environment in this folder.
```bash
$ cd /path/to/this/folder
$ python3 -m venv venv
```

Update ```pip``` and install the required Python packages.
```bash
$ source venv/bin/activate
$ pip install --upgrade pip setuptools wheel
$ pip install sklearn scipy pyod tensorflow keras pandas
```

We include a ```requirements.txt``` file with the exact version number we used for the reported benchmarks.
If you wish, you can ignore the last command and install the identical versions of these packages with:
```bash
$ pip install -r requirements.txt
```

Verify that all tests for ```pyclam``` finish successfully.
```bash
$ python -m unittest discover
```

This verification should take roughly one minute, and all tests should pass.

You now have a proper virtual environment to run our code.

## Running benchmarks

First, download all datasets used for benchmarks.
```bash
$ python datasets.py
```

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
