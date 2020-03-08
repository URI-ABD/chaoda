# Clustered Hierarchical Anomaly and Outlier Detection Algorithms

To reproduce the results shown in the paper, you will need to run ```bash reproduce.sh``` from terminal.
We ran this on Ubuntu 18.04 with Python 3.6+.

If you would like to not use the bash script, you need to:

    * install the required packages in requirements.txt,
    * and run ```python3 reproduce.py``` from terminal. 

This takes a few hours.
You might want a coffee, some popcorn and/or a dedicated workstation.

When this is done, navigate to the plots folder and see the auc-vs-depth and lfd-vs-depth plots along with a .txt file containing the LaTeX formatted table of results and comparisons.

#### P.S.

The Random Walk algorithm takes, by far, the longest to run.
We have turned it off by default.
If you would like to see that run, just un-comment line 167 in methods.py and line 163 in reproduce.py.
