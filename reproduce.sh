#!/bin/bash

venv_dir=".chaoda"
if [ -d $venv_dir ]
then
    source $venv_dir/bin/activate
else
    python3 -m venv $venv_dir
    source $venv_dir/bin/activate
    pip install --upgrade pip setuptools
    pip install -r requirements.txt
fi

cd src && python reproduce.py && cd ..
deactivate
