#!/bin/bash

export NOTEBOOK_PATH=${PWD}/notebooks
export NOTEBOOK_PORT=${EXPOSED_PORT:-18881}

jupyter-notebook --config=${PWD}/.jupyter_notebook_config.py --allow-root --ip 0.0.0.0 --no-browser --notebook-dir=$NOTEBOOK_PATH --port=$NOTEBOOK_PORT

