# Reproduction of reproduce_paper_results.sh
# Without execution of eval_main.py
# To just download and prepare the data

#!/bin/bash

# Save current path.
CURR_DIR=$PWD

# Download the data to the same locations with the current script.
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR="${BASE_DIR}/data"
mkdir -p "${DATA_DIR}"
mkdir -p "${DATA_DIR}/wmt19_metric_task_results"
curl -o "${DATA_DIR}/annotations.zip" http://storage.googleapis.com/gresearch/kobe/data/annotations.zip
curl -o "${DATA_DIR}/wmt19_metric_task_results/sys-level_scores_metrics.csv" http://storage.googleapis.com/gresearch/kobe/data/wmt19_metric_task_results/sys-level_scores_metrics.csv

# Unzip the annotations.
cd "${DATA_DIR}"
unzip "${DATA_DIR}/annotations.zip"