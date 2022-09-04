#!/bin/bash
set -e -o pipefail -x
source /home/xywang/scripts/init_conda.sh
conda activate pytorch
python /home/xywang/SpaCE/project_1-master/task/TaskForSingleSentenceClassification.py