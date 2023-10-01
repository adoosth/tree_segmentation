#!/bin/bash
#SBATCH -o ./log/experiment-%J.txt
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 12:00:00

echo "Activating conda..."
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate PointMAE

CONFIG_FILE=$1
echo "CONFIG_FILE="$CONFIG_FILE

python local_train.py $CONFIG_FILE