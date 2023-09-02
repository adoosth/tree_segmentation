#!/bin/bash
#SBATCH -o ./log/experiment-%J.txt

############################################################
# Utils                                                    #
############################################################

#module load conda
module load gcc/9.3.0
#module load cuda/11.0.3

echo "Activating conda..."
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate PointMAE

# Distributed parameters
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "srun --ntasks $WORLD_SIZE python cluster_instance.py $@"
srun --ntasks $WORLD_SIZE python cluster_instance.py $@
echo -ne "Done @"
hostname
echo ""