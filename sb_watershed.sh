#!/bin/bash
#SBATCH -o ./log/watershed/%J.txt

############################################################
# Utils                                                    #
############################################################

module load gcc/9.3.0
module load singularity

# run singularity and bind script and data directories
singularity run -B watershed:$HOME,data:/data ./container/R.sif

# Xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

Rscript watershed.R /data/syn-multi-thin/test/20230907-160753-syn-multi-thin-2-ULS_26.laz ~/results/syn-multi-thin-2-ULS_26.laz