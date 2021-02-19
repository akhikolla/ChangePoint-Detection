#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=0-00:10:00
#SBATCH --array=1-16

argv=$(awk "NR==${SLURM_ARRAY_TASK_ID}" plot_arg.txt)

srun python3 .py $argv
