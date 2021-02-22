#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=0-15:72:00
#SBATCH --array=1-130


argv=$(awk "NR==${SLURM_ARRAY_TASK_ID}" ts.txt)

srun python3 newaccuracy.py $argv
