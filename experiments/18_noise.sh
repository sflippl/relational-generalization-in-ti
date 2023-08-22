#!/bin/sh
#SBATCH --job-name=noise18
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=sl4742@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-179
#SBATCH --output=slurm/slurm-%A_%a.out

module load anaconda3-2019.03
conda activate principles-of-ti
python experiments/18_noise.py $SLURM_ARRAY_TASK_ID
