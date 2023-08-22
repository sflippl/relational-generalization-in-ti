#!/bin/sh
#SBATCH --job-name=exp14
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mail-user=sl4742@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-159
#SBATCH --output=slurm/slurm-%A_%a.out
#SBATCH --exclude=ax[01]

module load anaconda3-2019.03
conda activate principles-of-ti
python experiments/14_crossentropy-tp.py $SLURM_ARRAY_TASK_ID
