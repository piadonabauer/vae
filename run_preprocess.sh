#!/bin/bash
#SBATCH --job-name=nersemble_preprocess
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load StdEnv cuda opencv python scipy-stack
source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate

srun python data/processing/preprocess_nersemble.py \
  --image-size 128