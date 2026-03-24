#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-05:00:00
#SBATCH --output=job_%j.out

# load modules if needed
module load StdEnv cuda opencv python scipy-stack 

# activate environment
source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate

# run your script
cd /home/piado/projects/aip-lindell/piado/vae/Open-Sora
python3 scripts/vae/train.py configs/vae/train/wan_multiview_finetune.py