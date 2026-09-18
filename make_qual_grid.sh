#!/usr/bin/env bash
#SBATCH --job-name=qual_grid
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/qual_grid_%j.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/qual_grid_%j.err

set -euo pipefail
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
source "$VAE_VENV"

python /home/piado/projects/aip-lindell/piado/vae/paper/figures/make_qualitative_panels.py
echo "Done."
