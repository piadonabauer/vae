#!/bin/bash
# Wrapper script to set up environment and run VAE training
# Usage: bash train_vae.sh [config_path] [additional_args]

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENSORA_ROOT="$SCRIPT_DIR/Open-Sora"
VENV_PATH="$SCRIPT_DIR/snth"

# Activate virtual environment if it exists
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtual environment: $VENV_PATH"
fi

# Set Python path to include Open-Sora
export PYTHONPATH="${OPENSORA_ROOT}:${PYTHONPATH}"

# Default config if not provided
CONFIG="${1:-configs/vae/train/wan_multiview_finetune.py}"

echo "=========================================="
echo "VAE Training Script"
echo "=========================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Config: $CONFIG"
echo "Working dir: $OPENSORA_ROOT"
echo "=========================================="
echo ""

# Run the training script
cd "$OPENSORA_ROOT"
python scripts/diffusion/train.py "$CONFIG" "${@:2}"
