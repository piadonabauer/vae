#!/bin/bash
# =============================================================================
# test_temporal_ideas.sh  —  9-way sweep over temporal-compression quality ideas
#
# Variants (SLURM array index):
#   0  baseline          all ideas off, no discriminator
#   1  strided_lora       use_strided_temporal_lora
#   2  pos_embed          use_temporal_latent_pos_embed
#   3  refinement         use_temporal_refinement
#   4  latent_attn        use_latent_temporal_attention
#   5  ctx_warmup         use_latent_context_warmup
#   6  frame_feedback     use_decoder_frame_feedback  (+ grad_clip=0.5)
#   7  disc_3d            no new methods + discriminator_choice=Train (3D PatchGAN)
#   8  disc_4d            no new methods + discriminator_choice=TrainMultiview4D
#
# Data:  all_people_one_expression — all people, one sequence each, 2 views,
#        9 temporal frames, 128px, temporal_compression=True.
# Steps: ~1500 optimizer updates per run.
#
# Batch-size fallback:  tries bs=32 first; on OOM reduces to 16→8→4→1.
#   Accumulation steps are raised to keep effective batch ≈ 32 (bs=1 is last resort).
#
# Submit:  sbatch --array=0-8 scripts/vae/test_temporal_ideas.sh
# One run: sbatch --array=4   scripts/vae/test_temporal_ideas.sh
# =============================================================================
#SBATCH --job-name=temporal_ideas
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-8
#SBATCH --exclude=kn051

set -uo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
source /home/piado/projects/aip-lindell/piado/vae/snth/bin/activate
cd /home/piado/projects/aip-lindell/piado/vae/Open-Sora

export TRITON_CACHE_DIR="$SLURM_TMPDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/.torchinductor"
export PYTORCH_KERNEL_CACHE_PATH="$SLURM_TMPDIR/.pytorch_kernels"

TRAIN=scripts/vae/train.py
CFG=configs/vae/train/wan_multiview_finetune.py
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# ── Variant table ────────────────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-0}

declare -A NAMES=(
    [0]="baseline"
    [1]="strided_temporal_lora"
    [2]="temporal_latent_pos_embed"
    [3]="temporal_refinement"
    [4]="latent_temporal_attention"
    [5]="latent_context_warmup"
    [6]="decoder_frame_feedback"
    [7]="disc_3d_patchgan"
    [8]="disc_4d_multiview"
)

declare -A IDEA_OVERRIDES=(
    [0]=""
    [1]="--model.use_strided_temporal_lora True"
    [2]="--model.use_temporal_latent_pos_embed True"
    [3]="--model.use_temporal_refinement True"
    [4]="--model.use_latent_temporal_attention True"
    [5]="--model.use_latent_context_warmup True"
    [6]="--model.use_decoder_frame_feedback True --grad_clip 0.5"
    [7]="--discriminator_choice Train --gen_disc_weight 0.1"
    [8]="--discriminator_choice TrainMultiview4D --gen_disc_weight 0.1"
)

NAME="temporal_test_${NAMES[$IDX]}"
OVERRIDES="${IDEA_OVERRIDES[$IDX]}"

echo "======================================================================="
echo " Job array index : $IDX"
echo " Variant name    : $NAME"
echo " Extra overrides : $OVERRIDES"
echo "======================================================================="
nvidia-smi

# ── Epochs → ~1500 optimizer steps ──────────────────────────────────────────
# all_people_one_expression has ~390 train .pt files (all participants minus val).
# steps_per_epoch = floor(390 / batch_size).
# We want steps ≈ 1500, so epochs ≈ ceil(1500 * batch_size / 390).
APPROX_SAMPLES=390
TARGET_STEPS=1500
compute_epochs() {
    local bs=$1
    echo $(( (TARGET_STEPS * bs + APPROX_SAMPLES - 1) / APPROX_SAMPLES ))
}

# ── Common config (all variants) ─────────────────────────────────────────────
COMMON_OVERRIDES=(
    --data_preset               all_people_one_expression
    --model.temporal_compression True
    --FAST_MODE                 False
    --save_ckpt                 False
    --eval_every                20
    --full_eval_every           500
    --log_schedule_steps        "[]"
    --log_every                 10
    --experiment_name           "$NAME"
)

# ── OOM-aware launcher ────────────────────────────────────────────────────────
# Tries batch sizes in order; on OOM retries with a smaller batch + higher
# accumulation to keep effective batch ≈ 32.  Any non-OOM failure aborts.

TMP_LOG="/tmp/${NAME}_${SLURM_JOB_ID:-$$}.log"
SUCCESS=0

for bs in 32 16 8 4 1; do
    # Effective batch ≈ 32; last-resort bs=1 uses accum=1 (just make it run).
    case $bs in
        32) accum=1  ;;
        16) accum=2  ;;
         8) accum=4  ;;
         4) accum=8  ;;
         1) accum=1  ;;
    esac

    epochs=$(compute_epochs $bs)
    echo ""
    echo "--- Trying batch_size=$bs  accumulation_steps=$accum  epochs=$epochs ---"

    rm -f "$TMP_LOG"
    set +e
    accelerate launch \
        --num_processes   1 \
        --num_machines    1 \
        --dynamo_backend  no \
        --mixed_precision bf16 \
        --main_process_port "$MASTER_PORT" \
        "$TRAIN" "$CFG" \
        "${COMMON_OVERRIDES[@]}" \
        --batch_size         $bs \
        --accumulation_steps $accum \
        --epochs             $epochs \
        ${OVERRIDES} \
        2>&1 | tee "$TMP_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS=1
        echo "✓  Completed: $NAME  (batch_size=$bs, accum=$accum, epochs=$epochs)"
        break
    fi

    # Distinguish OOM from other errors
    if grep -qi "out of memory\|cuda oom\|outofmemoryerror\|cudaerroroutofmemory" "$TMP_LOG" 2>/dev/null; then
        echo "⚠  OOM at batch_size=$bs — retrying with smaller batch..."
        # Re-use the same master port (process is dead, port is free again)
        continue
    else
        echo "✗  Non-OOM failure (exit $EXIT_CODE) at batch_size=$bs — aborting."
        cat "$TMP_LOG" | tail -40
        exit $EXIT_CODE
    fi
done

if [ $SUCCESS -eq 0 ]; then
    echo "✗  All batch sizes failed for $NAME. Check GPU memory and logs."
    exit 1
fi

rm -f "$TMP_LOG"
echo "Done: $NAME"
