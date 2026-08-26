#!/usr/bin/env bash
#SBATCH --job-name=paper_sweep
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-18:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-9%4

# Paper rerun sweep: E1 (rate-quality 2x2 + per-view references) and E5 (unfreeze).
# All runs share ONE fixed protocol (see paper/02_experiments.md):
#   128px, T=9, V=2, all_people_one_expression, effective batch 64, 170 epochs,
#   lr 5e-4 constant, LoRA rank 32, no discriminator, EMA eval.
# The only thing that changes between arms is the model config.
#
# Arms:
#   1  E1a  per-view reference (independent_views), TC off
#   2  E1b  per-view reference, TC on
#   3  E1c  fused latent (cross-attention), TC off
#   4  E1d  fused latent, TC on            <- headline point
#   5  E5b  E1d + unfreeze full encoder (incl. strided time convs)
#   6  E5c  E5b + full_finetune_decoder    <- nothing frozen anywhere
#   7  E1z  zero-shot eval only (epochs=0 + final_eval) -- pretrained Wan floor.
#           NOTE: not tested end-to-end; if epochs=0 skips final_eval, run arm 1
#           with --epochs 1 --save_ckpt False instead and ignore the train step.
#   8  E11a E1d + latent widened 16->32 channels
#   9  E11b E1d + latent widened 16->64 channels
#           The widen arms are the positive capacity test: if joint view+temporal
#           compression recovers with a wider latent, the bottleneck really is the
#           rate. VAE only, no diffusion retraining. New channels start as a no-op
#           (step 0 = the 16-ch model). No warm start from E1b -- the boundary
#           conv shapes differ.
#
# Two-stage discipline: run every arm with OVERFIT=1 first (single_sequence, must
# reach near-perfect reconstruction) before launching the real generalization run.
# The overfit gate catches implementation problems for the cost of a few GPU-hours.
#
# Run length & comparability -- READ THIS:
# - Overfit gate: stops itself once epoch-mean train PSNR >= 35 for 3 consecutive
#   epochs (stop_at_train_psnr), capped at 2000 epochs. PASS/FAIL only, the gate
#   is never compared numerically between arms. (If an arm looks visually perfect
#   but plateaus just under 35, judge by eye before declaring it failed -- the
#   gate exists to catch breakage, not to be a benchmark.)
# - Generalization: FIXED budget for every arm -- 170 epochs at effective batch 64
#   = identical optimizer updates AND identical samples seen, no matter which
#   micro-batch the OOM ladder lands on (bs*accum is held at 64). full_eval_every
#   is counted in optimizer updates, so all arms are evaluated at the SAME update
#   steps. Never early-stop a generalization arm and never change TRAIN_EPOCHS for
#   one arm only -- both would make the numbers incomparable.
#
# Wall clock: jobs request 18h (easier to schedule than >20h). Each job submits a
# dependent successor (afterany) before training; on completion it writes a .DONE
# marker so leftover successors exit immediately. CHAIN_LEFT (default 3) caps the
# chain, i.e. up to 4 x 18h per arm. Checkpoints are per-epoch, resume is automatic.
#
# Staged init (default for the joint arms 4-6): warm-start from the converged E1b
# per-view TC checkpoint -- temporal axis first, view axis second. The script finds
# the E1b checkpoint automatically, or set INIT_CKPT=/path/to/epochN-.... Missing
# fusion keys keep their fresh (zero) init; the view attention is re-randomized.
# INIT_CKPT=none disables warm start (that is ablation E7b).
# ORDER MATTERS: arm 2 must finish before arms 4-6 start.
#
# Usage (from /project, Slurm rejects /home submits):
#   sbatch --export=ALL,TASK=4,OVERFIT=1 ./run_paper_sweep.sh   # stage 1: overfit gate
#   sbatch --export=ALL,TASK=4 ./run_paper_sweep.sh             # stage 2: real run
#   sbatch ./run_paper_sweep.sh                                 # all arms (stage 2)

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
DRY_RUN="${DRY_RUN:-0}"

OVERFIT="${OVERFIT:-0}"
INIT_CKPT="${INIT_CKPT:-}"

if [[ "$OVERFIT" == "1" ]]; then
  # Stage-1 overfit gate: one sequence, batch 1, no val set. Cheap; the only
  # question is whether train PSNR goes ~perfect. No warm start here on purpose:
  # the gate should test the architecture itself.
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-2000}"
  BATCH_LADDER=( "1:1" )
else
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-170}"
  # batch:accum pairs to try after an OOM, largest first. Effective batch stays 64.
  BATCH_LADDER=( "16:4" "8:8" "4:16" "2:32" )
fi

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$TASK" ]]; then
  echo "Set TASK=1..7 or submit as array job"; exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OPEN_SORA_ROOT}/slurm_logs"
  module --force purge
  module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/9.2.1.18 opencv python/3.11.5 scipy-stack cmake python-build-bundle/2025b
  # shellcheck source=/dev/null
  source "$VAE_VENV"
  export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.triton"
  export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/.torchinductor"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

cd "$OPEN_SORA_ROOT"

# wandb is part of the protocol: every run must log. Fail fast instead of
# silently training without a record.
if [[ "${WANDB_MODE:-}" != "offline" && -z "${WANDB_API_KEY:-}" ]] \
   && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "ERROR: wandb is not configured (no WANDB_API_KEY, no ~/.netrc entry)."
  echo "Run 'wandb login' or export WANDB_API_KEY before submitting."
  exit 1
fi

# Shared protocol flags. T=9 on purpose: latent T'=3 = frame 0 + two 4-frame
# chunks, so bleeding within chunks AND across a boundary are both measurable.
if [[ "$OVERFIT" == "1" ]]; then
  # 1 clip, no val set. The run stops itself once the gate target is held.
  DATA_ARGS=( --data_preset single_sequence
              --stop_at_train_psnr 35 --stop_at_train_psnr_consecutive 3 )
else
  DATA_ARGS=(
    --data_preset all_people_one_expression
    --dataset_presets.all_people_one_expression.expected_views 2
    --dataset_presets.all_people_one_expression.skip_mismatched_views True
    --val_dataset_presets.all_people_one_expression.expected_views 2
    --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
  )
fi

COMMON=(
  --bucket_config "{'128px_ar1:1': {9: (1.0, 1)}}"
  "${DATA_ARGS[@]}"
  --model.view_in 2
  --model.use_lora True
  --model.use_lora_after True
  --model.lora_rank 32
  --discriminator_choice none
  # Neutralize the divergence guard's early stop (threshold 0 can never trip):
  # (a) a slow-learning arm must never be killed -- the fixed budget is the
  #     protocol, and (b) the guard's start epoch derives from steps-per-epoch,
  #     i.e. from the MICRO-batch, so arms on different rungs of the OOM ladder
  #     would get different guard behavior. Keep train_psnr_guard True: the
  #     epoch-PSNR aggregation it drives also feeds the overfit-gate stop.
  --train_psnr_guard_threshold 0
  --epochs "$TRAIN_EPOCHS"
  --wandb True
  --optimization False
  --FAST_MODE False
  --save_ckpt True
  # Logging cadence, tuned for the actual run length: 358 train samples at
  # effective batch 64 = 5 updates/epoch = ~850 updates over 170 epochs.
  # Scalars: dense early schedule, then every 20 updates (~4 epochs) -> ~50 points.
  # Images: same early schedule, then geometric backoff (x1.5, capped) -> ~20 grids.
  # Full eval: every 50 updates (~10 epochs) -> ~17 val evals per run; the val set
  # is only 10 clips so this is cheap, and best-val selection needs the density.
  # All cadences count OPTIMIZER UPDATES, so eval/log points align across arms.
  --log_every 20
  --log_schedule_steps "[1,2,3,5,8,12,20,30,50,75,100,150,200]"
  --image_log_growth_factor 1.5
  --image_log_max_interval 2000
  --full_eval_every 50
  --eval_num_samples 0
  --fixed_seq_eval_every_epochs 10
  # Qualitative comparability: vis samples are picked by sorted dataset order
  # (first 3 distinct train participants / first 3 val clips = p018, p030, p038),
  # NOT from the shuffled batch -- identical people in every run and every arm.
  --num_reconstruction_vis_samples 3
)

MODEL_ARGS=()
case "$TASK" in
  1)
    run_name="paper_E1a_perview_tcF"
    MODEL_ARGS=( --model.independent_views True --model.temporal_compression False )
    ;;
  2)
    run_name="paper_E1b_perview_tcT"
    MODEL_ARGS=( --model.independent_views True --model.temporal_compression True )
    ;;
  3)
    run_name="paper_E1c_fused_tcF"
    MODEL_ARGS=( --model.fusion_mode cross_attention --model.use_viewwise_decoder_lora True
                 --model.temporal_compression False )
    ;;
  4)
    run_name="paper_E1d_fused_tcT"
    MODEL_ARGS=( --model.fusion_mode cross_attention --model.use_viewwise_decoder_lora True
                 --model.temporal_compression True )
    ;;
  5)
    run_name="paper_E5b_fused_tcT_unfreeze_enc"
    MODEL_ARGS=( --model.fusion_mode cross_attention --model.use_viewwise_decoder_lora True
                 --model.temporal_compression True
                 --model.train_spatial True --model.freeze_temporal False )
    ;;
  6)
    run_name="paper_E5c_fused_tcT_unfreeze_all"
    MODEL_ARGS=( --model.fusion_mode cross_attention --model.use_viewwise_decoder_lora True
                 --model.temporal_compression True
                 --model.train_spatial True --model.freeze_temporal False
                 --model.full_finetune_decoder True )
    ;;
  7)
    run_name="paper_E1z_perview_zeroshot"
    MODEL_ARGS=( --model.independent_views True --model.temporal_compression False
                 --epochs 0 --save_ckpt False --final_eval True )
    ;;
  8)
    run_name="paper_E11a_fused_tcT_widen32"
    MODEL_ARGS=( --model.fusion_mode cross_attention --model.use_viewwise_decoder_lora True
                 --model.temporal_compression True
                 --model.latent_widen_to 32 )
    ;;
  9)
    run_name="paper_E11b_fused_tcT_widen64"
    MODEL_ARGS=( --model.fusion_mode cross_attention --model.use_viewwise_decoder_lora True
                 --model.temporal_compression True
                 --model.latent_widen_to 64 )
    ;;
  *)
    echo "Unknown TASK=$TASK"; exit 1 ;;
esac

[[ "$OVERFIT" == "1" ]] && run_name="${run_name}_overfit"

# Finished arms leave a marker; leftover chained successors exit immediately.
DONE_MARKER="${OPEN_SORA_ROOT}/outputs/${run_name}.DONE"
if [[ -f "$DONE_MARKER" ]]; then
  echo "[chain] ${run_name} already done (${DONE_MARKER}); nothing to do."
  exit 0
fi

# Submit the successor BEFORE training starts, so a wall-clock kill cannot break
# the chain. The successor resumes from the newest checkpoint (logic below) or
# exits via the marker if this job finishes the arm.
CHAIN_LEFT="${CHAIN_LEFT:-3}"
if [[ -n "${SLURM_JOB_ID:-}" && "$CHAIN_LEFT" -gt 0 && "$DRY_RUN" != "1" ]]; then
  sbatch --dependency="afterany:${SLURM_JOB_ID}" \
         --export="ALL,TASK=${TASK},OVERFIT=${OVERFIT},CHAIN_LEFT=$((CHAIN_LEFT - 1)),INIT_CKPT=${INIT_CKPT}" \
         "$0" \
    && echo "[chain] successor queued (CHAIN_LEFT=$((CHAIN_LEFT - 1)))" \
    || echo "[chain] WARNING: could not queue successor; resubmit by hand if the job times out"
fi

# Staged init for the joint arms (4-6): default to the best E1b checkpoint unless
# the caller pins INIT_CKPT or disables it with INIT_CKPT=none. Skipped in overfit
# mode (the gate tests the architecture, not the curriculum).
WARMSTART_ARGS=()
if [[ "$OVERFIT" != "1" && "$TASK" =~ ^[456]$ && "$INIT_CKPT" != "none" ]]; then
  if [[ -z "$INIT_CKPT" ]]; then
    best_ep=-1
    for d in "${OPEN_SORA_ROOT}/outputs/paper_E1b_perview_tcT__job"*/; do
      [[ -d "$d" ]] || continue
      ck=$(ls "$d" 2>/dev/null | grep "^epoch" | sort -V | tail -1)
      [[ -z "$ck" ]] && continue
      ep=$(echo "$ck" | grep -oP 'epoch\K[0-9]+' || echo 0)
      if (( ep > best_ep )); then best_ep=$ep; INIT_CKPT="${d}${ck}"; fi
    done
  fi
  if [[ -n "$INIT_CKPT" ]]; then
    echo "[init] warm start from: $INIT_CKPT"
    WARMSTART_ARGS=( --load "$INIT_CKPT" --load_optimizer False
                     --model.reinit_view_attention_after_load True )
  else
    echo "[init] WARNING: no E1b checkpoint found -- falling back to Wan-only init."
    echo "[init] Run TASK=2 first, or pass INIT_CKPT explicitly (INIT_CKPT=none to silence)."
  fi
fi

experiment_name="$run_name"
[[ -n "${SLURM_JOB_ID:-}" ]] && experiment_name="${run_name}__job${SLURM_JOB_ID}_t${TASK}"
MASTER_PORT=$((21000 + (${SLURM_JOB_ID:-$$} % 20000) + TASK))
export MASTER_PORT MASTER_ADDR=127.0.0.1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0

# Resume if an earlier job for this arm left a checkpoint behind. A resume beats
# the warm start (the run already contains it).
LOAD_CKPT=""
best_epoch=-1
for prev_dir in "${OPEN_SORA_ROOT}/outputs/${run_name}__job"*/; do
  [[ -d "$prev_dir" ]] || continue
  ck=$(ls "$prev_dir" 2>/dev/null | grep "^epoch" | sort -V | tail -1)
  [[ -z "$ck" ]] && continue
  ep=$(echo "$ck" | grep -oP 'epoch\K[0-9]+' || echo 0)
  if (( ep > best_epoch )); then best_epoch=$ep; LOAD_CKPT="${prev_dir}${ck}"; fi
done
if [[ -n "$LOAD_CKPT" ]]; then
  echo "[resume] epoch ${best_epoch}: $LOAD_CKPT"
  LOAD_ARGS=( --load "$LOAD_CKPT" --load_optimizer False )
else
  LOAD_ARGS=( "${WARMSTART_ARGS[@]}" )
fi

echo "TASK=${TASK}  ${run_name}  epochs=${TRAIN_EPOCHS}  overfit=${OVERFIT}"
echo "model args: ${MODEL_ARGS[*]}"

for spec in "${BATCH_LADDER[@]}"; do
  IFS=':' read -r bs acc <<< "$spec"
  echo "── batch=${bs} accum=${acc} (effective $((bs * acc))) ──"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "accelerate launch ... --batch_size $bs --accumulation_steps $acc"; exit 0
  fi
  log_tail=$(mktemp)
  set +e
  accelerate launch \
    --num_processes 1 --num_machines 1 --dynamo_backend no --mixed_precision bf16 \
    --main_process_port "$MASTER_PORT" \
    scripts/vae/train.py "${OPEN_SORA_ROOT}/${CONFIG}" \
    --experiment_name "$experiment_name" --wandb_expr_name "$run_name" \
    --wandb_project wan_multiview_vae_paper \
    --batch_size "$bs" --accumulation_steps "$acc" \
    "${LOAD_ARGS[@]}" \
    "${COMMON[@]}" "${MODEL_ARGS[@]}" 2>&1 | tee "$log_tail"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then
    rm -f "$log_tail"
    touch "$DONE_MARKER"
    echo "done (b=${bs} a=${acc}); marker: $DONE_MARKER"
    exit 0
  fi
  if ! grep -qiE 'CUDA out of memory|OutOfMemoryError' "$log_tail"; then
    rm -f "$log_tail"; exit "$rc"
  fi
  rm -f "$log_tail"
  echo "OOM at batch=${bs}; retrying smaller"
  MASTER_PORT=$((MASTER_PORT + 1)); export MASTER_PORT
done
echo "all batch sizes OOM'd"; exit 1
