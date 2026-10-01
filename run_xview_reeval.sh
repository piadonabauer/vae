#!/usr/bin/env bash
#SBATCH --job-name=xview_reeval
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-2:00:00
#SBATCH --output=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/piado/projects/aip-lindell/piado/vae/Open-Sora/slurm_logs/%x_%A_%a.err
#SBATCH --array=1-33%4

# Re-eval cluster-B (and E1d + E1c control) under the *current* eval pipeline
# so all arms share one GT XView reference (target ≈ 0.907 for V=2).
# Eval-only: epochs=0, load best/latest ckpt, final_eval=True.
#
# Usage:
#   sbatch ./run_xview_reeval.sh
#   sbatch --export=ALL,TASK=13 ./run_xview_reeval.sh   # one arm

set -euo pipefail

OPEN_SORA_ROOT="${OPEN_SORA_ROOT:-/home/piado/projects/aip-lindell/piado/vae/Open-Sora}"
CONFIG="${CONFIG:-configs/vae/train/wan_multiview_finetune.py}"
VAE_VENV="${VAE_VENV:-/home/piado/projects/aip-lindell/piado/vae/snth/bin/activate}"
MANIFEST="${OPEN_SORA_ROOT}/../paper/xview_reeval_manifest.txt"

TASK="${TASK:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$TASK" ]]; then
  echo "Set TASK or submit as array"; exit 1
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

# Build / refresh manifest (src_dir relative to outputs/)
python3 - <<'PY' > "$MANIFEST"
import glob, os, json, re
root = os.environ.get("OPEN_SORA_ROOT", "/home/piado/projects/aip-lindell/piado/vae/Open-Sora")
out = os.path.join(root, "outputs")
rows = []
# Always include E1c (cluster-A control) + E1d (headline) + every GT>=0.94 arm
for path in sorted(glob.glob(os.path.join(out, "paper_E*/eval_metrics.jsonl"))):
    d = os.path.dirname(path)
    base = os.path.basename(d)
    if "_overfit" in base:
        continue
    gt = None
    best_p = -1
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("kind") != "full_eval":
                continue
            m = o.get("metrics") or {}
            if "xview_sim_gt" not in m:
                continue
            gt = m["xview_sim_gt"]
            p = m.get("psnr_mean", m.get("psnr")) or -1
            if p > best_p:
                best_p = p
    if gt is None:
        continue
    keep = (gt >= 0.94) or ("E1d_fused_tcT__" in base) or ("E1c_fused_tcF__" in base)
    if not keep:
        continue
    eps = sorted(
        [e for e in glob.glob(d + "/epoch*") if os.path.isdir(os.path.join(e, "model"))],
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )
    if not eps:
        continue
    # Prefer highest epoch (best PSNR almost always near end for these arms)
    ckpt = eps[-1]
    # short arm id: paper_E3d_no_emb_no_lora
    arm = base.split("__job")[0]
    rows.append((arm, base, ckpt, f"{gt:.3f}", f"{best_p:.2f}"))

# Deduplicate by arm name: keep the job dir we already selected (latest listing wins via sort)
by = {}
for r in rows:
    by[r[0]] = r
rows = [by[k] for k in sorted(by)]
for i, (arm, src, ckpt, gt, psnr) in enumerate(rows, 1):
    print(f"{i}\t{arm}\t{src}\t{ckpt}\t{gt}\t{psnr}")
print(f"# {len(rows)} arms", file=__import__("sys").stderr)
PY

N=$(grep -cve '^#' "$MANIFEST" || true)
LINE=$(awk -F'\t' -v t="$TASK" '$1==t {print}' "$MANIFEST")
if [[ -z "$LINE" ]]; then
  echo "TASK=$TASK out of range (manifest has $N arms)"; exit 1
fi
IFS=$'\t' read -r IDX ARM SRC_BASE CKPT OLD_GT OLD_PSNR <<< "$LINE"
echo "[reeval] TASK=$TASK arm=$ARM src=$SRC_BASE"
echo "[reeval] ckpt=$CKPT  old_GT=$OLD_GT old_best_PSNR=$OLD_PSNR"

# Reconstruct model CLI flags from the source run's config.txt
CFG_JSON="${OPEN_SORA_ROOT}/outputs/${SRC_BASE}/config.txt"
mapfile -t MODEL_ARGS < <(python3 - <<PY
import json
cfg = json.load(open("${CFG_JSON}"))
m = cfg.get("model", {})
args = []
def add(flag, val):
    if val is None: return
    if isinstance(val, bool):
        args.append(f"--model.{flag}"); args.append("True" if val else "False")
    else:
        args.append(f"--model.{flag}"); args.append(str(val))

for k in [
    "fusion_mode", "temporal_compression", "independent_views",
    "use_viewwise_decoder_lora", "use_view_embedding", "use_view_group_fusion",
    "lora_rank", "latent_widen_to", "use_lora_before", "use_lora_after",
    "full_finetune_decoder", "freeze_temporal", "train_spatial",
    "use_noncausal_decode", "use_temporal_reflection_pad",
    "use_temporal_side_channel", "use_learned_cache_update",
    "use_subframe_pos_emb", "use_decoder_temporal_attention",
    "view_in",
]:
    if k in m and m[k] is not None:
        add(k, m[k])

td = cfg.get("temporal_diff_loss_weight")
if td:
    print("--temporal_diff_loss_weight"); print(str(td))
disc = cfg.get("discriminator_choice")
if disc and disc != "none":
    print("--discriminator_choice"); print(str(disc))
    gdw = cfg.get("gen_disc_weight")
    if gdw is not None:
        print("--gen_disc_weight"); print(str(gdw))
# loss weights that differ from default
perc = cfg.get("perceptual_loss_weight")
if perc is not None and float(perc) != 1.5:
    print("--perceptual_loss_weight"); print(str(perc))
    print("--vae_loss_config.perceptual_loss_weight"); print(str(perc))
kl = cfg.get("kl_loss_weight")
if kl is None and isinstance(cfg.get("vae_loss_config"), dict):
    kl = cfg["vae_loss_config"].get("kl_loss_weight")
if kl is not None and float(kl) != 1e-6:
    print("--kl_loss_weight"); print(str(kl))
    print("--vae_loss_config.kl_loss_weight"); print(str(kl))
for a in args:
    print(a)
print("--data_preset"); print(cfg.get("data_preset") or "all_people_one_expression")
vin = m.get("view_in") or 2
print("--_view_in"); print(str(vin))
PY
)

run_name="xview_reeval_${ARM#paper_}"
experiment_name="${run_name}"
[[ -n "${SLURM_JOB_ID:-}" ]] && experiment_name="${run_name}__job${SLURM_JOB_ID}_t${TASK}"
MASTER_PORT=$((22000 + (${SLURM_JOB_ID:-$$} % 20000) + TASK))
export MASTER_PORT MASTER_ADDR=127.0.0.1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0

# Pull data_preset + view_in out of MODEL_ARGS
DATA_PRESET="all_people_one_expression"
VIEW_IN=2
FILTERED=()
skip=0
pending=""
for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
  if (( skip )); then
    case "$pending" in
      data_preset) DATA_PRESET="${MODEL_ARGS[$i]}" ;;
      view_in) VIEW_IN="${MODEL_ARGS[$i]}" ;;
    esac
    skip=0; pending=""; continue
  fi
  if [[ "${MODEL_ARGS[$i]}" == "--data_preset" ]]; then skip=1; pending=data_preset; continue; fi
  if [[ "${MODEL_ARGS[$i]}" == "--_view_in" ]]; then skip=1; pending=view_in; continue; fi
  FILTERED+=("${MODEL_ARGS[$i]}")
done
MODEL_ARGS=("${FILTERED[@]}")

COMMON=(
  --bucket_config "{'128px_ar1:1': {9: (1.0, 1)}}"
  --epochs 0
  --save_ckpt False
  --final_eval True
  --wandb_min_steps_before_init -1
  --eval_batch_size 1
  --vae_target_range "[-1,1]"
)

DATA_ARGS=( --data_preset "$DATA_PRESET" )
if [[ "$DATA_PRESET" == "all_people_one_expression" ]]; then
  DATA_ARGS+=(
    --dataset_presets.all_people_one_expression.expected_views "$VIEW_IN"
    --dataset_presets.all_people_one_expression.skip_mismatched_views True
    --val_dataset_presets.all_people_one_expression.expected_views "$VIEW_IN"
    --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
  )
fi
# E0 ceiling uses all_people train but same EMO-1 val — keep source preset as-is
if [[ "$DATA_PRESET" == "all_people" ]]; then
  DATA_ARGS+=(
    --val_dataset_presets.all_people_one_expression.expected_views 2
    --val_dataset_presets.all_people_one_expression.skip_mismatched_views True
  )
fi

echo "model args: ${MODEL_ARGS[*]}"
echo "data: ${DATA_ARGS[*]}  view_in=$VIEW_IN"
echo "load: $CKPT"

accelerate launch \
  --num_processes 1 --num_machines 1 --dynamo_backend no --mixed_precision bf16 \
  --main_process_port "$MASTER_PORT" \
  scripts/vae/train.py "${OPEN_SORA_ROOT}/${CONFIG}" \
  --experiment_name "$experiment_name" --wandb_expr_name "$run_name" \
  --wandb_project wan_multiview_vae_paper \
  --batch_size 8 --accumulation_steps 8 \
  --load "$CKPT" --load_optimizer False \
  "${COMMON[@]}" "${DATA_ARGS[@]}" "${MODEL_ARGS[@]}"

echo "DONE $ARM -> $experiment_name"
touch "${OPEN_SORA_ROOT}/outputs/${run_name}.DONE"
