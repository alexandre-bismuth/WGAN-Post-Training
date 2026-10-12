#!/bin/bash
# Held-out generator CE across the unseen-arm ES step checkpoints (gen_ce_curve.py).
# One GPU: teacher-forced forward over N fixed 500-msg windows for the anchor + every
# step0005..0050 proj checkpoint of the 3 unseen lanes (31 rows). No simulator, no wide
# book — likelihood only. Output: a small JSON on home (per-window CE + KL per row).
#
# Lustre checklist: breadcrumb-only ckpt discovery (exact stepNNNN paths passed in);
# squashfs day staging on $TMPDIR; ${SLURM_JOB_ID}-keyed output; no loggers; no agents.
#
#SBATCH --job-name=gen_ce_curve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=03:00:00
#SBATCH --output=logs/gen_ce_curve_%j.out
set -uo pipefail
umask 002
export TMPDIR=/tmp

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
MAMBA="$EXP/lobmamba"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$MAMBA"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LD_LIBRARY_PATH="$CP/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CP/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CP/lib/python3.12/site-packages/nvidia/cublas/lib:$CP/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export MAMBA3_EPS_COMPAT=0    # s28730 anchors: fixed-norm weights, compat must be OFF

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_googonly_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-01,2026-02}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
CKPT_ROOT="${CKPT_ROOT:-/lustre/projects/public/shared/post_training_GAN}"
LANES="${LANES:-s0=$CKPT_ROOT/s5b_mnode_5493387/unseen_s0,s1=$CKPT_ROOT/s5b_mnode_5493388/unseen_s1,s2=$CKPT_ROOT/s5b_mnode_5493389/unseen_s2}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "FATAL: helper missing"; exit 1; }
[ -f "$CKPT_SRC/latest_checkpoint.json" ] || { echo "FATAL: anchor breadcrumb missing"; exit 1; }

WORK="/tmp/gen_ce_${SLURM_JOB_ID}"; mkdir -p "$WORK"
DEST="$EXP/runs/gen_ce_curve_${SLURM_JOB_ID}"; mkdir -p "$DEST"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"

# Survive the ~30-40s post-start scratch wipe (see _run_test_eval.sbatch header).
sleep 75
if [ ! -f "$NODE_CKPT/latest_checkpoint.json" ] || [ ! -d "$INFER_DATA_DIR_NODE" ]; then
  echo "[gen_ce] WARN: node scratch wiped after start — restaging once"
  infer_squashfs_cleanup || true
  mkdir -p "$WORK" "$NODE_CKPT" || { echo "FATAL: /tmp unusable"; exit 7; }
  infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR" || { echo "FATAL: restage failed"; exit 7; }
  rsync -a "$CKPT_SRC/" "$NODE_CKPT/" || { echo "FATAL: restage ckpt failed"; exit 7; }
  sleep 75
  { [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ]; } \
    || { echo "FATAL: scratch wiped AGAIN"; exit 7; }
fi

# Eval-day filter: same RTH-variant staging as _run_test_eval.sbatch (panel days only).
if [ -n "${EVAL_DAYS:-}" ]; then
  FILT="$WORK/eval_days"; mkdir -p "$FILT"
  for DAY in ${EVAL_DAYS//,/ }; do
    for f in "$INFER_DATA_DIR_NODE"/*"$DAY"*34200000_57600000*message_10_proc.npy \
             "$INFER_DATA_DIR_NODE"/*"$DAY"*34200000_57600000*orderbook_10_proc.npy; do
      [ -e "$f" ] && ln -sf "$f" "$FILT/"
    done
  done
  NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
  NE=$(echo "$EVAL_DAYS" | tr ',' ' ' | wc -w)
  echo "[gen_ce] EVAL_DAYS filter -> $NF/$NE RTH day(s)"
  [ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE EVAL_DAYS readable — refusing partial eval"; exit 1; }
  INFER_DATA_DIR_NODE="$FILT"
fi

export CUDA_VISIBLE_DEVICES=0
PYTMP="$WORK/pytmp"; mkdir -p "$PYTMP"
TMPDIR="$PYTMP" timeout "${CE_TIMEOUT:-9000}" python -u -m eggroll_gan.eval.gen_ce_curve --run \
  --data_dir "$INFER_DATA_DIR_NODE" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
  --lanes "$LANES" --steps "${STEPS:-5,10,15,20,25,30,35,40,45,50}" \
  --n_windows "${N_WINDOWS:-512}" --chunk "${CE_CHUNK:-16}" \
  --out "$DEST/gen_ce_curve.json" \
  2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
RC=${PIPESTATUS[0]}

chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/gen_ce_curve_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[gen_ce] SUMMARY rc=$RC -> $DEST"
exit $RC
