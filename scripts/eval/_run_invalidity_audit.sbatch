#!/bin/bash
# Invalidity audit (2026-06-25) — the TRUE proportion of generated messages that cannot be matched on
# a LOB, for the pretrained anchor (mamba3_78M_sp500iso_fixednorm_s28730, eps-compat OFF). Replaces the
# mislabelled `num_errors` (a top-N "unchanged" no-op count, NOT invalidity) with an ENGINE-TRUTHFUL
# per-message verdict read off the JAX-LOB engine's actual effect (resting-qty removed for cancels,
# trade qty for executions). Runs the SAME verdict on the REAL continuation as a 0-baseline gate.
#
# One GH200, ONE short generation pass. Deep L500 init (REQUIRED: a shallow init re-introduces the very
# top-of-book-invisibility confound that corrupts num_errors). Held-out month 2026-02 (out-of-sample).
#
# Lustre 9-point checklist: no find/recursive ls (data_dir is a per-job symlink farm of exactly the 8
# RTH days); output keyed by ${SLURM_JOB_ID}; all run output on $TMPDIR (=/tmp, wipe-safe) with one
# final rsync; SINGLE job; weights + wide-book + data staged node-local once; breadcrumb ckpt load;
# no wandb/tensorboard; no persistent agent.
#
#SBATCH --job-name=invalidity_audit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=00:50:00
#SBATCH --output=logs/invalidity_audit_%j.out
set -uo pipefail
umask 002

export TMPDIR=/tmp                      # scratch-wipe-safe (the /local/user regression); sleep-75 tripwire below

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
MAMBA="$EXP/lobmamba"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$MAMBA"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LD_LIBRARY_PATH="$CP/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CP/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CP/lib/python3.12/site-packages/nvidia/cublas/lib:$CP/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export MAMBA3_EPS_COMPAT=0               # s28730 anchor: fixed RMSNorm — set unconditionally
export L2_STATE_N=500                    # full-depth Tier-1 no-op cross-check (verdict itself is depth-independent)

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_sp500iso_fixednorm_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-02}"
# The 8 held-out RTH days we have clean L500 init for (data/wide_L500_2026-02/GOOG).
EVAL_DAYS="${EVAL_DAYS:-2026-02-02,2026-02-03,2026-02-04,2026-02-05,2026-02-06,2026-02-09,2026-02-10,2026-02-11}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_2026-02/GOOG}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "FATAL: helper missing"; exit 1; }

N_POP="${N_POP:-64}"; N_COND="${N_COND:-500}"; N_GEN="${N_GEN:-500}"

WORK="${TMPDIR:-/tmp}/invalidity_audit_${SLURM_JOB_ID}"; mkdir -p "$WORK"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/"

# Survive the ~30-40s post-start scratch wipe: wait it out, verify scratch survived (fail-fast tripwire).
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "FATAL: node scratch wiped after start — resubmit"; exit 7; }
echo "[invalidity_audit] node scratch verified stable (WORK=$WORK)"

# Build a data_dir of EXACTLY the RTH variant of the 8 days we have L500 init for (get_dataset uses every
# day in data_dir and FATALs on a missing wide-book match). Mirrors the eval EVAL_DAYS filter.
FILT="$WORK/eval_days"; mkdir -p "$FILT"
for D in ${EVAL_DAYS//,/ }; do
  for f in "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*message_10_proc.npy \
           "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*orderbook_10_proc.npy; do
    [ -e "$f" ] && ln -sf "$f" "$FILT/"
  done
done
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NE=$(echo "$EVAL_DAYS" | tr ',' ' ' | wc -w)
echo "[invalidity_audit] EVAL_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE EVAL_DAYS readable in farm — refusing partial audit"; exit 1; }

export CUDA_VISIBLE_DEVICES=0
STAMP='{ print strftime("[%H:%M:%S]"), $0; fflush() }'

OUT_JSON="$WORK/invalidity.json"
timeout 2700 python -u -m eggroll_gan.eval.invalidity_audit --run \
  --data_dir "$FILT" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
  --wide_book_dir "$NODE_WIDE" --wide_levels 500 \
  --n_pop "$N_POP" --n_cond "$N_COND" --n_gen "$N_GEN" --seed 0 \
  --out "$OUT_JSON" 2>&1 | awk "$STAMP"
RC=${PIPESTATUS[0]}
echo "[invalidity_audit] audit rc=$RC"

# persist to Lustre (one rsync at the end)
DEST="$EXP/runs/invalidity_audit_${SLURM_JOB_ID}"
mkdir -p "$DEST"
rsync -a "$OUT_JSON" "$DEST/" 2>/dev/null || true
chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/invalidity_audit_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[invalidity_audit] DONE rc=$RC -> $DEST"
exit $RC
