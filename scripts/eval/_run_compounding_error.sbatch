#!/bin/bash
# Compounding-error token dump + analysis (2026-07-31) — per-position divergence curves,
# anchor vs post-trained picks, on the sealed 28-day panel universe. One GH200, one
# generation program (CRN across rows), tokens only (no LOB-Bench, no panel scoring), then
# the CPU analysis (per-position KL over phase alphabets, entropy normalization, cross-half
# floor, z-drift, day-clustered bootstrap) in the same job.
#
# Env knobs: EGG_SPECS "label=proj_dir,..." / FULL_SPECS "label=full_dir:step,..." rows,
# N_EVAL_CTX (4096 main / 64 smoke), GEN_CHUNK (512 main / 0 smoke), PANEL_TIMEOUT.
# Universe defaults = the sealed-test panel (SPLIT_JSON panel_days join, wide_L500_panel28).
#
# Lustre 9-point checklist: breadcrumb-only ckpt discovery; no find/recursive ls (day farm
# is a per-job symlink farm of exactly EVAL_DAYS); ${SLURM_JOB_ID}-keyed output; all run
# output on $TMPDIR (=/tmp) with one final rsync; single job; weights+wide-book+data staged
# node-local once; no loggers; no persistent agent.
#
#SBATCH --job-name=compounding_error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=09:30:00
#SBATCH --output=logs/compounding_error_%j.out
set -uo pipefail
umask 002

export TMPDIR=/tmp                      # scratch-wipe-safe; sleep-75 tripwire below

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
MAMBA="$EXP/lobmamba"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$MAMBA"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LD_LIBRARY_PATH="$CP/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CP/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CP/lib/python3.12/site-packages/nvidia/cublas/lib:$CP/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export MAMBA3_EPS_COMPAT="${MAMBA3_EPS_COMPAT:-0}"   # s28730 anchors: fixed RMSNorm
export L2_STATE_N="${L2_STATE_N:-50}"

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_googonly_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-01,2026-02}"
SPLIT_JSON="${SPLIT_JSON:-$EXP/docs/results/unseen_arm/panel_ab_split.json}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_panel28/$STOCK}"
EVAL_DAYS="${EVAL_DAYS:-$(python -c "import json;print(','.join(json.load(open('$SPLIT_JSON'))['panel_days']))")}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "FATAL: helper missing"; exit 1; }

N_EVAL_CTX="${N_EVAL_CTX:-4096}"; N_COND="${N_COND:-500}"; N_GEN="${N_GEN:-500}"
GEN_CHUNK="${GEN_CHUNK:-512}"

# Training env leaking into an eval program breaks the dataset build (manifest masks
# eval days -> abort). Same guard as the chain eval wrappers.
unset AB_SPLIT_JSON UNSEEN_MANIFEST PRETRAIN_SEEN_MANIFEST DATA_MODE DATA_NPY_DIR TRAIN_DAYS 2>/dev/null || true

WORK="${TMPDIR:-/tmp}/compounding_error_${SLURM_JOB_ID}"; mkdir -p "$WORK"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
rsync -aL "$WIDE_BOOK_DIR/" "$NODE_WIDE/"   # -L: panel28 merged dir is symlinks

# Survive the ~30-40s post-start scratch wipe: wait, verify, fail fast (resubmit on 7).
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "FATAL: node scratch wiped after start — resubmit"; exit 7; }
echo "[compounding_error] node scratch verified stable (WORK=$WORK)"

# Day farm: exactly the RTH variant of EVAL_DAYS (get_dataset uses every day in data_dir
# and FATALs on a missing wide-book match; partial farm => refuse, not a partial result).
FILT="$WORK/eval_days"; mkdir -p "$FILT"
for D in ${EVAL_DAYS//,/ }; do
  for f in "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*message_10_proc.npy \
           "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*orderbook_10_proc.npy; do
    [ -e "$f" ] && ln -sf "$f" "$FILT/"
  done
done
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NE=$(echo "$EVAL_DAYS" | tr ',' ' ' | wc -w)
echo "[compounding_error] EVAL_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE EVAL_DAYS readable in farm — refusing partial run"; exit 1; }

# Model rows: EGG_SPECS proj rows (breadcrumb check) + FULL_SPECS full-ckpt rows (staged
# node-local, own decoder head). Same contract as _run_test_eval.sbatch.
EGG_ARGS=()
if [ -n "${EGG_SPECS:-}" ]; then
  for spec in ${EGG_SPECS//,/ }; do
    lbl="${spec%%=*}"; dir="${spec#*=}"
    if [ -f "$dir/latest_checkpoint.json" ]; then
      EGG_ARGS+=(--eggroll "$lbl=$dir")
    else
      echo "[compounding_error] WARN: spec '$lbl' -> '$dir' has no breadcrumb — skipping row"
    fi
  done
  [ "${#EGG_ARGS[@]}" -gt 0 ] || { echo "FATAL: EGG_SPECS set but no valid ckpt dirs"; exit 1; }
fi
if [ -n "${FULL_SPECS:-}" ]; then
  for spec in ${FULL_SPECS//,/ }; do
    lbl="${spec%%=*}"; rest="${spec#*=}"; dir="${rest%:*}"; st="${rest##*:}"
    if [ -f "$dir/latest_checkpoint.json" ]; then
      NODE_FULL="$WORK/full_${lbl}"; mkdir -p "$NODE_FULL"
      rsync -aL "$dir/" "$NODE_FULL/" || { echo "FATAL: staging full ckpt '$lbl' failed"; exit 1; }
      EGG_ARGS+=(--full_ckpt "$lbl=$NODE_FULL:$st")
    else
      echo "[compounding_error] WARN: full spec '$lbl' -> '$dir' has no breadcrumb — skipping row"
    fi
  done
fi

export CUDA_VISIBLE_DEVICES=0
PYTMP="$WORK/pytmp"; mkdir -p "$PYTMP"
STAMP='{ print strftime("[%H:%M:%S]"), $0; fflush() }'

TMPDIR="$PYTMP" timeout "${PANEL_TIMEOUT:-30600}" python -u -m eggroll_gan.eval.compounding_error --run \
  --data_dir "$FILT" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
  --out_dir "$WORK/out" "${EGG_ARGS[@]}" \
  --n_cond "$N_COND" --n_gen "$N_GEN" --n_eval_ctx "$N_EVAL_CTX" --wide_levels 50 \
  --wide_book_dir "$NODE_WIDE" --gen_chunk "$GEN_CHUNK" --shard off --seed 0 \
  2>&1 | awk "$STAMP"
RC=${PIPESTATUS[0]}
echo "[compounding_error] generation rc=$RC"

RC_AN=1
if [ "$RC" -eq 0 ] || [ "$RC" -eq 3 ]; then     # analyze even on a merge_noop gate FAIL (rc=3)
  # NB: knob is GROUP_SPECS, not GROUPS — GROUPS is a bash special array that bash
  # re-initializes at startup, silently discarding any value exported into the job env.
  TMPDIR="$PYTMP" timeout "${ANALYSIS_TIMEOUT:-3600}" python -u -m eggroll_gan.eval.compounding_error_analysis \
    --dir "$WORK/out" --n_boot "${N_BOOT:-200}" --bins "${BINS:-20}" \
    ${GROUP_SPECS:+--groups "$GROUP_SPECS"} \
    2>&1 | awk "$STAMP"
  RC_AN=${PIPESTATUS[0]}
  echo "[compounding_error] analysis rc=$RC_AN"
fi

DEST="$EXP/runs/compounding_error_${SLURM_JOB_ID}"
mkdir -p "$DEST"
rsync -a "$WORK/out/" "$DEST/" 2>/dev/null || true
chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/compounding_error_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[compounding_error] SUMMARY gen_rc=$RC analysis_rc=$RC_AN -> $DEST"
[ "$RC" -eq 0 ] && [ "$RC_AN" -ne 0 ] && exit 5
exit $RC
