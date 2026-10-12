#!/bin/bash
# Directional-accuracy mid-price dump + analysis (2026-08-05) — downstream forecasting
# skill at horizons {100,250,500,1000} messages, anchor vs post-trained headline picks, on
# the sealed 28-day panel universe. One GH200, one generation program (CRN across rows),
# mids only (no LOB-Bench, no panel scoring), then the CPU analysis (per-horizon sign
# accuracy, paired deltas vs anchor, day-clustered bootstrap) in the same job.
#
# Data windowing/init stays at the sealed 500/500 (wide-book alignment is only valid at
# n_cond+n_gen=1000); the generator scan runs to HORIZON_MAX=1000 and true mids past the
# window end are read straight from the staged raw book day files.
#
# Env knobs: EGG_SPECS "label=proj_dir,..." / FULL_SPECS "label=full_dir:step,..." rows,
# N_EVAL_CTX (2048 main / 64 smoke), GEN_CHUNK (512 main / 0 smoke), HORIZON_MAX,
# HORIZONS, PANEL_TIMEOUT. Universe defaults = the sealed-test panel (SPLIT_JSON
# panel_days join, wide_L500_panel28). Per-row mids npz are written as rows complete, so a
# PANEL_TIMEOUT kill still rsyncs the finished rows.
#
# PACING (measured on job 5918381, 2,048 ctx x 1,000-msg horizon, 1 GH200): fixed
# overhead 5.5 min (squashfs mounts + rsync + sleep-75 + prep + XLA compile), then
# 15.0 min PER ROW; the CPU analysis is seconds. ETA = 5.5 + 15*n_rows minutes.
# Walls below follow ETA x1.5 with a >=30 min floor:
#   1 row (split seed job)  ETA ~21 min  -> 01:00:00
#   2 rows (anchor+canary)  ETA ~36 min  -> 01:30:00
#   12 rows (single program) ETA ~3.1 h  -> 04:45:00  (this script's default)
#   smoke, 3 rows @64 ctx    ETA ~6 min  -> 00:40:00
#
# SPLIT MODE (one job per row, ~5x faster wall-clock at ~40% more GPU-h): sound because
# the window draw + rng_grid are seed-deterministic (identical contexts/keys across jobs;
# only TF32 chaos differs — unbiased rollout noise, validated cross-program on the
# compounding runs at ~1%). Anchor job: ROWS=merge_noop + one EGG_SPECS pick (carries the
# in-program merge_noop canary). Seed jobs: EGG_SPECS=<one pick> SKIP_ANCHOR=1 (in-job
# analysis is skipped — no anchor in-dir). Merge at the end:
#   python -m eggroll_gan.eval.directional_accuracy_analysis --dir runs/<anchor>,runs/<s0>,...
#
# Lustre 9-point checklist: breadcrumb-only ckpt discovery; no find/recursive ls (day farm
# is a per-job symlink farm of exactly EVAL_DAYS); ${SLURM_JOB_ID}-keyed output; all run
# output on $TMPDIR (=/tmp) with one final rsync; single job; weights+wide-book+data staged
# node-local once; no loggers; no persistent agent.
#
#SBATCH --job-name=directional_accuracy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=04:45:00
#SBATCH --output=logs/directional_accuracy_%j.out
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

N_EVAL_CTX="${N_EVAL_CTX:-2048}"; N_COND="${N_COND:-500}"; N_GEN="${N_GEN:-500}"
HORIZON_MAX="${HORIZON_MAX:-1000}"; HORIZONS="${HORIZONS:-100,250,500,1000}"
GEN_CHUNK="${GEN_CHUNK:-512}"
ROWS="${ROWS:-}"; SKIP_ANCHOR="${SKIP_ANCHOR:-}"

# Training env leaking into an eval program breaks the dataset build (manifest masks
# eval days -> abort). Same guard as the chain eval wrappers.
unset AB_SPLIT_JSON UNSEEN_MANIFEST PRETRAIN_SEEN_MANIFEST DATA_MODE DATA_NPY_DIR TRAIN_DAYS 2>/dev/null || true

WORK="${TMPDIR:-/tmp}/directional_accuracy_${SLURM_JOB_ID}"; mkdir -p "$WORK"
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
echo "[directional_accuracy] node scratch verified stable (WORK=$WORK)"

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
echo "[directional_accuracy] EVAL_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
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
      echo "[directional_accuracy] WARN: spec '$lbl' -> '$dir' has no breadcrumb — skipping row"
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
      echo "[directional_accuracy] WARN: full spec '$lbl' -> '$dir' has no breadcrumb — skipping row"
    fi
  done
fi

export CUDA_VISIBLE_DEVICES=0
PYTMP="$WORK/pytmp"; mkdir -p "$PYTMP"
STAMP='{ print strftime("[%H:%M:%S]"), $0; fflush() }'

# 4h python ceiling < 04:45 wall: a slow program is killed with 45 min of analysis+rsync
# time to spare (finished rows are already on disk, so the partial dump still lands).
TMPDIR="$PYTMP" timeout "${PANEL_TIMEOUT:-14400}" python -u -m eggroll_gan.eval.directional_accuracy --run \
  --data_dir "$FILT" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
  --out_dir "$WORK/out" "${EGG_ARGS[@]}" \
  --n_cond "$N_COND" --n_gen "$N_GEN" --horizon_max "$HORIZON_MAX" --horizons "$HORIZONS" \
  --n_eval_ctx "$N_EVAL_CTX" --wide_levels 50 \
  --wide_book_dir "$NODE_WIDE" --gen_chunk "$GEN_CHUNK" --shard off --seed 0 \
  ${ROWS:+--rows "$ROWS"} ${SKIP_ANCHOR:+--skip_anchor} \
  2>&1 | awk "$STAMP"
RC=${PIPESTATUS[0]}
echo "[directional_accuracy] generation rc=$RC"

RC_AN=1
if [ -n "$SKIP_ANCHOR" ]; then
  RC_AN=0                                       # split mode: merged analysis runs later
  echo "[directional_accuracy] split mode — analysis deferred to the merged call"
elif [ "$RC" -eq 0 ] || [ "$RC" -eq 3 ]; then   # analyze even on a merge_noop gate FAIL (rc=3)
  TMPDIR="$PYTMP" timeout "${ANALYSIS_TIMEOUT:-3600}" python -u -m eggroll_gan.eval.directional_accuracy_analysis \
    --dir "$WORK/out" --n_boot "${N_BOOT:-500}" \
    2>&1 | awk "$STAMP"
  RC_AN=${PIPESTATUS[0]}
  echo "[directional_accuracy] analysis rc=$RC_AN"
fi

DEST="$EXP/runs/directional_accuracy_${SLURM_JOB_ID}"
mkdir -p "$DEST"
rsync -a "$WORK/out/" "$DEST/" 2>/dev/null || true
chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/directional_accuracy_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[directional_accuracy] SUMMARY gen_rc=$RC analysis_rc=$RC_AN -> $DEST"
[ "$RC" -eq 0 ] && [ "$RC_AN" -ne 0 ] && exit 5
exit $RC
