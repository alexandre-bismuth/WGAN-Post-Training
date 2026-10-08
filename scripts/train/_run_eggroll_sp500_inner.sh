#!/bin/bash
# Per-node body for the SP500 corpus-matched 2-node (8-GPU) multi-host run.
# Same process model as _run_eggroll_multinode_inner.sh (srun launches ONE
# copy per node; jax.distributed rendezvous into one global ('pop',) mesh;
# only jax.process_index()==0 writes). Data staging differs and lives in
# _sp500_stage_lib.sh (shared with the data-gate job):
#   - L10: pooled SP500 combined farm filtered to TICKS x TRAIN_DAYS
#   - wide books: single npz squashfs image, node-local staged + mounted,
#     flat symlink farm (get_dataset pairs TICKER-AWARE since 78e4465)
#   - completeness gate RELAXED: >= SP500_MIN_COVERAGE (default 0.95) of
#     ticker-days present (missing list printed); incomplete pairs dropped.
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
export MAMBA3_EPS_COMPAT=0
PYX=$CP/bin/python3

PROC="${SLURM_PROCID:-0}"
IS_P0=$([ "$PROC" = "0" ] && echo 1 || echo 0)

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_sp500iso_fixednorm_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
MONTHS="${MONTHS:-2026-01}"
TRAIN_DAYS="${TRAIN_DAYS:-2026-01-02,2026-01-05,2026-01-06,2026-01-07,2026-01-08,2026-01-09,2026-01-12,2026-01-13,2026-01-14,2026-01-15,2026-01-16,2026-01-20,2026-01-21,2026-01-22,2026-01-23,2026-01-26,2026-01-27,2026-01-28,2026-01-29,2026-01-30}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
WIDE_NPZ_IMAGE="${WIDE_NPZ_IMAGE:-/lustre/projects/public/shared/wide_L500_npz_2026-01.squashfs}"
SP500_TICKER_FILE="${SP500_TICKER_FILE:-$EXP/tools/data_build/sp500/train_tickers_2026-01.txt}"
STAGE_LIB="$EXP/scripts/train/_sp500_stage_lib.sh"
[ -f "$SQUASHFS_HELPERS" ] || { echo "[p$PROC] FATAL: helper missing"; exit 1; }
[ -f "$STAGE_LIB" ] || { echo "[p$PROC] FATAL: stage lib missing"; exit 1; }

# Ticker list: env SP500_TICKERS (comma) overrides the file (one per line or
# space/comma separated).
if [ -n "${SP500_TICKERS:-}" ]; then
  TICKS="$SP500_TICKERS"
else
  [ -f "$SP500_TICKER_FILE" ] || { echo "[p$PROC] FATAL: ticker file missing: $SP500_TICKER_FILE"; exit 1; }
  TICKS=$(tr ',\n' '  ' < "$SP500_TICKER_FILE" | xargs | tr ' ' ',')
fi
NT=$(echo "$TICKS" | tr ',' ' ' | wc -w)

# Node-local, proc-keyed scratch. Only p0 writes the shared dests.
# QUOTA SPLIT: trainer out_dir -> Lustre CKPT storage (post_training_GAN,
# checkpoints ONLY); home DEST keeps logs + eval-history breadcrumb copy.
WORK="${TMPDIR:-/tmp}/sp500_mnode_${SLURM_JOB_ID}_p${PROC}"; mkdir -p "$WORK/logs"
DEST="$EXP/runs/sp500_mnode_${SLURM_JOB_ID}"
CKPT_ROOT="${CKPT_ROOT:-/lustre/projects/public/shared/post_training_GAN}"
CKPT_DEST="$CKPT_ROOT/sp500_mnode_${SLURM_JOB_ID}"
[ "$IS_P0" = 1 ] && mkdir -p "$DEST" "$CKPT_DEST"

source "$SQUASHFS_HELPERS"
source "$STAGE_LIB"
trap 'stage_sp500_cleanup || true; infer_squashfs_cleanup || true' EXIT

echo "[p$PROC] staging: $NT tickers x $(echo "$TRAIN_DAYS" | tr ',' ' ' | wc -w) days, image=$(basename "$WIDE_NPZ_IMAGE")"
stage_sp500_data "$WORK" || { echo "[p$PROC] FATAL: SP500 staging failed"; exit 1; }
NODE_WIDE="$SP500_WIDE"

NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"; rsync -a "$CKPT_SRC/" "$NODE_CKPT/"

# Survive the ~30-40s post-start scratch wipe; /tmp is not wiped. Fail fast if it was.
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$SP500_FILT" ] && [ -d "$NODE_WIDE" ] \
  && mountpoint -q "$SP500_WIDE_MNT"; } \
  || { echo "[p$PROC] FATAL: node scratch wiped after start — resubmit"; exit 7; }
echo "[p$PROC] node scratch verified stable (WORK=$WORK, pairs=$SP500_N_PAIRS/$SP500_N_EXPECT)"
INFER_DATA_DIR_NODE="$SP500_FILT"

SEED="${SEED:-0}"; FC="${FC:-none}"; TAG="${TAG:-sp500_s${SEED}}"
NCTXPOOL="${NCTXPOOL:-256}"
if [ "$NCTXPOOL" -lt "${ESQ:-128}" ]; then echo "[p$PROC] FATAL: NCTXPOOL($NCTXPOOL) < Q(${ESQ:-128})"; exit 1; fi
OUT="$WORK/$TAG"; LG="$WORK/logs/$TAG.log"
ES_COMMON="--run --distributed --scope proj --critic_input ${CRITIC_INPUT:-learned} --rank 4 --sigma 0.003 --lr ${LR:-0.0005} --solver ${SOLVER:-adamw} \
  --G ${ESG:-16} --Q ${ESQ:-128} --crit_batch ${CRIT_BATCH:-256} --n_cond 500 --n_gen 500 --n_g_steps ${NSTEPS:-50} --n_d_steps 5 \
  --wide_book_dir $NODE_WIDE --wide_levels 500 \
  --kl_coef 0.1 --kl_ref anchor --kl_anneal const --kl_chunk 1 --feat_chunk ${FEAT_CHUNK:-4} \
  --n_ctx_pool $NCTXPOOL --n_eval_ctx ${NEVAL:-128} --pool_scope cont --pool_refresh_every 10 \
  --eval_every 5 --ckpt_every 5 --no_goodhart_stop --keep_step_ckpts --shard on \
  --enc_proj_dim ${ENC_PROJ_DIM:-8} --enc_whiten ${ENC_WHITEN:-robust} \
  --data_dir $INFER_DATA_DIR_NODE --ckpt_dir $NODE_CKPT --ckpt_step $CKPT_STEP"

# 15-min safety rsync — ONLY p0. logs -> home DEST; checkpoints -> Lustre CKPT_DEST.
sync_one()   { rsync -a "$WORK/logs" "$DEST/" 2>/dev/null
               rsync -a "$WORK/$TAG" "$CKPT_DEST/" 2>/dev/null; }
sync_final() { sync_one
               cp "$WORK/$TAG/latest_checkpoint.json" "$DEST/eval_history_${TAG}.json" 2>/dev/null; }
SYNC=""
if [ "$IS_P0" = 1 ]; then ( while true; do sleep 900; sync_one; done ) & SYNC=$!; fi

echo "[p$PROC] launching $TAG (seed=$SEED fc=$FC tickers=$NT G=${ESG:-16} Q=${ESQ:-128} shard=on dist=on n_gpu_local=$(nvidia-smi -L 2>/dev/null | wc -l))"
# LANE_TIMEOUT default 9600s < the 3h wall: trainer gets SIGTERM'd with ~20min
# left so sync_final still ships every step ckpt.
timeout ${LANE_TIMEOUT:-9600} python -u -m eggroll_gan.training.train_eggroll_gan_s5 \
  $ES_COMMON --seed "$SEED" --fitness_control "$FC" --out_dir "$OUT" --r1_gamma "${R1G:-10}" \
  > >(awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' > "$LG") 2>&1
RC=$?
echo "[p$PROC] $TAG rc=$RC"

[ -n "$SYNC" ] && { kill "$SYNC" 2>/dev/null || true; }
if [ "$IS_P0" = 1 ]; then
  sync_final || true
  chmod -R g+rwX "$DEST" 2>/dev/null || true
  chmod -R g+rwX "$CKPT_DEST" 2>/dev/null || true
fi
chmod g+rw "$EXP/logs/sp500_mnode_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "==== [p$PROC] $TAG tail (rc=$RC) ===="
tail -10 "$LG" 2>/dev/null
echo "[p$PROC] SUMMARY $TAG:$RC -> logs:$DEST ckpts:$CKPT_DEST/$TAG"
exit "$RC"
