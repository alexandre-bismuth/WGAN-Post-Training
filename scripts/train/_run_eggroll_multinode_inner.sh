#!/bin/bash
# Per-node body for the 2-node (8-GPU) multi-host S5b run. srun launches ONE copy of this on EACH
# node (--ntasks-per-node=1); jax.distributed.initialize() (via --distributed, fired at the trainer
# module top) rendezvouses the 2 processes into ONE global ('pop',) device mesh over all 8 GPUs.
#
# Each process: (1) stages its OWN node-local /tmp farm + ckpt + wide-book (no cross-node race),
# (2) launches the trainer with --shard on --distributed. The trainer gates ALL filesystem writes
# on jax.process_index()==0 (p0), so only SLURM_PROCID 0 holds real outputs and is the ONLY proc
# that rsyncs to the shared DEST. NO retry here: a crash must propagate (srun --kill-on-bad-exit)
# rather than leave one process hung on the next collective.
#
# Reads knobs from the environment (exported by the sbatch, propagated by srun --export=ALL):
#   SEED FC TAG ESG ESQ CRIT_BATCH R1G NSTEPS CRITIC_INPUT FEAT_CHUNK NCTXPOOL NEVAL LR SOLVER
#   LANE_TIMEOUT CKPT_SRC CKPT_STEP STOCK MONTHS WIDE_BOOK_DIR TRAIN_DAYS SHARD_DIR SQUASHFS_HELPERS
set -uo pipefail
umask 002
export TMPDIR=/tmp

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
MAMBA="$EXP/lobmamba"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$MAMBA"
# Overridable (gate j5489734 OOM, 2026-07-04): the ens5 critic d-step wants a single
# 38.5GiB buffer (exactly 5/3 x the proven ens3 ~23GiB) and with grow-on-demand BFC
# regions (PREALLOCATE=false) the request failed while the pool map was only ~1/3 full
# -> region fragmentation, not raw capacity. The unseen-arm launcher sets
# XLA_PREALLOCATE=true (one arena up-front; BFC defrags internally). Allocator-only:
# zero numeric change.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.9}"
export LD_LIBRARY_PATH="$CP/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CP/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CP/lib/python3.12/site-packages/nvidia/cublas/lib:$CP/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export MAMBA3_EPS_COMPAT=0

PROC="${SLURM_PROCID:-0}"
IS_P0=$([ "$PROC" = "0" ] && echo 1 || echo 0)

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_sp500iso_fixednorm_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-01}"
# MULTI-TICKER post-training (2026-07-02): set MULTI_TICKERS="GOOG,AAPL,..." to train on a
# pooled corpus. Farm = the helper's SP500 combined mode (all tickers, TICKER_DATE prefixes)
# filtered to TICKERS x TRAIN_DAYS; wide books = per-ticker npz snapshot dirs merged flat
# (get_dataset pairs them TICKER-AWARE since commit 78e4465). WIDE_BOOK_PARENT holds one
# subdir per ticker (L500 npz snapshots — width 500 is load-bearing).
MULTI_TICKERS="${MULTI_TICKERS:-}"
WIDE_BOOK_PARENT="${WIDE_BOOK_PARENT:-$EXP/data/wide_L500_2026-01}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_2026-01/GOOG}"
TRAIN_DAYS="${TRAIN_DAYS:-2026-01-02,2026-01-05,2026-01-06,2026-01-07,2026-01-08,2026-01-09,2026-01-12,2026-01-13,2026-01-14,2026-01-15,2026-01-16,2026-01-20,2026-01-21,2026-01-22,2026-01-23,2026-01-26,2026-01-27,2026-01-28,2026-01-29,2026-01-30}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "[p$PROC] FATAL: helper missing"; exit 1; }

# Node-local, proc-keyed scratch (two nodes never share a /tmp path). Only p0 writes the shared dests.
# QUOTA SPLIT (2026-07-02): the trainer out_dir ($WORK/$TAG) is pure checkpoint data (step00XX/,
# latest s5_*.msgpack, latest_checkpoint.json breadcrumb) -> it rsyncs to the Lustre CKPT storage
# dir (post_training_GAN — checkpoints ONLY, nothing else lives there). Home DEST keeps logs/ + a
# tiny copy of the breadcrumb (history incl. cross_entropy) for curves. The trainer never touches
# Lustre directly (everything lands on /tmp first; Lustre only sees p0's rsync).
WORK="${TMPDIR:-/tmp}/s5b_mnode_${SLURM_JOB_ID}_p${PROC}"; mkdir -p "$WORK/logs"
DEST="$EXP/runs/s5b_mnode_${SLURM_JOB_ID}"
CKPT_ROOT="${CKPT_ROOT:-/lustre/projects/public/shared/post_training_GAN}"
CKPT_DEST="$CKPT_ROOT/s5b_mnode_${SLURM_JOB_ID}"
[ "$IS_P0" = 1 ] && mkdir -p "$DEST" "$CKPT_DEST"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
# DATA_MODE=lustre_npy (unseen100 arm): the corpus is the pretraining data_npy dir
# (flat, fully-decompressed .npy, canonical RTH names) — read it directly from
# Lustre via per-day symlinks; no squashfs mounts, no /tmp staging of the farm.
# Pool refreshes read ~135 MB per refresh (512 windows x 1000 msgs) — Lustre-polite.
DATA_MODE="${DATA_MODE:-squashfs}"
if [ "$DATA_MODE" = "lustre_npy" ]; then
  DATA_NPY_DIR="${DATA_NPY_DIR:-/lustre/projects/shared/quant/Mamba3_GOOG_pretraining_runs/data_npy/GOOG}"
  [ -d "$DATA_NPY_DIR" ] || { echo "[p$PROC] FATAL: DATA_NPY_DIR missing"; exit 1; }
  INFER_DATA_DIR_NODE="$DATA_NPY_DIR"
elif [ -n "$MULTI_TICKERS" ]; then
  infer_squashfs_setup "SP500" "$MONTHS" "$SHARD_DIR"   # combined farm, all tickers
else
  infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
fi
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"; rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
if [ -n "$MULTI_TICKERS" ]; then
  for T in ${MULTI_TICKERS//,/ }; do
    [ -d "$WIDE_BOOK_PARENT/$T" ] || { echo "[p$PROC] FATAL: no wide-book dir for $T under $WIDE_BOOK_PARENT"; exit 1; }
    rsync -a "$WIDE_BOOK_PARENT/$T/" "$NODE_WIDE/"       # flat merge; names carry TICKER_DATE
  done
else
  rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/"
fi
[ -d "$NODE_WIDE" ] || { echo "[p$PROC] FATAL: WIDE_BOOK_DIR unavailable"; exit 1; }

# Survive the ~30-40s post-start scratch wipe before launch; /tmp is not wiped. Fail fast (cheap) if it was.
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "[p$PROC] FATAL: node scratch wiped after start (ckpt $( [ -f "$NODE_CKPT/latest_checkpoint.json" ] && echo ok || echo GONE ), farm $( [ -d "$INFER_DATA_DIR_NODE" ] && echo ok || echo GONE ), wide $( [ -d "$NODE_WIDE" ] && echo ok || echo GONE )) — resubmit"; exit 7; }
echo "[p$PROC] node scratch verified stable (WORK=$WORK)"

# TRAIN_DAYS x TICKERS filter (RTH variant of the days we have wide-book init for);
# refuse a partial mount. Single-ticker: patterns match only $STOCK's files (canonical
# names are TICKER_DATE-prefixed). Multi-ticker: require EVERY (ticker, day) pair.
FILT="$WORK/train_days"; mkdir -p "$FILT"
TICKS="${MULTI_TICKERS:-$STOCK}"
if [ "$DATA_MODE" = "lustre_npy" ]; then
  # single readdir of the Lustre dir (Lustre-polite: no per-day glob scans),
  # then in-shell filename matching against TRAIN_DAYS x TICKS.
  ALLF=$(ls "$INFER_DATA_DIR_NODE")
  for T in ${TICKS//,/ }; do
    for D in ${TRAIN_DAYS//,/ }; do
      for f in $(echo "$ALLF" | grep -E "^${T}_.*${D}.*34200000_57600000.*(message|orderbook)_10_proc\.npy$"); do
        ln -sf "$INFER_DATA_DIR_NODE/$f" "$FILT/"
      done
    done
  done
else
  for T in ${TICKS//,/ }; do
    for D in ${TRAIN_DAYS//,/ }; do
      for f in "$INFER_DATA_DIR_NODE"/${T}_*"$D"*34200000_57600000*message_10_proc.npy \
               "$INFER_DATA_DIR_NODE"/${T}_*"$D"*34200000_57600000*orderbook_10_proc.npy; do
        [ -e "$f" ] && ln -sf "$f" "$FILT/"
      done
    done
  done
fi
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NT=$(echo "$TICKS" | tr ',' ' ' | wc -w)
NE=$(( $(echo "$TRAIN_DAYS" | tr ',' ' ' | wc -w) * NT ))
echo "[p$PROC] TRAIN_DAYS filter -> $NF/$NE RTH ticker-day(s) ($NT ticker(s)) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "[p$PROC] FATAL: only $NF/$NE ticker-days readable in farm — refusing partial run"; exit 1; }
INFER_DATA_DIR_NODE="$FILT"

SEED="${SEED:-0}"; FC="${FC:-none}"; TAG="${TAG:-mn_s${SEED}}"
# Context pool MUST be >= Q. Divisibility (global n_dev=8): n_ctx_pool % 8 == 0 AND n_pool=(n_ctx_pool+
# n_eval_ctx) % 8 == 0 AND G*Q % 8 == 0. Defaults: 256%8=0, 256+128=384%8=0, 16*128=2048%8=0.
NCTXPOOL="${NCTXPOOL:-256}"
if [ "$NCTXPOOL" -lt "${ESQ:-128}" ]; then echo "[p$PROC] FATAL: NCTXPOOL($NCTXPOOL) < Q(${ESQ:-128})"; exit 1; fi
OUT="$WORK/$TAG"; LG="$WORK/logs/$TAG.log"
ES_COMMON="--run --distributed --scope proj --critic_input ${CRITIC_INPUT:-learned} --rank 4 --sigma ${SIGMA:-0.003} --lr ${LR:-0.0005} --solver ${SOLVER:-adamw} \
  --G ${ESG:-16} --Q ${ESQ:-128} --crit_batch ${CRIT_BATCH:-256} --n_cond 500 --n_gen 500 --n_g_steps ${NSTEPS:-50} --n_d_steps 5 \
  --wide_book_dir $NODE_WIDE --wide_levels 500 \
  --kl_coef 0.1 --kl_ref anchor --kl_anneal const --kl_chunk 1 --feat_chunk ${FEAT_CHUNK:-4} \
  --n_ctx_pool $NCTXPOOL --n_eval_ctx ${NEVAL:-128} --pool_scope cont --pool_refresh_every ${POOL_REFRESH:-10} \
  --ens_size ${ENS_SIZE:-3} \
  --eval_every ${EVAL_EVERY:-5} --ckpt_every ${CKPT_EVERY:-5} --no_goodhart_stop --keep_step_ckpts --shard on \
  --enc_proj_dim ${ENC_PROJ_DIM:-0} --enc_whiten ${ENC_WHITEN:-std} \
  --data_dir $INFER_DATA_DIR_NODE --ckpt_dir $NODE_CKPT --ckpt_step $CKPT_STEP"

# 15-min safety rsync — ONLY p0 (the only process the trainer lets write checkpoints).
# logs -> home DEST; checkpoints (the whole out_dir) -> Lustre CKPT_DEST. The periodic ckpt mirror
# is cheap insurance against a node loss late in the run (only NEW step dirs transfer each pass;
# a file caught mid-write is re-copied next pass and finalised by sync_final).
sync_one()   { rsync -a "$WORK/logs" "$DEST/" 2>/dev/null
               rsync -a "$WORK/$TAG" "$CKPT_DEST/" 2>/dev/null; }
sync_final() { sync_one
               cp "$WORK/$TAG/latest_checkpoint.json" "$DEST/eval_history_${TAG}.json" 2>/dev/null; }
SYNC=""
if [ "$IS_P0" = 1 ]; then ( while true; do sleep 900; sync_one; done ) & SYNC=$!; fi

echo "[p$PROC] launching $TAG (seed=$SEED fc=$FC G=${ESG:-16} Q=${ESQ:-128} crit_batch=${CRIT_BATCH:-256} shard=on dist=on n_gpu_local=$(nvidia-smi -L 2>/dev/null | wc -l))"
# LANE_TIMEOUT default 9600s (2h40m) < the 3h sbatch wall: on overrun the trainer gets SIGTERM'd
# with ~20min left so sync_final still ships every step ckpt (a wall-kill would lose /tmp entirely).
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
chmod g+rw "$EXP/logs/s5b_mnode_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "==== [p$PROC] $TAG tail (rc=$RC) ===="
tail -10 "$LG" 2>/dev/null
echo "[p$PROC] SUMMARY $TAG:$RC -> logs:$DEST ckpts:$CKPT_DEST/$TAG"
# Propagate the real RC so srun --kill-on-bad-exit tears down BOTH processes on a crash (no hang on
# the next jax.distributed collective). Timeout (124) on both near-simultaneously ends the job cleanly.
exit "$RC"
