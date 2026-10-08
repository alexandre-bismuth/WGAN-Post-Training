#!/bin/bash
# Per-node body for the multi-host GRPO (policy-gradient) run — the PG twin of _run_eggroll_multinode_inner.sh.
# srun launches ONE copy on EACH node (--ntasks-per-node=1); jax.distributed.initialize() (via
# --distributed at the trainer module top) rendezvouses the processes into ONE global ('pop',) mesh over
# all GPUs. Same learned critic + d_step as the ES arm (train_grpo_head imports them from
# train_eggroll_gan_s5); only the optimizer differs (ES -> on-policy PG). Writes gated on
# jax.process_index()==0; only SLURM_PROCID 0 rsyncs to the shared DEST. NO retry (crash must propagate).
#
# Reads knobs from the environment (exported by the sbatch, propagated by srun --export=ALL):
#   SEED FC TAG ESG ESQ CRIT_BATCH NSTEPS CRITIC_INPUT FEAT_CHUNK NCTXPOOL NEVAL LR SOLVER LAMBDA
#   PG_MB PG_CHUNK LANE_TIMEOUT CKPT_SRC CKPT_STEP STOCK MONTHS WIDE_BOOK_DIR TRAIN_DAYS SHARD_DIR SQUASHFS_HELPERS
#
# GRPO vs ES arg deltas: DROP --rank --sigma --r1_gamma --kl_ref --kl_chunk --keep_step_ckpts
# --no_goodhart_stop; ADD --adv grpo --pg_microbatch --pg_chunk --goodhart_patience 999. lr is the PG
# step size (default 1e-4), NOT comparable to the ES eta. GRPO KL is always to the anchor.
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

PROC="${SLURM_PROCID:-0}"
IS_P0=$([ "$PROC" = "0" ] && echo 1 || echo 0)

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_sp500iso_fixednorm_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-01}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_2026-01/GOOG}"
TRAIN_DAYS="${TRAIN_DAYS:-2026-01-02,2026-01-05,2026-01-06,2026-01-07,2026-01-08,2026-01-09,2026-01-12,2026-01-13,2026-01-14,2026-01-15,2026-01-16,2026-01-20,2026-01-21,2026-01-22,2026-01-23,2026-01-26,2026-01-27,2026-01-28,2026-01-29,2026-01-30}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "[p$PROC] FATAL: helper missing"; exit 1; }

# Node-local, proc-keyed scratch (nodes never share a /tmp path). DEST is shared (/home); only p0 writes it.
WORK="${TMPDIR:-/tmp}/grpo_mnode_${SLURM_JOB_ID}_p${PROC}"; mkdir -p "$WORK/logs"
DEST="$EXP/runs/grpo_mnode_${SLURM_JOB_ID}"; [ "$IS_P0" = 1 ] && mkdir -p "$DEST"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"; rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"; rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/"
[ -d "$NODE_WIDE" ] || { echo "[p$PROC] FATAL: WIDE_BOOK_DIR unavailable"; exit 1; }

# Survive the ~30-40s post-start scratch wipe before launch; /tmp is not wiped. Fail fast if it was.
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "[p$PROC] FATAL: node scratch wiped after start (ckpt $( [ -f "$NODE_CKPT/latest_checkpoint.json" ] && echo ok || echo GONE ), farm $( [ -d "$INFER_DATA_DIR_NODE" ] && echo ok || echo GONE ), wide $( [ -d "$NODE_WIDE" ] && echo ok || echo GONE )) — resubmit"; exit 7; }
echo "[p$PROC] node scratch verified stable (WORK=$WORK)"

# TRAIN_DAYS filter (RTH variant of the days we have wide-book init for); refuse a partial mount.
FILT="$WORK/train_days"; mkdir -p "$FILT"
for D in ${TRAIN_DAYS//,/ }; do
  for f in "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*message_10_proc.npy \
           "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*orderbook_10_proc.npy; do
    [ -e "$f" ] && ln -sf "$f" "$FILT/"
  done
done
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NE=$(echo "$TRAIN_DAYS" | tr ',' ' ' | wc -w)
echo "[p$PROC] TRAIN_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "[p$PROC] FATAL: only $NF/$NE TRAIN_DAYS readable in farm — refusing partial run"; exit 1; }
INFER_DATA_DIR_NODE="$FILT"

SEED="${SEED:-0}"; FC="${FC:-none}"; TAG="${TAG:-grpo_s${SEED}}"
# Context pool MUST be >= Q. Divisibility: G*Q, n_eval_ctx, n_ctx_pool, n_pool=(n_ctx_pool+n_eval_ctx)
# all % n_dev == 0 for BOTH 8 (2-node) and 16 (4-node): defaults 16*256=4096, 256, 512, 768 all %16=0.
NCTXPOOL="${NCTXPOOL:-512}"
if [ "$NCTXPOOL" -lt "${ESQ:-256}" ]; then echo "[p$PROC] FATAL: NCTXPOOL($NCTXPOOL) < Q(${ESQ:-256})"; exit 1; fi
OUT="$WORK/$TAG"; LG="$WORK/logs/$TAG.log"
# GRPO command: same learned critic; PG optimizer. lr=PG step (1e-4 natural). goodhart OFF (noisy composite).
GRPO_COMMON="--run --distributed --scope proj --critic_input ${CRITIC_INPUT:-learned} --adv grpo --solver ${SOLVER:-adamw} --lr ${LR:-1e-4} \
  --G ${ESG:-16} --Q ${ESQ:-256} --crit_batch ${CRIT_BATCH:-256} --pg_microbatch ${PG_MB:-32} --pg_chunk ${PG_CHUNK:-2} \
  --n_cond 500 --n_gen 500 --n_g_steps ${NSTEPS:-50} --n_d_steps 5 \
  --wide_book_dir $NODE_WIDE --wide_levels 500 \
  --kl_coef ${LAMBDA:-0.1} --kl_anneal const --feat_chunk ${FEAT_CHUNK:-4} \
  --n_ctx_pool $NCTXPOOL --n_eval_ctx ${NEVAL:-256} --pool_scope cont --pool_refresh_every 10 \
  --eval_every 5 --ckpt_every 5 --goodhart_patience 999 --shard on \
  --data_dir $INFER_DATA_DIR_NODE --ckpt_dir $NODE_CKPT --ckpt_step $CKPT_STEP"

# 15-min safety rsync — ONLY p0. Periodic mirror stays LIGHT (skip step-keyed ckpts); the FINAL sync
# keeps them so posthoc_select can re-select over the kept steps (GRPO save_ckpt writes step dirs at
# ckpt_every, same as ES).
sync_one()   { rsync -a --exclude='step[0-9]*' "$WORK/logs" "$WORK/$TAG" "$DEST/" 2>/dev/null; }
sync_final() { rsync -a "$WORK/logs" "$WORK/$TAG" "$DEST/" 2>/dev/null; }
SYNC=""
if [ "$IS_P0" = 1 ]; then ( while true; do sleep 900; sync_one; done ) & SYNC=$!; fi

echo "[p$PROC] launching $TAG (seed=$SEED fc=$FC adv=grpo G=${ESG:-16} Q=${ESQ:-256} lam=${LAMBDA:-0.1} crit_batch=${CRIT_BATCH:-256} shard=on dist=on n_gpu_local=$(nvidia-smi -L 2>/dev/null | wc -l))"
timeout ${LANE_TIMEOUT:-27000} python -u -m eggroll_gan.training.train_grpo_head \
  $GRPO_COMMON --seed "$SEED" --fitness_control "$FC" --out_dir "$OUT" \
  > >(awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' > "$LG") 2>&1
RC=$?
echo "[p$PROC] $TAG rc=$RC"

[ -n "$SYNC" ] && { kill "$SYNC" 2>/dev/null || true; }
if [ "$IS_P0" = 1 ]; then
  sync_final || true
  chmod -R g+rwX "$DEST" 2>/dev/null || true
fi
chmod g+rw "$EXP/logs/grpo_mnode_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "==== [p$PROC] $TAG tail (rc=$RC) ===="
tail -150 "$LG" 2>/dev/null
echo "[p$PROC] SUMMARY $TAG:$RC -> $DEST"
exit "$RC"
