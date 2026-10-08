#!/bin/bash
# TRUE CORPUS-MATCHED SP500 post-training (2026-07-02): the 2-node 8-GPU
# sharded ES run on the FULL s28730 pretraining corpus for 2026-01 —
# pooled L10 combined farm (all tickers) + the fleet-built L500 snapshot
# npz image. Identical process model to _run_eggroll_multinode.sbatch
# (1 task/node, jax.distributed via hostname coordinator, rank-0-gated
# writes); only the data staging differs (see _sp500_stage_lib.sh).
#
# Launch (after the data-gate job passes):
#   SEED=0 TAG=sp500_s0 ENC_PROJ_DIM=8 ENC_WHITEN=robust \
#     sbatch -t 03:00:00 scripts/train/_run_eggroll_sp500.sbatch
#   FC=shuffle TAG=sp500_shuf0 ... for the shuffle-null control.
#
# Divisibility (n_dev=8): G*Q=2048%8=0; n_eval_ctx=128%8=0; NCTXPOOL=256%8=0,
# n_pool=384%8=0; feat_chunk=4 | per-shard 256; kl_chunk=1.
#
# Lustre 9-point: breadcrumb resume; no find/recursive ls; ${SLURM_JOB_ID}-
# keyed dirs; output on /tmp + rsync; weights + npz image staged node-local
# once PER NODE (single sequential Lustre read each); no wandb/tb.
#
#SBATCH --job-name=sp500_mnode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --time=03:00:00
#SBATCH --output=logs/sp500_mnode_%j.out
set -uo pipefail
umask 002

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
INNER="$EXP/scripts/train/_run_eggroll_sp500_inner.sh"
[ -f "$INNER" ] || { echo "FATAL: inner script missing: $INNER"; exit 1; }

# JAX multi-host coordinator — proven the cluster recipe: first node's HOSTNAME
# (routable on Slingshot) + fixed port; explicit process counts from SLURM.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export JAX_COORDINATOR_ADDRESS="$MASTER_ADDR:${MASTER_PORT:-29500}"
export GPUS_PER_NODE=4
export JAX_COORDINATOR_TIMEOUT_MS="${JAX_COORDINATOR_TIMEOUT_MS:-600000}"

# Re-export knobs for the srun-launched inner script (srun --export=ALL).
export SEED FC TAG ESG ESQ CRIT_BATCH R1G NSTEPS CRITIC_INPUT FEAT_CHUNK NCTXPOOL NEVAL LR SOLVER LANE_TIMEOUT ENC_PROJ_DIM ENC_WHITEN
export CKPT_SRC CKPT_STEP MONTHS TRAIN_DAYS SHARD_DIR SQUASHFS_HELPERS
export SP500_TICKER_FILE SP500_TICKERS WIDE_NPZ_IMAGE SP500_MIN_COVERAGE

echo "[sp500-mnode] job $SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_JOB_NODELIST coord=$JAX_COORDINATOR_ADDRESS"
echo "[sp500-mnode] Q=${ESQ:-128} G=${ESG:-16} crit_batch=${CRIT_BATCH:-256} NCTXPOOL=${NCTXPOOL:-256} critic=${CRITIC_INPUT:-learned} nsteps=${NSTEPS:-50} enc_proj=${ENC_PROJ_DIM:-8} whiten=${ENC_WHITEN:-robust}"

srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=map_gpu:0,1,2,3 \
     --kill-on-bad-exit=1 --export=ALL --label bash "$INNER"
RC=$?
echo "[sp500-mnode] srun rc=$RC"
chmod g+rw "$EXP/logs/sp500_mnode_${SLURM_JOB_ID}.out" 2>/dev/null || true
exit 0
