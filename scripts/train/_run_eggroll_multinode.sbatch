#!/bin/bash
# S5b proj-scope MULTI-NODE (2-node, 8-GPU) sharded ES run — the A2 path to Q=128 (2026-06-30).
#
# WHY: Q=128 (G=16 -> 2048 rollouts/step) OOMs at 512/shard on ONE node (4 GPU) inside the KL pass
# (31 GiB: 2 backbone passes/rollout over a 26k-token window; forward-only, so remat does not help).
# 256/shard is the proven-fitting size (Q=64 1-node smoke 5432787 ran clean). 2 nodes x 4 GPU = 8
# shards -> 2048/8 = 256/shard at Q=128. The trainer's multi-host plumbing already exists:
# jax.distributed.initialize() at the module top (via --distributed), ONE global ('pop',) mesh over
# all 8 GPUs, rank-0 (jax.process_index()==0) gating on EVERY filesystem write, multi-host eval +
# all-gather. dist_utils shard_pop/replicate_tree gate on mesh.devices.size>1 (the A1 fix), which is
# also true in multi-host, so the same sharded inputs are built.
#
# Process model: 1 task/node (SLURM_NTASKS=2), each owning its node's 4 GPUs; jax.distributed
# auto-detects coordinator/process_id/num_processes from SLURM. The per-node body is _run_s5b_
# multinode_inner.sh, launched via `srun bash <file>` (NOT a heredoc on stdin, whose broadcast to
# all tasks is not guaranteed). Each node stages its OWN /tmp farm (no cross-node race); only
# SLURM_PROCID 0 rsyncs to the shared DEST (p1 is rank-gated to zero writes). --kill-on-bad-exit so
# a crash on one process tears down both (no hang on the next collective). NO in-script retry.
#
# Divisibility (n_dev=8): G*Q=2048%8=0; n_eval_ctx=128%8=0; NCTXPOOL=256%8=0, n_pool=384%8=0;
# feat_chunk=4 | per-shard 256; kl_chunk=1. Knobs (env): SEED FC TAG ESG ESQ CRIT_BATCH R1G NSTEPS
# CRITIC_INPUT FEAT_CHUNK NCTXPOOL NEVAL LR SOLVER LANE_TIMEOUT (+ data overrides).
#
# Lustre 9-point: breadcrumb resume; no find/recursive ls; ${SLURM_JOB_ID}-keyed dirs; output on
# /tmp + rsync; weights+data staged node-local once PER NODE; no wandb/tb; no persistent agents.
#
#SBATCH --job-name=s5b_mnode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --time=07:45:00
#SBATCH --output=logs/s5b_mnode_%j.out
set -uo pipefail
umask 002

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
INNER="$EXP/scripts/train/_run_eggroll_multinode_inner.sh"
[ -f "$INNER" ] || { echo "FATAL: inner script missing: $INNER"; exit 1; }

# JAX multi-host coordinator. The bare SLURM auto-detect picks a non-routable interface IP on
# Slingshot (-> gRPC "connection refused", job 5432902). Proven the cluster recipe (run_train.py +
# train_full_autoreg.batch): coordinator = first node's HOSTNAME (routable) + fixed port; explicit
# num_processes/process_id from SLURM (per-task, set by srun). GPUS_PER_NODE -> local_device_ids.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export JAX_COORDINATOR_ADDRESS="$MASTER_ADDR:${MASTER_PORT:-29500}"
export GPUS_PER_NODE=4
export JAX_COORDINATOR_TIMEOUT_MS="${JAX_COORDINATOR_TIMEOUT_MS:-600000}"   # generous for farm-staging skew

# Re-export so the inner script (run by srun on each node) sees the knobs at runtime. (`VAR=x sbatch`
# puts them in this batch env; srun --export=ALL is the default and propagates them to the tasks.)
export SEED FC TAG ESG ESQ CRIT_BATCH R1G NSTEPS CRITIC_INPUT FEAT_CHUNK NCTXPOOL NEVAL LR SOLVER LANE_TIMEOUT ENC_PROJ_DIM ENC_WHITEN
export CKPT_SRC CKPT_STEP STOCK MONTHS WIDE_BOOK_DIR TRAIN_DAYS SHARD_DIR SQUASHFS_HELPERS MULTI_TICKERS WIDE_BOOK_PARENT
# unseen100 headline arm (2026-07-03): lustre_npy data mode + unseen-slot mask + pool/critic knobs
# SIGMA=0 turns a lane into the critic-only separability gate (fakes == anchor rollouts).
export POOL_REFRESH ENS_SIZE DATA_MODE DATA_NPY_DIR UNSEEN_MANIFEST SIGMA
# Eval-side selection mask must never reach the trainer's dataset builder (it is mutually
# exclusive with UNSEEN_MANIFEST there) — the launcher scopes it, this is defense in depth.
unset PRETRAIN_SEEN_MANIFEST 2>/dev/null || true
# XLA allocator overrides (gate j5489734 ens5 d-step OOM; see inner script note)
export XLA_PREALLOCATE XLA_MEM_FRACTION

echo "[mnode] job $SLURM_JOB_ID nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS nodelist=$SLURM_JOB_NODELIST coord=$JAX_COORDINATOR_ADDRESS"
echo "[mnode] Q=${ESQ:-128} G=${ESG:-16} crit_batch=${CRIT_BATCH:-256} NCTXPOOL=${NCTXPOOL:-256} critic=${CRITIC_INPUT:-learned} nsteps=${NSTEPS:-50}"

# GPU exposure: the srun STEP must request GPUs (--gres) AND map all 4 to the single per-node task
# (--gpu-bind=map_gpu:0,1,2,3) or each task sees only CPU (job 5433052 "Unknown backend: 'gpu'").
# Exact recipe proven by lobmamba/train_full_autoreg.batch at 256 nodes (1 task/node, 4 GPUs/task).
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=map_gpu:0,1,2,3 \
     --kill-on-bad-exit=1 --export=ALL --label bash "$INNER"
RC=$?
echo "[mnode] srun rc=$RC"
chmod g+rw "$EXP/logs/s5b_mnode_${SLURM_JOB_ID}.out" 2>/dev/null || true
exit 0
