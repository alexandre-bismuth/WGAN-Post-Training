#!/bin/bash
# GRPO (policy-gradient) MULTI-NODE run — the PG twin of _run_eggroll_multinode.sbatch, for the
# GRPO-vs-EGGROLL comparison on the same learned critic (2026-07-01).
#
# WHY multinode: G=16 x Q=256 = 4096 rollouts/step. The proj PG backward at --shard on runs the
# sharded single-call path (M split across devices); at 512/device (2 nodes) it is ~OOM-borderline
# (train_grpo_head notes 73.9 GB at 512), so the real shape prefers 4 nodes (4096/16 = 256/device,
# the proven-fitting size). The learned critic's d-step is NOT sharded -> --crit_batch caps it.
# Divisibility holds for BOTH 8 (2-node) and 16 (4-node) devices: G*Q=4096, n_eval=256, NCTXPOOL=512,
# n_pool=768 all %16=0. Submit `sbatch --nodes=2` for a smoke, `--nodes=4` for the memory-safe prod.
#
# Process model / coordinator / GPU-bind: identical to the ES multinode sbatch (see its header).
# Lustre 9-point: breadcrumb resume; no find/recursive ls; ${SLURM_JOB_ID}-keyed dirs; output on
# /tmp + rsync; weights+data staged node-local once PER NODE; no wandb/tb; no persistent agents.
#
#SBATCH --job-name=grpo_mnode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --time=07:45:00
#SBATCH --output=logs/grpo_mnode_%j.out
set -uo pipefail
umask 002

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
INNER="$EXP/scripts/train/_run_grpo_multinode_inner.sh"
[ -f "$INNER" ] || { echo "FATAL: inner script missing: $INNER"; exit 1; }

# JAX multi-host coordinator = first node HOSTNAME (routable on Slingshot) + fixed port; explicit
# num_processes/process_id from SLURM (per-task, set by srun). GPUS_PER_NODE -> local_device_ids.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export JAX_COORDINATOR_ADDRESS="$MASTER_ADDR:${MASTER_PORT:-29500}"
export GPUS_PER_NODE=4
export JAX_COORDINATOR_TIMEOUT_MS="${JAX_COORDINATOR_TIMEOUT_MS:-600000}"

# Re-export so the inner (run by srun on each node) sees the knobs. GRPO knobs: LAMBDA (kl_coef),
# PG_MB (pg_microbatch), PG_CHUNK; Q defaults to 256, NCTXPOOL 512, NEVAL 256, LR 1e-4.
export SEED FC TAG ESG ESQ CRIT_BATCH NSTEPS CRITIC_INPUT FEAT_CHUNK NCTXPOOL NEVAL LR SOLVER LAMBDA PG_MB PG_CHUNK LANE_TIMEOUT
export CKPT_SRC CKPT_STEP STOCK MONTHS WIDE_BOOK_DIR TRAIN_DAYS SHARD_DIR SQUASHFS_HELPERS

echo "[grpo-mnode] job $SLURM_JOB_ID nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS nodelist=$SLURM_JOB_NODELIST coord=$JAX_COORDINATOR_ADDRESS"
echo "[grpo-mnode] adv=grpo Q=${ESQ:-256} G=${ESG:-16} lambda=${LAMBDA:-0.1} crit_batch=${CRIT_BATCH:-256} NCTXPOOL=${NCTXPOOL:-512} NEVAL=${NEVAL:-256} critic=${CRITIC_INPUT:-learned} nsteps=${NSTEPS:-50} lr=${LR:-1e-4}"

srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=map_gpu:0,1,2,3 \
     --kill-on-bad-exit=1 --export=ALL --label bash "$INNER"
RC=$?
echo "[grpo-mnode] srun rc=$RC"
chmod g+rw "$EXP/logs/grpo_mnode_${SLURM_JOB_ID}.out" 2>/dev/null || true
exit 0
