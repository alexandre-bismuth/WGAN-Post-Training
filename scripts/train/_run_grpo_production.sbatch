#!/bin/bash
# GRPO-proj PRODUCTION run (task #39) — the matched policy-gradient comparison arm vs EGGROLL-proj.
# Derived VERBATIM from _run_eggroll_production.sbatch (same anchor s28730 / eps-compat 0 / wide_L500 Jan
# data staging / scratch-wipe survival / Lustre-safe machinery); ONLY the train command + lanes differ:
#   eggroll_gan.training.train_grpo_head --scope proj --adv grpo  (canonical GRPO z-score advantage,
#   single on-policy step => importance ratio==1 => clip a no-op; exact KL(pi||anchor) trust region).
#
# Matched TOTAL sample budget vs EGGROLL-proj (~20k rollouts), each method at its own optimal shape:
# G=8 x Q=50 = 400 rollouts/step x 50 steps = 20,000 rollouts, 500/500, kl_coef 0.1 ANCHOR trust region.
# PG-correct allocation: small group G=8 only stabilizes the z-score advantage baseline; many contexts
# Q=50 give the gradient its diversity. Deliberately NOT EGGROLL's G=16xQ=32 — ES needs a large group
# to rank its population, PG does not; matched budget, each method at its natural shape is the fairer
# comparison than identical hyperparameters. 50 steps brackets the convergence horizon EGGROLL revealed
# (best by ~5-20, plateau ~40); best/ harvested post-hoc via soup. 4 lanes on 1 node = 3 seeds (error
# bars) + 1 shuffle-advantage control (attribution null, paired with seed 0). lr=1e-4 is THE PG knob
# (not the ES eta). goodhart_patience 999 => run the full NSTEPS, harvest best/ post-hoc. Each lane =
# 1 GPU, --shard off. pg_microbatch=2: the PG backward needs ONE contiguous alloc of
# ~MB*n_gen*d activations (~24.5 GiB at MB=2); under PREALLOCATE=false (on-demand, frag-prone)
# MB=4's ~49 GiB block could not be carved out when 4 lanes ran concurrently -> all-lane OOM
# on job 5275741. MB=2 halves the contiguous request; the grad is an exact linear sum over
# rollouts (advantages group-normalised BEFORE slicing) so MB only changes chunking, not result.
#
# SMOKE usage (gate the machinery FIRST — MUST mirror the 4-lane PROD topology: the 1-lane
# smoke 5273703 fit MB=4 and gave false confidence; the failure was concurrency+fragmentation):
#   NSTEPS=4 NEVAL=32 PG_CHUNK=1 sbatch --time=00:45:00 \
#       scripts/train/_run_grpo_production.sbatch     # default LANES = the real 4-lane shape
# PROD usage (after the smoke is green; ~18 h/lane at the measured ~3.2 s/rollout, 24 h QOS wall):
#   sbatch --time=23:00:00 scripts/train/_run_grpo_production.sbatch
#
# Lustre 9-point checklist: breadcrumb-only resume; no find/recursive ls; ${SLURM_JOB_ID}-keyed dirs;
# all run output on $TMPDIR with 15-min safety rsync + final rsync; weights+data staged node-local
# once; no wandb/tensorboard; no persistent agents.
#
#SBATCH --job-name=grpo_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=06:00:00
#SBATCH --output=logs/grpo_prod_%j.out
set -uo pipefail
umask 002

# RAM-backed /tmp from the start (the scratch-wipe regression wipes /local/user ~30-40s in; /tmp is not).
export TMPDIR=/tmp

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
MAMBA="$EXP/lobmamba"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$MAMBA"
# PREALLOCATE: the STYLIZED critic fits its 48.7 GiB pg_fn block on 1 GPU under the on-demand
# allocator (job 5280503 OK). The BACKBONE critic additionally runs feats_fn (Mamba3-78M fwd)
# immediately before pg_fn (train_grpo_head.py:554); feats_fn's transients fragment the on-demand
# heap so the 48.7 GiB *contiguous* pg_fn block can no longer be carved despite ~68 GiB aggregate
# free -> all-lane OOM at 48.7 GiB (jobs 5283266, 5284778). PG_MB is NOT the lever (the block is
# n_gen-driven, ~MB-independent). Fix: run backbone with XLA_PREALLOC=true (one BFC arena that
# coalesces feats_fn's freed transients). If that is still tight, shard each seed across GPUs
# (--shard on, fewer concurrent lanes) to split the pop axis -> per-GPU pg_fn drops ~n_dev x.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PREALLOC:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEMFRAC:-0.9}
# The proj backward emits a giant transpose (loop_transpose_fusion) that XLA's autotuner cannot
# handle (INTERNAL "Autotuning failed" on every smoke's first attempt). Disable autotuning from the
# start (the retry already did) so the first attempt uses a default algorithm and skips the ~5-min
# failed search; memory is then bounded by --pg_microbatch.
export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
export LD_LIBRARY_PATH="$CP/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CP/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CP/lib/python3.12/site-packages/nvidia/cublas/lib:$CP/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# s28730 anchor trained with FIXED RMSNorm epsilons -> eps-compat OFF, set UNCONDITIONALLY (a stray
# inherited MAMBA3_EPS_COMPAT=1 from the submit shell would silently corrupt the forward pass).
export MAMBA3_EPS_COMPAT=0

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_sp500iso_fixednorm_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-01}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_2026-01/GOOG}"
TRAIN_DAYS="${TRAIN_DAYS:-2026-01-02,2026-01-05,2026-01-06,2026-01-07,2026-01-08,2026-01-09,2026-01-12,2026-01-13,2026-01-14,2026-01-15,2026-01-16,2026-01-20,2026-01-21,2026-01-22,2026-01-23,2026-01-26,2026-01-27,2026-01-28,2026-01-29,2026-01-30}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "FATAL: helper missing"; exit 1; }

WORK="${TMPDIR:-/tmp}/grpo_prod_${SLURM_JOB_ID}"; mkdir -p "$WORK/logs"
DEST="$EXP/runs/grpo_production_${SLURM_JOB_ID}"; mkdir -p "$DEST"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
NODE_WIDE=""
if [ -n "${WIDE_BOOK_DIR:-}" ]; then
  NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
  rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/"
fi
[ -n "$NODE_WIDE" ] || { echo "FATAL: WIDE_BOOK_DIR unset — refusing an L10 run (train/eval mismatch)"; exit 1; }

# Survive the ~30-40s post-start scratch wipe BEFORE launching the lanes: wait, verify survival.
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "FATAL: node scratch wiped after start (ckpt $( [ -f "$NODE_CKPT/latest_checkpoint.json" ] && echo ok || echo GONE ), farm $( [ -d "$INFER_DATA_DIR_NODE" ] && echo ok || echo GONE ), wide $( [ -d "$NODE_WIDE" ] && echo ok || echo GONE )) — resubmit"; exit 7; }
echo "[grpo] node scratch verified stable (WORK=$WORK)"

# Fail FAST + clearly if JAX cannot see a GPU on this node (transient CUDA-init failure / GPU-less
# alloc). The bad-node mode is JAX falling back to 1 CPU device, then a backend='gpu' jit erroring
# deep in compile (job 5273620 on nid010505) — catch it here so the result is a clean exit-7
# "resubmit" rather than 4 confusing per-lane tracebacks.
if ! python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 7)" 2>/dev/null; then
  echo "FATAL: JAX sees no GPU on $(hostname) (CUDA init failed / no GPU bound) — resubmit"; exit 7
fi
echo "[grpo] JAX GPU visible on $(hostname)"

# TRAIN_DAYS filter: get_dataset(test_split=0.0) needs a wide-book match for every day in data_dir;
# keep only the RTH variant of the days we have init for. Refuse a partial mount (job 5195629 mode).
FILT="$WORK/train_days"; mkdir -p "$FILT"
for D in ${TRAIN_DAYS//,/ }; do
  for f in "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*message_10_proc.npy \
           "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*orderbook_10_proc.npy; do
    [ -e "$f" ] && ln -sf "$f" "$FILT/"
  done
done
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NE=$(echo "$TRAIN_DAYS" | tr ',' ' ' | wc -w)
echo "[grpo] TRAIN_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE TRAIN_DAYS readable in farm (mount/wipe issue) — refusing partial run"; exit 1; }
INFER_DATA_DIR_NODE="$FILT"

# Canonical GRPO (arXiv:2501.12948): z-score advantage, exact anchor-KL trust region, one on-policy
# step/batch. proj scope => the head stays frozen at the anchor; the PG gradient flows into the proj
# kernels. goodhart_patience 999 => no early stop (harvest best/ post-hoc, EGGROLL-matched).
GRPO_COMMON="--run --scope proj --adv ${ADV:-grpo} --lr ${LR:-1e-4} --solver ${SOLVER:-adamw} \
  --critic_input ${CRITIC_INPUT:-learned} \
  --ens_size ${ENS_SIZE:-3} --enc_proj_dim ${ENC_PROJ_DIM:-0} --enc_whiten ${ENC_WHITEN:-std} \
  --G ${GG:-8} --Q ${QQ:-50} --n_cond ${NCOND:-500} --n_gen ${NGEN:-500} --n_g_steps ${NSTEPS:-50} --n_d_steps 5 \
  --wide_book_dir $NODE_WIDE --wide_levels 500 \
  --kl_coef ${KL_COEF:-0.1} --kl_anneal const --feat_chunk ${FEAT_CHUNK:-2} --pg_chunk ${PG_CHUNK:-1} \
  --pg_microbatch ${PG_MB:-2} \
  --n_ctx_pool 64 --n_eval_ctx ${NEVAL:-128} --pool_scope cont --pool_refresh_every 10 \
  --eval_every ${EVAL_EVERY:-5} --ckpt_every ${CKPT_EVERY:-5} --goodhart_patience 999 --shard off \
  --data_dir $INFER_DATA_DIR_NODE --ckpt_dir $NODE_CKPT --ckpt_step $CKPT_STEP"

LANE_TIMEOUT="${LANE_TIMEOUT:-75600}"   # 21 h safety cap per lane (~18 h expected at 400 rollouts/step x 50; breadcrumb-resumable)
run_lane() {  # $1 gpu  $2 seed  $3 fitness_control  $4 tag  [$5 lr override]
  local gpu="$1" seed="$2" fc="$3" tag="$4" lr_ovr="${5:-}"
  local out="$WORK/$tag" lg="$WORK/logs/$tag.log"
  CUDA_VISIBLE_DEVICES=$gpu timeout "$LANE_TIMEOUT" python -u -m eggroll_gan.training.train_grpo_head \
    $GRPO_COMMON --seed "$seed" --fitness_control "$fc" --out_dir "$out" ${lr_ovr:+--lr $lr_ovr} \
    > >(awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' > "$lg") 2>&1
  local rc=$?
  echo "[grpo] lane $tag (seed=$seed fc=$fc) rc=$rc"
  # One retry on a CRASH (not a timeout): autotune_level=0 fallback; breadcrumb resume.
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    echo "[grpo] lane $tag retrying with --xla_gpu_autotune_level=0 (resume from breadcrumb)"
    CUDA_VISIBLE_DEVICES=$gpu XLA_FLAGS="--xla_gpu_autotune_level=0" timeout 1800 \
      python -u -m eggroll_gan.training.train_grpo_head \
      $GRPO_COMMON --seed "$seed" --fitness_control "$fc" --out_dir "$out" --resume_dir "$out" ${lr_ovr:+--lr $lr_ovr} \
      > >(awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' >> "$lg") 2>&1
    rc=$?
    echo "[grpo] lane $tag retry rc=$rc"
  fi
  echo "$rc" > "$WORK/logs/$tag.rc"
}

# Lane spec: comma-separated `gpu:seed:fitness_control:tag[:lr]`. Default = 3 GRPO seeds + shuffle
# control (paired with seed 0), mirroring the EGGROLL-proj prod lanes for a 1:1 comparison.
LANES="${LANES:-0:0:none:g_s0,1:1:none:g_s1,3:2:none:g_s2,2:0:shuffle:g_ctrl}"

# Fail FAST if the alloc holds fewer GPUs than the lane spec needs. Each lane pins
# CUDA_VISIBLE_DEVICES to a distinct physical GPU; an under-allocated job (e.g. only 1-2 GPUs bound)
# would silently pack >1 lane onto one device, init + run the forward fine under PREALLOCATE=false,
# then collide on the ~24.5 GiB PG backward and OOM deep in step 1 — the failure mode of job 5275741.
NEED_GPU=0
for entry in ${LANES//,/ }; do IFS=: read -r g _ <<< "$entry"; [ "$g" -gt "$NEED_GPU" ] && NEED_GPU="$g"; done
HAVE_GPU=$(python -c "import jax; print(sum(1 for d in jax.devices() if d.platform=='gpu'))" 2>/dev/null || echo 0)
if [ "$HAVE_GPU" -le "$NEED_GPU" ]; then
  echo "FATAL: LANES needs $((NEED_GPU+1)) GPUs (max index $NEED_GPU) but JAX sees only $HAVE_GPU on $(hostname) — lanes would share a GPU and OOM; fix --gpus-per-node and resubmit"; exit 7
fi
echo "[grpo] GPU count OK: $HAVE_GPU visible, LANES needs $((NEED_GPU+1))"

TAGS=(); PIDS=()
for entry in ${LANES//,/ }; do
  IFS=: read -r gpu seed fc tag lane_lr <<< "$entry"
  run_lane "$gpu" "$seed" "$fc" "$tag" "$lane_lr" &
  PIDS+=($!); TAGS+=("$tag")
done
echo "[grpo] launched ${#TAGS[@]} lane(s): ${TAGS[*]}"

sync_dirs() { local d=("$WORK/logs") t; for t in "${TAGS[@]}"; do d+=("$WORK/$t"); done
             rsync -a --exclude='step[0-9]*' "${d[@]}" "$DEST/" 2>/dev/null; }

( while true; do sleep 900; sync_dirs; done ) &
SYNC=$!

wait "${PIDS[@]}"
kill "$SYNC" 2>/dev/null || true

sync_dirs || true
chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/grpo_prod_${SLURM_JOB_ID}.out" 2>/dev/null || true

echo "==== [grpo] lane tails ===="
RCS=""
for t in "${TAGS[@]}"; do
  rc="$(cat "$WORK/logs/$t.rc" 2>/dev/null || echo 9)"
  echo "---- $t (rc=$rc) ----"
  tail -10 "$WORK/logs/$t.log" 2>/dev/null
  RCS="$RCS $t:$rc"
done
echo "[grpo] SUMMARY rcs:$RCS -> $DEST"
exit 0
