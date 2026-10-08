#!/bin/bash
# GRPO chain arm — the optimiser-comparison twin of the EGGROLL chain (submitted by
# scripts/submit_chain.sh with OPTIMIZER=grpo). Machinery is _run_grpo_production.sbatch's
# proven single-node topology (per-GPU lanes, --shard off, PREALLOC/autotune/PG-microbatch
# lessons); the DATA recipe is the chain's (lustre_npy corpus + UNSEEN_MANIFEST window mask
# + wide_L500 deep-init snapshots + TRAIN_DAYS filter, exactly as
# _run_eggroll_multinode_inner.sh), so the PG arms train on the SAME window universe as the
# EGGROLL arms. Emits the chain grid layout the SEL eval scores:
#   $CKPT_ROOT/grpo_chain_${SLURM_JOB_ID}/<tag>/stepNNNN/latest_checkpoint.json
#
# Lane spec (env LANES): comma-separated `gpu:seed:fitness_control:tag[:lr]`; default =
# 3 GRPO seeds + a shuffle-advantage control paired with seed 0. Each lane is one GPU.
# kl_coef default 0.3 (the GRPO-arm trust-region choice; EGGROLL uses 0.1 with rank-σ̄).
#
# SMOKE:  NSTEPS=4 NEVAL=32 LANES="0:0:none:smoke" sbatch --time=00:59:00 <this file>
# PROD:   via submit_chain.sh (OPTIMIZER=grpo ARMS="g_s0:8:50:0 g_s1:8:50:1 ...")
# RESUME: RESUME_SRC=<prior job's CKPT_DEST, e.g. $CKPT_ROOT/grpo_chain_NNNNNNN> restages each
#         lane's tag dir (breadcrumb + step grid) into $WORK before launch; lanes with a
#         breadcrumb continue from it (NSTEPS is the TOTAL step target, so pass the new
#         cumulative target, e.g. NSTEPS=100 to extend a 50-step run). The prior step grid
#         rides along, so the NEW job's CKPT_DEST holds the FULL grid for one SEL pass.
#         Estimator-exact across the seam (stateless per-step RNG; pool refresh reconstructed;
#         AdamW state restored) EXCEPT the critic's own optimizer moments, which restart fresh
#         (critic weights ARE restored; n_d_steps=5/step re-warms them within a few steps).
#
# Lustre 9-point checklist: breadcrumb-only resume; no find/recursive ls; ${SLURM_JOB_ID}-
# keyed dirs; all run output on $TMPDIR with 15-min safety rsync; weights staged node-local
# once; corpus read directly from the flat pre-decompressed npy dir (Lustre-polite);
# no wandb/tensorboard; no persistent agents.
#
#SBATCH --job-name=grpo_chain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=23:30:00
#SBATCH --output=logs/grpo_chain_%j.out
set -uo pipefail
umask 002
export TMPDIR=/tmp
# Eval-side selection mask must never reach the trainer's dataset builder (mutually
# exclusive with UNSEEN_MANIFEST there) — launcher scopes it; defense in depth.
unset PRETRAIN_SEEN_MANIFEST 2>/dev/null || true

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
MAMBA="$EXP/lobmamba"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$MAMBA"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEMFRAC:-0.9}"
# The proj backward's giant transpose defeats XLA autotuning (INTERNAL "Autotuning failed"
# on every first attempt) — disable from the start, as the production runs do.
export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
export LD_LIBRARY_PATH="$CP/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CP/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CP/lib/python3.12/site-packages/nvidia/cublas/lib:$CP/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export MAMBA3_EPS_COMPAT="${MAMBA3_EPS_COMPAT:-0}"

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_googonly_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_unseen100/$STOCK}"
DATA_MODE="${DATA_MODE:-lustre_npy}"
[ "$DATA_MODE" = "lustre_npy" ] || { echo "FATAL: this chain arm supports DATA_MODE=lustre_npy only"; exit 1; }
DATA_NPY_DIR="${DATA_NPY_DIR:-/lustre/projects/shared/quant/Mamba3_GOOG_pretraining_runs/data_npy/GOOG}"
[ -d "$DATA_NPY_DIR" ] || { echo "FATAL: DATA_NPY_DIR missing: $DATA_NPY_DIR"; exit 1; }
[ -n "${TRAIN_DAYS:-}" ] || { echo "FATAL: TRAIN_DAYS unset (the launcher exports it)"; exit 1; }

WORK="${TMPDIR:-/tmp}/grpo_chain_${SLURM_JOB_ID}"; mkdir -p "$WORK/logs"
DEST="$EXP/runs/grpo_chain_${SLURM_JOB_ID}"; mkdir -p "$DEST"
CKPT_ROOT="${CKPT_ROOT:-/lustre/projects/public/shared/post_training_GAN}"
CKPT_DEST="$CKPT_ROOT/grpo_chain_${SLURM_JOB_ID}"; mkdir -p "$CKPT_DEST"

NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/"

# Survive the ~30-40s post-start scratch wipe BEFORE launching lanes; /tmp is not wiped.
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "FATAL: node scratch wiped after start — resubmit"; exit 7; }

if ! python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 7)" 2>/dev/null; then
  echo "FATAL: JAX sees no GPU on $(hostname) — resubmit"; exit 7
fi
echo "[grpo-chain] JAX GPU visible on $(hostname)"

# TRAIN_DAYS filter over the flat npy corpus: ONE readdir, then in-shell name matching.
FILT="$WORK/train_days"; mkdir -p "$FILT"
ALLF=$(ls "$DATA_NPY_DIR")
for D in ${TRAIN_DAYS//,/ }; do
  for f in $(echo "$ALLF" | grep -E "^${STOCK}_.*${D}.*34200000_57600000.*(message|orderbook)_10_proc\.npy$"); do
    ln -sf "$DATA_NPY_DIR/$f" "$FILT/"
  done
done
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NE=$(echo "$TRAIN_DAYS" | tr ',' ' ' | wc -w)
echo "[grpo-chain] TRAIN_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE TRAIN_DAYS readable — refusing partial run"; exit 1; }
INFER_DATA_DIR_NODE="$FILT"

# Chain-matched GRPO command: proj scope, anchor-KL trust region, robust-critic recipe
# forwarded from the launcher env, step-keyed ckpt grid for the SEL eval.
GRPO_COMMON="--run --scope proj --adv ${ADV:-grpo} --lr ${LR:-1e-5} --lr_warmup ${LR_WARMUP:-5} --solver ${SOLVER:-adamw} \
  --critic_input ${CRITIC_INPUT:-learned} \
  --ens_size ${ENS_SIZE:-5} --enc_proj_dim ${ENC_PROJ_DIM:-8} --enc_whiten ${ENC_WHITEN:-robust} \
  --crit_batch ${CRIT_BATCH:-256} \
  --G ${GG:-8} --Q ${QQ:-50} --n_cond ${NCOND:-500} --n_gen ${NGEN:-500} --n_g_steps ${NSTEPS:-50} --n_d_steps 5 \
  --wide_book_dir $NODE_WIDE --wide_levels 500 \
  --kl_coef ${KL_COEF:-0.3} --kl_anneal const --feat_chunk ${FEAT_CHUNK:-2} --pg_chunk ${PG_CHUNK:-1} \
  --pg_microbatch ${PG_MB:-2} --gen_chunk ${GRPO_GEN_CHUNK:-0} \
  --n_ctx_pool ${GRPO_CTX_POOL:-64} --n_eval_ctx ${NEVAL:-128} --pool_scope cont --pool_refresh_every ${POOL_REFRESH:-10} \
  --eval_every ${EVAL_EVERY:-5} --ckpt_every ${CKPT_EVERY:-5} --goodhart_patience 999 --keep_step_ckpts --shard off \
  --data_dir $INFER_DATA_DIR_NODE --ckpt_dir $NODE_CKPT --ckpt_step $CKPT_STEP"

LANE_TIMEOUT="${LANE_TIMEOUT:-82800}"   # 23 h cap/lane (breadcrumb-resumable)
# A crashing lane resumes from its breadcrumb. LANE_RETRIES bounds how many times; each
# retry gets LANE_RETRY_TIMEOUT. Defaults reproduce the historical single 1 h retry. Raise
# both when the failure mode is periodic (e.g. allocator fragmentation OOM every few
# steps): a fresh process then ratchets the breadcrumb forward once per retry.
LANE_RETRIES="${LANE_RETRIES:-1}"
LANE_RETRY_TIMEOUT="${LANE_RETRY_TIMEOUT:-3600}"
# Minimum job-wall runway needed to make a retry worth starting (JAX compile + >=1 step).
LANE_MIN_RUNWAY="${LANE_MIN_RUNWAY:-3000}"
runway_left() {  # seconds of job wall remaining, or a large number if Slurm didn't say
  [ -n "${SLURM_JOB_END_TIME:-}" ] || { echo 999999; return; }
  echo $(( SLURM_JOB_END_TIME - $(date +%s) ))
}
run_lane() {  # $1 gpu  $2 seed  $3 fitness_control  $4 tag  [$5 lr override]
  local gpu="$1" seed="$2" fc="$3" tag="$4" lr_ovr="${5:-}"
  local out="$WORK/$tag" lg="$WORK/logs/$tag.log"
  local rflag=""
  [ -f "$out/latest_checkpoint.json" ] && rflag="--resume_dir $out"
  CUDA_VISIBLE_DEVICES=$gpu timeout "$LANE_TIMEOUT" python -u -m eggroll_gan.training.train_grpo_head \
    $GRPO_COMMON --seed "$seed" --fitness_control "$fc" --out_dir "$out" $rflag ${lr_ovr:+--lr $lr_ovr} \
    > >(awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' > "$lg") 2>&1
  local rc=$?
  echo "[grpo-chain] lane $tag (seed=$seed fc=$fc) rc=$rc"
  local try=0 left
  while [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$try" -lt "$LANE_RETRIES" ]; do
    left="$(runway_left)"
    if [ "$left" -lt "$LANE_MIN_RUNWAY" ]; then
      echo "[grpo-chain] lane $tag stopping retries: ${left}s wall left < ${LANE_MIN_RUNWAY}s runway"
      break
    fi
    try=$(( try + 1 ))
    echo "[grpo-chain] lane $tag retry $try/$LANE_RETRIES (resume from breadcrumb, ${left}s wall left)"
    CUDA_VISIBLE_DEVICES=$gpu timeout "$LANE_RETRY_TIMEOUT" python -u -m eggroll_gan.training.train_grpo_head \
      $GRPO_COMMON --seed "$seed" --fitness_control "$fc" --out_dir "$out" --resume_dir "$out" ${lr_ovr:+--lr $lr_ovr} \
      > >(awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' >> "$lg") 2>&1
    rc=$?
    echo "[grpo-chain] lane $tag retry $try rc=$rc"
  done
  echo "$rc" > "$WORK/logs/$tag.rc"
}

LANES="${LANES:-0:0:none:g_s0,1:1:none:g_s1,3:2:none:g_s2,2:0:shuffle:g_ctrl}"
NEED_GPU=0
for entry in ${LANES//,/ }; do IFS=: read -r g _ <<< "$entry"; [ "$g" -gt "$NEED_GPU" ] && NEED_GPU="$g"; done
HAVE_GPU=$(python -c "import jax; print(sum(1 for d in jax.devices() if d.platform=='gpu'))" 2>/dev/null || echo 0)
if [ "$HAVE_GPU" -le "$NEED_GPU" ]; then
  echo "FATAL: LANES needs $((NEED_GPU+1)) GPUs but JAX sees $HAVE_GPU — resubmit"; exit 7
fi

TAGS=(); PIDS=()
for entry in ${LANES//,/ }; do
  IFS=: read -r gpu seed fc tag lane_lr <<< "$entry"
  if [ -n "${RESUME_SRC:-}" ]; then
    if [ -f "$RESUME_SRC/$tag/latest_checkpoint.json" ]; then
      mkdir -p "$WORK/$tag"
      rsync -a "$RESUME_SRC/$tag/" "$WORK/$tag/"
      echo "[grpo-chain] lane $tag restaged from $RESUME_SRC/$tag (resume)"
    else
      echo "FATAL: RESUME_SRC set but no breadcrumb at $RESUME_SRC/$tag — refusing a silent cold start"
      exit 1
    fi
  fi
  run_lane "$gpu" "$seed" "$fc" "$tag" "$lane_lr" &
  PIDS+=($!); TAGS+=("$tag")
done
echo "[grpo-chain] launched ${#TAGS[@]} lane(s): ${TAGS[*]}"

# 15-min safety sync: logs (no step dirs) -> home DEST; full tag dirs (WITH the step grid,
# only new step dirs transfer each pass) -> Lustre CKPT_DEST for the SEL eval.
sync_dirs() { local t
              rsync -a "$WORK/logs" "$DEST/" 2>/dev/null
              for t in "${TAGS[@]}"; do rsync -a "$WORK/$t" "$CKPT_DEST/" 2>/dev/null; done; }
( while true; do sleep 900; sync_dirs; done ) &
SYNC=$!

wait "${PIDS[@]}"
kill "$SYNC" 2>/dev/null || true
sync_dirs || true
chmod -R g+rwX "$DEST" "$CKPT_DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/grpo_chain_${SLURM_JOB_ID}.out" 2>/dev/null || true

echo "==== [grpo-chain] lane tails ===="
RCS=""
for t in "${TAGS[@]}"; do
  rc="$(cat "$WORK/logs/$t.rc" 2>/dev/null || echo 9)"
  echo "---- $t (rc=$rc) ----"
  tail -10 "$WORK/logs/$t.log" 2>/dev/null
  RCS="$RCS $t:$rc"
done
echo "[grpo-chain] SUMMARY rcs:$RCS -> grids in $CKPT_DEST"
exit 0
