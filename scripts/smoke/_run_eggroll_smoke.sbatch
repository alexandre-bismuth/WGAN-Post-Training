#!/bin/bash
# S5b GPU smoke (2026-06-13) — DEEP-INIT (clean L500) retraining pre-flight. Submit only with
# explicit user approval.
#
# Validates the in/out-projection LoRA (do_Tmm fold) scope AND the deep-init training path that
# production (_run_eggroll_production.sbatch) will use, on ONE GH200, against the NEW anchor
# (mamba3_78M_sp500iso_fixednorm_s28730, eps-compat OFF):
#   Phase A — eggroll_gan.s5b_proj_gates --rollout at the smoke horizon (64 ctx / 16 gen, L10):
#             sigma=0 bit-exact LoRA fold (default + forced-highest matmul precision) + rollout
#             faithfulness diagnostic + sigma>0 differs + 3-step ES moves ONLY MM kernels + fused
#             feats+KL. This is a MATH gate (init-independent), so it stays cheap at L10.
#   Phase B — train_eggroll_gan_s5 --scope proj at the PRODUCTION horizon (n_cond=500 / n_gen=500)
#             with the clean L500 deep init (--wide_book_dir + --wide_levels 500): 8 steps, kl on,
#             eval/ckpt every 4, then RESUME +4 from the breadcrumb -> proj ckpt/resume round-trip.
#             The trainer's CPU checks (incl. C11-C14 proj glue) run automatically before --run.
#
# WHY 500/500 here (the old smoke was 64/16): the wide-book npz is indexed by per-file window id
# with seq_start = seq_idx*n_messages and local_seq_indices built at n_messages=1000
# (lobster_dataloader.py:835). At 64/16 every snapshot mis-maps and silently falls back to L10 —
# exactly the train/eval mismatch this whole pre-flight exists to prevent. Per-step cost is tiny at
# G=8xQ=4 (~32 rollouts); the 500-msg compile is the only real cost (~a few min — cf. eval job
# 5196129 which compiled + ran 256 contexts at 500/500 in ~6 min), so it fits the 1h15 wall.
#
# Smoke confirms, before the 6h x 4-GPU production: (a) wide-book pairs 1:1 on the SquashFS Jan farm,
# (b) >0 covered windows ([prep] coverage log), (c) sim_nOrders=1000 init builds, (d) the 500-msg
# rollout compiles + runs FINITE, (e) keep-rate / no-drain (trainer logs mean kept msgs), (f) ckpt +
# resume round-trip, (g) per-step wall time (extrapolate to confirm the production wall). best/ is
# written only when the composite improves (a dynamics outcome); the eval has a latest/ fallback.
#
# Lustre 9-point checklist: breadcrumb-only resume (no ls); no find/recursive ls; output dirs keyed
# by ${SLURM_JOB_ID}; all run output on $TMPDIR (=/tmp, wipe-safe) with one final rsync; single job;
# weights + wide-book + data staged node-local once; no wandb/tensorboard; no persistent agents.
#
#SBATCH --job-name=s5b_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=02:00:00
#SBATCH --output=logs/s5b_smoke_%j.out
set -uo pipefail
umask 002

# Mount/stage to RAM-backed /tmp from the start. The scratch-wipe regression wipes /local/user
# (the default TMPDIR) ~30-40s after start; /tmp is NOT wiped, so a fresh mount there is clean
# (proven by eval job 5196129, 8-day, no wipe WARN). The sleep-75 verify below fails fast if even
# /tmp is reaped (cheap to resubmit) rather than the eval's heavier restage path.
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
# New anchor (s28730) was trained with FIXED RMSNorm epsilons -> eps-compat OFF. Set UNCONDITIONALLY
# (not ${VAR:-0}) so a stray inherited MAMBA3_EPS_COMPAT=1 (the legacy s46050 setting) propagated by
# sbatch from the submit shell cannot silently corrupt the forward pass. There is no legacy use here.
export MAMBA3_EPS_COMPAT=0

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_sp500iso_fixednorm_s28730}"; CKPT_STEP="${CKPT_STEP:-28730}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-01}"
# RTH days we have clean L500 init for (data/wide_L500_2026-01/GOOG). The SquashFS Jan farm carries
# BOTH hour-variants (RTH 34200000_57600000 + extended 24900000_57900000) + all days; we filter to
# these RTH days so the wide-book pairs 1:1 by date (the extended variant's window indexing does not
# match our RTH-keyed snapshots). A few days = thousands of covered windows >> n_pool here.
TRAIN_DAYS="${TRAIN_DAYS:-2026-01-02,2026-01-05,2026-01-06,2026-01-07}"
WIDE_BOOK_DIR="${WIDE_BOOK_DIR:-$EXP/data/wide_L500_2026-01/GOOG}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "FATAL: helper missing"; exit 1; }

WORK="${TMPDIR:-/tmp}/s5b_smoke_${SLURM_JOB_ID}"; mkdir -p "$WORK"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
# Clean L500 deep-init snapshots for the sim book (~1 GB for GOOG 2026-01). REQUIRED for Phase B:
# without them the sim starts at L10 and generated flow drains the book (the train/eval mismatch).
NODE_WIDE=""
if [ -n "${WIDE_BOOK_DIR:-}" ]; then
  NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
  rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/"
fi
[ -n "$NODE_WIDE" ] || { echo "FATAL: WIDE_BOOK_DIR unset — refusing an L10 smoke (would not test the deep-init path)"; exit 1; }

# Survive the ~30-40s post-start scratch wipe: wait it out, verify scratch + ckpt + farm + wide-book
# all survived. /tmp is not wiped, so this is a fail-fast tripwire (cheap to resubmit).
sleep 75
{ [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ] && [ -d "$NODE_WIDE" ]; } \
  || { echo "FATAL: node scratch wiped after start (ckpt $( [ -f "$NODE_CKPT/latest_checkpoint.json" ] && echo ok || echo GONE ), farm $( [ -d "$INFER_DATA_DIR_NODE" ] && echo ok || echo GONE ), wide $( [ -d "$NODE_WIDE" ] && echo ok || echo GONE )) — resubmit"; exit 7; }
echo "[s5b_smoke] node scratch verified stable (WORK=$WORK)"

# TRAIN_DAYS filter (mirrors the eval EVAL_DAYS filter). get_dataset(test_split=0.0) uses EVERY day
# in data_dir and needs a wide-book match for each (else FATAL); keep only the RTH variant of the
# days we have init for. Refuse a partial mount (the job 5195629 failure mode).
FILT="$WORK/train_days"; mkdir -p "$FILT"
for D in ${TRAIN_DAYS//,/ }; do
  for f in "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*message_10_proc.npy \
           "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*orderbook_10_proc.npy; do
    [ -e "$f" ] && ln -sf "$f" "$FILT/"
  done
done
NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
NE=$(echo "$TRAIN_DAYS" | tr ',' ' ' | wc -w)
echo "[s5b_smoke] TRAIN_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
[ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE TRAIN_DAYS readable in farm (mount/wipe issue) — refusing partial smoke"; exit 1; }
INFER_DATA_DIR_NODE="$FILT"

export CUDA_VISIBLE_DEVICES=0
STAMP='{ print strftime("[%H:%M:%S]"), $0; fflush() }'

# ---------------- EGGROLL phases A/B/B2 — skippable via SKIP_EGGROLL=1 (re-test Phase C only) --------
# (A/B/B2 went green in job 5281914; SKIP_EGGROLL=1 re-tests just the GRPO arm without re-burning them.)
OUT_B="$WORK/s5b_train"
RC_A=0; RC_B=0; RC_B2=0
if [ -n "${SKIP_EGGROLL:-}" ]; then
  echo "[s5b_smoke] SKIP_EGGROLL=1: EGGROLL phases A/B/B2 already green in 5281914 -> Phase C only"
else
# ---------------- Phase A: proj-fold gates (sigma=0 strong gate etc.) — L10, 64/16 ----------------
timeout ${TO_A:-1020} python -u -m eggroll_gan.tests.s5b_proj_gates --rollout \
  --data_dir "$INFER_DATA_DIR_NODE" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
  --n_pop 4 --n_cond 64 --n_gen 16 --sigma 0.01 --es_steps 3 --seed 0 \
  2>&1 | awk "$STAMP"
RC_A=${PIPESTATUS[0]}
echo "[s5b_smoke] phase A rc=$RC_A"

# ---------------- Phase B: proj-scope trainer smoke + resume — DEEP INIT, 500/500 ----------------
OUT_B="$WORK/s5b_train"
RC_B=1; RC_B2=1
if [ "$RC_A" -eq 0 ]; then
  timeout ${TO_B:-1800} python -u -m eggroll_gan.training.train_eggroll_gan_s5 --run --scope proj \
    --critic_input ${CRITIC_INPUT:-learned} \
    --data_dir "$INFER_DATA_DIR_NODE" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
    --wide_book_dir "$NODE_WIDE" --wide_levels 500 \
    --out_dir "$OUT_B" --G ${ESG:-8} --Q ${ESQ:-4} --rank 4 --sigma 0.01 --lr 0.001 --solver adamw \
    --n_cond 500 --n_gen 500 --n_g_steps 8 --n_d_steps 5 --r1_gamma ${R1_GAMMA:-0} \
    --kl_coef 0.1 --kl_ref anchor --kl_chunk 1 --feat_chunk ${FEAT_CHUNK:-4} \
    --n_ctx_pool ${NCTXPOOL:-16} --n_eval_ctx ${NEVAL:-4} --pool_scope cont --pool_refresh_every 4 \
    --eval_every 4 --ckpt_every 4 --no_goodhart_stop \
    --enc_proj_dim ${ENC_PROJ_DIM:-0} --enc_whiten ${ENC_WHITEN:-std} \
    --shard off --seed 0 2>&1 | awk "$STAMP"
  RC_B=${PIPESTATUS[0]}
  echo "[s5b_smoke] phase B (train 8 steps, 500/500 deep-init, refresh@4) rc=$RC_B"

  if [ "$RC_B" -eq 0 ] && [ -f "$OUT_B/latest_checkpoint.json" ]; then
    timeout ${TO_B2:-900} python -u -m eggroll_gan.training.train_eggroll_gan_s5 --run --scope proj \
      --critic_input ${CRITIC_INPUT:-learned} \
      --data_dir "$INFER_DATA_DIR_NODE" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
      --wide_book_dir "$NODE_WIDE" --wide_levels 500 \
      --out_dir "$OUT_B" --resume_dir "$OUT_B" --G ${ESG:-8} --Q ${ESQ:-4} --rank 4 --sigma 0.01 --lr 0.001 \
      --solver adamw --n_cond 500 --n_gen 500 --n_g_steps 16 --n_d_steps 5 --r1_gamma ${R1_GAMMA:-0} \
      --kl_coef 0.1 --kl_ref anchor --kl_chunk 1 --feat_chunk ${FEAT_CHUNK:-4} \
      --n_ctx_pool ${NCTXPOOL:-16} --n_eval_ctx ${NEVAL:-4} --pool_scope cont --pool_refresh_every 4 \
      --eval_every 4 --ckpt_every 4 --no_goodhart_stop \
      --enc_proj_dim ${ENC_PROJ_DIM:-0} --enc_whiten ${ENC_WHITEN:-std} \
      --shard off --seed 0 2>&1 | awk "$STAMP"
    RC_B2=${PIPESTATUS[0]}
    echo "[s5b_smoke] phase B2 (resume 9-16, exercises fire-point continue) rc=$RC_B2"
  fi
fi
fi   # end SKIP_EGGROLL gate

# ---------------- Phase C: GRPO-proj trainer smoke (learned critic, SHARED node-local staging) --------
# Mirrors the GRPO production command (train_grpo_head --scope proj --adv grpo) at smoke size, reusing
# the SAME staged data/ckpt/wide-book. Proves the critic wiring in the policy-
# gradient arm (the StylizedFeats real-replay->phi + fake-phi path that Phase B exercises for EGGROLL).
# Gated on Phase A (gates + staging OK) so a Phase-B-specific issue does not mask the GRPO arm.
OUT_C="$WORK/grpo_train"
RC_C=1
if [ -n "${SKIP_GRPO:-}" ]; then RC_C=0; echo "[s5b_smoke] SKIP_GRPO=1 — skipping GRPO Phase C (not this experiment)"; fi
if [ "$RC_A" -eq 0 ] && [ -z "${SKIP_GRPO:-}" ]; then
  timeout ${TO_C:-1200} python -u -m eggroll_gan.training.train_grpo_head --run --scope proj --adv grpo \
    --critic_input ${CRITIC_INPUT:-learned} \
    --data_dir "$INFER_DATA_DIR_NODE" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
    --wide_book_dir "$NODE_WIDE" --wide_levels 500 \
    --out_dir "$OUT_C" --G 8 --Q 4 --lr 1e-4 --solver adamw \
    --n_cond 500 --n_gen 500 --n_g_steps 4 --n_d_steps 5 \
    --kl_coef 0.1 --kl_anneal const --feat_chunk 4 --pg_chunk 1 --pg_microbatch 2 \
    --gen_chunk ${GRPO_GEN_CHUNK:-0} \
    --n_ctx_pool 16 --n_eval_ctx 4 --pool_scope cont --pool_refresh_every 4 \
    --eval_every 4 --ckpt_every 4 --goodhart_patience 999 \
    --shard off --seed 0 2>&1 | awk "$STAMP"
  RC_C=${PIPESTATUS[0]}
  echo "[s5b_smoke] phase C (GRPO-proj, 4 steps, 500/500) rc=$RC_C"
fi

# ---------------- persist results to Lustre (one rsync at the end) ----------------
DEST="$EXP/runs/s5b_smoke_${SLURM_JOB_ID}"
mkdir -p "$DEST"
rsync -a "$OUT_B/" "$DEST/s5b_train/" 2>/dev/null || true
rsync -a "$OUT_C/" "$DEST/grpo_train/" 2>/dev/null || true
chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/s5b_smoke_${SLURM_JOB_ID}.out" 2>/dev/null || true

echo "[s5b_smoke] SUMMARY: A(gates)=$RC_A B(eggroll)=$RC_B B2(resume)=$RC_B2 C(grpo)=$RC_C -> $DEST"
[ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ] && [ "$RC_B2" -eq 0 ] && [ "$RC_C" -eq 0 ]
exit $?
