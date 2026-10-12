#!/bin/bash
# Held-out TEST-SET eval (2026-06-10) — thesis-prelim arbiter table (task #12).
# Usage:  sbatch --nodes=1 --export=ALL,PROD_DIR=$EXP/runs/s5b_production_<jobid> \
#                eggroll_gan/_run_test_eval.sbatch
#
# Rows (test_eval.py, ONE GPU): anchor / decoding bar (tau=0.8, top50, W3 winner) /
# EGGROLL-proj seeds 0-2 / drift control — all on IDENTICAL 256 held-out contexts from the
# HELD-OUT month 2026-02 (training+sweep used 2026-01; 2026-02 is temporally out-of-sample),
# CRN across rows, distributional panel + bootstrap CIs, anchor-normalized.
# Built-in GPU gate: merge_noop row must be BIT-exact vs the anchor row (same program).
# L2_STATE_N=50 (env, import-time snapshot width) + --wide_levels 50 (sim book depth):
# unchanged_rate becomes the deep inert-message proxy (real ~2% at depth 50).
# Per-lane ckpt: <lane>/best if its breadcrumb exists (best-composite), else <lane> (latest).
# After the panel, the SAME job runs LOB-Bench (task #13) on the saved rollouts (CPU only):
# lobbench_adapter -> LOBSTER CSV farm on $TMPDIR (never Lustre — ~6k small files) ->
# run_lobbench (venv python) -> lobbench_results.json + lobbench_table.md rsync'd with the rest.
#
# Lustre checklist: breadcrumb-only ckpt discovery; no find/recursive ls; ${SLURM_JOB_ID} dirs;
# output on $TMPDIR, ONE final rsync; weights+data node-local; no loggers; no agents.
#
# Node-local scratch hazard (jobs 5150840/5161966/5162358/5162462 — four nodes, every GPU
# shape incl. the one that worked 2026-06-10 daytime): since ~2026-06-10 23:00 the prolog-
# created /local/user/<uid> is wiped ~30-40s after job start and CANNOT be recreated (mkdir
# under /local/user denied) -> cluster-side regression, reported to HPC team. Resource shape
# is IRRELEVANT to the bug; the WORKING mitigation (proven by 5162462) is: (a) sleep 75s then
# VERIFY scratch survived; self-heal once — restage on $TMPDIR if possible, else FALL BACK TO
# /tmp (RAM-backed, counts vs job mem cgroup; fine for few-GB staging) — before touching the
# GPU, (b) python tempfile dir under job-scoped $WORK/pytmp. If even /tmp is reaped, exit 7.
# 1 GPU is all the eval uses (single-device panel; LOB-Bench is CPU).
#
#SBATCH --job-name=test_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=01:30:00
#SBATCH --output=logs/test_eval_%j.out
set -uo pipefail
umask 002

# Mount/stage to RAM-backed /tmp from the start. The scratch-wipe regression wipes
# /local/user (the default TMPDIR) ~30-40s in; the existing restage-from-wiped path then
# re-mounts the squashfuse farm flakily (only 1 of 8 days came back readable, job 5195629).
# /tmp is NOT wiped, so a FRESH mount there is clean (proven by the interactive build). The
# sleep-75 + restage logic below stays as a fallback in case even /tmp gets reaped.
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
# L2_STATE_N sets ONLY the saved-l2 width (inference_no_errcorr.l2_state_n, import-time).
# The model's per-step book view is always book_l2[0:40] (top-10) regardless — proven by
# job 5176000 (L2_STATE_N=10) being bit-identical in msgs to 5174597 (=50).
export L2_STATE_N="${L2_STATE_N:-50}"
# MAMBA3_EPS_COMPAT=1 restores the pre-2026-04-22 RMSNorm epsilons (BCNorm eps=d_state,
# out_norm eps=d_inner) that the s46050 anchor was TRAINED with (upstream "epsilon trap"
# fix ffe45d66 changed the forward pass under those weights; without compat the anchor
# emits ~43% START-token garbage messages). Default ON for this ckpt.
export MAMBA3_EPS_COMPAT="${MAMBA3_EPS_COMPAT:-1}"

# PROD_DIR (EGGROLL production runs) is OPTIONAL. If unset, the panel runs the
# anchor (+ decoding bar) alone — used for the clean-init anchor re-measurement.

CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_megacap8_26tok_s46050}"; CKPT_STEP="${CKPT_STEP:-46050}"
STOCK="${STOCK:-GOOG}"; MONTHS="${MONTHS:-2026-02}"
SHARD_DIR="${SHARD_DIR:-/lustre/projects/public/data/lob_preproc_sp500_squashfs}"
SQUASHFS_HELPERS="${SQUASHFS_HELPERS:-/lustre/projects/public/data/recon_2026-05/sp500_L500/pipeline_patched/pipeline/_squashfs_helpers.sh}"
[ -f "$SQUASHFS_HELPERS" ] || { echo "FATAL: helper missing"; exit 1; }

WORK="${TMPDIR:-/tmp}/test_eval_${SLURM_JOB_ID}"; mkdir -p "$WORK"
# DEST_BASE: home runs/ by default; the 28-day tiered panel fleet (512/1024 ctx x ~46 rows)
# saves ~10+ GB of rollout npz -> point it at Lustre post_training_GAN (home quota is tight).
DEST="${DEST_BASE:-$EXP/runs}/test_eval_${SLURM_JOB_ID}"; mkdir -p "$DEST"
source "$SQUASHFS_HELPERS"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR"
NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT"
rsync -a "$CKPT_SRC/" "$NODE_CKPT/"
# Optional wide/linf book snapshots for sim init (~36 MB for GOOG 2026-01). Without them
# the sim starts with only L10 liquidity and generated flow drains the book (the WS-
# discrepancy root cause). WIDE_BOOK_DIR = Lustre source dir of *linf_snapshots*.npz.
NODE_WIDE=""
if [ -n "${WIDE_BOOK_DIR:-}" ]; then
  NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE"
  rsync -aL "$WIDE_BOOK_DIR/" "$NODE_WIDE/"   # -L: panel28 merged dir is symlinks
fi

# Survive the ~30-40s post-start scratch wipe (see header): wait it out, verify, heal once.
# Self-heal restages on $TMPDIR if it is still usable, else falls back to /tmp.
sleep 75
if [ ! -f "$NODE_CKPT/latest_checkpoint.json" ] || [ ! -d "$INFER_DATA_DIR_NODE" ]; then
  echo "[test_eval] WARN: node scratch wiped after start (ckpt $( [ -f "$NODE_CKPT/latest_checkpoint.json" ] && echo ok || echo GONE ), farm $( [ -d "$INFER_DATA_DIR_NODE" ] && echo ok || echo GONE )) — restaging once"
  infer_squashfs_cleanup || true
  if ! mkdir -p "$WORK" 2>/dev/null; then
    OLDWORK="$WORK"
    export TMPDIR=/tmp
    WORK="/tmp/test_eval_${SLURM_JOB_ID}"
    echo "[test_eval] WARN: $OLDWORK unusable — FALLING BACK to $WORK (RAM-backed /tmp)"
    mkdir -p "$WORK" || { echo "FATAL: even /tmp unusable"; exit 7; }
  fi
  NODE_CKPT="$WORK/ckpt"; mkdir -p "$NODE_CKPT" || { echo "FATAL: cannot recreate $NODE_CKPT"; exit 7; }
  infer_squashfs_setup "$STOCK" "$MONTHS" "$SHARD_DIR" || { echo "FATAL: restage squashfs failed"; exit 7; }
  rsync -a "$CKPT_SRC/" "$NODE_CKPT/" || { echo "FATAL: restage ckpt failed"; exit 7; }
  if [ -n "${WIDE_BOOK_DIR:-}" ]; then
    NODE_WIDE="$WORK/wide_book"; mkdir -p "$NODE_WIDE" || { echo "FATAL: cannot recreate $NODE_WIDE"; exit 7; }
    rsync -a "$WIDE_BOOK_DIR/" "$NODE_WIDE/" || { echo "FATAL: restage wide books failed"; exit 7; }
  fi
  sleep 75
  { [ -f "$NODE_CKPT/latest_checkpoint.json" ] && [ -d "$INFER_DATA_DIR_NODE" ]; } \
    || { echo "FATAL: scratch wiped AGAIN after restage — node-local storage unusable on this node"; exit 7; }
fi
echo "[test_eval] node scratch verified stable (WORK=$WORK)"

# Held-out eval day filter. get_dataset is called with test_split=0.0 -> it uses EVERY day in
# the data_dir and requires wide-book init for each. We have clean L500 init only for the RTH
# (34200000_57600000) variant of specific days, so build a filtered data_dir with exactly those
# message/orderbook pairs. This avoids the missing-init FATAL on days past our raw-archive cut
# AND drops the extended-hours variant, whose window indexing does not match our RTH-keyed snapshots.
if [ -n "${EVAL_DAYS:-}" ]; then
  FILT="$WORK/eval_days"; mkdir -p "$FILT"
  for D in ${EVAL_DAYS//,/ }; do
    for f in "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*message_10_proc.npy \
             "$INFER_DATA_DIR_NODE"/*"$D"*34200000_57600000*orderbook_10_proc.npy; do
      [ -e "$f" ] && ln -sf "$f" "$FILT/"
    done
  done
  NF=$(ls -1 "$FILT"/*message*.npy 2>/dev/null | wc -l)
  NE=$(echo "$EVAL_DAYS" | tr ',' ' ' | wc -w)
  echo "[test_eval] EVAL_DAYS filter -> $NF/$NE RTH day(s) in $FILT"
  # Refuse a partial eval: if the farm didn't fully mount, fail fast rather than silently
  # producing a 1-day result (the job 5195629 failure mode).
  [ "$NF" -eq "$NE" ] || { echo "FATAL: only $NF/$NE EVAL_DAYS readable in farm (mount/wipe issue) — refusing partial eval"; exit 1; }
  INFER_DATA_DIR_NODE="$FILT"
fi

# Per-lane EGGROLL checkpoint rows (OPTIONAL): only when PROD_DIR is set. With no
# PROD_DIR, EGG_ARGS stays empty and the panel runs the anchor (+ decoding bar)
# alone — the clean-init anchor re-measurement.
EGG_ARGS=()
if [ -n "${EGG_SPECS:-}" ]; then
  # Explicit row list: comma-separated `label=dir` entries. Used when the rows are souped and/or
  # span multiple training jobs (soups + per-solver controls do NOT live under one PROD_DIR). Each
  # dir is a self-contained proj checkpoint located by its own latest_checkpoint.json breadcrumb
  # (no ls/find — Lustre rule). Pass the FINAL ckpt dir directly (e.g. <run>/m_ctrl/best, <soup>).
  for spec in ${EGG_SPECS//,/ }; do
    lbl="${spec%%=*}"; dir="${spec#*=}"
    if [ -f "$dir/latest_checkpoint.json" ]; then
      EGG_ARGS+=(--eggroll "$lbl=$dir")
    else
      echo "[test_eval] WARN: spec '$lbl' -> '$dir' has no breadcrumb — skipping row"
    fi
  done
  [ "${#EGG_ARGS[@]}" -gt 0 ] || { echo "FATAL: EGG_SPECS set but no valid ckpt dirs (breadcrumb missing)"; exit 1; }
elif [ -n "${PROD_DIR:-}" ]; then
  for t in es003_s0 es003_s1 es003_s2 ctrl_shuf; do
    if [ -f "$PROD_DIR/$t/best/latest_checkpoint.json" ]; then
      EGG_ARGS+=(--eggroll "$t=$PROD_DIR/$t/best")
    elif [ -f "$PROD_DIR/$t/latest_checkpoint.json" ]; then
      echo "[test_eval] WARN: $t has no best/ ckpt — falling back to latest"
      EGG_ARGS+=(--eggroll "$t=$PROD_DIR/$t")
    else
      echo "[test_eval] WARN: lane $t missing entirely — skipping row"
    fi
  done
  [ "${#EGG_ARGS[@]}" -gt 0 ] || { echo "FATAL: PROD_DIR set but no EGGROLL ckpts under $PROD_DIR"; exit 1; }
else
  echo "[test_eval] no PROD_DIR/EGG_SPECS set — anchor-only run (anchor + decoding bar)"
fi

# FULL_SPECS (OPTIONAL, additive): comma-separated `label=ckpt_dir:step` FULL-checkpoint
# rows (e.g. the continued-pretraining control) rolled out in the SAME program as the
# anchor — CRN-paired, no cross-program jitter. Each dir is staged node-local like the
# anchor (breadcrumb check first; -aL dereferences the home->Lustre symlink).
if [ -n "${FULL_SPECS:-}" ]; then
  for spec in ${FULL_SPECS//,/ }; do
    lbl="${spec%%=*}"; rest="${spec#*=}"; dir="${rest%:*}"; st="${rest##*:}"
    if [ -f "$dir/latest_checkpoint.json" ]; then
      NODE_FULL="$WORK/full_${lbl}"; mkdir -p "$NODE_FULL"
      rsync -aL "$dir/" "$NODE_FULL/" || { echo "FATAL: staging full ckpt '$lbl' failed"; exit 1; }
      EGG_ARGS+=(--full_ckpt "$lbl=$NODE_FULL:$st")
    else
      echo "[test_eval] WARN: full spec '$lbl' -> '$dir' has no breadcrumb — skipping row"
    fi
  done
fi

export CUDA_VISIBLE_DEVICES=0
PYTMP="$WORK/pytmp"; mkdir -p "$PYTMP" || { echo "FATAL: node-local scratch gone pre-panel"; exit 1; }
# Optional overrides: TOP_N (-1 = full sampling, matches the upstream run_inference default),
# ROWS (comma list, anchor always kept), WIDE_LEVELS (7500 with linf snapshots; sim
# book_depth/nOrders) — used by the WS-discrepancy controls and proper-init evals.
EXTRA_ARGS=()
[ -n "${TOP_N:-}" ]    && EXTRA_ARGS+=(--top_n "$TOP_N")
[ -n "${ROWS:-}" ]     && EXTRA_ARGS+=(--rows "$ROWS")
[ -n "$NODE_WIDE" ]    && EXTRA_ARGS+=(--wide_book_dir "$NODE_WIDE")
WIDE_LEVELS="${WIDE_LEVELS:-50}"
# Tiered 28-day panel fleet (pre-reg §5): N_EVAL_CTX=512 selection rows / 1024 reported
# rows; GEN_CHUNK=512 keeps large tiers inside the proven ~512-rollout/GPU regime (exact
# per-rollout slicing). Timeouts scale with ctx tier — override alongside sbatch -t.
N_EVAL_CTX="${N_EVAL_CTX:-256}"
GEN_CHUNK="${GEN_CHUNK:-0}"
TMPDIR="$PYTMP" timeout "${PANEL_TIMEOUT:-4500}" python -u -m eggroll_gan.eval.test_eval --run \
  --data_dir "$INFER_DATA_DIR_NODE" --ckpt_dir "$NODE_CKPT" --ckpt_step "$CKPT_STEP" \
  --out_dir "$WORK/out" "${EGG_ARGS[@]}" --bar_temp 0.8 \
  --n_cond 500 --n_gen 500 --n_eval_ctx "$N_EVAL_CTX" --n_boot 500 --wide_levels "$WIDE_LEVELS" \
  --gen_chunk "$GEN_CHUNK" \
  --shard off --seed 0 "${EXTRA_ARGS[@]}" \
  2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
RC=${PIPESTATUS[0]}

# ---- LOB-Bench scoring on the saved rollouts (CPU; task #13) ----
# AB_SPLIT_JSON set (tiered 28-day panel fleet, pre-reg §5): score ONLY the AB_SPLIT
# (default A/selection) days via ab_split_scoring + the pre-committed split file — the
# sealed B read stays a deliberate manual step. Unset: legacy full-panel scoring.
RC_LB=1
if [ "$RC" -eq 0 ] && [ "${SKIP_LB:-0}" = "1" ]; then
  # Caller scores the saved rollouts itself (e.g. per-row fan-out with its own --split/
  # --drop_first/--out_tag). Generation-only mode.
  RC_LB=0
  echo "[test_eval] SKIP_LB=1 — in-job LOB-Bench scoring skipped"
elif [ "$RC" -eq 0 ] && [ -n "${AB_SPLIT_JSON:-}" ]; then
  VENVPY="$EXP/third_party/lobbench_venv/bin/python"
  ABROWS="anchor"
  EGGS="${EGG_SPECS:-}"
  for spec in ${EGGS//,/ }; do ABROWS="$ABROWS,${spec%%=*}"; done
  FULLS="${FULL_SPECS:-}"
  for spec in ${FULLS//,/ }; do ABROWS="$ABROWS,${spec%%=*}"; done
  TMPDIR="$PYTMP" timeout "${LB_TIMEOUT:-5400}" "$VENVPY" "$EXP/scripts/eval/ab_split_scoring.py" \
    --eval_dir "$WORK/out" --ctx_days "$WORK/out/ctx_days.json" --split "${AB_SPLIT:-A}" \
    --rows "$ABROWS" --work "$WORK/ab" --split_json "$AB_SPLIT_JSON" \
    --levels 10 --ticker "$STOCK" \
    2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
  RC_LB=${PIPESTATUS[0]}
elif [ "$RC" -eq 0 ]; then
  VENVPY="$EXP/third_party/lobbench_venv/bin/python"
  TMPDIR="$PYTMP" timeout "${AD_TIMEOUT:-1200}" "$VENVPY" "$EXP/eggroll_gan/eval/lobbench_adapter.py" \
    --eval_dir "$WORK/out" --out_dir "$WORK/lobbench" --levels 10 --ticker "$STOCK" \
    2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
  RC_AD=${PIPESTATUS[0]}
  if [ "$RC_AD" -eq 0 ]; then
    TMPDIR="$PYTMP" timeout "${LB_TIMEOUT:-5400}" "$VENVPY" -m eggroll_gan.eval.run_lobbench \
      --bench_dir "$WORK/lobbench" --out_dir "$WORK/out/lobbench_scores" \
      --skip_cond 0 --skip_impact 1 \
      2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
    RC_LB=${PIPESTATUS[0]}
  else
    echo "[test_eval] adapter rc=$RC_AD — LOB-Bench skipped"
  fi
else
  echo "[test_eval] panel rc=$RC — diag: scratch root $( [ -d "${TMPDIR:-/tmp}" ] && echo present || echo MISSING ), WORK $( [ -d "$WORK" ] && echo present || echo MISSING )"
fi

rsync -a "$WORK/out/" "$DEST/" 2>/dev/null || true
chmod -R g+rwX "$DEST" 2>/dev/null || true
chmod g+rw "$EXP/logs/test_eval_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[test_eval] SUMMARY panel_rc=$RC lobbench_rc=$RC_LB -> $DEST"
exit $RC
