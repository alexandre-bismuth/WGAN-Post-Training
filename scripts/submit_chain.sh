#!/bin/bash
# Unified train -> selection -> test chain launcher for EGGROLL-GAN post-training arms.
# One script for ANY arm configuration: G/Q geometry, seeds, shuffle nulls, critic, corpus
# and eval universe are all parameters (defaults in scripts/env.sh, every one overridable
# by exporting the variable first).
#
# Stages per invocation:
#   T*   one training job per arm (sequential afterany chain; nodes auto-sized from G*Q)
#   SEL  one eval job: anchor + every arm's step-ckpt grid in ONE program on the selection
#        contexts (positions 0..SEL_CTX-1); per-arm WS-21 argmin -> chain_picks.json
#   TEST one eval job (afterok:SEL): anchor + picks + merge_noop in ONE program on the
#        TEST_CTX draw, scored on the disjoint tail (drop_first SEL_CTX); native LOB-Bench
#        CIs + same-program paired deltas -> chain_test_results.md
#
# All wall times are ETA * 1.5, computed from the arm count / step count (models in env.sh).
# Verdicts and job gating are DIRECTORY-based (training sbatch exit codes are meaningless);
# the eval wrappers sanitize training-only env (UNSEEN_MANIFEST etc.) so corpus settings
# can never leak into the eval dataset build.
#
# Usage
#   ARMS="<tag>:<G>:<Q>[:<seed>[:<fc>]] ..."  (fc: none|shuffle; default seed 0, fc none)
#
#   # dry-run (prints every sbatch line, submits nothing)
#   ARMS="g32q64:32:64 g64q32:64:32" bash scripts/submit_chain.sh
#   # submit an iso-compute pair + evals
#   ARMS="g32q64:32:64 g64q32:64:32" bash scripts/submit_chain.sh --submit
#   # one 4x-budget arm (8192 rollouts/step -> 8 nodes, auto-sized)
#   ARMS="g64q128:64:128" bash scripts/submit_chain.sh --submit
#   # seed-1 arm plus a matched shuffle null
#   ARMS="s1:16:128:1 null1:16:128:1:shuffle" bash scripts/submit_chain.sh --submit
#   # different critic / corpus / anchor: override env first
#   ENC_WHITEN=std UNSEEN_MANIFEST="" DATA_NPY_DIR=/path/to/npy \
#     ARMS="alt:16:128" bash scripts/submit_chain.sh --submit
#   # evals only, over grids from earlier training jobs
#   GRIDS="g32q64=$CKPT_ROOT/s5b_mnode_<jid>/g32q64,..." bash scripts/submit_chain.sh --eval-only --submit
#   # training only (no evals queued)
#   ARMS="..." bash scripts/submit_chain.sh --train-only --submit
#   # selection only (no TEST queued — batched-selection workflow; picks merge later via
#   # the TEST wrapper's comma-separated PICKS_JSON)
#   GRIDS="..." STEP_GRID="0016 0018 0020" bash scripts/submit_chain.sh --eval-only --sel-only --submit
#   # decoupled paper protocol: selection on its own universe, sealed test on the full panel
#   SEL_SPLIT_JSON=docs/reference/sel_split_unseen100.json \
#     SEL_WIDE_BOOK=data/wide_L500_unseen100/GOOG SEL_MONTHS=<months> \
#     PRETRAIN_SEEN_MANIFEST=data/unseen_manifest/unseen_manifest_v1.npz \
#     SEL_CTX=2048 TEST_CTX=4096 ARMS="..." bash scripts/submit_chain.sh --submit
#   # GRPO optimiser-comparison arms (same corpus/critic/eval; up to 4 arms = 4 single-GPU
#   # lanes in ONE 1-node job; all arms must share G:Q; fc column = shuffle-advantage null)
#   OPTIMIZER=grpo ARMS="g_s0:8:50:0 g_s1:8:50:1 g_s2:8:50:2 g_ctrl:8:50:0:shuffle" \
#     bash scripts/submit_chain.sh --submit
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
cd "$EXP"

SUBMIT=0; EVAL_ONLY=0; TRAIN_ONLY=0; SEL_ONLY=0
for a in "$@"; do
  case "$a" in
    --submit) SUBMIT=1 ;;
    --eval-only) EVAL_ONLY=1 ;;
    --train-only) TRAIN_ONLY=1 ;;
    --sel-only) SEL_ONLY=1 ;;
    *) echo "FATAL: unknown flag '$a' (--submit | --eval-only | --train-only | --sel-only)"; exit 1 ;;
  esac
done
[ "$EVAL_ONLY" -eq 1 ] && [ "$TRAIN_ONLY" -eq 1 ] && { echo "FATAL: pick one of --eval-only/--train-only"; exit 1; }
[ "$SEL_ONLY" -eq 1 ] && [ "$TRAIN_ONLY" -eq 1 ] && { echo "FATAL: --sel-only conflicts with --train-only"; exit 1; }

mins_to_wall () { printf "%02d:%02d:00" $(( $1 / 60 )) $(( $1 % 60 )); }

sub () {  # sub <label> <sbatch args...> -> echoes job id (or DRY)
  local LBL=$1; shift
  echo "[chain] $LBL: sbatch $*" >&2
  if [ "$SUBMIT" -eq 1 ]; then
    local JID
    JID=$(sbatch "$@" | awk '{print $NF}')
    echo "$(date +%F_%T) $LBL $JID" >> "$TRACK"
    echo "[chain] -> $LBL = job $JID" >&2
    echo "$JID"
  else
    echo "DRY"
  fi
}

TRACK="$EXP/logs/chain_jobids.txt"
mkdir -p "$EXP/logs"

# ── arms: parse + validate geometry ──────────────────────────────────────────
declare -a A_TAG A_G A_Q A_SEED A_FC A_LR A_NODES
NARMS=0
if [ "$EVAL_ONLY" -eq 0 ]; then
  [ -n "${ARMS:-}" ] || { echo "FATAL: set ARMS=\"tag:G:Q[:seed[:fc[:lr]]] ...\""; exit 1; }
  OPTIMIZER="${OPTIMIZER:-eggroll}"
  for spec in $ARMS; do
    IFS=: read -r TAG G Q SEED FC ALR <<< "$spec"
    SEED="${SEED:-0}"; FC="${FC:-none}"
    [[ "$G" =~ ^[0-9]+$ && "$Q" =~ ^[0-9]+$ ]] || { echo "FATAL: bad arm spec '$spec'"; exit 1; }
    if [ "$OPTIMIZER" = "grpo" ]; then
      # single-GPU lanes, --shard off: no device-divisibility constraints; G need not be even
      # (no antithetic pairs in PG); pool floor is the GRPO ctx pool, not NCTXPOOL.
      (( ${GRPO_CTX_POOL:-64} >= Q )) || { echo "FATAL: $TAG: GRPO_CTX_POOL=${GRPO_CTX_POOL:-64} < Q=$Q"; exit 1; }
      NODES=1
    else
      (( G % 2 == 0 )) || { echo "FATAL: $TAG: G=$G must be even (antithetic pairs)"; exit 1; }
      (( NCTXPOOL >= Q )) || { echo "FATAL: $TAG: NCTXPOOL=$NCTXPOOL < Q=$Q"; exit 1; }
      PER_NODE=$(( SHARD_ROLLOUTS * GPUS_PER_NODE ))
      NODES=$(( (G * Q + PER_NODE - 1) / PER_NODE ))
      NDEV=$(( NODES * GPUS_PER_NODE ))
      (( G * Q % NDEV == 0 ))    || { echo "FATAL: $TAG: G*Q=$((G*Q)) %% devices=$NDEV != 0"; exit 1; }
      (( NEVAL % NDEV == 0 ))    || { echo "FATAL: $TAG: NEVAL=$NEVAL %% devices=$NDEV != 0"; exit 1; }
      (( NCTXPOOL % NDEV == 0 )) || { echo "FATAL: $TAG: NCTXPOOL=$NCTXPOOL %% devices=$NDEV != 0"; exit 1; }
      (( (NCTXPOOL + NEVAL) % NDEV == 0 )) || { echo "FATAL: $TAG: pool+eval %% devices=$NDEV != 0"; exit 1; }
    fi
    if [ -n "${ALR:-}" ] && [ "$OPTIMIZER" != "grpo" ]; then
      echo "FATAL: $TAG: per-arm lr field is grpo-only (eggroll arms take the global env)"; exit 1
    fi
    A_TAG[NARMS]=$TAG; A_G[NARMS]=$G; A_Q[NARMS]=$Q
    A_SEED[NARMS]=$SEED; A_FC[NARMS]=$FC; A_LR[NARMS]="${ALR:-}"; A_NODES[NARMS]=$NODES
    NARMS=$(( NARMS + 1 ))
    echo "[chain] arm $TAG: G=$G Q=$Q ($((G*Q)) rollouts/step) seed=$SEED fc=$FC${ALR:+ lr=$ALR} -> $NODES node(s) [$OPTIMIZER]"
  done
  if [ "$OPTIMIZER" = "grpo" ]; then
    (( NARMS <= 4 )) || { echo "FATAL: OPTIMIZER=grpo takes <= 4 arms (one GPU lane each)"; exit 1; }
    for (( i = 1; i < NARMS; i++ )); do
      { [ "${A_G[i]}" = "${A_G[0]}" ] && [ "${A_Q[i]}" = "${A_Q[0]}" ]; } \
        || { echo "FATAL: grpo arms must share one G:Q (lane env is job-wide)"; exit 1; }
    done
  fi
else
  [ -n "${GRIDS:-}" ] || { echo "FATAL: --eval-only needs GRIDS=\"tag=grid_dir[,...]\""; exit 1; }
  NARMS=$(awk -F, '{print NF}' <<< "$GRIDS")
fi

# ── preflight (fail fast on anything the jobs would need) ────────────────────
[ -f "$CKPT_SRC/latest_checkpoint.json" ] || { echo "FATAL: anchor breadcrumb missing: $CKPT_SRC"; exit 1; }
if [ "$EVAL_ONLY" -eq 0 ]; then
  if [ -n "$UNSEEN_MANIFEST" ]; then
    [ -f "$UNSEEN_MANIFEST" ] || { echo "FATAL: UNSEEN_MANIFEST missing: $UNSEEN_MANIFEST"; exit 1; }
  fi
  [ -d "$WIDE_BOOK_TRAIN" ] || { echo "FATAL: WIDE_BOOK_TRAIN missing: $WIDE_BOOK_TRAIN"; exit 1; }
  if [ -n "$TRAIN_DAYS_JSON" ]; then
    [ -f "$TRAIN_DAYS_JSON" ] || { echo "FATAL: TRAIN_DAYS_JSON missing: $TRAIN_DAYS_JSON"; exit 1; }
    TRAIN_DAYS=$("$PYBIN" -c "import json;print(','.join(json.load(open('$TRAIN_DAYS_JSON'))['days']))")
    export TRAIN_DAYS
  fi
fi
if [ "$TRAIN_ONLY" -eq 0 ]; then
  [ -f "$SPLIT_JSON" ] || { echo "FATAL: SPLIT_JSON missing: $SPLIT_JSON"; exit 1; }
  [ -d "$WIDE_BOOK_EVAL" ] || { echo "FATAL: WIDE_BOOK_EVAL missing: $WIDE_BOOK_EVAL"; exit 1; }
  EVAL_DAYS=$("$PYBIN" -c "import json;print(','.join(json.load(open('$SPLIT_JSON'))['panel_days']))")
  [ -f "$SEL_SPLIT_JSON" ] || { echo "FATAL: SEL_SPLIT_JSON missing: $SEL_SPLIT_JSON"; exit 1; }
  [ -d "$SEL_WIDE_BOOK" ] || { echo "FATAL: SEL_WIDE_BOOK missing: $SEL_WIDE_BOOK"; exit 1; }
  SEL_EVAL_DAYS=$("$PYBIN" -c "import json;print(','.join(json.load(open('$SEL_SPLIT_JSON'))['panel_days']))")
  if [ -n "${PRETRAIN_SEEN_MANIFEST:-}" ]; then
    [ -f "$PRETRAIN_SEEN_MANIFEST" ] || { echo "FATAL: PRETRAIN_SEEN_MANIFEST missing: $PRETRAIN_SEEN_MANIFEST"; exit 1; }
  fi
fi
# PRETRAIN_SEEN_MANIFEST is a SEL-job-only mask. Scope it NOW: the training jobs inherit
# the launcher env via --export=ALL, and the trainer's shared dataset builder would trip
# the UNSEEN/SEEN mutual-exclusion assert if both manifests reached it.
SEL_SEEN_MANIFEST="${PRETRAIN_SEEN_MANIFEST:-}"
unset PRETRAIN_SEEN_MANIFEST 2>/dev/null || true
echo "[chain] preflight OK"

# ── T*: training arms (sequential; each arm may use several nodes) ───────────
GRIDS_OUT=""
if [ "$EVAL_ONLY" -eq 0 ]; then
  export WIDE_BOOK_DIR="$WIDE_BOOK_TRAIN"
  if [ "${OPTIMIZER:-eggroll}" = "grpo" ]; then
    # ONE 1-node job; each arm = one GPU lane. Wall from the measured ~3.2 s/rollout/GPU.
    GG="${A_G[0]}"; QQ="${A_Q[0]}"
    LANES=""
    for (( i = 0; i < NARMS; i++ )); do
      LANES="$LANES,$i:${A_SEED[i]}:${A_FC[i]}:${A_TAG[i]}${A_LR[i]:+:${A_LR[i]}}"
    done
    LANES="${LANES#,}"
    export GG QQ LANES
    GRPO_STEP_MIN="${GRPO_STEP_MIN:-$(( (GG * QQ * 32 + 599) / 600 ))}"
    TMINS=$(( (TRAIN_STAGE_MIN + NSTEPS * GRPO_STEP_MIN) * 3 / 2 ))
    (( TMINS > 1410 )) && TMINS=1410            # 23.5 h QOS ceiling; lanes resume via breadcrumb
    TRAIN_WALL=$(mins_to_wall "$TMINS")
    export LANE_TIMEOUT="${LANE_TIMEOUT:-$(( TMINS * 60 * 9 / 10 ))}"
    JID=$(sub "T_grpo" -t "$TRAIN_WALL" --nodes=1 --export=ALL scripts/train/_run_grpo_chain.sbatch)
    for (( i = 0; i < NARMS; i++ )); do
      GRIDS_OUT="$GRIDS_OUT,${A_TAG[i]}=$CKPT_ROOT/grpo_chain_${JID}/${A_TAG[i]}"
    done
    TRAIN_DEPS="$JID"
    [ "$SUBMIT" -eq 1 ] && sleep 30
  else
    TRAIN_WALL=$(mins_to_wall $(( (TRAIN_STAGE_MIN + NSTEPS * TRAIN_STEP_MIN) * 3 / 2 )))
    export LANE_TIMEOUT="${LANE_TIMEOUT:-$(( (TRAIN_STAGE_MIN + NSTEPS * TRAIN_STEP_MIN) * 60 * 13 / 10 ))}"
    PREV=""
    for (( i = 0; i < NARMS; i++ )); do
      export TAG="${A_TAG[i]}" ESG="${A_G[i]}" ESQ="${A_Q[i]}" SEED="${A_SEED[i]}" FC="${A_FC[i]}"
      DEP=(); [ -n "$PREV" ] && [ "$PREV" != "DRY" ] && DEP=(--dependency=afterany:$PREV)
      JID=$(sub "T_${TAG}" -t "$TRAIN_WALL" --nodes="${A_NODES[i]}" "${DEP[@]}" --export=ALL \
            scripts/train/_run_eggroll_multinode.sbatch)
      GRIDS_OUT="$GRIDS_OUT,${TAG}=$CKPT_ROOT/s5b_mnode_${JID}/${TAG}"
      PREV="$JID"
      [ "$SUBMIT" -eq 1 ] && sleep 30
    done
    TRAIN_DEPS="$PREV"
  fi
  GRIDS_OUT="${GRIDS_OUT#,}"
else
  GRIDS_OUT="$GRIDS"
  TRAIN_DEPS=""
fi
[ "$TRAIN_ONLY" -eq 1 ] && { echo "[chain] train-only: done ($NARMS arm(s))"; exit 0; }

# ── SEL + TEST (eval env assembled here; wrappers additionally sanitize) ─────
export DEST_BASE
export CHAIN_ARMS="$GRIDS_OUT"
unset UNSEEN_MANIFEST DATA_MODE DATA_NPY_DIR TRAIN_DAYS 2>/dev/null || true

# Decoupled protocol: selection on its own day universe (SEL_SPLIT_JSON) while the sealed
# test scores the FULL TEST_CTX draw on the panel (drop_first 0, no prefix-nesting gate).
# Auto-enabled when the two split files differ; TEST_DECOUPLED=0/1 forces either mode.
TEST_SPLIT_JSON="$SPLIT_JSON"
TEST_EVAL_DAYS="$EVAL_DAYS"
if [ "${TEST_DECOUPLED:-auto}" = "auto" ]; then
  if [ "$SEL_SPLIT_JSON" != "$TEST_SPLIT_JSON" ]; then TEST_DECOUPLED=1; else TEST_DECOUPLED=0; fi
fi

N_CKPTS=$(( NSTEPS / CKPT_EVERY ))
# An explicit STEP_GRID overrides the uniform CKPT_EVERY grid in the SEL wrapper — size
# the wall from the grid actually scored, not from the modulo arithmetic.
if [ -n "${STEP_GRID:-}" ]; then
  N_CKPTS=$(wc -w <<< "$STEP_GRID")
  export STEP_GRID
fi
SEL_ROWS=$(( 1 + N_CKPTS * NARMS ))
SEL_MINS=$(( (EVAL_STAGE_MIN + SEL_ROWS * SEL_ROW_MIN) * 3 / 2 ))
if (( SEL_MINS > 1410 )); then
  echo "[chain] WARN: SEL wall ${SEL_MINS}min capped at the 23.5h QOS ceiling (${SEL_ROWS} rows) — prefer batching arms via --eval-only/--sel-only" >&2
  SEL_MINS=1410
fi
SEL_WALL=$(mins_to_wall "$SEL_MINS")
# The eval wrapper's internal panel timeout must scale with the ROW COUNT, not sit at its fixed
# 18000s default — a denser step grid outgrew it (job 5717708: 41 rows x ~8.6 min > 5 h, killed
# 6 rows short). rows x SEL_ROW_MIN x 1.3, floored at the old default; explicit env still wins.
USER_PANEL_TIMEOUT="${PANEL_TIMEOUT:-}"
export PANEL_TIMEOUT="${USER_PANEL_TIMEOUT:-$(( SEL_ROWS * SEL_ROW_MIN * 60 * 13 / 10 > 18000 ? SEL_ROWS * SEL_ROW_MIN * 60 * 13 / 10 : 18000 ))}"
# SEL job env: the selection universe. PRETRAIN_SEEN_MANIFEST (captured + unset at
# preflight so it can NEVER reach a training job) is re-exported for THIS job only and
# restricts the draw to pretraining-seen slots.
export EVAL_DAYS="$SEL_EVAL_DAYS" SPLIT_JSON="$SEL_SPLIT_JSON"
export WIDE_BOOK_DIR="$SEL_WIDE_BOOK" MONTHS="$SEL_MONTHS"
[ -n "$SEL_SEEN_MANIFEST" ] && export PRETRAIN_SEEN_MANIFEST="$SEL_SEEN_MANIFEST"
DEP_SEL=(); [ -n "$TRAIN_DEPS" ] && [ "$TRAIN_DEPS" != "DRY" ] && DEP_SEL=(--dependency=afterany:$TRAIN_DEPS)
SEL=$(sub SEL -t "$SEL_WALL" --nodes=1 "${DEP_SEL[@]}" --export=ALL scripts/eval/_run_chain_sel_eval.sbatch)
[ "$SUBMIT" -eq 1 ] && sleep 30
if [ "$SEL_ONLY" -eq 1 ]; then
  echo "[chain] sel-only: SEL=$SEL (picks: $DEST_BASE/test_eval_${SEL}/chain_picks.json)"
  exit 0
fi

# TEST job env: the sealed panel universe. PRETRAIN_SEEN_MANIFEST must NOT reach the test
# job — the panel days are absent from the manifest and the mask would FATAL there.
export EVAL_DAYS="$TEST_EVAL_DAYS" SPLIT_JSON="$TEST_SPLIT_JSON"
export WIDE_BOOK_DIR="$WIDE_BOOK_EVAL" MONTHS="$EVAL_MONTHS" TEST_DECOUPLED
unset PRETRAIN_SEEN_MANIFEST 2>/dev/null || true
export PICKS_JSON="$DEST_BASE/test_eval_${SEL}/chain_picks.json"
export SEL_EVAL_DIR="$DEST_BASE/test_eval_${SEL}"
# Extra same-program rows (null pick / CPT control) are generated and scored too —
# count them into the wall, or the TEST job gets killed rows short.
N_EXTRA=0
[ -n "${EXTRA_EGG_SPECS:-}" ] && N_EXTRA=$(( N_EXTRA + $(awk -F, '{print NF}' <<< "$EXTRA_EGG_SPECS") ))
[ -n "${FULL_SPECS:-}" ] && N_EXTRA=$(( N_EXTRA + $(awk -F, '{print NF}' <<< "$FULL_SPECS") ))
TEST_ROWS=$(( NARMS + 2 + N_EXTRA ))
TEST_MINS=$(( (EVAL_STAGE_MIN + TEST_ROWS * TEST_GENROW_MIN + (NARMS + 1 + N_EXTRA) * TEST_SCOREROW_MIN) * 3 / 2 ))
if (( TEST_MINS > 1410 )); then
  echo "[chain] WARN: TEST wall ${TEST_MINS}min capped at the 23.5h QOS ceiling" >&2
  TEST_MINS=1410
fi
TEST_WALL=$(mins_to_wall "$TEST_MINS")
export PANEL_TIMEOUT="${USER_PANEL_TIMEOUT:-$(( TEST_ROWS * TEST_GENROW_MIN * 60 * 13 / 10 > 18000 ? TEST_ROWS * TEST_GENROW_MIN * 60 * 13 / 10 : 18000 ))}"
DEP_TEST=(); [ "$SEL" != "DRY" ] && DEP_TEST=(--dependency=afterok:$SEL --kill-on-invalid-dep=yes)
TEST=$(sub TEST -t "$TEST_WALL" --nodes=1 "${DEP_TEST[@]}" --export=ALL scripts/eval/_run_chain_test_eval.sbatch)

if [ "$SUBMIT" -eq 1 ]; then
  chmod g+rw "$TRACK" 2>/dev/null || true
  echo "[chain] chain: train=[${TRAIN_DEPS:-eval-only}] -> SEL=$SEL -> TEST=$TEST (tracked in $TRACK)"
  echo "[chain] result: $DEST_BASE/test_eval_${TEST}/chain_test_results.md"
else
  echo "[chain] DRY RUN (add --submit). Walls: train=${TRAIN_WALL:--} sel=$SEL_WALL test=$TEST_WALL"
fi
