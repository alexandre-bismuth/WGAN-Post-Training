#!/bin/bash
# Single-program re-scoring of the unseen-arm PRIMARY B universe (window-disjoint,
# positions 512-2047 = 1536 windows / 28 days). Companion CPU job to one GPU generation
# job (JOB_GEN) that produced anchor + s0pick + s1pick + s2pick + merge_noop in ONE
# program over the 2048-ctx seed-0 CRN draw.
#
# WHY: the sealed final_unseal numbers paired s0/s1 against job 5506605's anchor but s2
# against job 5509802's anchor (two TF32 program replicates ~5.5% apart), and the paper's
# per-feature deltas had no paired CIs at all. Here every row shares ONE anchor rollout
# set and ONE bootstrap draw universe:
#   - one ab_split_scoring (= one lob_bench score_row) per row over the full B universe,
#   - block_bootstrap (patched: per-feature replicate matrices) per row,
#   - paired deltas per seed AND a 3-seed-average paired delta (aggregate + per feature),
#   - figure-data JSON + markdown for the paper plots.
#
# Gates (abort before any scoring): ctx integrity, merge_noop bit-exact (G1), and the
# G-BY plumbing check (by_day+drop_first(0) rescore == JOB_GEN's stored pooled daysA).
# The A-day canary (picks beat anchor on open A windows) is WARN-only.
#
# Lustre checklist: exact-path reads only (rollout npz + jsons under the JOB_GEN dir);
# CSV farms on $TMPDIR; outputs keyed by ${SLURM_JOB_ID}/JOB_GEN; no find/recursive ls;
# no loggers; no agents.
#
# N-row generalization (paper protocol): row sets + geometry are env-driven —
#   ROWS_SEEDS="h_s0pick ... h_s9pick" ROWS_CONTROL="cpt nullpick" N_CTX=4096 DROP_FIRST=0
#   TAG_PREFIX=bbW_ OUT_MD=... FIG_JSON=... UNIVERSE="..." TITLE="..."
# Defaults reproduce the legacy 3-seed unseen-arm program. Wall: ~row_count x score-min /
# SCORE_PAR + bboot; override with sbatch -t for large programs.
#
#SBATCH --job-name=rescoreB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=180G
#SBATCH --gpus-per-node=0
#SBATCH --partition=workq
#SBATCH --time=08:00:00
#SBATCH --output=logs/rescoreB_%j.out
set -uo pipefail
umask 002
EXP="${EXP:-$SLURM_SUBMIT_DIR}"
cd "$EXP"
PT=/lustre/projects/public/shared/post_training_GAN/panel28_evals
PY="$EXP/third_party/lobbench_venv/bin/python"
SJ="$EXP/docs/results/unseen_arm/panel_ab_split.json"
JG="${JOB_GEN:?JOB_GEN required (the single-program generation job id)}"
D="$PT/test_eval_$JG"
# Row sets + universe geometry are env-driven (defaults = the legacy 3-seed unseen-arm
# program). For chain-generated programs set TAG_PREFIX=bbW_ so the by-day rescoring
# never clobbers the chain TEST job's pooled lobbench_scores_testW_* dirs.
ROWS_SEEDS="${ROWS_SEEDS:-s0pick s1pick s2pick}"       # treated picks (averaged)
ROWS_CONTROL="${ROWS_CONTROL:-cpt_s0}"                 # control rows (may be "")
N_CTX="${N_CTX:-2048}"                                 # rollout draw size of the program
DROP_FIRST="${DROP_FIRST:-512}"                        # 0 = decoupled full-universe scoring
NKEEP=$(( N_CTX - DROP_FIRST ))
TAG_PREFIX="${TAG_PREFIX:-testW_}"
SCORE_PAR="${SCORE_PAR:-5}"
OUT_MD="${OUT_MD:-$EXP/docs/results/unseen_arm/rescoreB_single_program.md}"
FIG_JSON="${FIG_JSON:-$EXP/docs/figures/unseen_arm/rescoreB_data.json}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/tmp
W="/tmp/rescoreB_${SLURM_JOB_ID}"; mkdir -p "$W"
LOGD="$EXP/logs/rescoreB_${SLURM_JOB_ID}"; mkdir -p "$LOGD"; chmod g+rwX "$LOGD"
FAIL="$W/FAIL"; : > "$FAIL"
echo "[rescoreB] start $(date) host=$(hostname) gen_job=$JG"

# ---------- G0: integrity gates ------------------------------------------------
"$PY" - "$D" "$SJ" "$N_CTX" "$ROWS_SEEDS" "$ROWS_CONTROL" <<'EOF' || { echo "[rescoreB] G0 GATE FAILURE — aborting"; exit 2; }
import json, os, sys
import numpy as np
D, SJ, NCTX = sys.argv[1], sys.argv[2], int(sys.argv[3])
SEEDS = sys.argv[4].split()
CTRLS = sys.argv[5].split()
sp = json.load(open(SJ)); pan = set(sp["A_selection"]) | set(sp["B_sealed"])
cd = json.load(open(f"{D}/ctx_days.json"))
assert cd["seed"] == 0 and cd["n_eval_ctx"] == NCTX, (cd["seed"], cd["n_eval_ctx"])
assert all(x in pan for x in cd["day"]), "ctx outside the split-file day panel"
for r in ["real", "anchor"] + SEEDS + CTRLS:
    p = f"{D}/rollout_{r}.npz"; assert os.path.isfile(p), f"missing {p}"
# G1 merge_noop bit-exactness: read the legacy gate flag when present, else verify
# directly from the rollouts (chain-generated programs carry no test_eval_results.json).
# G1_DONOR (dir path): for a program that died before its merge_noop row, accept the
# canary from a same-config replicate program — valid ONLY if the donor's real stream is
# bit-identical to this program's (proves the same seed-0 draw), and the donor's
# merge_noop is bit-exact vs the DONOR's own anchor (merge integrity is program-local).
def _bitexact(x, y, what):
    assert set(x.files) == set(y.files), (what, x.files, y.files)
    for k in x.files:
        assert np.array_equal(x[k], y[k]), f"{what} differs on '{k}'"
try:
    res = json.load(open(f"{D}/test_eval_results.json"))
    mn = res.get("rows", {}).get("merge_noop")
    assert mn and mn.get("gate_bitexact") is True, "G1 merge_noop not bit-exact"
except FileNotFoundError:
    if os.path.isfile(f"{D}/rollout_merge_noop.npz"):
        _bitexact(np.load(f"{D}/rollout_anchor.npz"),
                  np.load(f"{D}/rollout_merge_noop.npz"), "G1 merge_noop vs anchor")
    elif os.environ.get("G1_DONOR"):
        DD = os.environ["G1_DONOR"]
        _bitexact(np.load(f"{D}/rollout_real.npz"),
                  np.load(f"{DD}/rollout_real.npz"), "G1 donor real vs this real")
        _bitexact(np.load(f"{DD}/rollout_anchor.npz"),
                  np.load(f"{DD}/rollout_merge_noop.npz"), "G1 donor merge_noop vs donor anchor")
        print(f"[G0] G1 via donor replicate {DD} (real bit-identical; donor canary bit-exact)")
    else:
        raise AssertionError(f"no merge_noop evidence in {D} and no G1_DONOR set")
print(f"[G0] ctx={NCTX} seed=0, rollouts complete, G1 merge_noop bit-exact")
# WARN-only canary (legacy A-day programs only): picks should beat anchor on open A windows
if os.path.isfile(f"{D}/lobbench_scores_daysA/lobbench_results.json"):
    try:
        da = json.load(open(f"{D}/lobbench_scores_daysA/lobbench_results.json"))
        wa = da["anchor"]["summary"]["wasserstein"][0][0]
        for r in SEEDS:
            ws = da[r]["summary"]["wasserstein"][0][0]
            msg = "OK" if ws < wa else "WARNING: pick does NOT beat anchor on A canary"
            print(f"[canary] daysA {r}: anchor {wa:.5f} pick {ws:.5f} ({100*(ws-wa)/wa:+.2f}%) {msg}")
    except Exception as e:
        print(f"[canary] WARN: could not read daysA canary ({e}) — continuing")
EOF
[ $? -eq 0 ] || exit 2

# ---------- G-BY: by_day plumbing gate (by_day scoring must equal pooled) --------
# Reference source, in preference order: the legacy stored pooled daysA scores, else the
# chain TEST job's stored pooled testW_anchor (same split/drop as this scoring), else
# skip with a WARN (no independent pooled reference available).
GBY_MODE=""
if [ -n "${SKIP_GBY:-}" ]; then
  echo "[rescoreB] G-BY skipped by SKIP_GBY env (gate already verified on this program by a prior job)"
elif [ -f "$D/lobbench_scores_daysA/lobbench_results.json" ]; then GBY_MODE="daysA";
elif [ "$TAG_PREFIX" != "testW_" ] && [ -f "$D/lobbench_scores_testW_anchor/lobbench_results.json" ]; then GBY_MODE="testW"; fi
if [ -n "$GBY_MODE" ]; then
  mkdir -p "$W/gby"
  if [ "$GBY_MODE" = "daysA" ]; then
    GBY_SPLIT=(--split A --drop_first 0); GBY_REF="$D/lobbench_scores_daysA/lobbench_results.json"
  else
    GBY_SPLIT=(--split ALL --drop_first "$DROP_FIRST"); GBY_REF="$D/lobbench_scores_testW_anchor/lobbench_results.json"
  fi
  "$PY" scripts/eval/ab_split_scoring.py \
    --eval_dir "$D" --ctx_days "$D/ctx_days.json" \
    "${GBY_SPLIT[@]}" --split_json "$SJ" --rows anchor --by_day \
    --work "$W/gby" --out_tag gbycheck --levels 10 --ticker GOOG > "$LOGD/GBY.log" 2>&1 \
    || { echo "[rescoreB] G-BY rescore crashed — see $LOGD/GBY.log"; exit 2; }
  "$PY" - "$D" "$GBY_REF" <<'EOF' || { echo "[rescoreB] G-BY GATE FAILURE (by_day != stored pooled reference) — aborting"; exit 2; }
import json, math, sys
D, REF = sys.argv[1], sys.argv[2]
new = json.load(open(f"{D}/lobbench_scores_gbycheck/lobbench_results.json"))["anchor"]
ref = json.load(open(REF))["anchor"]
wa, wb = new["summary"]["wasserstein"][0][0], ref["summary"]["wasserstein"][0][0]
assert math.isfinite(wa) and abs(wa - wb) < 1e-9, f"WS-21 {wa} != {wb}"
print(f"[G-BY] by_day scoring == stored pooled reference (WS-21 {wa:.10f}) — plumbing verified")
EOF
  [ $? -eq 0 ] || exit 2
else
  echo "[rescoreB] G-BY WARN: no pooled reference in $D — plumbing gate skipped"
fi

# ---------- fan-out: row-tasks (score -> count-gate -> bboot) -------------------
TASKS="$W/tasks.txt"; : > "$TASKS"
# NO_ANCHOR_TASK=1: per-row fan-out jobs score ONLY their listed rows (the anchor is
# banked by a sibling job); pdeltas below still read the anchor's banked bboot json.
[ -z "${NO_ANCHOR_TASK:-}" ] && echo anchor >> "$TASKS"
for r in $ROWS_SEEDS $ROWS_CONTROL; do echo "$r" >> "$TASKS"; done
run_task () {
  local ROW=$1
  local TAG="${TAG_PREFIX}${ROW}"  # NB: keep on its own line — a single `local A=.. B=$A`
  local WD="$W/${TAG}"             # does NOT see A when evaluating B (bash local gotcha)
  local LOG="$LOGD/${ROW}.log"
  # Resume: a row is complete iff its bboot json persisted to $D (the bench workdir needed
  # by bboot lives on node-local $W, so score-done-but-bboot-missing rows must fully re-run).
  # Makes resubmission after a wall kill idempotent per row.
  if [ -s "$D/lobbench_scores_${TAG}/bboot_${ROW}.json" ]; then
    echo "[rescoreB] task $ROW resume-skip (bboot json present) $(date +%T)"
    return
  fi
  mkdir -p "$WD"
  "$PY" scripts/eval/ab_split_scoring.py \
    --eval_dir "$D" --ctx_days "$D/ctx_days.json" \
    --split ALL --drop_first "$DROP_FIRST" --by_day --rows "$ROW" --split_json "$SJ" \
    --work "$WD" --out_tag "$TAG" --levels 10 --ticker GOOG > "$LOG" 2>&1 \
    || { echo "score_${ROW}" >> "$FAIL"; return; }
  if [ "$DROP_FIRST" -eq 0 ]; then
    grep -q "split ALL: ${N_CTX}/${N_CTX}" "$LOG" || { echo "count_${ROW}" >> "$FAIL"; return; }
  else
    grep -q "drop_first ${DROP_FIRST}: ${N_CTX} -> ${NKEEP}" "$LOG" || { echo "count_${ROW}" >> "$FAIL"; return; }
  fi
  "$PY" -m eggroll_gan.eval.block_bootstrap \
    --bench_dir "$WD/bench_${TAG}" --row "$ROW" --n_boot 500 \
    --check_against "$D/lobbench_scores_${TAG}/lobbench_results.json" \
    --out "$D/lobbench_scores_${TAG}/bboot_${ROW}.json" >> "$LOG" 2>&1 \
    || echo "bboot_${ROW}" >> "$FAIL"
  echo "[rescoreB] task $ROW done $(date +%T)"
}
export -f run_task; export W D PY SJ LOGD FAIL EXP TAG_PREFIX DROP_FIRST N_CTX NKEEP
xargs -a "$TASKS" -L1 -P "$SCORE_PAR" bash -c 'run_task "$@"' _
echo "[rescoreB] scoring+bboot done $(date +%T); failures: $(wc -l < "$FAIL")"
[ -s "$FAIL" ] && { echo "[rescoreB] row-task failures:"; cat "$FAIL"; }

# ---------- paired deltas (per row, all vs the SAME in-job anchor) --------------
PD="$PT/rescoreB_${SLURM_JOB_ID}_pdeltas"; mkdir -p "$PD"
bb () { echo "$D/lobbench_scores_${TAG_PREFIX}$1/bboot_$1.json"; }
for r in $ROWS_SEEDS $ROWS_CONTROL; do
  "$PY" scripts/analysis/paired_delta_from_bboot.py \
    --row "$(bb $r)" --ref "$(bb anchor)" --label "P_${r}-anchor" \
    --out "$PD/P_${r}.json" >> "$LOGD/pdeltas.log" 2>&1 || echo "pdelta_${r}" >> "$FAIL"
done

# ---------- seed-average delta + figure data + markdown -------------------------
FIGD="$(dirname "$FIG_JSON")"; mkdir -p "$FIGD"
OUTMD="$OUT_MD"
"$PY" scripts/analysis/aggregate_rescoreB.py \
  --scores_root "$D" --fig_json "$FIG_JSON" --out_md "$OUTMD" \
  --gen_job "$JG" --score_job "$SLURM_JOB_ID" \
  --seed_rows "$(tr ' ' ',' <<< "$ROWS_SEEDS")" \
  --control_rows "$(tr ' ' ',' <<< "$ROWS_CONTROL")" \
  --tag_prefix "$TAG_PREFIX" \
  ${UNIVERSE:+--universe "$UNIVERSE"} ${TITLE:+--title "$TITLE"} \
  >> "$LOGD/aggregate.log" 2>&1 \
  || echo "aggregate" >> "$W/FAIL"

# ---------- wrap up --------------------------------------------------------------
chmod -R g+rwX "$LOGD" "$PD" "$FIGD" 2>/dev/null
chmod -R g+rwX "$D"/lobbench_scores_${TAG_PREFIX}* "$D/lobbench_scores_gbycheck" 2>/dev/null
chmod g+rw "$OUTMD" "$EXP/logs/rescoreB_${SLURM_JOB_ID}.out" 2>/dev/null
if [ -s "$FAIL" ]; then
  echo "[rescoreB] COMPLETED WITH FAILED PASSES:"; cat "$FAIL"; exit 1
fi
echo "[rescoreB] ALL DONE $(date) -> $OUTMD"
