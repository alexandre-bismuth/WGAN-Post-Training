#!/bin/bash
# Score-only SEL salvage: run the chain-SEL scoring fan-out + argmin aggregation over rollouts
# that ALREADY EXIST in a test_eval dir (a SEL whose generation completed but whose scoring
# stage was killed — job 5717708's mode: PANEL_TIMEOUT hit AFTER all live-arm rollouts landed).
# Generation is NOT rerun; the rollouts stay one-program CRN-comparable. ctx_days.json may be
# ported from ANY same-config eval's dir — the seed-0 idx draw is deterministic (verified for
# 5717708 by the bit-identical rollout_real.npz md5 vs 5716070).
#
# The scoring + aggregation blocks mirror _run_chain_sel_eval.sbatch VERBATIM (same
# ab_split_scoring invocation, same argmin python), so the picks are protocol-identical.
#
# Env (exported by the caller): SEL_DIR (the test_eval dir holding rollout_*.npz + ctx_days),
# CHAIN_ARMS="tag=grid_dir[,...]" (arms to score — arms may be a SUBSET of what was generated),
# NSTEPS CKPT_EVERY (or STEP_GRID), SPLIT_JSON, SEL_CTX, STOCK, LOBBENCH_PY.
#
#SBATCH --job-name=sel_salvage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --time=02:30:00
#SBATCH --output=logs/sel_salvage_%j.out
set -uo pipefail
umask 002

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
cd "$EXP"
PY="${LOBBENCH_PY:-$EXP/third_party/lobbench_venv/bin/python}"
SJ="${SPLIT_JSON:?SPLIT_JSON must be exported by the caller}"
SRC="${SEL_DIR:?SEL_DIR must be exported by the caller}"
SEL_CTX="${SEL_CTX:-512}"
STEPS="${STEP_GRID:-$(seq -f "%04g" "${CKPT_EVERY:-5}" "${CKPT_EVERY:-5}" "${NSTEPS:-50}" | tr '\n' ' ')}"
[ -n "${CHAIN_ARMS:-}" ] || { echo "[salvage] VERDICT: FAIL (CHAIN_ARMS not set)"; exit 2; }

# ---- gate: every requested row's rollout + the shared files exist ------------------------
ROWS=""
MISSING=0
for f in ctx_days.json rollout_real.npz rollout_anchor.npz; do
  [ -f "$SRC/$f" ] || { echo "[salvage] MISSING $SRC/$f"; MISSING=1; }
done
for spec in ${CHAIN_ARMS//,/ }; do
  ARM="${spec%%=*}"
  for s in $STEPS; do
    ROWS="$ROWS,${ARM}_st$s"
    [ -f "$SRC/rollout_${ARM}_st$s.npz" ] || { echo "[salvage] MISSING rollout_${ARM}_st$s.npz"; MISSING=1; }
  done
done
ROWS="${ROWS#,}"
[ "$MISSING" -eq 0 ] || { echo "[salvage] VERDICT: FAIL (rollouts incomplete in $SRC)"; exit 1; }
echo "[salvage] gate OK: $(echo "$ROWS" | tr ',' '\n' | wc -l) row(s) + anchor in $SRC, steps [$STEPS]"

# ---- scoring fan-out (verbatim _run_chain_sel_eval.sbatch) -------------------------------
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGD="$EXP/logs/sel_salvage_${SLURM_JOB_ID}"; mkdir -p "$LOGD"
W="/tmp/sel_salvage_${SLURM_JOB_ID}"; mkdir -p "$W"
FAIL="$W/FAIL"; : > "$FAIL"
TASKS="$W/tasks.txt"; : > "$TASKS"
echo "anchor" >> "$TASKS"
for r in ${ROWS//,/ }; do echo "$r" >> "$TASKS"; done

run_row () {
  local R=$1
  local TAG
  TAG="sel_${R}"
  mkdir -p "$W/$R"
  "$PY" scripts/eval/ab_split_scoring.py \
    --eval_dir "$SRC" --ctx_days "$SRC/ctx_days.json" \
    --split ALL --split_json "$SJ" --rows "$R" --drop_first 0 \
    --work "$W/$R" --out_tag "$TAG" --levels 10 --ticker "${STOCK:-GOOG}" \
    > "$LOGD/$TAG.log" 2>&1 || echo "rc_$R" >> "$FAIL"
  grep -q "split ALL: ${SEL_CTX}/${SEL_CTX}" "$LOGD/$TAG.log" || echo "count_$R" >> "$FAIL"
}
export -f run_row; export W SRC PY SJ LOGD FAIL STOCK EXP SEL_CTX
xargs -a "$TASKS" -L1 -P 8 bash -c 'run_row "$@"' _
echo "[salvage] scoring done; failed tasks: $(wc -l < "$FAIL")"
[ -s "$FAIL" ] && { echo "[salvage] failed rows:"; cat "$FAIL"; }

# ---- aggregate: per-arm WS-21 curves -> argmin picks (verbatim) --------------------------
"$PY" - "$SRC" "$CHAIN_ARMS" "$STEPS" <<'EOF'
import json, math, os, sys
SRC, ARMS, STEPS_S = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = [int(s) for s in STEPS_S.split()]
GRIDS = dict(s.split("=", 1) for s in ARMS.split(","))

def ws(row):
    p = f"{SRC}/lobbench_scores_sel_{row}/lobbench_results.json"
    v = json.load(open(p))[row]["summary"]["wasserstein"][0][0]
    assert math.isfinite(v), f"{row}: WS not finite"
    return v

anc = ws("anchor")
picks, lines = {}, [f"# Chain selection curves ({len(STEPS)} steps/arm, salvage job {os.environ.get('SLURM_JOB_ID','?')})",
                    "", f"anchor WS-21 = {anc:.5f} (same program)", ""]
for arm, grid in GRIDS.items():
    curve = {s: ws(f"{arm}_st{s:04d}") for s in STEPS}
    best = min(curve, key=curve.get)
    picks[arm] = {"step": best, "ws21": curve[best], "ws21_anchor": anc,
                  "delta": curve[best] - anc, "rel": (curve[best] - anc) / anc,
                  "ckpt": f"{grid}/step{best:04d}", "curve": {str(s): curve[s] for s in STEPS}}
    lines += [f"## {arm}", "", "| step | WS-21 | delta vs anchor |", "|---|---|---|"]
    for s in STEPS:
        tag = " <- PICK" if s == best else ""
        lines.append(f"| {s} | {curve[s]:.5f} | {curve[s]-anc:+.5f}{tag} |")
    lines += ["", f"- pick: st{best:04d}, WS-21 {curve[best]:.5f} "
                  f"({100*(curve[best]-anc)/anc:+.2f}% vs anchor)", ""]
json.dump(picks, open(f"{SRC}/chain_picks.json", "w"), indent=1)
open(f"{SRC}/chain_selection_curves.md", "w").write("\n".join(lines) + "\n")
for f in ("chain_picks.json", "chain_selection_curves.md"):
    os.chmod(f"{SRC}/{f}", 0o664)
print("\n".join(lines))
EOF
AGG_RC=$?

chmod -R g+rwX "$SRC" "$LOGD" 2>/dev/null || true
chmod g+rw "$EXP/logs/sel_salvage_${SLURM_JOB_ID}.out" 2>/dev/null || true

if [ "$AGG_RC" -eq 0 ] && [ ! -s "$FAIL" ] && [ -f "$SRC/chain_picks.json" ]; then
  echo "[salvage] VERDICT: PASS ($SRC/chain_picks.json)"
  exit 0
fi
echo "[salvage] VERDICT: FAIL (agg_rc=$AGG_RC fails=$(wc -l < "$FAIL"))"
exit 1
