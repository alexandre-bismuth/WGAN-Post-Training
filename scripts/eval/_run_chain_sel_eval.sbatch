#!/bin/bash
# Chain SELECTION eval (submitted by scripts/submit_chain.sh with afterany on the training
# arms). Generates, in ONE program (CRN, no cross-program TF32 jitter): anchor + the full
# step-ckpt grid of every arm in CHAIN_ARMS over the SEL_CTX-context seed-0 CRN draw = the
# selection universe (draw positions 0..SEL_CTX-1 of the TEST_CTX draw — same seed, same
# day panel, so the prefix is identical; the test job verifies this). Scores every row on
# ALL selection windows (per-row fan-out, per-row --out_tag so no shared output dir can be
# clobbered) and writes chain_picks.json = per-arm WS-21 argmin step.
#
# afterok on the TRAINING jobs is meaningless (the training sbatch always exits 0), so this
# job verifies the ckpt grids from the DIRECTORIES (exact-path breadcrumb stats, no ls).
# Likewise the final PASS/FAIL verdict is computed from what actually landed in DEST — not
# from child exit codes or log noise: rollouts present + all rows scored finite + picks
# written => PASS.
#
# Env (from the launcher): CHAIN_ARMS="tag=grid_dir[,...]", EVAL_DAYS, CKPT_SRC/CKPT_STEP,
# MAMBA3_EPS_COMPAT, STOCK, MONTHS, WIDE_BOOK_DIR (eval snapshots), DEST_BASE, SPLIT_JSON,
# SEL_CTX, NSTEPS, CKPT_EVERY.
#
#SBATCH --job-name=chain_sel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --output=logs/chain_sel_%j.out
set -uo pipefail
umask 002

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
cd "$EXP"
PY="${LOBBENCH_PY:-$EXP/third_party/lobbench_venv/bin/python}"
SJ="${SPLIT_JSON:?SPLIT_JSON must be exported by the launcher}"
DEST="$DEST_BASE/test_eval_${SLURM_JOB_ID}"
SEL_CTX="${SEL_CTX:-512}"
STEPS="${STEP_GRID:-$(seq -f "%04g" "${CKPT_EVERY:-5}" "${CKPT_EVERY:-5}" "${NSTEPS:-50}" | tr '\n' ' ')}"

# ---- gate 1 (directory truth, not training rc): every arm's step grid complete -----------
[ -n "${CHAIN_ARMS:-}" ] || { echo "[chain-sel] VERDICT: FAIL (CHAIN_ARMS not set)"; exit 2; }
MISSING=0
NARMS=0
for spec in ${CHAIN_ARMS//,/ }; do
  GRID="${spec#*=}"
  NARMS=$((NARMS + 1))
  for s in $STEPS; do
    if [ ! -f "$GRID/step$s/latest_checkpoint.json" ]; then
      echo "[chain-sel] MISSING ckpt breadcrumb: $GRID/step$s"; MISSING=1
    fi
  done
done
[ "$MISSING" -eq 0 ] || { echo "[chain-sel] VERDICT: FAIL (training grid incomplete)"; exit 2; }
echo "[chain-sel] gate 1 OK: $NARMS arm(s), steps [$STEPS]"

# ---- rows: anchor (implicit) + one row per (arm, step) ------------------------------------
EGG_SPECS=""
ROWS=""
for spec in ${CHAIN_ARMS//,/ }; do
  ARM="${spec%%=*}"
  GRID="${spec#*=}"
  for s in $STEPS; do
    EGG_SPECS="$EGG_SPECS,${ARM}_st$s=$GRID/step$s"
    ROWS="$ROWS,${ARM}_st$s"
  done
done
export EGG_SPECS="${EGG_SPECS#,}"
export ROWS="${ROWS#,}"
export SKIP_LB=1 N_EVAL_CTX="$SEL_CTX" GEN_CHUNK="${GEN_CHUNK:-512}" PANEL_TIMEOUT="${PANEL_TIMEOUT:-18000}"
# Training-side env must NOT reach the eval program: a non-empty UNSEEN_MANIFEST makes the
# shared dataset builder (s3_es_rollout) mask eval days against the training manifest,
# which aborts on out-of-corpus eval days. PRETRAIN_SEEN_MANIFEST is the deliberate
# exception (decoupled paper protocol): it restricts the SELECTION draw to pretraining-
# seen slots — disjoint by construction from all post-training data.
unset AB_SPLIT_JSON UNSEEN_MANIFEST DATA_MODE DATA_NPY_DIR TRAIN_DAYS 2>/dev/null || true
# TEST-only row env must not leak extra rows into the selection program.
unset FULL_SPECS EXTRA_EGG_SPECS 2>/dev/null || true
[ -n "${PRETRAIN_SEEN_MANIFEST:-}" ] && \
  echo "[chain-sel] selection universe masked to pretraining-SEEN slots: $PRETRAIN_SEEN_MANIFEST"

# ---- generation: the proven test_eval body as a child (same SLURM_JOB_ID => same DEST) ---
bash scripts/eval/_run_test_eval.sbatch
CHILD_RC=$?
echo "[chain-sel] child test_eval rc=$CHILD_RC (informational — verdict comes from DEST)"

WORKOUT="/tmp/test_eval_${SLURM_JOB_ID}/out"
SRC="$WORKOUT"; [ -f "$SRC/ctx_days.json" ] || SRC="$DEST"

# ---- gate 2 (directory truth): every rollout landed --------------------------------------
MISSING=0
for f in ctx_days.json rollout_real.npz rollout_anchor.npz; do
  [ -f "$SRC/$f" ] || { echo "[chain-sel] MISSING $SRC/$f"; MISSING=1; }
done
for r in ${ROWS//,/ }; do
  [ -f "$SRC/rollout_$r.npz" ] || { echo "[chain-sel] MISSING rollout_$r.npz"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "[chain-sel] VERDICT: FAIL (rollouts incomplete in $SRC)"; exit 1; }
echo "[chain-sel] gate 2 OK: all rows + real present in $SRC"

# ---- scoring fan-out: one task per row, per-row out_tag (clobber-proof) ------------------
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGD="$EXP/logs/chain_sel_${SLURM_JOB_ID}"; mkdir -p "$LOGD"
W="/tmp/chain_sel_score_${SLURM_JOB_ID}"; mkdir -p "$W"
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
echo "[chain-sel] scoring done; failed tasks: $(wc -l < "$FAIL")"
[ -s "$FAIL" ] && { echo "[chain-sel] failed rows:"; cat "$FAIL"; }

# ---- aggregate: per-arm WS-21 curves -> argmin picks -------------------------------------
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
picks, lines = {}, [f"# Chain selection curves ({len(STEPS)} steps/arm, job {os.environ.get('SLURM_JOB_ID','?')})",
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

# ---- persist scores + picks to DEST, then verdict from DEST ------------------------------
if [ "$SRC" != "$DEST" ]; then
  rsync -a "$SRC"/lobbench_scores_sel_* "$SRC/chain_picks.json" "$SRC/chain_selection_curves.md" \
    "$DEST/" 2>/dev/null || true
fi
chmod -R g+rwX "$DEST" "$LOGD" 2>/dev/null || true
chmod g+rw "$EXP/logs/chain_sel_${SLURM_JOB_ID}.out" 2>/dev/null || true

if [ "$AGG_RC" -eq 0 ] && [ ! -s "$FAIL" ] && [ -f "$DEST/chain_picks.json" ]; then
  if "$PY" -c "
import json, os
p = json.load(open('$DEST/chain_picks.json'))
assert p and all(os.path.isfile(v['ckpt'] + '/latest_checkpoint.json') for v in p.values())
for a, v in sorted(p.items()):
    print(f'[chain-sel] pick {a} -> {v[\"ckpt\"]}')"; then
    echo "[chain-sel] VERDICT: PASS ($DEST)"
    exit 0
  fi
fi
echo "[chain-sel] VERDICT: FAIL (agg_rc=$AGG_RC fails=$(wc -l < "$FAIL") picks=$( [ -f "$DEST/chain_picks.json" ] && echo present || echo missing ))"
exit 1
