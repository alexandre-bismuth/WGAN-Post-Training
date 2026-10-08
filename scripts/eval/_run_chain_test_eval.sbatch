#!/bin/bash
# Chain TEST eval (submitted by scripts/submit_chain.sh with --dependency=afterok:<SEL>).
#
# Reads the per-arm picks from the selection job's chain_picks.json, then generates in ONE
# program (CRN, no cross-program TF32 jitter): anchor + every pick + merge_noop
# (bit-exactness canary) over the TEST_CTX-context seed-0 CRN draw, and scores the
# window-disjoint TEST universe (draw positions SEL_CTX..TEST_CTX-1) with LOB-Bench native
# summary CIs (99%, marginal/unpaired; quick-answer mode) + same-program paired point
# deltas. For paper-grade paired window-bootstrap CIs run
# scripts/eval/_run_rescoreB_score.sbatch on the saved rollouts instead.
#
# Validity gates (all directory/content-based — the final PASS/FAIL verdict is computed
# from what landed in DEST, never from child exit codes or log noise):
#   T1 rollouts complete; T2 ctx integrity: seed 0, n=TEST_CTX and (coupled mode) first
#   SEL_CTX contexts == the selection job's draw (so --drop_first removes exactly the
#   selection universe); T3 merge_noop bit-exact vs anchor; T4 all rows scored finite.
#
# TEST_DECOUPLED=1 (paper protocol): selection lived on a DIFFERENT day universe, so the
# prefix-nesting half of T2 is skipped and the FULL TEST_CTX draw is scored (drop_first 0).
#
# Env (from the launcher): PICKS_JSON (comma-separated list of chain_picks.json files —
# batched-selection picks merge here) SEL_EVAL_DIR SPLIT_JSON SEL_CTX TEST_CTX
# TEST_DECOUPLED + the shared eval data env (EVAL_DAYS CKPT_SRC CKPT_STEP
# MAMBA3_EPS_COMPAT STOCK MONTHS WIDE_BOOK_DIR DEST_BASE). Optional extra rows in the
# SAME program: EXTRA_EGG_SPECS="label=proj_ckpt_dir,..." (e.g. the selection-null pick)
# and FULL_SPECS="label=full_ckpt_dir:step,..." (e.g. the CPT control).
#
#SBATCH --job-name=chain_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --output=logs/chain_test_%j.out
set -uo pipefail
umask 002

EXP="${EXP:-$SLURM_SUBMIT_DIR}"
cd "$EXP"
PY="${LOBBENCH_PY:-$EXP/third_party/lobbench_venv/bin/python}"
SJ="${SPLIT_JSON:?SPLIT_JSON must be exported by the launcher}"
DEST="$DEST_BASE/test_eval_${SLURM_JOB_ID}"
SEL_CTX="${SEL_CTX:-512}"
TEST_CTX="${TEST_CTX:-2048}"
TEST_DECOUPLED="${TEST_DECOUPLED:-0}"
if [ "$TEST_DECOUPLED" = "1" ]; then
  DROPN=0; NTEST=$TEST_CTX          # full universe: selection lived elsewhere
else
  DROPN=$SEL_CTX; NTEST=$(( TEST_CTX - SEL_CTX ))
fi

# ---- picks (written by the selection job(s); afterok guarantees the chained one PASSed) --
# One row per arm per json: "<arm>pick=<picked step ckpt dir>". PICKS_JSON may be a comma
# list of chain_picks.json files (batched selection); arm names must be globally unique.
for pj in ${PICKS_JSON//,/ }; do
  [ -f "$pj" ] || { echo "[chain-test] VERDICT: FAIL (picks missing: $pj)"; exit 2; }
done
EGG_SPECS=$("$PY" -c "
import json
specs = []
for pj in '$PICKS_JSON'.split(','):
    p = json.load(open(pj))
    specs += [f'{a}pick={v[\"ckpt\"]}' for a, v in sorted(p.items())]
labels = [s.split('=', 1)[0] for s in specs]
assert len(labels) == len(set(labels)), f'duplicate arm names across picks files: {labels}'
print(','.join(specs))")
[ -n "$EGG_SPECS" ] || { echo "[chain-test] VERDICT: FAIL (empty picks in $PICKS_JSON)"; exit 2; }
PICKROWS=""
for spec in ${EGG_SPECS//,/ }; do
  DIR="${spec#*=}"
  [ -f "$DIR/latest_checkpoint.json" ] || { echo "[chain-test] VERDICT: FAIL (no breadcrumb: $DIR)"; exit 2; }
  PICKROWS="$PICKROWS,${spec%%=*}"
done
PICKROWS="${PICKROWS#,}"
echo "[chain-test] picks: $EGG_SPECS"

# ---- optional extra rows in the SAME program (null pick, CPT control, ...) ---------------
EXTRA_ROWS=""
if [ -n "${EXTRA_EGG_SPECS:-}" ]; then
  for spec in ${EXTRA_EGG_SPECS//,/ }; do
    DIR="${spec#*=}"
    [ -f "$DIR/latest_checkpoint.json" ] || { echo "[chain-test] VERDICT: FAIL (extra row no breadcrumb: $DIR)"; exit 2; }
    EXTRA_ROWS="$EXTRA_ROWS,${spec%%=*}"
  done
  EGG_SPECS="$EGG_SPECS,$EXTRA_EGG_SPECS"
  echo "[chain-test] extra proj rows: $EXTRA_EGG_SPECS"
fi
if [ -n "${FULL_SPECS:-}" ]; then
  for spec in ${FULL_SPECS//,/ }; do
    rest="${spec#*=}"; DIR="${rest%:*}"
    [ -f "$DIR/latest_checkpoint.json" ] || { echo "[chain-test] VERDICT: FAIL (full row no breadcrumb: $DIR)"; exit 2; }
    EXTRA_ROWS="$EXTRA_ROWS,${spec%%=*}"
  done
  export FULL_SPECS
  echo "[chain-test] extra full-ckpt rows: $FULL_SPECS"
fi
EXTRA_ROWS="${EXTRA_ROWS#,}"

export EGG_SPECS
export ROWS="$PICKROWS${EXTRA_ROWS:+,$EXTRA_ROWS},merge_noop"
export SKIP_LB=1 N_EVAL_CTX="$TEST_CTX" GEN_CHUNK="${GEN_CHUNK:-512}" PANEL_TIMEOUT="${PANEL_TIMEOUT:-18000}"
# See _run_chain_sel_eval.sbatch: training env leaking into the eval program breaks the
# dataset build (UNSEEN_MANIFEST masks eval days -> abort).
unset AB_SPLIT_JSON UNSEEN_MANIFEST PRETRAIN_SEEN_MANIFEST DATA_MODE DATA_NPY_DIR TRAIN_DAYS 2>/dev/null || true

# Wall-kill insurance: the child persists /tmp -> DEST only at its very end, so a job
# killed at the QOS ceiling loses every rollout. Mirror the child workdir to DEST every
# 30 min (incremental: a finished npz copies once, then rsync skips it) so generation
# survives a kill and scoring is resumable via _run_rescoreB_score.sbatch on the saved
# rollouts. DEST is job-keyed; one dir-scan per pass is Lustre-polite.
( while sleep 1800; do
    rsync -a "/tmp/test_eval_${SLURM_JOB_ID}/out/" "$DEST/" 2>/dev/null || true
    chmod -R g+rwX "$DEST" 2>/dev/null || true
  done ) &
SYNC_PID=$!

# ---- generation: the proven test_eval body as a child (same SLURM_JOB_ID => same DEST) ---
bash scripts/eval/_run_test_eval.sbatch
CHILD_RC=$?
kill "$SYNC_PID" 2>/dev/null || true
echo "[chain-test] child test_eval rc=$CHILD_RC (informational — verdict comes from DEST)"

WORKOUT="/tmp/test_eval_${SLURM_JOB_ID}/out"
SRC="$WORKOUT"; [ -f "$SRC/ctx_days.json" ] || SRC="$DEST"

# ---- T1: rollouts complete ----------------------------------------------------------------
MISSING=0
for f in ctx_days.json rollout_real.npz rollout_anchor.npz rollout_merge_noop.npz; do
  [ -f "$SRC/$f" ] || { echo "[chain-test] MISSING $SRC/$f"; MISSING=1; }
done
for r in ${PICKROWS//,/ } ${EXTRA_ROWS//,/ }; do
  [ -f "$SRC/rollout_$r.npz" ] || { echo "[chain-test] MISSING rollout_$r.npz"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "[chain-test] VERDICT: FAIL (rollouts incomplete in $SRC)"; exit 1; }

# ---- T2 + T3: ctx integrity (prefix-nesting only in coupled mode); merge_noop bit-exact ---
"$PY" - "$SRC" "${SEL_EVAL_DIR:-}" "$SEL_CTX" "$TEST_CTX" "$TEST_DECOUPLED" <<'EOF' || { echo "[chain-test] VERDICT: FAIL (ctx/bit-exact gate)"; exit 1; }
import json, sys
import numpy as np
SRC, SELD, NSEL, NTOT, DECOUPLED = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                    int(sys.argv[4]), sys.argv[5] == "1")
cd = json.load(open(f"{SRC}/ctx_days.json"))
assert cd["seed"] == 0 and cd["n_eval_ctx"] == NTOT, (cd["seed"], cd["n_eval_ctx"])
if DECOUPLED:
    # Selection lived on a different day universe: no prefix to verify; the whole draw
    # is the sealed universe.
    print(f"[T2] ctx OK: seed 0, n={NTOT} (decoupled — full universe scored)")
else:
    sel = json.load(open(f"{SELD}/ctx_days.json"))
    assert sel["n_eval_ctx"] == NSEL, sel["n_eval_ctx"]
    assert cd["idx"][:NSEL] == sel["idx"] and cd["day"][:NSEL] == sel["day"], \
        "test ctx prefix != selection draw — drop_first would NOT remove the selection universe"
    print(f"[T2] ctx OK: seed 0, n={NTOT}, first {NSEL} == selection draw")
a = np.load(f"{SRC}/rollout_anchor.npz")
m = np.load(f"{SRC}/rollout_merge_noop.npz")
assert set(a.files) == set(m.files), (a.files, m.files)
for k in a.files:
    assert np.array_equal(a[k], m[k]), f"merge_noop differs from anchor on '{k}'"
print("[T3] merge_noop bit-exact vs anchor — program integrity OK")
EOF
[ $? -eq 0 ] || exit 1

# ---- scoring fan-out: anchor + picks on the window-disjoint tail --------------------------
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGD="$EXP/logs/chain_test_${SLURM_JOB_ID}"; mkdir -p "$LOGD"
W="/tmp/chain_test_score_${SLURM_JOB_ID}"; mkdir -p "$W"
FAIL="$W/FAIL"; : > "$FAIL"

run_row () {
  local R=$1
  local TAG
  TAG="testW_${R}"
  mkdir -p "$W/$R"
  "$PY" scripts/eval/ab_split_scoring.py \
    --eval_dir "$SRC" --ctx_days "$SRC/ctx_days.json" \
    --split ALL --split_json "$SJ" --rows "$R" --drop_first "$DROPN" \
    --work "$W/$R" --out_tag "$TAG" --levels 10 --ticker "${STOCK:-GOOG}" \
    > "$LOGD/$TAG.log" 2>&1 || echo "rc_$R" >> "$FAIL"
  if [ "$DROPN" -eq 0 ]; then
    grep -q "split ALL: ${TEST_CTX}/${TEST_CTX}" "$LOGD/$TAG.log" || echo "count_$R" >> "$FAIL"
  else
    grep -q "drop_first ${DROPN}: ${TEST_CTX} -> ${NTEST}" "$LOGD/$TAG.log" || echo "count_$R" >> "$FAIL"
  fi
}
export -f run_row; export W SRC PY SJ LOGD FAIL STOCK EXP SEL_CTX TEST_CTX NTEST DROPN
{ echo anchor; for r in ${PICKROWS//,/ } ${EXTRA_ROWS//,/ }; do echo "$r"; done; } | xargs -L1 -P 4 bash -c 'run_row "$@"' _
echo "[chain-test] scoring done; failed tasks: $(wc -l < "$FAIL")"
[ -s "$FAIL" ] && { echo "[chain-test] failed rows:"; cat "$FAIL"; }

# ---- T4 + results table: points, native 99% CIs, paired same-program deltas ---------------
"$PY" - "$SRC" "$PICKS_JSON" "$NTEST" "$TEST_DECOUPLED" "${EXTRA_ROWS:-}" <<'EOF'
import json, math, os, sys
SRC, PICKS, NTEST = sys.argv[1], sys.argv[2], int(sys.argv[3])
DECOUPLED = sys.argv[4] == "1"
EXTRA = [r for r in sys.argv[5].split(",") if r]
picks = {}
for pj in PICKS.split(","):
    picks.update(json.load(open(pj)))

def summary(row):
    r = json.load(open(f"{SRC}/lobbench_scores_testW_{row}/lobbench_results.json"))[row]["summary"]
    out = {}
    for m in ("wasserstein", "l1"):
        v, ci = r[m][0][0], r[m][0][1]
        assert math.isfinite(v), f"{row}/{m} not finite"
        out[m] = (v, ci)
    return out

pick_rows = [f"{a}pick" for a in sorted(picks)]
rows = {r: summary(r) for r in ["anchor"] + pick_rows + EXTRA}
res = {"rows": {r: {m: {"point": v[m][0], "native_ci99": list(v[m][1])} for m in v}
                for r, v in rows.items()},
       "picks": picks}
universe = (f"full universe, {NTEST} windows" if DECOUPLED
            else f"window-disjoint tail, {NTEST} windows")
lines = [f"# Chain TEST result ({universe})", "",
         "One program (CRN, merge_noop bit-exact), LOB-Bench NATIVE 99% CIs (marginal,",
         "unpaired); deltas are same-program paired points. For paired window-bootstrap",
         "CIs run _run_rescoreB_score.sbatch on these rollouts.", "",
         "| row | pick step | WS-21 | native 99% CI | dWS vs anchor | Mean L1 | native 99% CI | dL1 |",
         "|---|---|---|---|---|---|---|---|"]
aw, al = rows["anchor"]["wasserstein"][0], rows["anchor"]["l1"][0]
lines.append(f"| anchor | - | {aw:.5f} | [{rows['anchor']['wasserstein'][1][0]:.4f}, "
             f"{rows['anchor']['wasserstein'][1][1]:.4f}] | - | {al:.5f} | "
             f"[{rows['anchor']['l1'][1][0]:.4f}, {rows['anchor']['l1'][1][1]:.4f}] | - |")
for r in pick_rows + EXTRA:
    step = f"st{picks[r[:-4]]['step']:04d}" if r in pick_rows else "-"
    w, wc = rows[r]["wasserstein"]; l, lc = rows[r]["l1"]
    res["rows"][r]["delta_ws_rel"] = (w - aw) / aw
    res["rows"][r]["delta_l1_rel"] = (l - al) / al
    lines.append(f"| {r} | {step} | {w:.5f} | [{wc[0]:.4f}, {wc[1]:.4f}] | "
                 f"{100*(w-aw)/aw:+.2f}% | {l:.5f} | [{lc[0]:.4f}, {lc[1]:.4f}] | "
                 f"{100*(l-al)/al:+.2f}% |")
lines += ["", "Selection-universe curves: chain_selection_curves.md in the selection job's dir.", ""]
json.dump(res, open(f"{SRC}/chain_test_results.json", "w"), indent=1)
open(f"{SRC}/chain_test_results.md", "w").write("\n".join(lines) + "\n")
for f in ("chain_test_results.json", "chain_test_results.md"):
    os.chmod(f"{SRC}/{f}", 0o664)
print("\n".join(lines))
EOF
AGG_RC=$?

# ---- persist to DEST, then verdict from DEST -----------------------------------------------
if [ "$SRC" != "$DEST" ]; then
  rsync -a "$SRC"/lobbench_scores_testW_* "$SRC/chain_test_results.json" "$SRC/chain_test_results.md" \
    "$DEST/" 2>/dev/null || true
fi
chmod -R g+rwX "$DEST" "$LOGD" 2>/dev/null || true
chmod g+rw "$EXP/logs/chain_test_${SLURM_JOB_ID}.out" 2>/dev/null || true

if [ "$AGG_RC" -eq 0 ] && [ ! -s "$FAIL" ] && [ -f "$DEST/chain_test_results.json" ]; then
  echo "[chain-test] VERDICT: PASS ($DEST/chain_test_results.md)"
  exit 0
fi
echo "[chain-test] VERDICT: FAIL (agg_rc=$AGG_RC fails=$(wc -l < "$FAIL") results=$( [ -f "$DEST/chain_test_results.json" ] && echo present || echo missing ))"
exit 1
