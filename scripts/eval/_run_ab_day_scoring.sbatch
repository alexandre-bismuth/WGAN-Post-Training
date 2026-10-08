#!/bin/bash
#SBATCH --account=<slurm_account>
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=ab-score
#SBATCH --output=$EXP/logs/ab_score_%j.out
#
# A/B-split LOB-Bench SELECTION scoring for the GOOG arm (2026-07-02).
# Pure CPU re-scoring of saved test_eval rollouts (no GPU). For each seed's two
# eval dirs, runs ab_split_scoring.py on SPLIT (default A) over the step rows, then
# consolidate_ab_selection.py merges to a per-seed step->WS-21 curve + argmin winner.
# ctx_days.json is portable (all rollout_real.npz byte-identical, md5 ed760ac3...).
# Re-run with SPLIT=B for the sealed final pass (winners + soup only).
set -uo pipefail
umask 002
export TMPDIR=/tmp
EXP="${EXP:-$SLURM_SUBMIT_DIR}"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
cd "$EXP"
export PATH="$CP/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$EXP/lobmamba"
VENVPY="$EXP/third_party/lobbench_venv/bin/python"
SPLIT="${SPLIT:-A}"
CTX="$EXP/runs/test_eval_5474032/ctx_days.json"
[ -f "$CTX" ] || { echo "FATAL: ctx_days.json missing at $CTX"; exit 1; }
[ -x "$VENVPY" ] || { echo "FATAL: lobbench venv python missing"; exit 1; }

# seed:jid:rows  (s0 step40 lives as 'bbest' in 5459924; anchor scored once)
SPEC=(
  "s0:5459924:anchor,bs05,bs10,bs20,bs30,bs50,bbest"
  "s0:5464409:bs15,bs25,bs35,bs45"
  "s1:5474032:bs05,bs10,bs15,bs20,bs25"
  "s1:5474079:bs30,bs35,bs40,bs45,bs50"
  "s2:5474082:bs05,bs10,bs15,bs20,bs25"
  "s2:5474086:bs30,bs35,bs40,bs45,bs50"
  "shuf0:5474089:bs05,bs10,bs15,bs20,bs25"
  "shuf0:5474093:bs30,bs35,bs40,bs45,bs50"
)

rc_any=0
for line in "${SPEC[@]}"; do
  IFS=: read -r seed jid rows <<< "$line"
  ed="$EXP/runs/test_eval_$jid"
  if [ ! -f "$ed/rollout_real.npz" ]; then echo "SKIP $seed:$jid (no rollout_real)"; rc_any=1; continue; fi
  echo "==== scoring $seed  jid=$jid  split=$SPLIT  rows=$rows ===="
  "$VENVPY" scripts/eval/ab_split_scoring.py \
      --eval_dir "$ed" --ctx_days "$CTX" --split "$SPLIT" --rows "$rows" \
      --work "$TMPDIR/ab_${jid}_${SPLIT}" --ticker GOOG \
      2>&1 | sed "s/^/[$seed:$jid] /" || { echo "WARN scoring failed for $seed:$jid"; rc_any=1; }
  chmod -R g+rwX "$ed/lobbench_scores_days${SPLIT}" 2>/dev/null || true
done

echo "==== consolidating split $SPLIT ===="
"$VENVPY" scripts/eval/consolidate_ab_selection.py --split "$SPLIT" \
  --seeds "s0=5459924,5464409 s1=5474032,5474079 s2=5474082,5474086 shuf0=5474089,5474093" \
  --runs_dir "$EXP/runs" --out "$EXP/docs/results/robust_critic/selection_tables/ab_selection_${SPLIT}.md"
rc_cons=$?

chmod g+rw "$EXP/docs/results/robust_critic/selection_tables/ab_selection_${SPLIT}"* 2>/dev/null || true
chmod g+rw "$EXP/logs/ab_score_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "AB_SCORE_DONE split=$SPLIT scoring_rc_any=$rc_any consolidate_rc=$rc_cons"
exit $rc_cons
