#!/bin/bash
# Analysis-only re-run over a PERSISTED compounding-error token bank (runs/compounding_error_<job>).
# CPU companion to _run_compounding_error.sbatch — use when the in-job analysis failed or to
# re-analyze with different --groups / bins / boot without touching the GPU generation.
# Env: DIR (required), GROUP_SPECS ("name=row,row;..."), N_BOOT (200), BINS (20).
# NB: the knob is GROUP_SPECS — GROUPS is a bash special array; bash re-initializes it at
# startup and silently discards any value exported into the job environment.
# Lustre checklist: exact-path reads of one runs/ dir on home (VAST); no find/recursive ls;
# no loggers; single job; no persistent agent.
#SBATCH --job-name=ce_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gpus-per-node=0
#SBATCH --partition=workq
#SBATCH --time=01:00:00
#SBATCH --output=logs/ce_analysis_%j.out
set -uo pipefail
umask 002
EXP="${EXP:-$SLURM_SUBMIT_DIR}"
cd "$EXP"
CP="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
export PATH="$CP/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONPATH="$EXP:$EXP/lobmamba"
export TMPDIR=/tmp MPLBACKEND=Agg
DIR="${DIR:?DIR required (persisted compounding_error run dir)}"
STAMP='{ print strftime("[%H:%M:%S]"), $0; fflush() }'
echo "[ce_analysis] start $(date) dir=$DIR group_specs=${GROUP_SPECS:-<none>}"
python -u -m eggroll_gan.eval.compounding_error_analysis \
  --dir "$DIR" --n_boot "${N_BOOT:-200}" --bins "${BINS:-20}" \
  ${GROUP_SPECS:+--groups "$GROUP_SPECS"} 2>&1 | awk "$STAMP"
RC=${PIPESTATUS[0]}
chmod -R g+rwX "$DIR" 2>/dev/null || true
chmod g+rw "$EXP/logs/ce_analysis_${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "[ce_analysis] SUMMARY rc=$RC dir=$DIR"
exit $RC
