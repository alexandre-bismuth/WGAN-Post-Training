#!/bin/bash
# Central site + experiment configuration, sourced by scripts/submit_chain.sh.
# EVERY value is an env-overridable default: export the variable before invoking the
# launcher (or edit here) to retarget another site, anchor, corpus, or eval universe.
# Nothing in scripts/ should hardcode a user home, cluster root, or person.

# Experiment root: derived from this file's location (works from any clone path).
EXP="${EXP:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export EXP

# Python environments.
export GAN_CONDA_PREFIX="${GAN_CONDA_PREFIX:-/projects/public/data/quant/miniforge3}"
PYBIN="$GAN_CONDA_PREFIX/bin/python3"          # training-side python (JAX env)
LOBBENCH_PY="$EXP/third_party/lobbench_venv/bin/python"   # scoring-side python

# Durable storage root (checkpoints + eval rollouts; parallel FS, never home quota).
export CKPT_ROOT="${CKPT_ROOT:-/lustre/projects/public/shared/post_training_GAN}"
export DEST_BASE="${DEST_BASE:-$CKPT_ROOT/panel28_evals}"

# Frozen anchor (pretrained generator). EPS compat is checkpoint-dependent: 0 for the
# fixed-RMSNorm s28730 anchors, 1 for the legacy s46050 line.
export CKPT_SRC="${CKPT_SRC:-$EXP/checkpoints/mamba3_78M_googonly_s28730}"
export CKPT_STEP="${CKPT_STEP:-28730}"
export MAMBA3_EPS_COMPAT="${MAMBA3_EPS_COMPAT:-0}"
export STOCK="${STOCK:-GOOG}"

# Training corpus (defaults = the unseen-pretraining-epoch recipe). Set UNSEEN_MANIFEST=""
# to train WITHOUT the unseen-slot mask (any plain corpus); set TRAIN_DAYS_JSON="" to skip
# the day filter. These are TRAINING-side only — the eval wrappers unset them.
export DATA_MODE="${DATA_MODE:-lustre_npy}"
export DATA_NPY_DIR="${DATA_NPY_DIR:-/lustre/projects/shared/quant/Mamba3_GOOG_pretraining_runs/data_npy/$STOCK}"
export UNSEEN_MANIFEST="${UNSEEN_MANIFEST-$EXP/data/unseen_manifest/unseen_manifest_v1.npz}"
TRAIN_DAYS_JSON="${TRAIN_DAYS_JSON-$EXP/data/unseen_manifest/unseen100_days.json}"
export WIDE_BOOK_TRAIN="${WIDE_BOOK_TRAIN:-$EXP/data/wide_L500_unseen100/$STOCK}"

# Eval universe (defaults = the 28-day Jan+Feb panel with the pre-committed A/B split).
SPLIT_JSON="${SPLIT_JSON:-$EXP/docs/results/unseen_arm/panel_ab_split.json}"
export WIDE_BOOK_EVAL="${WIDE_BOOK_EVAL:-$EXP/data/wide_L500_panel28/$STOCK}"
export EVAL_MONTHS="${EVAL_MONTHS:-2026-01,2026-02}"
# Selection universe (may differ from the sealed-test universe). Defaults keep the legacy
# coupled behavior (selection = prefix of the test draw). Setting SEL_SPLIT_JSON to a
# different split file auto-enables the DECOUPLED test path in submit_chain: the sealed
# test then scores the FULL TEST_CTX draw (drop_first 0, no prefix-nesting gate). The
# paper protocol pairs sel_split_unseen100.json with PRETRAIN_SEEN_MANIFEST (exported at
# launch) so selection draws only pretraining-SEEN slots — disjoint by construction from
# all post-training data. PRETRAIN_SEEN_MANIFEST reaches ONLY the SEL job.
SEL_SPLIT_JSON="${SEL_SPLIT_JSON:-$SPLIT_JSON}"
export SEL_WIDE_BOOK="${SEL_WIDE_BOOK:-$WIDE_BOOK_EVAL}"
export SEL_MONTHS="${SEL_MONTHS:-$EVAL_MONTHS}"
# Selection/test context geometry. Coupled (legacy): positions 0..SEL_CTX-1 select,
# SEL_CTX..TEST_CTX-1 test. Decoupled: SEL_CTX windows on the selection universe,
# all TEST_CTX windows on the test universe.
export SEL_CTX="${SEL_CTX:-512}"
export TEST_CTX="${TEST_CTX:-2048}"

# Cluster geometry. SHARD_ROLLOUTS is the proven per-GPU resident-rollout ceiling — the
# launcher sizes --nodes from it (nodes = G*Q / (SHARD_ROLLOUTS * GPUS_PER_NODE)).
export SHARD_ROLLOUTS="${SHARD_ROLLOUTS:-256}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Critic + optimizer recipe (defaults = the robust-critic configuration).
export CRITIC_INPUT="${CRITIC_INPUT:-learned}"
export ENC_PROJ_DIM="${ENC_PROJ_DIM:-8}"
export ENC_WHITEN="${ENC_WHITEN:-robust}"
export ENS_SIZE="${ENS_SIZE:-5}"
export CRIT_BATCH="${CRIT_BATCH:-256}"
export NSTEPS="${NSTEPS:-50}"
export NCTXPOOL="${NCTXPOOL:-512}"
export POOL_REFRESH="${POOL_REFRESH:-5}"
export NEVAL="${NEVAL:-128}"
export CKPT_EVERY="${CKPT_EVERY:-5}"           # step-ckpt grid the selection eval scores
export XLA_PREALLOCATE="${XLA_PREALLOCATE:-true}"

# Wall-time model: wall = ETA * 1.5. Per-step minutes measured on the production shape
# (256 rollouts/shard); override for slower configs.
export TRAIN_STEP_MIN="${TRAIN_STEP_MIN:-3}"   # min/ES-step incl. d-steps (measured ~2.8)
export TRAIN_STAGE_MIN="${TRAIN_STAGE_MIN:-15}"
# Per-row minutes scale ~linearly with context count; the references were measured at
# SEL_CTX=512 (gen+score ~9.5 min/row) and TEST_CTX=2048 (gen+in-child panel metrics
# ~35 min/row — jobs 5846028/5846652 measured 34.6 @2048 / 65.7 @4096; the old 25-min
# figure excluded the panel-metrics CPU tail and undersized PANEL_TIMEOUT, killing
# 5846028 two rows short).
export SEL_ROW_MIN="${SEL_ROW_MIN:-$(( (10 * SEL_CTX + 511) / 512 ))}"
export TEST_GENROW_MIN="${TEST_GENROW_MIN:-$(( (35 * TEST_CTX + 2047) / 2048 ))}"
export TEST_SCOREROW_MIN="${TEST_SCOREROW_MIN:-$(( (20 * TEST_CTX + 2047) / 2048 ))}"
export EVAL_STAGE_MIN="${EVAL_STAGE_MIN:-25}"
