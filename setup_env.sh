#!/usr/bin/env bash
# Environment setup for the EGGROLL-GAN repository.
#
# Source this file (do not execute it) to put the first-party packages on
# PYTHONPATH and to (re)create the gymnax_exchange order-book engine symlink:
#
#     source setup_env.sh
#
# gymnax_exchange (the JAX limit-order-book simulator) is an external checkout
# that is not committed; it is symlinked into the model package so it imports as
# the top-level `gymnax_exchange`. Override GYMNAX_EXCHANGE_SRC to point at your
# own checkout if the default shared path is unavailable.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

: "${GYMNAX_EXCHANGE_SRC:=/lustre/projects/public/data/lob_pipeline/LOBS5/AlphaTrade/gymnax_exchange}"
if [ -d "$ROOT/lobmamba" ] && [ -e "$GYMNAX_EXCHANGE_SRC" ]; then
    ln -sfn "$GYMNAX_EXCHANGE_SRC" "$ROOT/lobmamba/gymnax_exchange"
elif [ ! -e "$GYMNAX_EXCHANGE_SRC" ]; then
    echo "setup_env.sh: warning: GYMNAX_EXCHANGE_SRC not found ($GYMNAX_EXCHANGE_SRC);" \
         "set it to your gymnax_exchange checkout" >&2
fi

# First-party packages: the repo root (for `import eggroll_gan`) and the pruned
# Mamba3 model package (for top-level `import lob` / `s5` / `preproc` / `utils`).
export PYTHONPATH="$ROOT:$ROOT/lobmamba${PYTHONPATH:+:$PYTHONPATH}"

# RMSNorm epsilon compatibility: 0 = fixed norms (the s28730 anchor, default);
# 1 = legacy pre-2026-04-22 epsilon trap (only for the retired s46050 anchor).
export MAMBA3_EPS_COMPAT="${MAMBA3_EPS_COMPAT:-0}"
