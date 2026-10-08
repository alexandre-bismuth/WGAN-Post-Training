# scripts/ — general train + eval tooling

Everything here is a **template**: no user homes, no person names, no per-experiment
hardcoding. Site + experiment defaults live in one place (`env.sh`); every value is an
env-overridable variable. Historical one-time launchers were removed 2026-07-15 (tracked
ones live in git history; untracked ones in `../archive/scripts_onetime_2026-07-15/`).

## The one launcher: `submit_chain.sh`

Runs a full arm → selection → test chain for ANY configuration:

```bash
# dry-run first (prints every sbatch line + computed nodes/walls, submits nothing)
ARMS="g32q64:32:64 g64q32:64:32" bash scripts/submit_chain.sh

# submit: iso-compute pair + evals
ARMS="g32q64:32:64 g64q32:64:32" bash scripts/submit_chain.sh --submit

# one 4x-budget arm (nodes auto-sized: 64*128 rollouts / 256-per-GPU -> 8 nodes)
ARMS="g64q128:64:128" bash scripts/submit_chain.sh --submit

# seeds + matched shuffle null:  tag:G:Q[:seed[:fc]]
ARMS="s1:16:128:1 null1:16:128:1:shuffle" bash scripts/submit_chain.sh --submit

# different critic / corpus / anchor: override env, same launcher
CRITIC_INPUT=stylized UNSEEN_MANIFEST="" DATA_NPY_DIR=/path/to/npy \
  ARMS="styl:16:128" bash scripts/submit_chain.sh --submit

# evals only, over grids from earlier trainings
GRIDS="myarm=<CKPT_ROOT>/s5b_mnode_<jobid>/myarm" bash scripts/submit_chain.sh --eval-only --submit

# training only
ARMS="..." bash scripts/submit_chain.sh --train-only --submit
```

What it does per invocation:

| stage | job | what |
|---|---|---|
| T_(tag) | `train/_run_eggroll_multinode.sbatch`, one per arm, chained `afterany` | EGGROLL post-training; `--nodes` auto-sized from G×Q at the proven `SHARD_ROLLOUTS`/GPU ceiling; geometry divisibility validated BEFORE submission |
| SEL | `eval/_run_chain_sel_eval.sbatch`, `afterany` on the arms | anchor + every arm's step-ckpt grid in ONE program on the selection contexts (draw positions 0..`SEL_CTX`−1); per-arm WS-21 argmin → `chain_picks.json` |
| TEST | `eval/_run_chain_test_eval.sbatch`, `afterok:SEL` | anchor + picks + merge_noop (bit-exact gate) in ONE program on the `TEST_CTX` draw; scored on the disjoint tail; native LOB-Bench CIs + same-program paired deltas → `chain_test_results.md` |

Design rules baked in (learned the hard way):

- **Wall times are ETA × 1.5**, computed from step/arm counts (per-step and per-row minute
  models in `env.sh`; override for slower configs).
- **Directory-based verdicts**: the training sbatch always exits 0, so the eval jobs verify
  checkpoint grids and their own outputs from the filesystem — PASS/FAIL is decided by what
  landed in DEST, never by child exit codes or log noise.
- **Env hygiene**: the eval wrappers `unset` training-only variables (`UNSEEN_MANIFEST`,
  `DATA_MODE`, ...) so a corpus setting can never leak into the eval dataset build.
- **Clobber-proof scoring**: per-row `--out_tag`, one scoring process per row.
- **Same-program rows**: anchor and models are always generated in one program (CRN), so
  deltas are free of cross-program TF32 jitter; merge_noop must be bit-exact vs anchor.
- Submissions are staggered (`sleep 30`) and checkpoint discovery is breadcrumb-only
  (`latest_checkpoint.json`) per the cluster filesystem rules.

## Configuration: `env.sh`

One file, three blocks — site (python env, storage roots, cluster geometry), experiment
(anchor, training corpus, eval universe), recipe (critic + optimizer knobs + wall models).
Override any variable in the calling environment; nothing else needs editing to retarget
a different cluster, anchor checkpoint, ticker, corpus, or eval panel.

## Workhorses (called by the chain; usable standalone)

- `train/_run_eggroll_multinode.sbatch` + `_inner.sh` — multi-node sharded EGGROLL trainer
  (any node count; JAX multi-host; node-local staging; rank-0-only writes).
- `train/_run_eggroll_sp500{,_inner}.sh|sbatch` + `_sp500_stage_lib.sh` — SquashFS
  multi-ticker corpus variant.
- `train/_run_grpo_*.sbatch|sh` — GRPO comparison-arm trainers (same critic).
- `eval/_run_test_eval.sbatch` — the generation workhorse: anchor + arbitrary checkpoint
  rows (`EGG_SPECS`/`FULL_SPECS`) on CRN contexts; `SKIP_LB=1` = generation-only.
- `eval/ab_split_scoring.py` — LOB-Bench scoring on saved rollouts with split/window
  control (`--split`, `--drop_first`, `--out_tag`).
- `eval/_run_rescoreB_score.sbatch` — paper-grade scoring: paired window-level block
  bootstrap CIs (use this instead of the quick native CIs for publication deltas).
- `eval/_run_ab_day_scoring.sbatch` + `consolidate_ab_selection.py` — day-split A/B
  selection protocol (legacy sealed-arm machinery).
- `eval/_run_gen_ce_curve.sbatch` — held-out generator CE across a run's step ckpts.
- `eval/_run_invalidity_audit.sbatch` — engine-truthful message-validity audit.
- `smoke/_run_eggroll_smoke.sbatch` — short smoke lane.
- `analysis/` — CPU/login-safe result aggregation + paper figures.

## Conventions

- Always submit from the experiment root (the launcher `cd`s there itself); `#SBATCH
  --output=logs/...` paths and `EXP="${EXP:-$SLURM_SUBMIT_DIR}"` derivation rely on it.
- Group-writable outputs everywhere (`umask 002` + `chmod g+rwX`).
- Job IDs are appended to `logs/chain_jobids.txt`.
