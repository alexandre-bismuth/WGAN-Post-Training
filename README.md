# Adversarial Post-Training for HFT Foundation Models

Code for the ICAIF '26 accepted paper *Adversarial Post-Training for HFT Foundation Models*.

A Wasserstein critic, trained online and from scratch on rollouts executed through the true,
non-differentiable matching engine, steers gradient-free updates — EGGROLL evolution strategies
or GRPO — of rank-4 LoRA factors on a frozen pre-trained Mamba3-78M limit-order-book (LOB)
generator under a KL trust region to the pre-trained anchor. No gradient ever flows into the
generator: the discrete token sampler and the discrete order-book simulator make the critic's
score non-differentiable with respect to the generator's parameters, so updates are estimated
from scalar rollout scores alone.

## Layout

```
eggroll_gan/          first-party package
  critic/             WGAN critic (raw-message and stylized-fact inputs, spectral norm)
  es/                 EGGROLL plumbing: population generator (LoRA scopes), fitness = critic − λ·KL
  training/           alternating D-backprop / G-ES loop; GRPO arm (train_grpo_head.py)
  baselines/          decoding-parameter bar; policy-gradient baselines
  eval/               held-out distributional panel, LOB-Bench bridge, selection,
                      directional-accuracy and per-position (compounding-error) diagnostics
  tests/              CPU-only gates (run on a login node, no GPU)
lobmamba/             vendored, pruned Mamba3 LOB model code (lob/, s5/)
HyperscaleES/         EGGROLL evolution-strategy library (vendored; upstream LICENSE included)
scripts/              canonical SLURM run scripts: train/, eval/, smoke/, analysis/
```

## Setup

```bash
source setup_env.sh   # PYTHONPATH + gymnax_exchange symlink + MAMBA3_EPS_COMPAT=0
```

- **Python**: JAX/Flax/Orbax stack (GPU); see `pyproject.toml`.
- **gymnax_exchange** (the JAX-LOB matching engine) is an external checkout from
  [AlphaTrade](https://github.com/KangOxford/AlphaTrade); point `GYMNAX_EXCHANGE_SRC` at your copy
  and `setup_env.sh` symlinks it into `lobmamba/`.
- **Data**: LOBSTER-format NASDAQ message streams, preprocessed to the 26-token encoding by
  `lobmamba/preproc.py`; wide-book L500 snapshots for deep initialization are built by
  `tools/data_build/`. Raw LOBSTER data is licensed and must be obtained separately.
- Cluster-specific absolute paths in `scripts/` are left as documentation of the exact runs;
  override the corresponding environment variables for your site.

## Running

```bash
sbatch scripts/train/_run_eggroll_production.sbatch   # EGGROLL post-training seeds (1 node)
sbatch scripts/train/_run_grpo_multinode.sbatch       # GRPO comparison arm
sbatch scripts/eval/_run_test_eval.sbatch             # sealed held-out panel + LOB-Bench (1 GPU)
sbatch scripts/eval/_run_directional_accuracy.sbatch  # downstream directional-accuracy eval
sbatch scripts/smoke/_run_eggroll_smoke.sbatch        # fast end-to-end gate

# CPU-only validation (no GPU required)
python -m eggroll_gan.tests.s2_head_sigma0
python -m eggroll_gan.tests.s3_es_rollout
python -m eggroll_gan.tests.s5b_proj_gates
```

Training/eval scripts write to `$TMPDIR` and rsync results to shared storage on success;
checkpoint discovery uses a `latest_checkpoint.json` breadcrumb.

## Evaluation protocol

Selection is performed offline over the checkpoint grid with LOB-Bench on an in-training-timeframe
window; the selected checkpoint is scored **once** on a sealed 28-day held-out panel (4,096 windows,
day-clustered bootstrap CIs, selection-matched random-perturbation null and compute-matched
continued-pre-training control). See the paper for the full protocol and results.

## Acknowledgement

The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Business, Innovation, Science and Trade (DBIST) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].

McIntosh-Smith, S., Alam S. R. and Woods, C. (2024). "Isambard-AI: a leadership class supercomputer optimised specifically for Artificial Intelligence". https://doi.org/10.48550/arXiv.2410.11199

