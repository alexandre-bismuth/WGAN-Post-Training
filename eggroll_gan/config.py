"""Central configuration for the EGGROLL-GAN post-training of the pretrained Mamba3 LOB generator.

Values are either read from the supplied checkpoint metadata or taken from the EGGROLL authors'
configs (HyperscaleES/llm_experiments/*.py); per-field comments record which.

References:
  - Overview: ../README.md ; method + data + eval recipe: docs/reference/, docs/results/deep_record_stages1-2.md
  - Anchor checkpoint: checkpoints/mamba3_78M_sp500iso_fixednorm_s28730 (step 28730, fixed RMSNorm).
  - EGGROLL defaults: HyperscaleES/llm_experiments/{general_do_evolution,sft_evolution}.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # exp_0626_EGGROLL_GAN
MAMBA_ROOT = os.path.join(EXP_ROOT, "lobmamba")                        # pruned Mamba3 model package (lob/ + s5/)
HYPERSCALEES_ROOT = os.path.join(EXP_ROOT, "HyperscaleES")             # EGGROLL repo (JAX)


@dataclass
class Paths:
    exp_root: str = EXP_ROOT
    mamba_root: str = MAMBA_ROOT
    hyperscalees_root: str = HYPERSCALEES_ROOT
    # Pretrained generator checkpoint (Orbax). Mamba3-78M anchor, fixed RMSNorm (MAMBA3_EPS_COMPAT=0).
    ckpt_dir: str = os.path.join(EXP_ROOT, "checkpoints", "mamba3_78M_sp500iso_fixednorm_s28730")
    ckpt_step: int = 28730
    # Run output dir; kept off Lustre during training (set to $TMPDIR at runtime).
    out_dir: str = os.path.join(EXP_ROOT, "runs")


# --------------------------------------------------------------------------------------
# Generator (pretrained Mamba3 78M). Architecture is fixed and must match the anchor checkpoint
# exactly to load: 78,539,423 params; d_model=1024, n_layers=6, blocks=16; 26tok; mode=none.
# Production anchor: fixed-RMSNorm s28730 (MAMBA3_EPS_COMPAT=0), trained on the SP500 set
# (incl. GOOG), 2022-01-01..2025-12-31. Read a checkpoint's training config at <ckpt>/<step>/metadata/metadata.
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Architecture (frozen — must match the checkpoint exactly to load).
    ssm_type: str = "mamba3"
    d_model: int = 1024
    ssm_size_base: int = 1024
    blocks: int = 16
    n_message_layers: int = 2
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    n_layers: int = 6              # fused SSM layers
    mamba3_d_state: int = 128
    mamba3_expand: int = 2
    mamba3_headdim: int = 64
    mamba3_chunk_size: int = 64
    mamba3_rope_fraction: float = 0.5
    activation_fn: str = "half_glu1"
    prenorm: bool = True
    batchnorm: bool = False        # => predict() uses {"params": ...} only (no batch_stats)
    bn_momentum: float = 0.95

    # Data / IO.
    use_book_data: bool = True
    merging: str = "padded"        # => model class BatchPaddedLobPredModel
    book_depth: int = 500
    book_transform: bool = True
    book_dim: int = 503            # L2 book representation dim (confirm at load time)

    # Tokenisation: the model imports `from lob import encoding`, which is the 26-token variant
    # (encoding.py == encoding_26tok.py; the --token_mode / TOKEN_MODE flag is a no-op).
    # In encoding.py: Vocab len == 2112; Message_Tokenizer.MSG_LEN == 26 (TOK_LENS sum == 26).
    # The checkpoint has a single decoder Dense: kernel (d_model=1024, vocab=2112) + bias (2112,).
    token_mode: str = "26tok"      # informational only (flag is a no-op); real encoding is 26tok
    vocab_size: int = 2112         # == len(encoding.Vocab())
    msg_len: int = 26              # tokens per message == Message_Tokenizer.MSG_LEN
    # Time tokens (positions TIME_START_I..TIME_END_I) are computed from delta_t, not sampled.
    time_start_i: int = 11         # Message_Tokenizer.TIME_START_I (26tok)
    time_end_i: int = 15           # Message_Tokenizer.TIME_END_I (26tok)
    msg_seq_len: int = 500         # messages of context

    # The decoder param path EGGROLL perturbs in the decoder-head scope.
    decoder_kernel_path: Tuple[str, ...] = ("decoder", "kernel")   # shape (d_model, vocab)
    decoder_bias_path: Tuple[str, ...] = ("decoder", "bias")       # shape (vocab,)


# --------------------------------------------------------------------------------------
# EGGROLL ES hyperparameters — from the HyperscaleES/llm_experiments defaults
# --------------------------------------------------------------------------------------
@dataclass
class EggrollConfig:
    # Low-rank perturbation rank. r=4 follows the App-M LoRA-on-projections recipe with a frozen
    # SSM core. G=512 is past the rank-saturation point min(G·r, m, n), so larger G only adds the
    # 1/G Monte-Carlo averaging term.
    rank: int = 4
    sigma: float = 1e-3            # base noise scale; swept over {0.003, 0.01, 0.03} (dominant knob)
    lr_scale: float = 1.0          # σ²√N scaling factor; the S5 path uses η≈1e-3 directly instead

    # Population N = parallel_generations_per_gpu * n_gpu. The EGGROLL authors use 1024/GPU on a
    # 100M model; this 78M model has far longer rollouts (~13k AR steps), so N per GPU is smaller.
    parallel_generations_per_gpu: int = 256   # per-GPU population; memory / wall-clock bound
    generations_per_prompt: int = 8           # == group_size for group-relative fitness norm
    noise_reuse: int = 1

    # es_map perturbation scope — three modes:
    #   "decoder_head": perturb only the decoder head; everything else excluded (the flags below).
    #   "interior": perturb the whole interior, freezing only the I/O projection matrices (input
    #     embeddings/projections + decoder head). Interior matmuls route through do_mm/do_Tmm
    #     (MM_PARAM); norms/biases/dt/A/D use PARAM. The I/O projections stay excluded because their
    #     low-rank B-factor scales with vocab (2112) / book (503): expensive, low leverage.
    #   "dense_mlp": perturb only the Dense/MLP projections (a middle rung).
    # The head kernel is 1024x2112 (~2.16M params); materialising N per-member copies costs ~8.9 GB
    # at N=1024, so the low-rank do_Tmm folding (A:Nx1024x1 + B:Nx2112x1 ~= 51 MB at N=1024) is used.
    perturb_scope: str = "decoder_head"       # one of: "decoder_head" | "interior" | "dense_mlp"
    perturb_decoder_kernel: bool = True       # MM_PARAM (via do_Tmm; Flax kernel is (in,out))
    perturb_decoder_bias: bool = True         # PARAM (full-rank)
    freeze_everything_else: bool = True       # used with "decoder_head"; False for "interior" (freeze I/O projections instead)
    freeze_io_projections: bool = True        # for "interior": keep input embeddings + decoder head excluded

    optimizer: str = "adamw"       # optax solver passed to init_noiser

    @property
    def population_size_hint(self) -> str:
        return "N = parallel_generations_per_gpu * n_gpu (e.g. 256*4=1024, target up to 1024*4=4096)"


# --------------------------------------------------------------------------------------
# Rollout / generation
# --------------------------------------------------------------------------------------
@dataclass
class RolloutConfig:
    n_msg_context: int = 500       # context messages
    n_msg_generate: int = 500      # messages to generate per rollout
    # Distribution matching uses true sampling (temperature 1.0); lower it if ES fitness variance
    # is too high. (The EGGROLL authors' RL uses greedy decoding for lower-variance fitness.)
    temperature: float = 1.0
    sample_top_n: int = 50         # top-k truncation for sampling
    tick_size: int = 100


# --------------------------------------------------------------------------------------
# Data (26tok LOBSTER, 8-megacap GOOG-led) — from the checkpoint's saved training config.
# EGGROLL-GAN trains/evaluates on the SAME data the 78M generator was pretrained on.
# Loader: lob.dataloading.create_lobster_prediction_dataset with token_mode='26tok',
#   mask_fn=LOBSTER_Dataset.no_mask, use_book_data=True, book_transform=True
#   (recipe: docs/reference/data_anchor_eval.md).
# Lustre safety: pre-stage shards to $TMPDIR before the loader starts; never scan the data dir.
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    dataset: str = "lobster-prediction"
    token_mode: str = "26tok"
    data_root: str = "/lustre/projects/data/lob_preproc_26tok"
    book_ablation: str = "real"
    msg_seq_len: int = 500
    book_depth: int = 500
    book_transform: bool = True
    train_date_range: Tuple[str, str] = ("2022-01-01", "2025-12-31")
    test_date_range: Tuple[str, str] = ("2026-01-01", "2026-01-31")  # held-out: GOOG Jan-2026 (LOB-Bench)
    # 8-megacap set the 78M was trained on (GOOG is the lead + standard eval/generation ticker).
    tickers: List[str] = field(default_factory=lambda: [
        "GOOG", "AAPL", "NVDA", "AMZN", "META", "TSLA", "MSFT", "AMD"])
    val_split: float = 0.01
    # The SquashFS SP500-sweep flags (SQUASHFS_MULTI_MODE / SQUASHFS_MONTHS / FORBID_RAW_NPYZST)
    # apply only when using the SP500 corpus. The 78M's data is the 26tok megacap set above, loaded
    # via the standard LOBSTER dataloader, so those flags are not needed here.


# --------------------------------------------------------------------------------------
# Discriminator (WGAN critic) — Mamba3 backbone + LoRA + scalar head
# --------------------------------------------------------------------------------------
@dataclass
class DiscriminatorConfig:
    # What the WGAN critic scores:
    #   "learned" (DEFAULT, the production robust recipe) — a small SN-regularised causal TCN that
    #     learns its representation from the per-step microstructure descriptor sequence [T, F_in].
    #     See critic/learned_critic.py.
    #   "raw" — the SAME TCN over the decoded-message FIELD sequence itself (event/dir/relative
    #     price/size/Δt/refs under lossless transforms, no book-derived channels): the raw generated
    #     data rather than hand-designed statistics of it. See critic/raw_message_features.py; its
    #     heavy-tailed channels make enc_whiten='robust' the recommended pairing.
    #   "backbone" (LEGACY) — the frozen Mamba3 backbone's pooled hidden state of the [ctx ; cont]
    #     token window. That objective coincides with pretraining, so it gave the critic little post-
    #     training signal beyond what the generator already optimised (and it is reward-hackable).
    # Retired modes (fixed hand-designed feature vectors: "stylized" marginal facts, "dynamical"
    # temporal functionals, "signature" path signatures) were removed with their critic/ modules —
    # recover from git history if ever needed.
    # Only the critic's input changes; the spectral-normed head, WGAN loss, Lipschitz constraint and
    # rank-σ̄ fitness are identical (make_critic adapts to the feature width / sequence shape).
    critic_input: str = "learned"
    # Backbone initialised from the same pretrained Mamba3 checkpoint.
    lora_rank: int = 8
    head_hidden: Tuple[int, ...] = (128,)
    pooling: str = "mean"          # mean-pool hidden states (or 'last'). The S5 trainer pools the
                                   # continuation positions only by default (--pool_scope cont): real &
                                   # fake share the context prefix, so whole-window pooling dilutes D.
    use_spectral_norm: bool = True # Lipschitz control (token-space GP is N/A: discrete tokens)
    r1_gamma: float = 0.0          # R1 gradient penalty on the critic's continuous feature input:
                                   # γ·½·E_real||∇_h D(h)||². 0 = off (bit-identical). Smooths D so the
                                   # generator can't climb sharp non-realistic directions.
    weight_decay: float = 1e-4
    lr: float = 1e-4
    label_smoothing: float = 0.0   # WGAN critic: no labels; kept for optional BCE escalation
    n_d_steps_per_g: int = 1       # update D <= G frequency (start 1, can lower D)
    conditional: bool = True       # input = [context ; continuation] (never continuation alone)
    # Lever 3 (--critic_input learned | raw): a small SN-regularised causal TCN over a per-step
    # channel sequence. Literature defaults — Bai 2018 (TCN: k=3, dilation 2^i,
    # 2 convs/residual block), Miyato 2018 (SN every layer), Heusel 2017 (TTUR critic lr, Adam β1=0).
    # All inert unless the critic input is a sequence mode (make_critic only reads them there).
    enc_type: str = "tcn"          # 'tcn' (dilated, default) | 'cnn' (dilation fixed 1, ablation)
    enc_layers: int = 7            # TCN blocks; L=7,k=3 -> receptive field 509 >= the 500-step window
    enc_hidden: int = 64           # channel width per block
    enc_pool: str = "attn"         # time pooling: 'attn' (default) | 'mean' | 'last' (never last for a critic)
    enc_kernel: int = 3            # causal conv kernel size
    enc_lr: float = 3e-4           # TTUR: critic lr faster than the generator step (Heusel 2017)
    enc_adam_b1: float = 0.0       # Adam β1 for the learned critic (TTUR / SN-GAN setting)
    # Strongest learned-critic config: multi-scale conv trunk + seed-diverse ensemble with a
    # pessimistic (mean−β·std) reward — capacity subject to robustness, since raw capacity accelerates
    # reward-hacking under frozen-generator ES/GRPO (Gao 2023). Multi-scale after MelGAN/HiFi-GAN
    # (Kumar 2019 / Kong 2020); conservative ensemble after Coste 2024. ens_size=1 + enc_scales=(1,)
    # reduces to the bare TCN.
    enc_scales: Tuple[int, ...] = (1, 2, 4)   # multi-scale avg-pool downsample factors (one TCN branch each)
    ens_size: int = 3                          # number of seed-diverse ensemble members (1 disables the ensemble)
    ens_pessimism: float = 1.0                 # β in the reward mean_k − β·std_k (member-disagreement penalty)
    # Broad critic: stop the ES reward collapsing onto the spread marginal (only spread improves while
    # OFI/returns/depth regress) without naming any benchmark channel.
    enc_proj_dim: int = 0          # >0: each ensemble member gets a fixed random projection F_in->enc_proj_dim
                                   #     of its input (Projected-GAN, Sauer 2021) so the members are
                                   #     projection-diverse — the generator can't satisfy the reward by
                                   #     fixing one channel. 0 = off (bit-identical to the current critic).
    enc_whiten: str = "std"        # learned-critic input standardiser: 'std' (mean/σ, default, bit-identical)
                                   #     | 'robust' (median/MAD, heavy-tail-safe — OFI has fat tails).


# --------------------------------------------------------------------------------------
# GAN training loop
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    kl_coef: float = 0.0           # lambda for KL(perturbed || ref) trust-region penalty in fitness
    n_g_steps: int = 1000
    eval_every: int = 25
    ckpt_every: int = 500          # >= 500 per the Lustre checkpoint-frequency rule
    seed: int = 0


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    eggroll: EggrollConfig = field(default_factory=EggrollConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    data: DataConfig = field(default_factory=DataConfig)
    disc: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT = Config()
