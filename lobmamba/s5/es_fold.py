"""EGGROLL ES low-rank forward fold (S5b) — dependency-free helpers.

An ES-perturbed forward pass threads an optional `es` pytree (None-defaulted at every
signature) down the model. `es` mirrors the params tree but contains ONLY the perturbed
projection kernels, each leaf replaced by pre-sampled LoRA factors:

    es = {"<module>": ..., "<proj_name>": {"kernel": {"A": (in, r), "B": (out, r)}}}

The factors are hoisted: sampled ONCE per ES step in `eggroll_gan.es_generator.build_proj_factors`
with HyperscaleES's own `get_lora_update_params` (the SAME per-leaf key + iterinfo + sigma/sqrt(rank)
that `EggRoll.do_updates` later uses to reconstruct the gradient), with the antithetic sigma sign
already folded into A. The model-side fold is therefore exactly `EggRoll.do_Tmm`:

    x @ W + x @ A @ B.T        (W is the Flax (in, out) kernel; x @ W is the UNCHANGED stock op)

Why hoisted instead of in-model RNG (plan S5b item 5's authorized optimisation):
  * `rank`/`noise_reuse` are shape-statics — inside a traced arg pytree under jit(vmap) they
    would concretize; hoisting removes them from the model interface entirely.
  * the model files stay free of any hyperscalees import (live-training path untouched).
  * the per-step RNG cost is paid once, not once per token step of the AR scan.

Guarantees relied on by the gates:
  * es is None (or has no entry for a projection) -> the stock module output is returned
    UNTOUCHED — byte-identical live-training code path.
  * sigma=0 -> A == 0 exactly (get_lora_update_params scales A by sigma) -> x @ 0 @ B.T == 0
    -> bit-exact no-op through the real rollout.
  * fold_kernel == EggRoll.do_Tmm bit-for-bit (checked in train_eggroll_gan_s5 CPU checks).
"""


def subtree(es, name):
    """None-safe child lookup: the es subtree for submodule `name`, or None if absent.
    Dict STRUCTURE is trace-time static, so this costs nothing under jit."""
    if es is None:
        return None
    return es.get(name)


def fold_kernel(es, name, x, base):
    """Return `base + x @ A @ B.T` when `es[name]["kernel"]` carries LoRA factors, else `base`
    unchanged. `base` must be the stock projection output for input `x` (i.e. x @ W [+ b])."""
    ab = subtree(es, name)
    if ab is None:
        return base
    ab = ab["kernel"]
    return base + (x @ ab["A"]) @ ab["B"].T
