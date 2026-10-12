"""S2 — EGGROLL plumbing: the reusable bridge between HyperscaleES's `EggRoll` noiser and a
*Flax* parameter pytree (the pretrained Mamba3 generator).

Two jobs:
  1. `import_hyperscalees()` — import ONLY the EGGROLL leaf modules we need
     (`noiser.eggroll`, `models.common`, + their bases) WITHOUT running the HyperscaleES
     package `__init__.py` files. The real `models/__init__.py` eagerly does
     `from . import ... rl, llm`, and `rl.py` imports `gymnax`, which is NOT installed in
     this env (this env has `gymnax_exchange`, a different package). HyperscaleES's RL
     environments are unused, so this side-steps that dependency entirely and keeps the git
     submodule pristine (no edits to the pinned commit).
  2. EGGROLL <-> Flax glue used by S2/S3:
       - `build_es_map`   : classify each Flax leaf PARAM / MM_PARAM / EMB_PARAM / EXCLUDED
       - `build_scan_map` : the `()`-leaf scan map EggRoll expects (see note below)
       - `build_es_tree_key` : per-leaf PRNG-key tree
       - `decoder_apply_stock` / `decoder_apply_eggroll` : the decoder-head forward, stock
         vs EGGROLL-wrapped (matmul -> `do_Tmm`, bias -> `get_noisy_standard`).

Why `do_Tmm` for the decoder head: the Mamba3 generator's head is `self.decoder = nn.Dense(d_output)`
(lob/lob_seq_model.py), whose kernel is `(in, out) = (d_model=1024, vocab=2112)` and which computes
`x @ kernel`. EggRoll's `do_Tmm(..., x)` computes exactly `x @ param (+ low-rank x@A@B.T)` for a
`(in, out)` `param` — the matching convention. (`do_mm` is for `(out, in)` matrices.)

Scan-map note: `simple_es_tree_key`/`do_updates` consume `scan_map` via `tree.map(f, params, ..., scan_map)`,
which uses `params`' treedef and `flatten_up_to` to hand each *array leaf position* the corresponding
`scan_map` value wholesale. So a non-scanned leaf wants the empty tuple `()` there (len 0 -> no split).
`jax.tree.map(lambda _: (), params)` produces exactly that (this mirrors how `common.py` builds it).

This module is import-cheap and CPU-safe (no checkpoint, no model build); it is the foundation S3
reuses to lift the decoder leaves to a per-member `in_axes=0` rollout.
"""
from __future__ import annotations

import os
import sys
import types
import importlib.util
from typing import Any

import jax
import jax.numpy as jnp

from ..config import HYPERSCALEES_ROOT
_HS_ROOT = os.path.join(HYPERSCALEES_ROOT, "src", "hyperscalees")


# ----------------------------------------------------------------------------------------
# (1) Import EGGROLL leaf modules without triggering the gymnax-dependent package __init__.
# ----------------------------------------------------------------------------------------
def _exec_leaf(mod_name: str, file_path: str):
    """Exec one .py file as a fully-qualified module and register it in sys.modules so that
    sibling relative imports (`from .base_noiser import Noiser`) resolve against it."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod          # register BEFORE exec so relative imports find it
    spec.loader.exec_module(mod)
    return mod


def import_hyperscalees() -> types.SimpleNamespace:
    """Return a namespace with the EGGROLL symbols we use. Idempotent."""
    if not os.path.isdir(_HS_ROOT):
        raise FileNotFoundError(
            f"HyperscaleES source not found at {_HS_ROOT}. Did the git submodule get checked out?")

    # Register stub parent packages (with __path__ so relative imports resolve) instead of
    # executing the real __init__.py files.
    for pkg, sub in (("hyperscalees", ""),
                     ("hyperscalees.noiser", "noiser"),
                     ("hyperscalees.models", "models")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [os.path.join(_HS_ROOT, sub) if sub else _HS_ROOT]
            sys.modules[pkg] = m

    # Load leaves in dependency order (base before dependent).
    if "hyperscalees.noiser.base_noiser" not in sys.modules:
        _exec_leaf("hyperscalees.noiser.base_noiser", os.path.join(_HS_ROOT, "noiser", "base_noiser.py"))
    eggroll = sys.modules.get("hyperscalees.noiser.eggroll") or \
        _exec_leaf("hyperscalees.noiser.eggroll", os.path.join(_HS_ROOT, "noiser", "eggroll.py"))
    if "hyperscalees.models.base_model" not in sys.modules:
        _exec_leaf("hyperscalees.models.base_model", os.path.join(_HS_ROOT, "models", "base_model.py"))
    common = sys.modules.get("hyperscalees.models.common") or \
        _exec_leaf("hyperscalees.models.common", os.path.join(_HS_ROOT, "models", "common.py"))
    base_noiser = sys.modules["hyperscalees.noiser.base_noiser"]
    base_model = sys.modules["hyperscalees.models.base_model"]

    return types.SimpleNamespace(
        EggRoll=eggroll.EggRoll,
        Noiser=base_noiser.Noiser,
        MLP=common.MLP, Linear=common.Linear, MM=common.MM, TMM=common.TMM, Parameter=common.Parameter,
        simple_es_tree_key=common.simple_es_tree_key,
        PARAM=common.PARAM, MM_PARAM=common.MM_PARAM, EMB_PARAM=common.EMB_PARAM, EXCLUDED=common.EXCLUDED,
        Model=base_model.Model, CommonInit=base_model.CommonInit, CommonParams=base_model.CommonParams,
        common=common, eggroll=eggroll,
    )


# ----------------------------------------------------------------------------------------
# (2) EGGROLL <-> Flax pytree glue.
# ----------------------------------------------------------------------------------------
def _key_path_strs(path) -> tuple:
    """Render a jax KeyPath (tuple of KeyEntry) as a tuple of plain strings, robust to the
    DictKey / GetAttrKey / SequenceKey / FlattenedIndexKey variants."""
    out = []
    for k in path:
        for attr in ("key", "name", "idx"):
            if hasattr(k, attr):
                out.append(str(getattr(k, attr)))
                break
        else:
            out.append(str(k))
    return tuple(out)


def build_es_map(params: Any, hs: types.SimpleNamespace,
                 *, perturb_decoder_kernel: bool = True, perturb_decoder_bias: bool = True) -> Any:
    """es_map pytree (same structure as `params`, INT leaves) for the "decoder-head only" scope:
        decoder/kernel -> MM_PARAM (low-rank do_Tmm), decoder/bias -> PARAM (full-rank), else EXCLUDED.

    Matching is on the trailing path `("decoder", "kernel"|"bias")`, so it is robust to any wrapping.
    For the eventual interior scope, swap this for a classifier that marks interior matmuls MM_PARAM
    and norms/biases PARAM while keeping the I/O projections EXCLUDED (see config.EggrollConfig)."""
    def classify(path, _leaf):
        ps = _key_path_strs(path)
        tail = ps[-2:]
        if tail == ("decoder", "kernel"):
            return hs.MM_PARAM if perturb_decoder_kernel else hs.EXCLUDED
        if tail == ("decoder", "bias"):
            return hs.PARAM if perturb_decoder_bias else hs.EXCLUDED
        return hs.EXCLUDED
    return jax.tree_util.tree_map_with_path(classify, params)


def build_es_map_proj(params: Any, hs: types.SimpleNamespace, *, perturb_glu: bool = True,
                      perturb_book_proj: bool = False, perturb_fused_encoder: bool = True) -> Any:
    """es_map for the S5b "in/out projection LoRA" scope (App-M S5/Galim recipe ported to Mamba):
    LoRA r=4 (MM_PARAM, routed via `do_Tmm`) on the interior PROJECTION KERNELS, and FREEZE everything
    else — the selective-scan core (A is activation-derived from `in_proj` so it has no leaf; Δ=`dt_bias`,
    `D`, `B_bias`/`C_bias`), all norms (`*/scale`, norm `bias`), all dense biases, and the I/O (input
    `embedding` + the `decoder` head). A pure-LoRA fine-tune: there are NO PARAM/full-rank folds.

    MM_PARAM kernels: SSM `seq/in_proj/kernel` + `seq/out_proj/kernel` (message/book/fused blocks), and —
    per toggle — `out2/kernel` (GLU), `fused_s5/encoder/kernel`, `book_encoder/projection/kernel`.
    Path-tail matching is robust to the outer batch-vmap wrapping.

    `perturb_book_proj` defaults to FALSE: `book_encoder/projection` (503->d_model) is the model's
    book-INPUT interface, so by the freeze-the-I/O rationale it belongs with the frozen input
    projections; its 503-dim B-factor cost is modest, so the toggle stays available as the first
    ablation to switch on if proj-scope separation stalls."""
    def classify(path, _leaf):
        ps = _key_path_strs(path)
        name = ps[-1]
        parent = ps[-2] if len(ps) >= 2 else ""
        # I/O projections (frozen): output head + input embedding.
        if (parent == "decoder") or (name == "embedding"):
            return hs.EXCLUDED
        if name == "kernel":
            if parent in ("in_proj", "out_proj"):
                return hs.MM_PARAM
            if parent == "out2":
                return hs.MM_PARAM if perturb_glu else hs.EXCLUDED
            if parent == "projection":                     # book input->d_model projection (interior)
                return hs.MM_PARAM if perturb_book_proj else hs.EXCLUDED
            if parent == "encoder" and "fused_s5" in ps:   # fused-stack input encoder (Dense; interior)
                return hs.MM_PARAM if perturb_fused_encoder else hs.EXCLUDED
            return hs.EXCLUDED                              # any other/unknown kernel: freeze (conservative)
        # selective-scan core, norms, biases: all frozen in this pure-LoRA recipe.
        return hs.EXCLUDED
    return jax.tree_util.tree_map_with_path(classify, params)


def build_scan_map(params: Any) -> Any:
    """Non-scanned scan map: the empty tuple `()` at every leaf position (see module docstring)."""
    return jax.tree.map(lambda _: (), params)


def build_es_tree_key(params: Any, base_key, hs: types.SimpleNamespace, scan_map: Any | None = None) -> Any:
    """Per-leaf PRNG-key tree, as EggRoll's matmul/standard ops expect for `base_key`."""
    if scan_map is None:
        scan_map = build_scan_map(params)
    return hs.simple_es_tree_key(params, base_key, scan_map)


# --- decoder-head forward: stock Flax vs EGGROLL-wrapped ------------------------------------
def decoder_apply_stock(kernel, bias, x):
    """Exactly what `flax.linen.Dense` computes for kernel (in,out) + bias (out,): `x @ kernel + bias`."""
    return x @ kernel + bias


def decoder_apply_eggroll(hs: types.SimpleNamespace, frozen_noiser_params, noiser_params,
                          kernel, bias, key_kernel, key_bias, iterinfo, x):
    """EGGROLL-wrapped decoder head: matmul via `do_Tmm` (+ low-rank perturbation), bias via
    `get_noisy_standard`. With `iterinfo=None` (or sigma=0) this reduces to `decoder_apply_stock`."""
    h = hs.EggRoll.do_Tmm(frozen_noiser_params, noiser_params, kernel, key_kernel, iterinfo, x)
    b = hs.EggRoll.get_noisy_standard(frozen_noiser_params, noiser_params, bias, key_bias, iterinfo)
    return h + b


if __name__ == "__main__":
    # Login-node smoke: confirm the import bridge works and the symbols are present.
    hs = import_hyperscalees()
    print("[es_plumbing] import_hyperscalees OK")
    print("  EggRoll        =", hs.EggRoll)
    print("  MLP            =", hs.MLP)
    print("  class consts   = PARAM=%d MM_PARAM=%d EMB_PARAM=%d EXCLUDED=%d"
          % (hs.PARAM, hs.MM_PARAM, hs.EMB_PARAM, hs.EXCLUDED))
    # Tiny structural sanity on a mock generator pytree.
    mock = {
        "message_encoder": {"layers_0": {"kernel": jnp.zeros((4, 4))}},
        "book_encoder": {"proj": {"kernel": jnp.zeros((4, 4))}},
        "fused_s5": {"blocks_0": {"A": jnp.zeros((4,))}},
        "decoder": {"kernel": jnp.zeros((4, 6)), "bias": jnp.zeros((6,))},
    }
    es_map = build_es_map(mock, hs)
    print("  es_map         =", jax.tree_util.tree_map(int, es_map))
    assert es_map["decoder"]["kernel"] == hs.MM_PARAM
    assert es_map["decoder"]["bias"] == hs.PARAM
    assert es_map["message_encoder"]["layers_0"]["kernel"] == hs.EXCLUDED
    assert es_map["fused_s5"]["blocks_0"]["A"] == hs.EXCLUDED
    print("[es_plumbing] es_map classification OK")

    # S5b proj es_map: only projection kernels MM_PARAM; SSM core / norms / biases / I/O EXCLUDED.
    z = lambda *s: jnp.zeros(s)
    proj_mock = {
        "message_encoder": {"encoder": {"embedding": z(2112, 1024)},
                            "layers_0": {"seq": {"in_proj": {"kernel": z(1024, 4480)},
                                                  "out_proj": {"kernel": z(2048, 1024)},
                                                  "out_norm": {"scale": z(2048)}, "dt_bias": z(32),
                                                  "D": z(32), "B_norm": {"scale": z(128)}},
                                          "out2": {"kernel": z(1024, 1024), "bias": z(1024)},
                                          "norm": {"scale": z(1024), "bias": z(1024)}}},
        "book_encoder": {"projection": {"kernel": z(503, 1024), "bias": z(1024)},
                         "pre_layers_0": {"seq": {"in_proj": {"kernel": z(503, 2253)}}}},
        "fused_s5": {"encoder": {"kernel": z(2048, 1024), "bias": z(1024)},
                     "layers_0": {"seq": {"out_proj": {"kernel": z(2048, 1024)}}}},
        "decoder": {"kernel": z(1024, 2112), "bias": z(2112)},
    }
    pm = build_es_map_proj(proj_mock, hs)
    me, be, fs = pm["message_encoder"], pm["book_encoder"], pm["fused_s5"]
    assert me["layers_0"]["seq"]["in_proj"]["kernel"] == hs.MM_PARAM
    assert me["layers_0"]["seq"]["out_proj"]["kernel"] == hs.MM_PARAM
    assert me["layers_0"]["out2"]["kernel"] == hs.MM_PARAM            # GLU (toggle on by default)
    assert be["projection"]["kernel"] == hs.EXCLUDED                  # book proj (DEFAULT OFF: input I/O)
    assert fs["encoder"]["kernel"] == hs.MM_PARAM                     # fused encoder (toggle on by default)
    assert fs["layers_0"]["seq"]["out_proj"]["kernel"] == hs.MM_PARAM
    pm_book = build_es_map_proj(proj_mock, hs, perturb_book_proj=True)
    assert pm_book["book_encoder"]["projection"]["kernel"] == hs.MM_PARAM  # ablation toggle still works
    # everything else frozen
    assert me["encoder"]["embedding"] == hs.EXCLUDED                  # input I/O
    assert pm["decoder"]["kernel"] == hs.EXCLUDED and pm["decoder"]["bias"] == hs.EXCLUDED  # output I/O
    assert me["layers_0"]["seq"]["out_norm"]["scale"] == hs.EXCLUDED  # norm
    assert me["layers_0"]["seq"]["dt_bias"] == hs.EXCLUDED            # Δ
    assert me["layers_0"]["seq"]["D"] == hs.EXCLUDED                  # D
    assert me["layers_0"]["out2"]["bias"] == hs.EXCLUDED              # dense bias
    assert me["layers_0"]["norm"]["bias"] == hs.EXCLUDED              # norm bias
    # toggles off -> those kernels freeze
    pm_off = build_es_map_proj(proj_mock, hs, perturb_glu=False, perturb_book_proj=False,
                               perturb_fused_encoder=False)
    assert pm_off["message_encoder"]["layers_0"]["out2"]["kernel"] == hs.EXCLUDED
    assert pm_off["book_encoder"]["projection"]["kernel"] == hs.EXCLUDED
    assert pm_off["fused_s5"]["encoder"]["kernel"] == hs.EXCLUDED
    assert pm_off["message_encoder"]["layers_0"]["seq"]["in_proj"]["kernel"] == hs.MM_PARAM  # always on
    print("[es_plumbing] build_es_map_proj classification OK")
