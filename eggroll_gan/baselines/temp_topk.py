"""Temperature / top-k decoding baseline of the UNCHANGED pretrained 78M anchor.

THE BAR the GAN post-training must beat (Caccia et al., "Language GANs Falling Short": tuned
MLE decoding often captures most of the distribution-match gain claimed by discrete-sequence GANs).
Sweep rollout temperature x sample_top_n of the frozen anchor on held-out GOOG contexts at the
500-msg thesis horizon, scored with the SAME eval_monitor stylized-fact harness the GAN runs use.
If no GAN/PG arm beats the best (temp, top_n) cell here, that is the headline negative result.
(LOB-Bench-proper scoring of the saved per-cell rollouts is the eval_compare follow-up.)

HOW TEMPERATURE ENTERS WITHOUT TOUCHING generate(): per token, generate() masks invalid tokens
(-1e9), log_softmaxes, truncates to the top-`sample_top_n` MASKED logits and renormalises
(validation_helpers.filter_valid_pred / sample_pred — temperature 1 only). But scaling the DECODER
HEAD {W/tau, b/tau} gives logits/tau BEFORE the mask, and:
  * -1e9 masking and log_softmax+renormalise commute with the scaling (softmax(x/tau) on the support);
  * the top-k SET is order-invariant under the monotone map x -> x/tau, so the truncation support is
    IDENTICAL to stock generate() at every position (policy_grad PG3/B1 prove it).
So sampling with head {W/tau, b/tau} IS exact temperature-tau sampling of the anchor. The sweep then
reuses the bit-validated ES rollout machinery: the temperature axis rides the per-rollout-head
"population" axis (G = len(temps) scaled heads, Q = eval contexts — one rollout call per top_n,
which is a static recompile axis).

CPU (no GPU / no model): self-checks B1..B4. GPU: --run sweeps and writes
<out_dir>/baseline_results.json + a markdown table; composite is normalized to the ANCHOR-DEFAULT
cell (temp=1.0, top_n=config default), so 1.0 = stock decoding and <1 = better than stock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from ..config import DEFAULT as CFG
from ..eval import eval_monitor as EM
from ..es.es_generator import (make_generate_es_sharded, gq_grid_indices, grid_rngs,
                           tile_dirs_over_Q, grid_repeat_contexts)

MSG_LEN = CFG.model.msg_len


# ----------------------------------------------------------------------------------------
# Pure helpers (CPU-testable).
# ----------------------------------------------------------------------------------------
def scaled_heads(kernel, bias, temps):
    """Temperature-tau sampling == stock sampling with head {W/tau, b/tau} (see module docstring).
    Returns the [n_temps, ...] population pytree the ES rollout machinery consumes."""
    ts = jnp.asarray(list(temps), dtype=jnp.float32)
    return {"kernel": kernel[None] / ts[:, None, None], "bias": bias[None] / ts[:, None]}


def cell_rows(M_arr, G, g):
    """Rows of temperature-cell g from a context-major [Q*G, ...] grid array (r = q*G + g)."""
    a = np.asarray(M_arr)
    Q = a.shape[0] // G
    return a.reshape((Q, G) + a.shape[1:])[:, g]


def default_grid(temps, topns, default_top_n):
    """Sweep grid with the anchor-default cell (temp=1.0, top_n=default) guaranteed present —
    it is the normalization baseline (composite 1.0 = stock decoding)."""
    ts = sorted({float(t) for t in temps} | {1.0})
    ks = sorted({int(n) for n in topns} | {int(default_top_n)}, key=lambda n: (n < 0, n))
    return ts, ks


# ----------------------------------------------------------------------------------------
# GPU sweep.
# ----------------------------------------------------------------------------------------
def run(args):
    from ..tests.s3_es_rollout import _prep_real_batch
    from ..training.train_eggroll_gan_s5 import _replay_real

    temps, topns = default_grid(args.temps, args.topns, CFG.rollout.sample_top_n)
    G, Q = len(temps), args.n_eval_ctx
    print(f"[baseline] temps={temps} topns={topns} (G={G} x Q={Q} rollouts per top_n) "
          f"n_cond={args.n_cond} n_gen={args.n_gen}", flush=True)

    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, Q,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels)
    inf = P["inf"]
    ETi = int(inf.EVENT_TYPE_i)
    kernel0, bias0 = P["kernel"], P["bias"]
    pop = scaled_heads(kernel0, bias0, temps)
    pop_grid = tile_dirs_over_Q(pop, Q)                                  # [Q*G, d, V] scaled heads
    ctx = dict(m=P["m_seq_inp"], b=P["b_seq_inp"], sim=P["sim_states_init"],
               ih=P["init_hidden_batched"], it=P["init_time_batched"])
    grid = {k: grid_repeat_contexts(v, G) for k, v in ctx.items()}
    rng_grid = grid_rngs(jax.random.PRNGKey(args.seed + 1), G, Q)        # CRN across top_n cells

    real_l2 = _replay_real(inf, P["sim_init"], P["sim_states_init"], P["m_seq_raw_cont"])
    real_et = P["m_seq_raw_cont"][..., ETi].astype(jnp.int32)
    _et = jax.jit(lambda md: md[..., ETi].astype(jnp.int32))

    baseline_metrics, results = None, []
    for top_n in topns:
        gen = make_generate_es_sharded(P["model"], P["batchnorm"], P["encoder"], int(top_n),
                                       P["tick_size"], args.n_gen, P["sim_init"],
                                       P["valid_mask_array"], conditional=True, shard=args.shard)
        out = gen(pop_grid, P["train_state"], grid["m"], grid["b"], grid["sim"],
                  rng_grid, grid["ih"], grid["it"])
        gl2_all = np.asarray(out[1])                                     # [Q*G, n_gen, W]
        get_all = np.asarray(_et(out[0]))                                # [Q*G, n_gen]
        nerr_all = np.asarray(out[2]).astype(np.float32)
        for g, tau in enumerate(temps):
            m = EM.stylized_fact_metrics(jnp.asarray(cell_rows(gl2_all, G, g)), real_l2,
                                         jnp.asarray(cell_rows(get_all, G, g)), real_et,
                                         n_levels=inf.l2_state_n)
            m = {k: float(v) for k, v in m.items()}
            if tau == 1.0 and int(top_n) == int(CFG.rollout.sample_top_n):
                baseline_metrics = m
            results.append(dict(temp=tau, top_n=int(top_n), metrics=m,
                                mean_num_errors=float(cell_rows(nerr_all, G, g).mean())))
            print(f"[baseline] top_n={top_n:>4} temp={tau:.2f}  "
                  + " ".join(f"{k}={v:.4g}" for k, v in m.items()), flush=True)

    assert baseline_metrics is not None, "anchor-default cell (temp=1.0, default top_n) missing"
    for r in results:
        r["composite_norm"] = float(EM.normalize_composite(r["metrics"], baseline_metrics))
    results.sort(key=lambda r: r["composite_norm"])
    best = results[0]

    os.makedirs(args.out_dir, exist_ok=True)
    payload = dict(stage="baseline_temp_topk", n_cond=args.n_cond, n_gen=args.n_gen,
                   n_eval_ctx=Q, seed=args.seed, temps=temps, topns=[int(n) for n in topns],
                   default_top_n=int(CFG.rollout.sample_top_n), ckpt_dir=args.ckpt_dir,
                   ckpt_step=args.ckpt_step, baseline_metrics=baseline_metrics, results=results,
                   best=dict(temp=best["temp"], top_n=best["top_n"],
                             composite_norm=best["composite_norm"]))
    with open(os.path.join(args.out_dir, "baseline_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = ["| top_n | temp | composite_norm | book_l1 | ret_corr | moment_l1 | event_l1 | mid_l1 | num_err |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['top_n']} | {r['temp']:.2f} | {r['composite_norm']:.4f} | "
                     f"{m['book_l1']:.4g} | {m['ret_corr']:.3f} | {m['moment_l1']:.4g} | "
                     f"{m['event_l1']:.4g} | {m['mid_l1']:.4g} | {r['mean_num_errors']:.2f} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "baseline_table.md"), "w") as f:
        f.write(f"# temp/top-k baseline (anchor-default = 1.0)\n\n{table}\n")
    print("\n" + table, flush=True)
    print(f"\n[baseline] BEST: top_n={best['top_n']} temp={best['temp']:.2f} "
          f"composite={best['composite_norm']:.4f} (THE BAR for the GAN/PG arms) -> {args.out_dir}",
          flush=True)
    return 0


# ----------------------------------------------------------------------------------------
# Login-node self-checks (no model).
# ----------------------------------------------------------------------------------------
def cpu_checks(seed=0):
    from .policy_grad import sampling_probs

    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[baseline] CPU checks — temperature fold / grid slicing / normalization", flush=True)
    k = jax.random.PRNGKey(seed)
    V, top_n, tau = 16, 5, 0.8
    logits = jax.random.normal(jax.random.fold_in(k, 1), (V,))
    vm = jnp.array([True] * 10 + [False] * 6)

    # (B1) head scaling == temperature through the FULL sampling chain (mask + topk + renorm).
    p_fold = sampling_probs(logits / tau, vm, top_n)         # scaled head -> logits/tau pre-mask
    p_temp = sampling_probs(logits, vm, top_n)
    p_temp = jnp.where(p_temp > 0, p_temp ** (1 / tau), 0.0)  # tau applied to the SAME support
    p_temp = p_temp / jnp.sum(p_temp)
    d1 = float(jnp.max(jnp.abs(p_fold - p_temp)))
    chk("(B1) scaled-head dist == temperature dist (same support)", d1 < 1e-6, f"max|d|={d1:.2e}")

    # (B2) scaled_heads: kernel/bias scaled together; tau=1.0 row exactly the anchor.
    kk = jax.random.normal(jax.random.fold_in(k, 2), (3, 4))
    bb = jax.random.normal(jax.random.fold_in(k, 3), (4,))
    sh = scaled_heads(kk, bb, [0.5, 1.0, 2.0])
    chk("(B2) scaled_heads rows = {W/t, b/t}; tau=1 row == anchor",
        float(jnp.max(jnp.abs(sh["kernel"][1] - kk))) == 0.0
        and float(jnp.max(jnp.abs(sh["kernel"][0] - kk / 0.5))) < 1e-6
        and float(jnp.max(jnp.abs(sh["bias"][2] - bb / 2.0))) < 1e-6)

    # (B3) cell_rows inverts the context-major grid layout (r = q*G + g).
    G, Q = 3, 4
    di, ci = gq_grid_indices(G, Q)
    enc = np.asarray(di) * 100 + np.asarray(ci)               # value encodes (g, q)
    for g in range(G):
        rows = cell_rows(enc, G, g)
        if not (np.all(rows // 100 == g) and np.array_equal(rows % 100, np.arange(Q))):
            chk("(B3) cell_rows slices one temperature across all contexts", False, f"g={g}")
            break
    else:
        chk("(B3) cell_rows slices one temperature across all contexts", True)

    # (B4) default cell present + self-normalizes to 1.0.
    ts, ks = default_grid([0.8, 1.1], [20], 50)
    base = dict(mid_l1=1e6, book_l1=2.0, ret_corr=0.5, moment_l1=0.1, event_l1=0.2)
    chk("(B4) default cell injected; composite(base, base) == 1.0",
        1.0 in ts and 50 in ks and abs(EM.normalize_composite(base, base) - 1.0) < 1e-6)

    print("[baseline] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="temp/top-k decoding baseline of the pretrained anchor")
    ap.add_argument("--run", action="store_true", help="run the GPU sweep; else CPU checks only")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "baseline_out"))
    ap.add_argument("--n_cond", type=int, default=500)
    ap.add_argument("--n_gen", type=int, default=500)
    ap.add_argument("--n_eval_ctx", type=int, default=64,
                    help="held-out contexts per cell (the metric-noise knob)")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.8, 0.9, 1.0, 1.1])
    ap.add_argument("--topns", type=int, nargs="+", default=[20, 50, 100, -1],
                    help="-1 = full valid distribution (sample_pred semantics)")
    ap.add_argument("--wide_levels", type=int, default=10)
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fails = cpu_checks(seed=args.seed)
    if fails:
        print(f"[baseline] CPU checks FAILED: {fails}"); sys.exit(1)
    if args.run:
        if not args.data_dir:
            print("[baseline] --run requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        sys.exit(run(args))
    print("\n[baseline] (skipped GPU sweep — pass --run on the GH200; CPU checks PASSED)")


if __name__ == "__main__":
    main()
