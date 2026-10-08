"""Raw decoded-message critic input (--critic_input raw).

PRINCIPLE: give the critic the generated data itself, not hand-designed statistics of it.
The 26-token message encoding is a bijection of the decoded field tuple (event type,
direction, relative price, size, Δt, order-reference fields): `inference_no_errcorr`
decodes generated tokens into EXACTLY the [T, 14] field layout the real corpus provides
(m_seq_raw), so the decoded-field sequence IS the raw message stream on both sides of the
real/fake comparison — no information is lost relative to the token stream, and no
human-crafted summary (OFI, imbalance, spread, ...) stands between the data and the critic.
Working at FIELD level rather than token level is deliberate: token IDs add only
liabilities — digit-group artifacts an ES generator can Goodhart without improving order
flow, a 26x longer sequence, and an embedding the small SN-TCN critic would have to learn.

CHANNELS — every trained-AND-SAMPLED field of the encoding, under lossless (monotone,
invertible) transforms only:
  * event type  -> 4 one-hots; direction -> {-1, +1}.
  * price       -> PRICE_i: the RELATIVE (stationary) price in ticks, the exact quantity
                   the price tokens encode. PRICE_ABS is excluded — it is NA in decoded
                   fakes and nonstationary in reals (a trivial real/fake tell with zero
                   realism content).
  * size        -> log1p.
  * Δt          -> log(DTs + 1e-9·DTns + 1ns): the full inter-arrival time, ns field
                   included. log (not log1p): LOB inter-arrivals live on a MULTIPLICATIVE
                   scale spanning ~1e-6..1e1 s, and log1p(x) ~= x below 1 s would flatten
                   all sub-second timing structure to ~0; the +1ns floor matches the
                   encoding's own resolution, keeping the map monotone and invertible.
  * refs        -> relative ref price, log1p ref size, log order age (t_msg − t_ref,
                   i.e. time-to-cancel/modify, same +1ns floor) + a ref_valid indicator.
                   NA-sentinel (-9999)
                   fields gate their channel to 0. Age uses the clocks both classes carry
                   (real recorded / fake Δt-accumulated) — a genuine generated quantity.
EXCLUDED, deliberately: ORDER_ID (engine bookkeeping, NA in decoded fakes) and absolute
time_s/time_ns as LEVELS — loss-masked in pretraining, derived at rollout, and their
real-vs-fake quantization mismatch is a consistency artifact, not realism (see the NOTE in
learned_critic.py). The L2 book is NOT used: this is the pure-message arm; the message
stream plus the shared init state determines the book. `rollout_descriptor` keeps the
(l2, msgs) contract signature and ignores l2.

CONTRACT: same featuriser surface as critic/learned_critic.py (batch_features,
FEATURE_NAMES, N_FEATURES, fit_normalizer, standardize), so the trainer's
`LearnedFeats._FEAT` plug point reuses the SN multi-scale TCN ensemble, random-projection
diversity, robust whitening, R1 and the KL trust region unchanged. price_rel and log_dt
are heavy-tailed -> pair this mode with enc_whiten='robust' (the production default).

CPU-safe: pure jnp, no model build. `__main__` runs the login-node self-test.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Sequence-aware standardiser shared with the learned-critic featuriser: identical contract,
# re-exported so this module satisfies the full _FEAT surface.
from .learned_critic import fit_normalizer, standardize          # noqa: F401

# Decoded-message field indices (mirror lobmamba/lob/inference_no_errcorr.py:55-68).
EVENT_TYPE_I = 1
DIRECTION_I = 2
PRICE_I = 4                    # relative (stationary) price in ticks — what the price tokens encode
SIZE_I = 5
DT_S_I, DT_NS_I = 6, 7
TIME_S_I, TIME_NS_I = 8, 9
PRICE_REF_I, SIZE_REF_I = 10, 11
TIME_S_REF_I, TIME_NS_REF_I = 12, 13

_NA_VAL = -9999.0              # lob.encoding.NA_VAL — missing-field sentinel (real AND decoded fake)

FEATURE_NAMES = (
    "et_new",                  # event one-hot: new limit order
    "et_cancel",               # event one-hot: cancel (partial)
    "et_delete",               # event one-hot: delete (full)
    "et_exec",                 # event one-hot: execution
    "signed_dir",              # direction {0,1} -> {-1,+1}
    "price_rel",               # relative price, ticks (stationary; PRICE_ABS deliberately excluded)
    "log_size",                # log1p order size
    "log_dt",                  # log(Δt + 1ns), Δt in seconds (s + ns fields combined)
    "ref_valid",               # 1 if the msg carries a complete order reference (modif/cancel/exec)
    "price_ref_rel",           # referenced order's relative price, ticks; 0 where no/NA ref
    "log_size_ref",            # log1p referenced order's size; 0 where no/NA ref
    "log_age_ref",             # log(t_msg - t_ref + 1ns) referenced-order age (time-to-cancel); 0 where no/NA ref
)
N_FEATURES = len(FEATURE_NAMES)


def _valid(x):
    """1.0 where the field is present: finite and not the NA sentinel."""
    return (jnp.isfinite(x) & (x != _NA_VAL)).astype(jnp.float32)


def rollout_descriptor(l2, msgs, *, n_levels: int, tick_size: float):
    """Decoded messages [T, 14] -> raw-field channel sequence [T, F_in]. Pure jnp; vmap externally.
    l2 / n_levels / tick_size are accepted for the featuriser contract and unused (pure-message mode)."""
    del l2, n_levels, tick_size
    m = jnp.asarray(msgs, jnp.float32)

    event = m[:, EVENT_TYPE_I]
    et = [(event == float(v)).astype(jnp.float32) for v in (1, 2, 3, 4)]

    direction = m[:, DIRECTION_I]
    signed_dir = jnp.where(_valid(direction) > 0, direction * 2.0 - 1.0, 0.0)

    price = m[:, PRICE_I]
    price_rel = jnp.where(_valid(price) > 0, price, 0.0)

    size = m[:, SIZE_I]
    log_size = jnp.where(_valid(size) > 0, jnp.log1p(jnp.maximum(size, 0.0)), 0.0)

    dt_ok = _valid(m[:, DT_S_I]) * _valid(m[:, DT_NS_I])
    dt = jnp.maximum(m[:, DT_S_I], 0.0) + 1e-9 * jnp.maximum(m[:, DT_NS_I], 0.0)
    log_dt = jnp.where(dt_ok > 0, jnp.log(dt + 1e-9), 0.0)

    pr, sr = m[:, PRICE_REF_I], m[:, SIZE_REF_I]
    ts_r, tns_r = m[:, TIME_S_REF_I], m[:, TIME_NS_REF_I]
    pr_ok, sr_ok = _valid(pr), _valid(sr)
    t_ok = _valid(m[:, TIME_S_I]) * _valid(m[:, TIME_NS_I]) * _valid(ts_r) * _valid(tns_r)
    ref_valid = pr_ok * sr_ok * t_ok
    price_ref_rel = jnp.where(pr_ok > 0, pr, 0.0)
    log_size_ref = jnp.where(sr_ok > 0, jnp.log1p(jnp.maximum(sr, 0.0)), 0.0)
    age = (m[:, TIME_S_I] - ts_r) + 1e-9 * (m[:, TIME_NS_I] - tns_r)
    log_age_ref = jnp.where(t_ok > 0, jnp.log(jnp.maximum(age, 0.0) + 1e-9), 0.0)

    desc = jnp.stack(et + [signed_dir, price_rel, log_size, log_dt,
                           ref_valid, price_ref_rel, log_size_ref, log_age_ref], axis=1)
    return jnp.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)   # [T, F_in]


def batch_features(l2, msgs, *, n_levels: int, tick_size: float):
    """Vectorised descriptor. l2: [N, T, W] (unused), msgs: [N, T, 14] -> [N, T, F_in]."""
    return jax.vmap(lambda a, b: rollout_descriptor(a, b, n_levels=n_levels, tick_size=tick_size))(l2, msgs)


# ========================================================================================
# Login-node self-test: field mapping, NA gating, standardiser, TCN forward + separation.
# ========================================================================================
def _synth_msgs(key, T, *, real: bool):
    """Synthetic decoded-message stream [T, 14]. 'real' has an LOB-like event mix (~45% new orders)
    with clustered (autocorrelated log-Δt) inter-arrivals and size persistence; 'fake' has a uniform
    event mix and i.i.d. uniform Δt / sizes — separable only through the raw field channels."""
    ks = jax.random.split(key, 8)
    if real:
        event = jax.random.choice(ks[0], jnp.arange(1.0, 5.0), (T,),
                                  p=jnp.array([0.45, 0.15, 0.15, 0.25]))
    else:
        event = jax.random.randint(ks[0], (T,), 1, 5).astype(jnp.float32)
    direction = jax.random.bernoulli(ks[1], 0.5, (T,)).astype(jnp.float32)
    price = jnp.round(jax.random.normal(ks[2], (T,)) * 5.0)
    if real:
        e = jax.random.normal(ks[3], (T,))
        def st(p, u):
            nv = 0.85 * p + u
            return nv, nv
        _, ar = jax.lax.scan(st, 0.0, e)
        dt = jnp.exp(ar - 6.0)                                   # clustered, heavy-tailed
        size = jnp.round(jnp.exp(jnp.abs(ar)) * 40.0) + 1.0
    else:
        dt = jax.random.uniform(ks[3], (T,)) * 5e-3
        size = jnp.round(jax.random.uniform(ks[4], (T,)) * 200.0) + 1.0
    dt_s = jnp.floor(dt)
    dt_ns = jnp.round((dt - dt_s) * 1e9)
    t = 36000.0 + jnp.cumsum(dt)
    t_s = jnp.floor(t)
    t_ns = jnp.round((t - t_s) * 1e9)
    # refs: NA on new orders (evt==1), else a valid (price, size, earlier-time) triple
    has_ref = event != 1.0
    age = jnp.exp(jax.random.normal(ks[5], (T,))) * 2.0
    tr = t - age
    tr_s = jnp.floor(tr)
    tr_ns = jnp.round((tr - tr_s) * 1e9)
    na = jnp.full((T,), _NA_VAL)
    pick = lambda v: jnp.where(has_ref, v, na)
    m = jnp.stack([na, event, direction, na, price, size, dt_s, dt_ns, t_s, t_ns,
                   pick(jnp.round(jax.random.normal(ks[6], (T,)) * 5.0)),
                   pick(jnp.round(jax.random.uniform(ks[7], (T,)) * 100.0) + 1.0),
                   pick(tr_s), pick(tr_ns)], axis=1)
    return m


def _cpu_self_test(seed=0):
    import optax
    from .learned_critic import LearnedCritic
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[raw_message_features] CPU self-test — raw-field descriptor + TCN compatibility", flush=True)
    k = jax.random.key(seed)
    T = 48
    EH, EL = 8, 2                 # tiny encoder: keep the compile footprint under the login thread limit

    m = _synth_msgs(jax.random.fold_in(k, 1), T, real=True)
    l2 = jnp.zeros((T, 40))
    d = rollout_descriptor(l2, m, n_levels=10, tick_size=100.0)
    chk("(R1) descriptor shape [T, F_in]", d.shape == (T, N_FEATURES), f"shape={tuple(d.shape)}")
    chk("(R2) descriptor finite", bool(jnp.all(jnp.isfinite(d))))
    i_rv = FEATURE_NAMES.index("ref_valid")
    new_rows = m[:, EVENT_TYPE_I] == 1.0
    chk("(R3) NA refs gate to 0 on new orders",
        bool(jnp.all(jnp.where(new_rows[:, None], d[:, i_rv:], 0.0) == 0.0)))
    chk("(R3b) refs live on non-new orders", bool(jnp.all(d[~new_rows][:, i_rv] == 1.0)))
    i_et = [FEATURE_NAMES.index(n) for n in ("et_new", "et_cancel", "et_delete", "et_exec")]
    chk("(R4) event one-hots partition every row",
        bool(jnp.all(jnp.sum(d[:, jnp.array(i_et)], axis=1) == 1.0)))
    i_age = FEATURE_NAMES.index("log_age_ref")
    chk("(R5) ref age channel live where ref present", bool(jnp.all(d[~new_rows][:, i_age] != 0.0)))
    # NA injection on a core field stays finite and zeroed
    m_na = m.at[3, PRICE_I].set(_NA_VAL)
    d_na = rollout_descriptor(l2, m_na, n_levels=10, tick_size=100.0)
    chk("(R6) NA core field -> 0, finite", bool(jnp.isfinite(d_na).all())
        and float(d_na[3, FEATURE_NAMES.index("price_rel")]) == 0.0)

    db = batch_features(jnp.stack([l2, l2]), jnp.stack([m, m]), n_levels=10, tick_size=100.0)
    chk("(R7) batch_features [N, T, F_in]", db.shape == (2, T, N_FEATURES), f"shape={tuple(db.shape)}")
    for robust in (False, True):
        mean, std = fit_normalizer(db, robust=robust)
        z = standardize(db, mean, std)
        chk(f"(R8{'r' if robust else ''}) standardiser finite (robust={robust})",
            mean.shape == (N_FEATURES,) and bool(jnp.all(jnp.isfinite(z))))

    # --- the TCN critic learns to separate real vs fake FROM RAW FIELDS ALONE ---
    def batch(key, n, real):
        ms = jnp.stack([_synth_msgs(kk, T, real=real) for kk in jax.random.split(key, n)])
        return batch_features(jnp.zeros((n, T, 40)), ms, n_levels=10, tick_size=100.0)
    real_x = batch(jax.random.fold_in(k, 2), 12, True)
    fake_x = batch(jax.random.fold_in(k, 3), 12, False)
    mean, std = fit_normalizer(real_x, robust=True)
    real_x, fake_x = standardize(real_x, mean, std), standardize(fake_x, mean, std)
    net = LearnedCritic(enc_hidden=EH, enc_layers=EL, enc_type="tcn", enc_pool="mean")
    v = net.init(jax.random.PRNGKey(1), jnp.zeros((1, T, N_FEATURES)), train=False)
    params = v["params"]; sn = {kk: vv for kk, vv in v.items() if kk != "params"}
    tx = optax.adam(3e-3); opt_state = tx.init(params)

    def auc(sr, sf):
        s = jnp.concatenate([sr, sf]); ranks = jnp.argsort(jnp.argsort(s)) + 1.0
        u = jnp.sum(ranks[:sr.shape[0]]) - sr.shape[0] * (sr.shape[0] + 1) / 2.0
        return float(u / (sr.shape[0] * sf.shape[0]))

    def dstep(params, sn, opt_state):
        feats = jnp.concatenate([real_x, fake_x], 0)
        def loss_fn(p):
            sc, mut = net.apply({"params": p, **sn}, feats, train=True, mutable=["sn"])
            return jnp.mean(sc[12:]) - jnp.mean(sc[:12]), mut
        (_, mut), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), {"sn": mut["sn"]}, opt_state

    auc0 = auc(net.apply({"params": params, **sn}, real_x, train=False),
               net.apply({"params": params, **sn}, fake_x, train=False))
    for _ in range(200):
        params, sn, opt_state = dstep(params, sn, opt_state)
    auc1 = auc(net.apply({"params": params, **sn}, real_x, train=False),
               net.apply({"params": params, **sn}, fake_x, train=False))
    chk("(R9) TCN separates real vs fake dynamics from raw fields (AUC rises)", auc1 > 0.9,
        f"auc0={auc0:.3f} -> auc1={auc1:.3f}")

    print("\n[raw_message_features] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"),
          flush=True)
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _cpu_self_test() else 0)
