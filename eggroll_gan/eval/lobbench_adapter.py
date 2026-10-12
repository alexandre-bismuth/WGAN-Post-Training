"""Rollout-npz -> LOBSTER-CSV adapter for LOB-Bench scoring.

Converts the saved test_eval rollouts (rollout_<row>.npz: decoded msgs [Q,T,14] + L2 books
[Q,T,W]) into the folder/file layout `data_loading.Simple_Loader` expects:

    <out_dir>/<row>/data_real/GOOG_<date>_message_real_id_<q>.csv          (+ orderbook)
    <out_dir>/<row>/data_gen /GOOG_<date>_message_real_id_<q>_gen_id_0.csv (+ orderbook)
    <out_dir>/<row>/data_cond/                                             (empty; optional)

Field mapping (decoded msg layout = inference_no_errcorr indices; matches encoding.py):
    time       = TIMEs + TIMEns (LOBSTER "sec.nanosec" string)
    event_type = EVENT_TYPE     (1 new / 2 cancel / 3 delete / 4 execute — LOBSTER codes)
    order_id   = ORDER_ID
    size       = SIZE
    price      = PRICE_ABS      (absolute integer price, same unit as the L2 book prices)
    direction  = 2*DIRECTION-1  (ours {0,1} = LOBSTER {-1,+1} via encoding.py (d+1)/2)
Rows whose event_type is not in {1..4} (padding/NA) are dropped, with the book row dropped
at the same index (lob_bench asserts len(messages) == len(book)).

CPU self-checks (--check) run a synthetic round-trip through lob_bench's OWN
load_message_df / load_book_df / Simple_Loader (requires the lobbench venv python).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# decoded-message field indices (inference_no_errcorr.py module constants)
OID_I, ET_I, DIR_I, PABS_I = 0, 1, 2, 3
SIZE_I, DTS_I, DTNS_I, TS_I, TNS_I = 5, 6, 7, 8, 9


def msgs_to_lobster(msgs):
    """[T, 14] decoded msgs -> (rows [T', 6] int64 cols, time strings [T']) with padding dropped."""
    m = np.asarray(msgs)
    keep = np.isin(m[:, ET_I].astype(np.int64), (1, 2, 3, 4))
    m = m[keep]
    ts = m[:, TS_I].astype(np.int64)
    tns = np.clip(m[:, TNS_I].astype(np.int64), 0, 999_999_999)
    time_str = [f"{s}.{ns:09d}" for s, ns in zip(ts, tns)]
    rows = np.stack([m[:, ET_I], m[:, OID_I], m[:, SIZE_I], m[:, PABS_I],
                     2 * m[:, DIR_I] - 1], axis=1).astype(np.int64)
    return rows, time_str, keep


def write_seq_csvs(msgs, l2, out_dir, fname_msg, fname_book, levels=10):
    """One sequence -> LOBSTER message + orderbook CSV (no header, no index)."""
    rows, time_str, keep = msgs_to_lobster(msgs)
    with open(os.path.join(out_dir, fname_msg), "w") as f:
        for t, (et, oid, sz, px, d) in zip(time_str, rows):
            f.write(f"{t},{et},{oid},{sz},{px},{d}\n")
    b = np.asarray(l2)[keep][:, : 4 * levels].astype(np.int64)
    with open(os.path.join(out_dir, fname_book), "w") as f:
        for r in b:
            f.write(",".join(str(x) for x in r) + "\n")
    return int(rows.shape[0])


def convert_row(npz_real, npz_gen, out_root, row, ticker="GOOG", date="2026-02-01", levels=10):
    """One eval row -> the Simple_Loader folder triplet. Returns (n_seqs, kept msg counts)."""
    d_real = os.path.join(out_root, row, "data_real")
    d_gen = os.path.join(out_root, row, "data_gen")
    d_cond = os.path.join(out_root, row, "data_cond")
    for d in (d_real, d_gen, d_cond):
        os.makedirs(d, exist_ok=True)
    rm, rl = npz_real["msgs"], npz_real["l2"]
    gm, gl = npz_gen["msgs"], npz_gen["l2"]
    assert rm.shape[0] == gm.shape[0], f"{row}: real Q={rm.shape[0]} vs gen Q={gm.shape[0]}"
    kept = []
    for q in range(rm.shape[0]):
        write_seq_csvs(rm[q], rl[q], d_real,
                       f"{ticker}_{date}_message_real_id_{q}.csv",
                       f"{ticker}_{date}_orderbook_real_id_{q}.csv", levels)
        kept.append(write_seq_csvs(gm[q], gl[q], d_gen,
                                   f"{ticker}_{date}_message_real_id_{q}_gen_id_0.csv",
                                   f"{ticker}_{date}_orderbook_real_id_{q}_gen_id_0.csv", levels))
    return rm.shape[0], kept


def main():
    ap = argparse.ArgumentParser(description="test_eval rollout npz -> LOB-Bench LOBSTER folders")
    ap.add_argument("--eval_dir", required=False, help="test_eval out dir with rollout_*.npz")
    ap.add_argument("--out_dir", required=False)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--ticker", default="GOOG")
    ap.add_argument("--date", default="2026-02-01")
    ap.add_argument("--check", action="store_true", help="synthetic round-trip self-checks")
    args = ap.parse_args()

    if args.check:
        sys.exit(cpu_checks())

    assert args.eval_dir and args.out_dir, "--eval_dir and --out_dir required"
    with open(os.path.join(args.eval_dir, "test_eval_results.json")) as f:
        rows = [r for r, v in json.load(f)["rows"].items() if "panel" in v]
    npz_real = np.load(os.path.join(args.eval_dir, "rollout_real.npz"))
    summary = {}
    for row in rows:
        p = os.path.join(args.eval_dir, f"rollout_{row}.npz")
        if not os.path.exists(p):
            print(f"[adapter] WARN no rollout npz for row '{row}' — skipped"); continue
        n, kept = convert_row(npz_real, np.load(p), args.out_dir, row,
                              ticker=args.ticker, date=args.date, levels=args.levels)
        summary[row] = dict(n_seqs=n, mean_kept_msgs=float(np.mean(kept)))
        print(f"[adapter] row '{row}': {n} seq pairs, mean kept msgs {np.mean(kept):.1f}")
    with open(os.path.join(args.out_dir, "adapter_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[adapter] done -> {args.out_dir}")


def cpu_checks():
    """Synthetic round-trip through lob_bench's OWN loaders (run with the lobbench venv python)."""
    import tempfile
    from ..config import EXP_ROOT as exp
    sys.path.insert(0, os.path.join(exp, "third_party", "lob_bench"))
    import data_loading as dl

    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    rng = np.random.default_rng(0)
    Q, T, W = 3, 40, 40
    msgs = np.zeros((Q, T, 14), np.float64)
    msgs[..., ET_I] = rng.integers(1, 5, (Q, T))
    msgs[..., OID_I] = rng.integers(1, 1000, (Q, T))
    msgs[..., SIZE_I] = rng.integers(1, 100, (Q, T))
    msgs[..., PABS_I] = rng.integers(990, 1010, (Q, T)) * 100
    msgs[..., DIR_I] = rng.integers(0, 2, (Q, T))
    msgs[..., TS_I] = 34200 + np.arange(T)
    msgs[..., TNS_I] = rng.integers(0, 10**9, (Q, T))
    msgs[0, 5, ET_I] = 0                                     # one padding row to drop
    l2 = rng.integers(1, 10**5, (Q, T, W)).astype(np.float64)

    with tempfile.TemporaryDirectory() as td:
        fake = {"msgs": msgs, "l2": l2}
        n, kept = convert_row(fake, fake, td, "rowx", levels=10)
        chk("(A1) convert_row writes Q seq pairs, drops padding",
            n == Q and kept[0] == T - 1 and kept[1] == T, f"kept={kept}")

        d = os.path.join(td, "rowx")
        m = dl.load_message_df(os.path.join(d, "data_real", "GOOG_2026-02-01_message_real_id_0.csv"))
        b = dl.load_book_df(os.path.join(d, "data_real", "GOOG_2026-02-01_orderbook_real_id_0.csv"))
        chk("(A2) lob_bench loaders parse: cols/len/types",
            list(m.columns) == ["time", "event_type", "order_id", "size", "price", "direction"]
            and len(m) == len(b) == T - 1 and b.shape[1] == 40
            and set(np.unique(m.direction)) <= {-1, 1}
            and float(m.time.iloc[1] - m.time.iloc[0]) < 2.0)

        loader = dl.Simple_Loader(os.path.join(d, "data_real"), os.path.join(d, "data_gen"),
                                  os.path.join(d, "data_cond"))
        ok = len(loader) == Q
        try:
            s0 = loader[0]
            ok = ok and s0 is not None
        except Exception as e:
            ok = False
            print(f"      loader[0] raised: {e}")
        chk("(A3) Simple_Loader pairs real/gen per real_id", ok, f"len={len(loader)}")

    print("[adapter] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    main()
