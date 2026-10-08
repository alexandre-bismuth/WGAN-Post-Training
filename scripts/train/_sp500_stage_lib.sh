# SP500 corpus-matched post-training: shared PER-NODE data staging (2026-07-02).
# Sourced by scripts/train/_run_eggroll_sp500_inner.sh AND
# scripts/train/_run_sp500_data_check.sbatch so the trainer and the data gate
# stage IDENTICALLY. Requires _squashfs_helpers.sh to be sourced already.
#
# stage_sp500_data <WORK> — populates:
#   $SP500_FILT      L10 ticker-day symlink farm (msg+book, RTH variant only)
#   $SP500_WIDE      flat wide-book npz symlink farm (L500 snapshots)
#   $SP500_N_PAIRS   number of complete (ticker, day) pairs staged
#   $SP500_N_EXPECT  n_tickers * n_days
# A (ticker, day) pair is staged ONLY if all three exist: RTH message npy,
# RTH L10 book npy (combined farm), and the L500 snapshot npz (image mount).
# Gate: coverage >= SP500_MIN_COVERAGE (default 0.95) else return 1, with the
# full missing list printed. get_dataset raises on any msg day-file without a
# wide npz, so incomplete pairs MUST NOT be linked.
#
# Env consumed: MONTHS SHARD_DIR TICKS TRAIN_DAYS WIDE_NPZ_IMAGE
#               SP500_MIN_COVERAGE PYX
# Lustre 9-point: 2 squashfuse mounts/node (L10 monthly shard + npz image,
# image pre-staged to node-local /tmp first: ONE sequential Lustre read);
# symlink farms on /tmp; no find/recursive ls on Lustre.

stage_sp500_data() {
    local WORK="$1"
    local proc="${SLURM_PROCID:-0}"

    # 1. pooled L10 combined farm (all tickers, TICKER_DATE-prefixed names)
    infer_squashfs_setup "SP500" "${MONTHS:?}" "${SHARD_DIR:?}" || return 1
    local farm="$INFER_DATA_DIR_NODE"

    # 2. wide-book npz image: stage to node-local /tmp, then mount
    local img="${WIDE_NPZ_IMAGE:?}"
    [ -f "$img" ] || { echo "[p$proc] FATAL: npz image missing: $img"; return 1; }
    local img_local="$WORK/$(basename "$img")"
    rsync -a "$img" "$img_local" || return 1
    SP500_WIDE_MNT="$WORK/wide_npz_mnt"; mkdir -p "$SP500_WIDE_MNT"
    if ! mountpoint -q "$SP500_WIDE_MNT" 2>/dev/null; then
        squashfuse "$img_local" "$SP500_WIDE_MNT" || return 1
    fi

    # 3. link complete (ticker, day) pairs; python for speed (one scandir,
    #    set lookups — no per-pair globs over the ~60k-entry farm)
    SP500_FILT="$WORK/train_days"; SP500_WIDE="$WORK/wide_book"
    mkdir -p "$SP500_FILT" "$SP500_WIDE"
    local gate
    gate=$("${PYX:?}" - "$farm" "$SP500_WIDE_MNT" "$SP500_FILT" "$SP500_WIDE" \
                        "${TICKS:?}" "${TRAIN_DAYS:?}" \
                        "${SP500_MIN_COVERAGE:-0.95}" <<'PYEOF'
import glob, os, sys
farm, mnt, filt, wide, ticks, days, min_cov = sys.argv[1:8]
tickers = [t for t in ticks.replace(',', ' ').split() if t]
days = [d for d in days.replace(',', ' ').split() if d]
RTH = '34200000_57600000'
# The combined L10 farm is HETEROGENEOUS by session window: the 7 home tickers
# carry the RTH tag 34200000_57600000; the ~478 generic SP500 tickers carry the
# full-session tag 24900000_57900000 (same tag their fleet-built L500 npz were
# indexed over). Accept the ticker's actual *_10_proc.npy tag; prefer RTH when
# both happen to be present so home tickers stay byte-identical to the GOOG arm.
have = {}
def _prefer(old, new):
    if old is None:
        return new
    return new if (RTH in new and RTH not in old) else old
with os.scandir(farm) as it:
    for e in it:
        name = e.name
        if not name.endswith('_10_proc.npy'):
            continue
        p = name.split('_')
        if len(p) < 2:
            continue
        kind = 'message' if 'message' in name else 'book'
        d = have.setdefault((p[0], p[1]), {})
        d[kind] = _prefer(d.get(kind), name)
n_pairs, missing = 0, []
for t in tickers:
    tdir = os.path.join(mnt, t)
    npz = {}
    if os.path.isdir(tdir):
        for f in glob.glob(os.path.join(tdir, f'{t}_*_L500_linf_snapshots_*.npz')):
            npz[os.path.basename(f).split('_')[1]] = f
    for d in days:
        pair = have.get((t, d), {})
        if 'message' in pair and 'book' in pair and d in npz:
            for f in (pair['message'], pair['book']):
                dst = os.path.join(filt, f)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(farm, f), dst)
            dst = os.path.join(wide, os.path.basename(npz[d]))
            if not os.path.lexists(dst):
                os.symlink(npz[d], dst)
            n_pairs += 1
        else:
            missing.append(f"{t}:{d}"
                           f"[{'M' if 'message' not in pair else ''}"
                           f"{'B' if 'book' not in pair else ''}"
                           f"{'W' if d not in npz else ''}]")
n_expect = len(tickers) * len(days)
cov = n_pairs / n_expect if n_expect else 0.0
print(f'STAGE {n_pairs} {n_expect} {cov:.4f}')
if missing:
    print(f'MISSING ({len(missing)}): ' + ' '.join(missing))
sys.exit(0 if cov >= float(min_cov) else 1)
PYEOF
)
    local rc=$?
    echo "$gate" | sed "s/^/[p$proc] /"
    SP500_N_PAIRS=$(echo "$gate" | awk '/^STAGE/{print $2}')
    SP500_N_EXPECT=$(echo "$gate" | awk '/^STAGE/{print $3}')
    if [ $rc -ne 0 ]; then
        echo "[p$proc] FATAL: ticker-day coverage below threshold" \
             "(${SP500_N_PAIRS}/${SP500_N_EXPECT}, need ${SP500_MIN_COVERAGE:-0.95})"
        return 1
    fi
    return 0
}

stage_sp500_cleanup() {
    if [ -n "${SP500_WIDE_MNT:-}" ] && mountpoint -q "$SP500_WIDE_MNT" 2>/dev/null; then
        fusermount -u "$SP500_WIDE_MNT" 2>/dev/null || true
    fi
}
