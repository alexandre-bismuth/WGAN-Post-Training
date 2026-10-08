from jax import config
config.update("jax_disable_jit", False) 
#config.update("jax_disable_jit", True)

from datetime import datetime
import functools
from glob import glob
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
from lob import train_helpers
import numpy as onp
import os
import sys
import pandas as pd
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
from utils import debug, info
import math
import time

import lob.validation_helpers as valh
import lob.evaluation as eval
import preproc as preproc
# from preproc import transform_L2_state_gpu
import lob.encoding as encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.encoding_1tok import (
    local_to_global_jax, global_to_local_jax, N_FIELDS, N_SPECIAL_TOKENS,
)
import chex

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
# from gym_exchange.environment.base_env.assets.action import OrderIdGenerator

REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

# indices for DECODED message fields
ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

l2_state_n = int(os.environ.get('L2_STATE_N', '10'))

# ENCODED TOKEN INDICES
# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')

# @jax.jit
# def init_msgs_from_l2(book: Union[pd.Series, onp.ndarray]) -> jnp.ndarray:
#     """"""
#     orderbookLevels = len(book) // 4  # price/quantity for bid/ask
#     data = jnp.array(book).reshape(int(orderbookLevels*2),2)
#     newarr = jnp.zeros((int(orderbookLevels*2),8))
#     initOB = newarr \
#         .at[:,3].set(data[:,0]) \
#         .at[:,2].set(data[:,1]) \
#         .at[:,0].set(1) \
#         .at[0:orderbookLevels*4:2,1].set(-1) \
#         .at[1:orderbookLevels*4:2,1].set(1) \
#         .at[:,4].set(0) \
#         .at[:,5].set(job.INITID) \
#         .at[:,6].set(34200) \
#         .at[:,7].set(0).astype('int32')
#     return initOB


def df_msgs_to_jnp(m_df: pd.DataFrame) -> jnp.ndarray:
    """"""
    m_df = m_df.copy()
    cols = ['Time', 'Type', 'OrderID', 'Quantity', 'Price', 'Side']
    if m_df.shape[1] == 7:
        cols += ["TradeID"]
    m_df.columns = cols
    m_df['TradeID'] = 0  #  TODO: should be TraderID for multi-agent support
    col_order=['Type','Side','Quantity','Price','TradeID','OrderID','Time']
    m_df = m_df[col_order]
    m_df = m_df[(m_df['Type'] != 6) & (m_df['Type'] != 7) & (m_df['Type'] != 5)]
    time = m_df["Time"].astype('string').str.split('.',expand=True)
    m_df[["TimeWhole","TimeDec"]] = time.astype('int32')
    m_df = m_df.drop("Time", axis=1)
    mJNP = jnp.array(m_df)
    return mJNP

@jax.jit
def msg_to_jnp(
        m_raw: jax.Array,
    ) -> jax.Array:
    """ Select only the relevant columns from the raw messages
        and rearrange for simulator.
    """
    m = m_raw.copy()
    
    return jnp.array([
        m[EVENT_TYPE_i],
        (m[DIRECTION_i] * 2) - 1,
        m[SIZE_i],
        m[PRICE_ABS_i],
        m[ORDER_ID_i], 
        0,  # TraderID
        m[TIMEs_i],
        m[TIMEns_i],
    ])

msgs_to_jnp = jax.jit(jax.vmap(msg_to_jnp))

# # NOTE: cannot jit due to side effects --> resolve later
# @jax.jit
# def reset_orderbook(
#         b: OrderBook,
#         l2_book: Optional[Union[pd.Series, onp.ndarray]] = None,
#     ) -> OrderBook:
#     """"""
#     b.bids = b.bids.at[:].set(-1)
#     b.asks = b.asks.at[:].set(-1)
#     b.trades = b.trades.at[:].set(-1)
#     if l2_book is not None:
#         msgs = init_msgs_from_l2(l2_book)
#         # NOTE: cannot jit due to side effects --> resolve later
#         # CONTINUE HERE....
#         b.process_orders_array(msgs)
#     return b

def copy_orderbook(
        b: OrderBook
    ) -> OrderBook:
    b_copy = OrderBook(cfg=b.cfg)
    b_copy.bids = b.bids.copy()
    b_copy.asks = b.asks.copy()
    b_copy.trades = b.trades.copy()
    return b_copy

def get_sim(
        init_l2_book: jax.Array,
        replay_msgs_raw: jax.Array,
        start_time: jax.Array,
        sim: OrderBook,
        # nOrders: int = 100,
        # nTrades: int = 100
        # sim_book_levels: int,
        # sim_queue_len: int,
    ) -> Tuple[OrderBook, jax.Array]:
    """
    """

    # reset simulator : args are (nOrders, nTrades)        
    #Set the ns component of the start time to 0 to ensure that init messages are before first message.
    # Only edge case is if first message and init are 0 ns - unlikely. 
    start_time=start_time.at[1].set(0)
    # init simulator at the start of the sequence
    sim_state = sim.reset(init_l2_book,start_time)
    # return sim, sim_state
    # replay sequence in simulator (actual)
    # so that sim is at the same state as the model
    replay = msgs_to_jnp(replay_msgs_raw)
    sim_state = sim.process_orders_array(sim_state, replay)
    return sim_state

get_sims_vmap = jax.jit(
    jax.vmap( 
        get_sim,
        in_axes=(0, 0,0,None),
        out_axes=(0),
    ),
    static_argnums=(3,)
)

def _replay_real_msgs_single(sim, state, real_msgs_raw, n_levels):
    """Replay real messages and extract L2 states at each step."""
    msgs_jnp = msgs_to_jnp(real_msgs_raw)

    def scan_body(carry, msg):
        state = sim.process_order_array(carry, msg)
        l2 = sim.get_L2_state(state, n_levels)
        return state, l2

    _, l2_states = jax.lax.scan(scan_body, state, msgs_jnp)
    return l2_states


def compute_gt_divergence(gen_l2, real_l2, tick_size):
    """Compute per-step divergence between generated and real book states."""
    gen_best_ask = gen_l2[:, 0]
    gen_best_bid = gen_l2[:, 2]
    real_best_ask = real_l2[:, 0]
    real_best_bid = real_l2[:, 2]

    gen_mid = (gen_best_ask + gen_best_bid) / 2.0
    real_mid = (real_best_ask + real_best_bid) / 2.0
    gen_spread = gen_best_ask - gen_best_bid
    real_spread = real_best_ask - real_best_bid

    mid_divergence = jnp.abs(gen_mid - real_mid) / tick_size
    spread_divergence = jnp.abs(gen_spread - real_spread) / tick_size

    gen_ask_vols = gen_l2[:, 1::4][:, :5]
    gen_bid_vols = gen_l2[:, 3::4][:, :5]
    real_ask_vols = real_l2[:, 1::4][:, :5]
    real_bid_vols = real_l2[:, 3::4][:, :5]

    vol_l1_ask = jnp.abs(gen_ask_vols - real_ask_vols).sum(axis=1)
    vol_l1_bid = jnp.abs(gen_bid_vols - real_bid_vols).sum(axis=1)

    return {
        'mid_divergence': mid_divergence,
        'spread_divergence': spread_divergence,
        'vol_l1_ask': vol_l1_ask,
        'vol_l1_bid': vol_l1_bid,
    }


def get_dataset(
        data_dir: str,
        n_messages: int,
        n_eval_messages: int,
        *,
        n_cache_files: int = 500,
        seed: int = 42,
        book_depth: int = 500,
        day_indeces: Optional[List[int]] = None,
        limit_seq: int = math.inf,
        test_split: float = 0.1,
        wide_book_dir: Optional[str] = None,
    ):
    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    if day_indeces is not None:
        #restricts the data to only include certain days.
        msg_files=[msg_files[i] for i in day_indeces]
        book_files=[book_files[i] for i in day_indeces]
    if test_split>0:
        n_test_files = max(1, int(len(msg_files) * test_split))
        msg_files = msg_files[-n_test_files:]
        book_files = book_files[-n_test_files:]

    # Filter out truncated .npy files (try mmap load, skip on failure)
    valid_pairs = []
    for mf, bf in zip(msg_files, book_files):
        ok = True
        for f in (mf, bf):
            try:
                a = onp.load(f, mmap_mode='r')
                _ = a.shape  # force header parse
                del a
            except Exception as e:
                print(f"[get_dataset] Skipping bad file {os.path.basename(f)}: {e}")
                ok = False
                break
        if ok:
            valid_pairs.append((mf, bf))
    if len(valid_pairs) < len(msg_files):
        print(f"[get_dataset] Kept {len(valid_pairs)}/{len(msg_files)} file pairs after validation")
    msg_files, book_files = zip(*valid_pairs) if valid_pairs else ([], [])

    # Build wide_book_files list by matching dates from book_files
    # Supports both .npy (per-event, e.g. L100) and .npz (per-sequence snapshots, e.g. L-inf)
    wide_book_files = None
    if wide_book_dir is not None:
        import re
        wide_book_files = []
        for bf in book_files:
            basename = os.path.basename(bf)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
            if date_match is None:
                raise ValueError(f"Cannot extract date from book file: {basename}")
            date_str = date_match.group(1)
            # TICKER-AWARE matching (multi-ticker farms, 2026-07-02): a merged
            # wide_book_dir holds several tickers' snapshots for the same date;
            # date-only matching would silently pair a GOOG day with the
            # alphabetically-first ticker's book. Canonical basenames start with
            # the ticker (<TICKER>_<date>_...), so prefer '<ticker>_*<date>*'.
            # Single-ticker dirs behave exactly as before via the fallback.
            ticker = basename.split('_')[0]

            def _match(ext_pat):
                cands = sorted(glob(os.path.join(
                    wide_book_dir, f'{ticker}_*{date_str}{ext_pat}')))
                if cands:
                    return cands
                # Date-only fallback (legacy single-ticker dirs) — accept ONLY
                # if unambiguous: all candidates share one filename prefix.
                cands = sorted(glob(os.path.join(wide_book_dir, f'*{date_str}{ext_pat}')))
                prefixes = {os.path.basename(c).split('_')[0] for c in cands}
                if len(prefixes) > 1:
                    raise FileNotFoundError(
                        f"wide_book_dir has {sorted(prefixes)} for {date_str} but none "
                        f"for ticker '{ticker}' — refusing ambiguous cross-ticker match")
                return cands

            wide_candidates = _match('*linf*snapshots*.npz') or _match('*orderbook*proc.npy')
            if len(wide_candidates) == 0:
                raise FileNotFoundError(
                    f"No wide book file found for date {date_str} in {wide_book_dir}")
            wide_book_files.append(wide_candidates[0])
        wb_ext = os.path.splitext(wide_book_files[0])[1] if wide_book_files else ''
        print(f"[get_dataset] Wide book files: {len(wide_book_files)} matched from {wide_book_dir} (format: {wb_ext})")

    ds = LOBSTER_Dataset(
        msg_files,
        n_messages=n_messages + n_eval_messages,
        mask_fn=LOBSTER_Dataset.inference_mask,
        seed=seed,
        n_cache_files=n_cache_files,
        randomize_offset=False,
        book_files=book_files,
        use_simple_book=True,
        book_transform=False,
        book_depth=book_depth,
        return_raw_msgs=True,
        inference=True, #this flag shifts the book to exclude the very first state b4 the 1st message.
        limit_seq_per_file=limit_seq,
        wide_book_files=wide_book_files,
    )
    return ds

def switch(
        condlist: Sequence[jax.Array],
        funclist: Sequence[Callable],
        operands: Any = None,
        *args, **kw
    ) -> Any:
    """ Convenience function for jax.lax.switch, assuming conditions in condlist are
        mutually exclusive cases. If an extra function is given in funclist,
        this will be applied if no condition is true.
    """
    # unroll the loop over a few args
    switch_i = sum([(i+1) * condlist[i] for i in range(len(condlist))])
    # last funclist element is the default function if no condition is true
    if len(condlist) == len(funclist) - 1:
        return jax.lax.switch(
            switch_i,
            (funclist[-1],
            *funclist[:-1]),
            *operands
        )
    elif len(condlist) == len(funclist):
        return jax.lax.switch(
            switch_i - 1,
            funclist,
            *operands
        )
    else:
        raise ValueError(f'Invalid number of conditions and functions, got {len(condlist)} and {len(funclist)}')
            

@jax.jit
def find_order_at_price_closest_time(
        side_array: jax.Array,    # (nOrders, 6)
        price: int,
        time_s: int,
        time_ns: int,
    ) -> jax.Array:
    """Find the active order at `price` whose timestamp is closest to
    (time_s, time_ns).  Returns the 6-element order row, or a
    NEGATIVE_RETURN_ID dummy if no active orders exist at that price.

    Column layout: 0=Price, 1=Qty, 2=OID, 3=TID, 4=Time_s, 5=Time_ns.
    Empty slots have all columns set to -1.

    Uses millisecond-precision time distance (int32-safe: max intraday
    diff ~23.4M ms, well within int32 range of ~2.1B).
    """
    # Mask: active orders at the target price (qty > 0 implies non-empty)
    price_match = (side_array[:, 0] == price) & (side_array[:, 1] > 0)

    # Time distance in milliseconds (int32-safe, no x64 dependency).
    # total_diff = (order_s - target_s) * 1000 + (order_ns - target_ns) // 1e6
    # Max |total_diff| for intraday LOB: ~23,400,000 ms ≪ 2,147,483,647
    diff_ms = (
        (side_array[:, 4] - time_s) * jnp.int32(1000)
        + (side_array[:, 5] - time_ns) // jnp.int32(1_000_000)
    )
    abs_diff = jnp.abs(diff_ms)

    # Non-matching rows get max distance so argmin ignores them
    masked_diff = jnp.where(price_match, abs_diff, jnp.iinfo(jnp.int32).max)
    best_idx = jnp.argmin(masked_diff)

    return jax.lax.cond(
        price_match.any(),
        lambda idx: side_array[idx],
        lambda idx: cst.NEGATIVE_RETURN_ID * jnp.ones((6,), dtype=jnp.int32),
        best_idx,
    )


def get_sim_msg(
        pred_msg_enc: jax.Array,
        sim: OrderBook,
        sim_state: LobState,
        mid_price: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Dict[str, Any]:
    """"""
    # decoded predicted message
    # pred_msg = tok.decode(pred_msg_enc, v).squeeze()
    msg_decoded = encoding.decode_msg(pred_msg_enc, encoder)
    # jax.debug.print('decoded predicted message: \n {}', msg_decoded)
    new_part = msg_decoded[: Message_Tokenizer.N_NEW_FIELDS]
    # ref part is not needed for the simulator logic
    # ref_part = pred_msg[Message_Tokenizer.N_NEW_FIELDS: ]

    event_type = msg_decoded[EVENT_TYPE_i]
    quantity = msg_decoded[SIZE_i]
    side = msg_decoded[DIRECTION_i]
    rel_price = msg_decoded[PRICE_i]
    delta_t_s = msg_decoded[DTs_i]
    delta_t_ns = msg_decoded[DTns_i]
    time_s = msg_decoded[TIMEs_i]
    time_ns = msg_decoded[TIMEns_i]

    rel_price_ref = msg_decoded[PRICE_REF_i]
    quantity_ref = msg_decoded[SIZE_REF_i]
    time_s_ref = msg_decoded[TIMEs_REF_i]
    time_ns_ref = msg_decoded[TIMEns_REF_i]

    p_abs = mid_price + rel_price * tick_size

    # get message for jax lob simulator
    # sim_msg = switch(
    #     (event_type == 1, (event_type == 2) | (event_type == 3), event_type == 4),
    #     (get_sim_msg_new, get_sim_msg_mod, get_sim_msg_exec, construct_dummy_sim_msg),
    #     (event_type, quantity, side, p_abs, time_s, time_ns, 
    #             rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
    #             new_order_id, sim, sim_state,
    #     )
    # )
    # --- Progressive order-ID resolution for cancellations ---
    # Level 1: exact timestamp match (original behavior)
    orig_order_L1 = sim.get_order_at_time(sim_state, side, time_s_ref, time_ns_ref)
    order_id_L1 = orig_order_L1[2]  # col 2 = OID

    # Level 2: price-based closest-time match (fallback)
    side_array = jax.lax.cond(
        side == 1,
        lambda a, b: b,   # side 1 = bids
        lambda a, b: a,   # side 0 = asks
        sim_state.asks, sim_state.bids,
    )
    orig_order_L2 = find_order_at_price_closest_time(
        side_array, p_abs, time_s_ref, time_ns_ref,
    )
    order_id_L2 = orig_order_L2[2]

    # Use L1 if it succeeded, otherwise fall back to L2
    order_id_ref = jnp.where(
        order_id_L1 != cst.NEGATIVE_RETURN_ID,
        order_id_L1,
        order_id_L2,
    )

    # Cancel/delete (type 2/3): use resolved ref ID; otherwise new_order_id
    order_id = jax.lax.cond(
        (event_type == 2) | (event_type == 3),
        lambda new_id, ref_id: ref_id,
        lambda new_id, ref_id: new_id,
        new_order_id, order_id_ref,
    )
    # jax.debug.print("{}",order_id)

    sim_msg = construct_sim_msg(
        event_type,  # type: execution
        side,  # side of execution
        quantity,
        p_abs,
        order_id,
        time_s,
        time_ns,
    )

    msg_decoded = msg_decoded.at[PRICE_ABS_i].set(p_abs) \
                             .at[ORDER_ID_i].set(order_id)

    # return dummy message instead if new_part contains NaNs
    return jax.lax.cond(
        jnp.isnan(new_part).any(),
        lambda sim_msg, msg_decoded: (construct_dummy_sim_msg(), msg_decoded),
        lambda sim_msg, msg_decoded: (sim_msg, msg_decoded),
        sim_msg, msg_decoded
    )

# event_type, side, quantity, price,order_id,trade(r)_id, time_s, time_ns
@jax.jit
def construct_sim_msg(
        event_type: int,
        side: int,
        quantity: int,
        price: int,
        order_id: int,
        time_s: int,
        time_ns: int,
    ):
    """ NOTE: trader ID is set to 0
    """
    return jnp.array([
        event_type,
        (side * 2) - 1,
        quantity,
        price,
        order_id, # order_id
        -88, 
        time_s,
        time_ns,
    ], dtype=jnp.int32)

@jax.jit
def construct_dummy_sim_msg(*args) -> jax.Array:
    return jnp.ones((8,), dtype=jnp.int32) * (-1)

@jax.jit
def construct_raw_msg(
        oid: Optional[int] = encoding.NA_VAL,
        event_type: Optional[int] = encoding.NA_VAL,
        direction: Optional[int] = encoding.NA_VAL,
        price_abs: Optional[int] = encoding.NA_VAL,
        price: Optional[int] = encoding.NA_VAL,
        size: Optional[int] = encoding.NA_VAL,
        delta_t_s: Optional[int] = encoding.NA_VAL,
        delta_t_ns: Optional[int] = encoding.NA_VAL,
        time_s: Optional[int] = encoding.NA_VAL,
        time_ns: Optional[int] = encoding.NA_VAL,
        price_ref: Optional[int] = encoding.NA_VAL,
        size_ref: Optional[int] = encoding.NA_VAL,
        time_s_ref: Optional[int] = encoding.NA_VAL,
        time_ns_ref: Optional[int] = encoding.NA_VAL,
    ):
    msg_raw = jnp.array([
        oid,
        event_type,
        direction,
        price_abs,
        price,
        size,
        delta_t_s,
        delta_t_ns,
        time_s,
        time_ns,
        price_ref,
        size_ref,
        time_s_ref,
        time_ns_ref,
    ])
    return msg_raw

@jax.jit
def rel_to_abs_price(
        p_rel: jax.Array,
        best_bid: jax.Array,
        best_ask: jax.Array,
        tick_size: int = 100,
    ) -> jax.Array:

    p_ref = (best_bid + best_ask) / 2
    p_ref = ((p_ref // tick_size) * tick_size).astype(jnp.int32)
    return p_ref + p_rel * tick_size

@jax.jit
def construct_orig_msg_enc(
        pred_msg_enc: jax.Array,
        #v: Vocab,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> jax.Array:
    """ Reconstructs encoded original message WITHOUT Delta t
        from encoded message string --> delta_t field is filled with NA_TOK
    """
    return jnp.concatenate([
        encoding.encode(jnp.array([1]), *encoder['event_type']),
        pred_msg_enc[slice(*valh.get_idx_from_field('direction'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('price_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size_ref'))],
        # NOTE: no delta_t here
        jnp.full(
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_s']] + \
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_ns']],
            Vocab.NA_TOK
        ),
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns_ref'))],
    ])

@jax.jit
def convert_msg_to_ref(
        pred_msg_enc: jax.Array,
    ) -> jax.Array:
    """ Converts encoded message to reference message part,
        i.e. (price, size, time) tokens
    """
    return jnp.concatenate([
        pred_msg_enc[slice(*valh.get_idx_from_field('price'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns'))],
    ])

def search_orig_msg(
        sim, sim_state, side, p_mod_raw, m_seq, pred_msg_enc, encoder, m_seq_raw
    ):
    vol = sim.get_volume_at_price(sim_state, side, p_mod_raw)
    ret_none = (vol==0)
    
    # if sim.get_volume_at_price(sim_state, side, p_mod_raw) == 0:
    #     debug('No volume at given price, discarding...')
    #     return None, None, None

    m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
    # ref part is only needed to match to an order ID
    # find original msg index location in the sequence (if it exists)
    orig_enc = construct_orig_msg_enc(pred_msg_enc, encoder)
    debug('reconstruct. orig_enc \n', orig_enc)

    sim_ids = sim.get_side_ids(sim_state, side)
    debug('sim IDs', sim_ids[sim_ids > 1])
    mask = get_invalid_ref_mask(m_seq_raw, p_mod_raw, sim_ids)
    orig_i, n_fields_removed = valh.try_find_msg(orig_enc, m_seq, mask)
    
    # didn't find matching original message
    if orig_i is None:
        if sim.get_volume_at_price(sim_state, side, p_mod_raw, True) == 0:
            debug('No init volume found', side, p_mod_raw)
            return None, None, None
        order_id = job.INITID
        # keep generated ref part, which we cannot validate
        orig_msg_found = orig_enc[-REF_LEN: ]
    
    # found matching original message
    else:
        # get order ID from raw data for simulator
        ORDER_ID_i = 0
        order_id = m_seq_raw[orig_i, ORDER_ID_i]
        # found original message: convert to ref part
        EVENT_TYPE_i = 1
        if m_seq_raw[orig_i, EVENT_TYPE_i] == 1:
            orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
        # found reference to original message
        else:
            # take ref fields from matching message
            orig_msg_found = jnp.array(m_seq[orig_i, -REF_LEN: ])

@jax.jit
def get_invalid_ref_mask(
        m_seq_raw: jax.Array,
        p_mod_raw: int,
        sim_ids: jax.Array
    ):
    """
    """
    PRICE_ABS_i = 3
    # filter sequence to prices matching the correct price level
    wrong_price_mask = (m_seq_raw[:, PRICE_ABS_i] != p_mod_raw)
    # filter to orders still in the book: order IDs from sim
    ORDER_ID_i = 0
    not_in_book_mask = jnp.isin(m_seq_raw[:, ORDER_ID_i], sim_ids, invert=True)
    mask = not_in_book_mask | wrong_price_mask
    return mask

@jax.jit
def add_times(
        a_s: jax.Array,
        a_ns: jax.Array,
        b_s: jax.Array,
        b_ns: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
    """ Adds two timestamps given as seconds and nanoseconds each (both fit in int32)
        and returns new timestamp, split into time_s and time_ns
    """
    a_ns = b_ns + a_ns
    extra_s = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    a_s = a_s + b_s + extra_s
    return a_s, a_ns

def _get_safe_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int,
    ) -> int:
    """
    """
    # get current mid price from simulator
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)

    # both negative: 0 ~> (ask + bid) / 2
    # ask negative:  1 ~> bid + tick_size
    # bid negative:  2 ~> ask - tick_size
    # both negative: 3 ~> 0
    case_i = (ask <= 0) * 1 + (bid <= 0) * 2
    
    p_mid = jax.lax.switch(
        case_i,
        (
            lambda ask, bid: (ask + bid) // 2,
            lambda ask, bid: bid + tick_size,
            lambda ask, bid: ask - tick_size,
            lambda ask, bid: 0,
        ),
        ask, bid
    )
    # round down to next valid tick
    p_mid = (p_mid // tick_size) * tick_size
    return p_mid

@partial(jax.jit, static_argnums=(0,))
def _get_new_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        p_mid_old: jax.Array,
        tick_size: int,
    ) -> jax.Array:
    """
    """
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)
    mid = ((((ask + bid) // 2) // tick_size) * tick_size)
    return jax.lax.cond(
        (ask <= 0) | (bid <= 0),
        lambda new, old: old,
        lambda new, old: new,
        mid, p_mid_old
    )

def _add_time_tokens(
        tok_seq_A: jax.Array,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        time_init_s: int,
        time_init_ns: int,
        delta_t_s_start_i: int,
        delta_t_s_end_i: int,
        delta_t_ns_start_i: int,
        delta_t_ns_end_i: int,
    ):
    """
    """
    # TODO: simplify --> separate function
    delta_t_s_toks = tok_seq_A[delta_t_s_start_i: delta_t_s_end_i]
    delta_t_ns_toks = tok_seq_A[delta_t_ns_start_i: delta_t_ns_end_i]
    # debug('delta_t_toks', delta_t_s_toks, delta_t_ns_toks)
    delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
    delta_t_s = encoding.combine_field(delta_t_s, 3)
    delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
    delta_t_ns = encoding.combine_field(delta_t_ns, 3)

    # debug('delta_t', delta_t_s, delta_t_ns)
    time_s_ret, time_ns_ret = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)
    # debug('time', time_s, time_ns)
    
    # encode time and add to sequence
    time_s = encoding.split_field(time_s_ret, 2, 3)
    time_s_toks = encoding.encode(time_s, *encoder['time'])
    time_ns = encoding.split_field(time_ns_ret, 3, 3)
    time_ns_toks = encoding.encode(time_ns, *encoder['time'])

    # debug('time_toks', time_s_toks, time_ns_toks)
    time_tokens=jnp.hstack([time_s_toks, time_ns_toks])
    return time_tokens, time_s_ret, time_ns_ret


def _generate_token(
        train_state : TrainState,
        model : nn.module,
        batchnorm : bool,
        valid_mask_array : jax.Array ,
        sample_top_n : int,
        es,            # EGGROLL ES LoRA factors (S5b) or None = stock path

        m_tok: jax.Array ,
        b_tok: jax.Array ,
        hidden: Tuple,
        token_index : int,
        rng,
    ):
    # syntactically valid tokens for current message position
    valid_mask = valh.get_valid_mask(valid_mask_array, token_index)
    # jax.debug.print("Calling apply model with token {} at index {}",m_tok,token_index)


    # TODO Turn hidden[4] to none here, and use dummy 0 for input.
    # if start_ema:
    #     hidden=hidden[:3]+(None,)

    # jax.debug.print("Start ema{}",start_ema)
    # print(start_ema)
    hidden, logits = valh.apply_model(hidden,
                              m_tok,
                              b_tok,
                              train_state,
                              model,
                              batchnorm,
                              False,
                              es)
    # jax.debug.print("{}",logits.shape)
    logits=logits[0]
    argsortedlogits=jnp.argsort(logits,descending=True)
    # jax.debug.print("Best tokens for index {} before the mask: \n {} \n best logits: \n {}",token_index,argsortedlogits,logits[0][argsortedlogits])
    
    
    # filter out (syntactically) invalid tokens for current position
    #TODO: check that the masking works correctly 

    if valid_mask is not None:
        logits = valh.filter_valid_pred(logits, valid_mask)
    

    
    # jax.debug.print("Best logits for index {} after the mask: \n {}",token_index,jnp.flip(jnp.argsort(logits)))


    # update sequence
    # NOTE: rng arg expects one element per batch element
    rng, rng_ = jax.random.split(rng)
    m_tok = valh.fill_predicted_tok( logits, sample_top_n, jnp.array([rng_]))
    return m_tok, hidden, token_index + 1, rng

def _make_generate_token_scannable(
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        valid_mask_array: jax.Array,
        sample_top_n: int,
        es=None,
    ):
    """
    """
    _partial = functools.partial(
        _generate_token, train_state, model, batchnorm, valid_mask_array, sample_top_n, es
    )
    # Skip jit when inside outer TP jit (nested jit device conflict)
    __generate_token = _partial if valh._TP_MESH is not None else jax.jit(_partial)

    def _generate_token_scannable(carry, xs):
        # m_seq, b_tok, mask_i, rng = carry
        m_tok, hidden, mask_i, rng = __generate_token(*carry)
        return (m_tok, carry[1], hidden, mask_i, rng), m_tok

    return _generate_token_scannable


def _generate_msg(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
        es,            # EGGROLL ES LoRA factors (S5b) or None = stock path

        m_init: jax.Array, #last token from prev message, or start tok.
        b_init: jax.Array, #last book state after prev message, or start book.
        n_msg_todo: int,
        p_mid: jax.Array,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        hidden: Tuple,
        time_i: jax.Array,
        # ema_start:bool,
        b_seq_real:Optional[jax.Array]=None,
    ) -> Tuple[jax.Array, LobState, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """
    """
    # jax.debug.print("Input token with {} msg todo \n {}",n_msg_todo, m_init)

    rng, rng_ = jax.random.split(rng)
    # treat as compile time constants
    with jax.ensure_compile_time_eval():
        l = Message_Tokenizer.MSG_LEN
        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    # 
    time_init_s = time_i[0]
    time_init_ns = time_i[1]

    # TODO: calculating time in case where generation is not sequentially left to right
    #       --> check if delta_t complete --> calc time once

    generate_token_scannable = _make_generate_token_scannable(
        train_state, model, batchnorm, valid_mask_array, sample_top_n, es
    )

    if debug_book:
        b_init=jnp.expand_dims(b_seq_real,0)

    # get next message: generate l tokens:
    # generate tokens until time is reached
    token_idx = 0
    # Pass the first token & book (last from prev msg or START)
    #Generate tokens up to the last delta t (before first abs time)
    gen_token_carry = (m_init, b_init,hidden, token_idx, rng_)
    # jax.debug.print("Book  going into scan token for with {} msg todo \n {}",n_msg_todo, b_init)
    (m_inter, b_inter, hidden, token_idx, rng_), tok_seq_A = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=time_s_start_i
    )
    tok_seq_A=jnp.squeeze(tok_seq_A)
    # fill the time tokens, retain the actual times, to generate the next message. 
    tok_seq_T, time_s, time_ns= _add_time_tokens(
        tok_seq_A,
        encoder,
        time_init_s,
        time_init_ns,
        delta_t_s_start_i,
        delta_t_s_end_i,
        delta_t_ns_start_i,
        delta_t_ns_end_i,
    )
    time_f=jnp.array([time_s, time_ns])

    # jax.debug.print("Calling apply model with time tokens {} with {} msgs to go",tok_seq_T,n_msg_todo)
    
    tok_seq_roll_thru_hidden=jnp.concatenate([tok_seq_A[-1:],tok_seq_T[:-1]])
    # jax.debug.print("Calling apply model with 'time' tokens {} with {} msgs to go",tok_seq_roll_thru_hidden,n_msg_todo)

    hidden,_=valh.apply_model(hidden,
                            tok_seq_roll_thru_hidden,
                            b_init,
                            train_state,
                            model,
                            batchnorm,
                            False,
                            es)

    # update mask index to skip time token positions
    token_idx = time_ns_end_i
    gen_token_carry = (tok_seq_T[-1:], b_init, hidden, token_idx, rng_)

    # finish message generation
    (m_final, b_final, hidden, token_idx, rng_), tok_seq_B = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=l-time_ns_end_i
    )

    tok_seq_B=jnp.squeeze(tok_seq_B)


    # Fully generated message.
    tok_seq_gen=jnp.concatenate([tok_seq_A,tok_seq_T,tok_seq_B])
    # order_id = id_gen.step()  # no order ID generator any more in v3 sim?
    order_id = n_msg_todo

    sim_msg, msg_decoded = get_sim_msg(
        tok_seq_gen,  # the generated message
        # m_seq[:-l],  # sequence without generated message
        # m_seq_raw[1:],   # raw data (same length as sequence without generated message)
        # None,
        sim,
        sim_state,
        mid_price = p_mid,
        new_order_id = order_id,
        tick_size = tick_size,
        encoder = encoder,
    )
    # def print_cond(string_,msg,n_msg_todo):
    #     if n_msg_todo==500:
    #         print(f"{string_} with {n_msg_todo} msg todo \n {msg}")

    # jax.debug.callback(print_cond, "sim_msg", sim_msg,n_msg_todo)

    # feed message to simulator, updating book state
    # jax.debug.callback(print_cond, "sim_state before", sim_state,n_msg_todo)

    sim_state = sim.process_order_array(sim_state, sim_msg)

    # jax.debug.callback(print_cond, "sim_state after", sim_state,n_msg_todo)


    # debug('trades', _trades)

    # get current mid price from simulator
    p_mid_new = _get_new_mid_price(sim, sim_state, p_mid, tick_size)
    # jax.debug.print('p_mid_new {}', p_mid_new)

    # price change in ticks
    p_change = ((p_mid_new - p_mid) // tick_size)#.astype(jnp.int32)

    # get new book state
    book_l2 = sim.get_L2_state(sim_state, l2_state_n)
    # l2_book_states.append(book_l2)

    # error if the new message does not change the book state
    # is_error = (book_l2 == b_seq[-1, 1:]).all()

    new_book_raw = jnp.concatenate([jnp.array([p_change]),time_f, book_l2[0:40]]).reshape(1,-1)
    # jax.debug.print("book shape with time and midprice sim for with {} msg todo \n {}",n_msg_todo,new_book_raw)

    b_final = preproc.transform_L2_state_gpu(new_book_raw, 500, 100)
    # jax.debug.print("book after transform after message, for with {} msg todo \n {}",n_msg_todo,b_final)
    # update book sequence

    n_msg_todo -= 1

    return msg_decoded, sim_state, m_final, tok_seq_gen, b_final, book_l2, p_mid_new, n_msg_todo, hidden, time_f

    
def _make_generate_msg_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
        es=None,
    ):
    """
    """
    _partial_msg = functools.partial(
        _generate_msg, sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book, es
    )
    # Skip jit when inside outer TP jit (nested jit device=[0] conflict)
    __generate_msg = _partial_msg if valh._TP_MESH is not None else jax.jit(_partial_msg, device=jax.devices()[0])

    def _generate_msg_scannable(gen_state, input):
        """ Wrapper for _generate_msg to be used with jax.lax.scan
        """
        b_seq_real=input
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time= gen_state
        rng, rng_ = jax.random.split(rng)
        
        msg_decoded, sim_state, m_seq, msg_token, b_seq, book_l2, p_mid, n_msg_todo,hidden, time = __generate_msg(
            m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng_, hidden, time, b_seq_real
        )
        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng,hidden, time), (msg_decoded, book_l2, msg_token, b_seq)
    return _generate_msg_scannable


# ─────────────────────────────────────────────────────────────────────
# 1tok inference path: one model call per message → 24 field logits
# ─────────────────────────────────────────────────────────────────────

def _sample_fields_1tok(field_logits_list, sample_top_n, rng):
    """Sample from 24 per-field logit distributions independently.

    Args:
        field_logits_list: list of 24 arrays, each (1, V_i) log-softmax
        sample_top_n: 1 for argmax, >1 for top-k, -1 for full distribution
        rng: JAX PRNG key
    Returns:
        (sampled_local, rng) where sampled_local is (24,) int32 local indices
    """
    samples = []
    for i, logits in enumerate(field_logits_list):
        rng, rng_ = jax.random.split(rng)
        logits_i = jnp.squeeze(logits)  # (1, 1, V_i) or (1, V_i) → (V_i,)
        # Block special tokens during generation
        # logits_i is already log_softmax from MultiFieldDecoder
        logits_i = logits_i.at[:N_SPECIAL_TOKENS].set(-jnp.inf)
        if sample_top_n == 1:
            chosen = jnp.argmax(logits_i)
        elif sample_top_n > 0:
            top_k_vals, top_k_idx = jax.lax.top_k(logits_i, sample_top_n)
            probs = jnp.exp(top_k_vals)
            probs = probs / probs.sum()  # renormalize after top-k
            chosen = jax.random.choice(rng_, top_k_idx, p=probs)
        else:
            probs = jnp.exp(logits_i)
            probs = probs / probs.sum()  # renormalize after blocking specials
            chosen = jax.random.choice(rng_, jnp.arange(logits_i.shape[0]), p=probs)
        samples.append(chosen)
    return jnp.stack(samples), rng


def _generate_msg_1tok(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,

        m_init: jax.Array,      # (1, 24) local indices — last generated or last cond msg
        b_init: jax.Array,      # (1, book_dim) — last book state
        n_msg_todo: int,
        p_mid: jax.Array,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        hidden: Tuple,
        time_i: jax.Array,
        b_seq_real: Optional[jax.Array] = None,
    ) -> Tuple:
    """Generate one complete message in a single model call (1tok mode)."""
    rng, rng_ = jax.random.split(rng)

    with jax.ensure_compile_time_eval():
        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    time_init_s = time_i[0]
    time_init_ns = time_i[1]

    if debug_book:
        b_init = jnp.expand_dims(b_seq_real, 0)

    # One model call → all 24 field logits
    hidden, field_logits = valh.apply_model_1tok(
        hidden, m_init, b_init, train_state, model, batchnorm, False)

    # Sample all 24 fields at once
    sampled_local, rng_ = _sample_fields_1tok(field_logits, sample_top_n, rng_)

    # Convert to global token IDs (same space as the 26tok encoding)
    sampled_global = local_to_global_jax(sampled_local)

    # Compute absolute time from delta_t + previous time (override model's time prediction)
    time_tokens, time_s, time_ns = _add_time_tokens(
        sampled_global,  # full message — _add_time_tokens indexes into it
        encoder,
        time_init_s, time_init_ns,
        delta_t_s_start_i, delta_t_s_end_i,
        delta_t_ns_start_i, delta_t_ns_end_i,
    )
    # Overwrite absolute time fields with computed values
    sampled_global = sampled_global.at[time_s_start_i:time_ns_end_i].set(time_tokens)
    time_f = jnp.array([time_s, time_ns])

    order_id = n_msg_todo

    # Decode and process through simulator (same as the 26tok path from here)
    sim_msg, msg_decoded = get_sim_msg(
        sampled_global,
        sim,
        sim_state,
        mid_price=p_mid,
        new_order_id=order_id,
        tick_size=tick_size,
        encoder=encoder,
    )

    sim_state = sim.process_order_array(sim_state, sim_msg)

    p_mid_new = _get_new_mid_price(sim, sim_state, p_mid, tick_size)
    p_change = ((p_mid_new - p_mid) // tick_size)

    book_l2 = sim.get_L2_state(sim_state, l2_state_n)
    new_book_raw = jnp.concatenate([jnp.array([p_change]), time_f, book_l2[0:40]]).reshape(1, -1)
    b_final = preproc.transform_L2_state_gpu(new_book_raw, 500, 100)

    # Next step input: convert back to local for the model
    sampled_local_out = global_to_local_jax(sampled_global)
    m_final = sampled_local_out.reshape(1, N_FIELDS)  # (1, 24)

    n_msg_todo -= 1

    return msg_decoded, sim_state, m_final, sampled_global, b_final, book_l2, p_mid_new, n_msg_todo, hidden, time_f


def _make_generate_msg_1tok_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
    ):
    __generate_msg = jax.jit(functools.partial(
        _generate_msg_1tok, sim, train_state, model, batchnorm,
        encoder, sample_top_n, tick_size, debug_book
    ), device=jax.devices()[0])

    def _generate_msg_scannable(gen_state, input):
        b_seq_real = input
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time = gen_state
        rng, rng_ = jax.random.split(rng)

        msg_decoded, sim_state, m_seq, msg_token, b_seq, book_l2, p_mid, n_msg_todo, hidden, time = __generate_msg(
            m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng_, hidden, time, b_seq_real
        )
        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time), (msg_decoded, book_l2, msg_token, b_seq)
    return _generate_msg_scannable


@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9, 13, 15), backend='gpu')
def generate_1tok(
        sim: OrderBook,              # static
        train_state: TrainState,
        model: nn.Module,            # static
        batchnorm: bool,             # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,           # static
        tick_size: int,              # static
        m_seq_cond: jax.Array,       # (n_cond_msgs+1, 24) local indices
        b_seq_cond: jax.Array,       # (n_cond_msgs+1, book_dim)
        n_msg_todo: int,             # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        init_hidden: Tuple,
        conditional: bool,           # static
        init_time: jax.Array,
        debug_book: bool = False,
        b_seq_real: Optional[jax.Array] = None,
        valid_mask_array: Optional[jax.Array] = None,  # unused for 1tok, kept for API compat
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    print("WARNING: Compiling generate_1tok, you should only see this once.")

    if not debug_book:
        b_seq_real = None

    if conditional:
        def roll_hidden_scan_1tok(carry, xs):
            m_seq, b_seq = xs
            h = carry
            h, _ = valh.apply_model_1tok(
                h, m_seq, b_seq, train_state, model, batchnorm, True)
            return h, None

        # Split conditioning: one message per scan step
        N = m_seq_cond[:-1].shape[0]  # n_cond_msgs
        m_seq_cond_split = m_seq_cond[:-1].reshape((N, 1, N_FIELDS))
        b_seq_cond_split = b_seq_cond[:-1].reshape((N, 1) + b_seq_cond.shape[1:])

        hidden_state, _ = jax.lax.scan(
            roll_hidden_scan_1tok, init_hidden, (m_seq_cond_split, b_seq_cond_split))
        init_token = m_seq_cond[-1:]   # (1, 24)
        init_book = b_seq_cond[-1:]    # (1, book_dim)
        init_time = jnp.asarray(valh.get_first_time_1tok(m_seq_cond, encoder))
    else:
        hidden_state = init_hidden
        init_token = m_seq_cond
        init_book = b_seq_cond

    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)

    generate_msg_scannable = _make_generate_msg_1tok_scannable(
        sim, train_state, model, batchnorm,
        encoder, sample_top_n, tick_size, debug_book,
    )
    gen_state, (msgs_decoded, l2_book_states, msgs_tokens, b_finals) = jax.lax.scan(
        generate_msg_scannable,
        (init_token, init_book, n_msg_todo, p_mid, sim_state, rng, hidden_state, init_time),
        length=n_msg_todo,
        xs=b_seq_real,
    )

    num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()
    b_finals = jnp.squeeze(b_finals, axis=1)        # (n_gen,1,503) -> (n_gen,503), matches real_cont_book

    return msgs_decoded, l2_book_states, num_errors, msgs_tokens, b_finals


generate_batched_1tok = jax.jit(
    jax.vmap(
        generate_1tok,
        in_axes=(
            None, None, None, None, None,
            None, None,    0,    0, None,
            0,       0,    0, None,    0,
            None,    0, None,
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9, 13, 15), backend='gpu'
)


@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9,13,15),backend='gpu')
def generate(
        sim: OrderBook,  # static
        train_state: TrainState,
        model: nn.Module,  # static
        batchnorm: bool,  # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,  # static
        tick_size: int,  # static
        m_seq_cond: jax.Array,
        b_seq_cond: jax.Array,
        n_msg_todo: int,  # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        init_hidden : Tuple,
        conditional : bool, # static
        init_time : jax.Array,
        debug_book: bool=False,
        b_seq_real: Optional[jax.Array]=None, #Must be very careful, these should only be used for debugging.
        valid_mask_array: Optional[jax.Array]=None,  # pre-computed syntax mask
        es=None,  # EGGROLL ES LoRA factors (S5b); None = stock path (byte-identical).
                  # With es, `model` must be the BatchPaddedLobPredModelES wrapper.
        # if eval_msgs given, also returns loss of predictions
        # e.g. to calculate perplexity
        # m_seq_eval: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:

    # id_gen = OrderIdGenerator()
    # l = Message_Tokenizer.MSG_LEN
    # v = Vocab()
    # vocab_len = len(v)
    # last_start_i = m_seq.shape[0] - l
    # l2_book_states = []
    # m_seq_raw = m_seq_raw.copy()
    # num_errors = 0
    print("WARNING: Compiling the generate function, you should only see this once.")
    # m_seq_cond=m_seq_cond.copy()
    # b_seq_cond=b_seq_cond.copy()

    if valid_mask_array is None:
        with jax.ensure_compile_time_eval():
            valid_mask_array = valh.syntax_validation_matrix()

    # valid_mask_array=None 
    # jax.debug.print("Note: Valid mask turned off in generate_token")

    if not debug_book:
        b_seq_real=None

    
    if conditional:
        def roll_hidden_scan(carry,xs):
            m_seq,b_seq=xs
            h=carry
            h,log=valh.apply_model(h,
                                m_seq, #All but the last token go in here to run fwd the hidden state.
                                b_seq, # All of the books, because last book needed for 21 1st toks of last message.
                                train_state,
                                model,
                                batchnorm,
                                True,
                                es)
            carry=h
            return carry, None
        

        print(m_seq_cond[:-1],b_seq_cond[:-1])
        # Split conditioning into N chunks for the scan.
        # N=n_cond_msgs (1 message per step) keeps attention matrices small,
        # avoiding O(L²) OOM for transformers while being harmless for S5.
        l = Message_Tokenizer.MSG_LEN
        total_cond_tokens = m_seq_cond[:-1].shape[0]
        N = total_cond_tokens // l   # 1 message per chunk
        chex.assert_is_divisible(total_cond_tokens, N)
        chex.assert_is_divisible(b_seq_cond[:-1].shape[0], N)
        m_seq_cond_split = m_seq_cond[:-1].reshape((N, -1))
        b_seq_cond_split = b_seq_cond[:-1].reshape((N, -1) + b_seq_cond[:-1].shape[1:])

        hidden_state,_ = jax.lax.scan(roll_hidden_scan, init_hidden, (m_seq_cond_split, b_seq_cond_split))
        init_token=m_seq_cond[-1:]
        init_book=b_seq_cond[-1:]
        init_time=jnp.asarray(valh.get_first_time(m_seq_cond,encoder))
        init_ema=False
    else:
        #If unconditional generation, then the initial token
        #  and book have one less dimension.   
        hidden_state=init_hidden
        init_ema=True
        # FIXME: Currently wrong and incomplete. 
        # Needs to just be START token and init book state.
        assert (m_seq_cond.ndim==1) & (m_seq_cond.shape[0]==1), "m_seq_cond needs to be a scalar (start tok?)"
        init_token=m_seq_cond
        init_book=b_seq_cond

    # jax.debug.print("hidden_state vs init hidden state {}",hidden_state==init_hidden)

    # get current mid price from simulator
    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)
    # jax.debug.print('generate - p_mid {}', p_mid)

    generate_msg_scannable = _make_generate_msg_scannable(
        sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book, es,
    )
    gen_state, (msgs_decoded, l2_book_states,msgs_tokens, b_finals) = jax.lax.scan(
        generate_msg_scannable,
        (init_token, init_book, n_msg_todo, p_mid, sim_state,rng, hidden_state,init_time),
        length=n_msg_todo,
        xs=b_seq_real,
    )
    (final_token, final_book,n_msg_todo, p_mid, sim_state, rng, hidden_state,final_time) = gen_state

    # all_msg_toks_gen=jnp.concatenate(msgs_tokens)



    # count errors when the message does not change the (visible) book state
    num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()
    b_finals = jnp.squeeze(b_finals, axis=1)        # (n_gen,1,503) -> (n_gen,503), matches real_cont_book

    return msgs_decoded, l2_book_states, num_errors, msgs_tokens, b_finals

generate_batched = jax.jit(
    jax.vmap(
        generate,
        in_axes=(
            None, None, None, None, None,
            None, None,    0,    0, None,
            0,       0,    0, None,    0,
            None,    0, None,
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9,13,15),backend='gpu'
)

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def calc_sequence_losses(
        m_seq,
        b_seq,
        state,
        model,
        batchnorm,
        n_inp_msgs,  # length of input sequence in messages
        valid_mask_array
    ):
    """ Takes a sequence of messages, and calculates cross-entropy loss for each message,
        based on the next message in the sequence.
    """
    @partial(jax.jit, static_argnums=(1,2))
    def moving_window(a: jax.Array, size: int, stride: int = 1):
        starts = jnp.arange(0, len(a) - size + 1, stride)
        return jax.vmap(
            lambda start: jax.lax.dynamic_slice(
                a,
                (start, *jnp.zeros(a.ndim-1, dtype=jnp.int32)),
                (size, *a.shape[1:])
            )
        )(starts)
    
    l = Message_Tokenizer.MSG_LEN

    @jax.jit
    def prep_single_inp(
            mask_i,
            na_mask,
            m_seq,
            b_seq,
        ):
        m_seq = m_seq.copy().reshape((-1, l))
        last_msg = jnp.where(
            na_mask,
            Vocab.HIDDEN_TOK,#Vocab.NA_TOK,
            m_seq[-1]
        )
        m_seq = m_seq.at[-1, :].set(last_msg).reshape(-1)
        m_seq, y = valh.mask_last_msg_in_seq(m_seq, mask_i)

        input = (m_seq, b_seq)
        integration_timesteps = (
            jnp.ones(len(m_seq), dtype=jnp.float32), 
            jnp.ones(len(b_seq), dtype=jnp.float32)
        )
        return input, integration_timesteps, y.astype(jnp.float32)
    prep_multi_input = jax.vmap(prep_single_inp, in_axes=(0, 0, None, None))

    @jax.jit
    def single_msg_losses(carry, inp):
        @partial(jax.jit, static_argnums=(0,))
        def na_mask_slice(last_non_masked_i):
            a = jnp.ones((l,), dtype=jnp.bool_)
            a = a.at[: last_non_masked_i+1].set(False)
            return a

        m_seq, b_seq, valid_mask = inp
        mask_idxs = jnp.concatenate([jnp.arange(0, TIME_START_I), jnp.arange(TIME_END_I, l)])
        na_masks = jnp.array([na_mask_slice(i) for i in range(TIME_START_I)] \
            + [na_mask_slice(i) for i in range(TIME_END_I, l)])

        bsz = 10
        assert 2*bsz >= mask_idxs.shape[0], f'bsz:{bsz}; msg len:{mask_idxs.shape[0]}'
        # split inference into two batches to avoid OOM
        input, integration_timesteps, y1 = prep_multi_input(mask_idxs[:bsz], na_masks[:bsz], m_seq, b_seq)
        logits1 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        input, integration_timesteps, y2 = prep_multi_input(mask_idxs[-bsz:], na_masks[-bsz:], m_seq, b_seq)
        logits2 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        
        logits = jnp.concatenate([logits1, logits2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        y = jnp.concatenate([y1, y2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        
        # filter out (syntactically) invalid tokens for current position
        if valid_mask is not None:
            logits = valh.filter_valid_pred(logits, valid_mask)

        losses = train_helpers.cross_entropy_loss(logits, y)
        return carry, losses

    m_seq = m_seq.reshape((-1, l))
    inputs = (
        moving_window(m_seq, n_inp_msgs),
        moving_window(b_seq, n_inp_msgs),
        jnp.repeat(
            jnp.expand_dims(
                jnp.delete(valid_mask_array, slice(TIME_START_I, TIME_END_I), axis=0),
                axis=0
            ),
            m_seq.shape[0] - n_inp_msgs + 1,
            axis=0
        )
    )
    last_i, losses = jax.lax.scan(
        single_msg_losses,
        init=0,
        xs=inputs
    )
    return losses

# def generate_single_rollout(
#         m_seq_inp,
#         b_seq_inp,
#         n_gen_msgs,
#         sim,
#         sim_state,
#         state,
#         model,
#         batchnorm,
#         encoder,
#         rng,
#     ):
    
#     rng, rng_ = jax.random.split(rng)

#     # generate predictions
#     m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate(
#         m_seq_inp,
#         b_seq_inp,
#         n_gen_msgs,
#         sim,
#         sim_state,
#         state,
#         model,
#         batchnorm,
#         encoder,
#         rng_,
#         sample_top_n=-1,  # sample from entire distribution
#     )

#     return (
#         m_seq_gen,
#         b_seq_gen,
#         {
#             'num_errors': num_errors,
#             'l2_book_states': l2_book_states,
#         }
#     )

# # sample from distribution of rollouts with same input an different rng keys
# generate_repeated_rollouts = jax.vmap(generate_single_rollout, in_axes=((None,)*9 + (0,)))
# # sample different rollouts with different input sequences (and different rng keys)
# generate_multiple_rollouts = jax.vmap(generate_single_rollout, in_axes=(0, 0, None, None, 0, None, None, None, None, 0))

def sample_new(
        n_samples: int,  # draw n random samples from dataset for evaluation
        batch_size: int,  # how many samples to process in parallel
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len_cond: int, #cond: should be 0 if uncond.
        n_cond_msgs: int, #cond: should be 0 if uncond
        n_gen_msgs: int, #gen
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        stock_symbol: str,
        n_vol_series: int = 500,
        # sim_book_levels: int = 20,
        # sim_queue_len: int = 100,
        # data_levels: int = 10,
        save_folder: str = './data_saved/',
        tick_size: int = 100,
        sample_top_n: int = -1,
        init_hidden: Optional[Tuple] = None,
        args: Optional[Any] = None,
        conditional: bool = True,
        v: Vocab = Vocab(),
        overfit_debug: bool = False,
        sample_indices: Optional[List[int]] = None,
        wide_levels: int = 10,
        token_mode: str = '24tok',
        gt_compare: bool = False,
    ):
    """
    """
    is_1tok = (token_mode == '1tok')
    assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
    if conditional is False:
        assert n_cond_msgs==0, "If conditional flag is false then cannot expect to have any messages for conditioning."
        if not is_1tok:
            assert seq_len_cond==0, "If conditional flag is false, then cannot have any tokens for conditioning."

    rng, rng_ = jax.random.split(rng)
    if sample_indices is not None:
        # Pre-determined indices (from multi-GPU rank splitting or HF-matched mode)
        flat = list(sample_indices)
        sample_i = [flat[i:i+batch_size] for i in range(0, len(flat), batch_size)]
    elif overfit_debug:
        sample_i = [list(range(batch_size))]
    else:
        sample_i = jax.random.choice(
            rng_,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_samples // batch_size, batch_size),
            replace=False
        ).tolist()
    rng, rng_ = jax.random.split(rng)

    # create folders to save the data if they don't exist yet
    Path(save_folder + f'/data_cond/').mkdir(exist_ok=True, parents=True)
    Path(save_folder + f'/data_real/').mkdir(exist_ok=True, parents=True)
    Path(save_folder + f'/data_gen/').mkdir(exist_ok=True, parents=True)


    if (init_hidden == None):
        ssm_type = getattr(args, 'ssm_type', 'gdn')
        if ssm_type == 'mamba3':
            m3_expand = getattr(args, 'mamba3_expand', 2)
            m3_headdim = getattr(args, 'mamba3_headdim', 64)
            m3_d_state = getattr(args, 'mamba3_d_state', 128)
            d_inner = m3_expand * args.d_model
            m3_nh = max(1, d_inner // m3_headdim)
            m3_rope_frac = getattr(args, 'mamba3_rope_fraction', 0.5)
            num_rope_angles = int(m3_d_state * m3_rope_frac) // 2
            # TP: use local head count for hidden state (runs inside shard_map)
            tp_size = getattr(args, 'tp_size', 1)
            m3_nh_carry = m3_nh // tp_size if tp_size > 1 else m3_nh
            init_hidden = model.initialize_carry(1,
                                            hidden_size=0,
                                            ssm_type='mamba3',
                                            n_message_layers=args.n_message_layers,
                                            n_book_pre_layers=args.n_book_pre_layers,
                                            n_book_post_layers=args.n_book_post_layers,
                                            n_fused_layers=args.n_layers,
                                            h_size_ema=args.d_model,
                                            n_heads=m3_nh_carry, headdim=m3_headdim,
                                            d_state=m3_d_state, num_rope_angles=num_rope_angles,
                                            d_book=getattr(args, 'd_book', 503))
        else:
            gdn_hd = getattr(args, 'gdn_head_dim', 128)
            gdn_nh = getattr(args, 'gdn_num_heads', None) or max(1, args.d_model // gdn_hd)
            gdn_hvd = gdn_hd * getattr(args, 'gdn_expand_v', 2)
            init_hidden = model.initialize_carry(1,
                                            hidden_size=0,
                                            n_message_layers=args.n_message_layers,
                                            n_book_pre_layers=args.n_book_pre_layers,
                                            n_book_post_layers=args.n_book_post_layers,
                                            n_fused_layers=args.n_layers,
                                            h_size_ema=args.d_model,
                                            num_heads=gdn_nh, head_dim=gdn_hd, head_v_dim=gdn_hvd,
                                            d_book=getattr(args, 'd_book', 503))

    # jax.debug.print("Init hidden is: \n {}",len(init_hidden))
    # Assumes only a single hidden state is given and needs to be duplicated. TODO Add a flag. 
    init_hidden_batched=jax.tree_util.tree_map(lambda x : jnp.resize(x,(batch_size,)+x.shape),init_hidden)


    #TODO: complete these options to make sure every case works and add some asserts. 
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_hidden ))
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_hidden_batched ))
    

    # init_time_batched=jax.tree_util.tree_map(lambda x : jnp.resize(x,(batch_size,)+x.shape),init_time)
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_time ))
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_time_batched ))
    # nOrders formula is pinned per wide_levels tier for LOBbench reproducibility.
    # Do NOT change — jax.random.choice in cancel_order fallback is array-length-
    # dependent, so any nOrders change silently shifts all _levels metrics.
    # L≤100: 2*levels+50 matches all published baselines (scaling law, soups, GDN, Mamba2).
    # L>100: levels+500 gives higher headroom to avoid simulator overflow.
    sim_nOrders = 2 * wide_levels + 50 if wide_levels <= 100 else wide_levels + 500
    sim_init = OrderBook(cfg=JAXLOB_Configuration(
        nOrders=sim_nOrders,
        book_depth=wide_levels,
        cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value,
    ))
    if wide_levels > 10:
        print(f"[sample_new] Wide init: {wide_levels} levels, nOrders={sim_nOrders}, "
              f"init_id_range=[{sim_init.cfg.init_id}, {sim_init.cfg.init_id - wide_levels*2}]")
    # all_metrics = []
    initial=True
    for batch_i in tqdm(sample_i):
        # print('BATCH', batch_i)
        # TODO: check if we can init the dataset without the raw data 
        #       if it's not needed 
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        # print("sample_new: M_seq_inputs:", m_seq)
        # print('m_seq.shape before jnp.array', onp.array(m_seq).shape)
        m_seq = jnp.array(m_seq)
        b_seq_pv = jnp.array(b_seq_pv)
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)

        # transform book to volume image representation for model
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)
        init_time_batched=b_seq_pv[:,0,1:3]


        #Add the start token
        #FIXME: Move this to the data loader using the inference mask. 
            # Done?
        # m_seq=jnp.concatenate([jnp.ones((batch_size,1),dtype=int)*v.START_TOK,m_seq],axis=1)

        print(m_seq.shape)
        # encoded data
        if is_1tok:
            # Dataset prepends a single START token → strip it before reshaping
            # to (batch, n_msgs, 24) so field positions align correctly.
            m_seq_flat = m_seq[:, 1:]  # drop START token at position 0
            n_clean = (m_seq_flat.shape[1] // N_FIELDS) * N_FIELDS
            n_total_msgs = n_clean // N_FIELDS
            m_seq_2d = m_seq_flat[:, :n_clean].reshape(batch_size, n_total_msgs, N_FIELDS)
            m_seq_2d = global_to_local_jax(m_seq_2d)  # broadcasts over (batch, n_msgs, 24)
            m_seq_inp = m_seq_2d[:, :n_cond_msgs+1]   # (batch, n_cond+1, 24)
            m_seq_eval = m_seq_flat[:, (n_cond_msgs+1)*N_FIELDS:]  # keep flat for debug/save
        else:
            m_seq_inp = m_seq[:, : seq_len_cond+1]
            m_seq_eval = m_seq[:, (seq_len_cond+1): ]
        # Debug prints to file
        # Set print options to show all array elements
        if overfit_debug:
            with open(f'debug_m_seq_inp_batch_{batch_i[0]}.txt', 'w') as f:
                print(f"m_seq_inp shape: {m_seq_inp.shape}", file=f)
                print(f"m_seq_inp:\n{m_seq_inp}", file=f)

            with open(f'debug_m_seq_eval_batch_{batch_i[0]}.txt', 'w') as f:
                print(f"m_seq_eval shape: {m_seq_eval.shape}", file=f)
                print(f"m_seq_eval:\n{m_seq_eval}", file=f)

        # Reset print options to default
        b_seq_inp = b_seq[: , : n_cond_msgs+1]
        b_seq_eval = b_seq[:, (n_cond_msgs+1):] 
        # true L2 data: remove price change column
        # shape: [batch, messages, levels]
        b_seq_pv_inp = onp.array(b_seq_pv[:, : n_cond_msgs+1, 3:])
        b_seq_pv_eval = onp.array(b_seq_pv[:, (n_cond_msgs+1):, 3:]) #Drop the midprice and times for logging purposes in lobster.

        # raw LOBSTER data
        m_seq_raw_inp = msg_seq_raw[:, : n_cond_msgs]
        m_seq_raw_eval = msg_seq_raw[:, n_cond_msgs: ]

        # initialise simulator
        sim_states_init = get_sims_vmap(
            book_l2_init,  # book state before any messages
            m_seq_raw_inp, # messages to replay to init sim
            init_time_batched,
            sim_init,
            # TODO: consider passing nOrders, nTrades
        )

        # book state after initialisation (replayed messages)
        # actually, this is already part of the input data --> only needed for comparison
        # l2_book_states_init = sim_init.get_L2_states_vmap(sim_states_init, l2_state_n)

        # run actual messages on sim_eval (once) to compare
        # convert m_seq_raw_eval to sim_msgs
        # msgs_eval = msgs_to_jnp(m_seq_raw_eval[: n_gen_msgs])
        # sim_state_eval, l2_book_states_eval, _ = sim_init.process_orders_array_l2(sim_state_init, msgs_eval, l2_state_n)

        if overfit_debug:
            debug_book=True
        else:
            debug_book=False
        if debug_book:
            real_book=jnp.concatenate([jnp.expand_dims(b_seq_inp[:,-1],axis=1),b_seq_eval[:,:-1]],axis=1)
            print(real_book.shape)
        else:
            real_book=None
        # print('m_seq_inp.shape', m_seq_inp.shape)
        # print('b_seq_inp.shape', b_seq_inp.shape)
        # print('sim_states_init.asks.shape', sim_states_init.asks.shape)
        # print('sim_states_init.bids.shape', sim_states_init.bids.shape)
        # print('sim_states_init.trades.shape', sim_states_init.trades.shape)
        # init_hidden_batched,init_time_batched,init_token_batched,init_book_batched=roll_batched(
        #     conditional, #Static             
        #     train_state,  # None map, static? 
        #     model, # static
        #     batchnorm, # static
        #     encoder,
        #     init_hidden_batched,
        #     m_seq_inp[:], # in_axis = 0
        #     b_seq_inp, # in_axis = 0
        #     init_time_batched,
        # )



        print('Before generation, real book is (should be none):', real_book)
        if initial:
            initial = False
            if is_1tok:
                # 1tok: no syntax mask needed, use generate_batched_1tok
                valid_mask_array = None
                generate_traced = generate_batched_1tok.trace(
                    sim_init,
                    train_state,
                    model,
                    batchnorm,
                    encoder,
                    sample_top_n,
                    tick_size,
                    m_seq_inp[:],       # (batch, n_cond+1, 24)
                    b_seq_inp,
                    n_gen_msgs,
                    sim_states_init,
                    jax.random.split(rng_, batch_size),
                    init_hidden_batched,
                    conditional,
                    init_time_batched,
                    debug_book,
                    real_book,
                    valid_mask_array,
                )
            else:
                is_transformer = getattr(args, 'model_type', 's5') == 'transformer'
                valid_mask_array = valh.syntax_validation_matrix(
                    block_start_tok=is_transformer)
                if valh._TP_MESH is not None:
                    # TP inference: wrap vmap(generate) in shard_map, then jit
                    # Static args captured by closure; dynamic args go through shard_map
                    from jax.experimental.shard_map import shard_map
                    from jax.sharding import PartitionSpec as P, NamedSharding

                    _sim_init = sim_init
                    _model = model
                    _batchnorm = batchnorm
                    _sample_top_n = sample_top_n
                    _tick_size = tick_size
                    _n_gen_msgs = n_gen_msgs
                    _conditional = conditional
                    _debug_book = debug_book

                    _vmap_axes = (
                        None, None, None, None, None,
                        None, None,    0,    0, None,
                        0,       0,    0, None,    0,
                        None,    0, None,
                    )

                    # Use raw (non-jitted) generate — inner @jax.jit(backend='gpu')
                    # conflicts with the outer jit(shard_map(...))
                    _generate_raw = generate.__wrapped__

                    # Unwrap ALL jitted functions called from generate's transitive
                    # closure that have backend='gpu' or device=... specifications.
                    # These conflict with the outer jit(in_shardings=4-device).
                    # The patch is restored after .trace() since the unwrapped
                    # callable is baked into the compiled XLA graph; later calls
                    # of generate_compiled don't need the Python attribute swap.
                    _saved_jits = {}
                    _jit_targets = [
                        (preproc, 'transform_L2_state_gpu'),
                    ]
                    try:
                        for _mod, _name in _jit_targets:
                            _fn = getattr(_mod, _name)
                            if hasattr(_fn, '__wrapped__'):
                                _saved_jits[(_mod, _name)] = _fn
                                setattr(_mod, _name, _fn.__wrapped__)

                        def _generate_tp(train_state, encoder, m_seq_inp, b_seq_inp,
                                         sim_states_init, rng, init_hidden_batched,
                                         init_time_batched, real_book, valid_mask_array):
                            return jax.vmap(_generate_raw, in_axes=_vmap_axes)(
                                _sim_init, train_state, _model, _batchnorm, encoder,
                                _sample_top_n, _tick_size, m_seq_inp, b_seq_inp,
                                _n_gen_msgs, sim_states_init, rng, init_hidden_batched,
                                _conditional, init_time_batched, _debug_book,
                                real_book, valid_mask_array)

                        _generate_sharded = shard_map(
                            _generate_tp, mesh=valh._TP_MESH,
                            in_specs=(P(),) * 10,
                            out_specs=(P(),) * 5,
                            check_rep=False)

                        _rep = NamedSharding(valh._TP_MESH, P())
                        generate_batched_tp = jax.jit(
                            _generate_sharded,
                            in_shardings=(_rep,) * 10,
                            out_shardings=(_rep,) * 5)

                        generate_traced = generate_batched_tp.trace(
                            train_state, encoder, m_seq_inp[:], b_seq_inp,
                            sim_states_init, jax.random.split(rng_, batch_size),
                            init_hidden_batched, init_time_batched, real_book,
                            valid_mask_array)
                    finally:
                        for (_mod, _name), _orig in _saved_jits.items():
                            setattr(_mod, _name, _orig)
                else:
                    generate_traced = generate_batched.trace(
                        sim_init,
                        train_state,
                        model,
                        batchnorm,
                        encoder,
                        sample_top_n,
                        tick_size,
                        m_seq_inp[:],
                        b_seq_inp,
                        n_gen_msgs,
                        sim_states_init,
                        jax.random.split(rng_, batch_size),
                        init_hidden_batched,
                        conditional,
                        init_time_batched,
                        debug_book,
                        real_book,
                        valid_mask_array,
                    )
            generate_lowered = generate_traced.lower()
            generate_compiled = generate_lowered.compile()

        start_time = time.time()
        if valh._TP_MESH is not None:
            # Replicate runtime args to match in_shardings
            _to_rep = lambda x: jax.device_put(x, _rep) if hasattr(x, 'shape') else x
            _rt = lambda t: jax.tree_util.tree_map(_to_rep, t)
            msgs_decoded, l2_book_states, num_errors, mgs_tokens, _b_finals = generate_compiled(
                _rt(train_state), _rt(encoder), _to_rep(m_seq_inp[:]),
                _to_rep(b_seq_inp), _rt(sim_states_init),
                _to_rep(jax.random.split(rng_, batch_size)),
                _rt(init_hidden_batched), _to_rep(init_time_batched),
                _rt(real_book), _to_rep(valid_mask_array))
        else:
            msgs_decoded, l2_book_states, num_errors, mgs_tokens, _b_finals = generate_compiled(
                train_state,
                encoder,
                m_seq_inp[:],
                b_seq_inp,
                sim_states_init,
                jax.random.split(rng_, batch_size),
                init_hidden_batched,
                init_time_batched,
                real_book,
                valid_mask_array,
            )
        end_time = time.time()
        print(f"Generation time for batch of size {batch_size}: {(end_time - start_time):.2f} seconds")
        rng, rng_ = jax.random.split(rng)
        # TODO: save as metadata
        print('num_errors', num_errors)

        # ── Ground-truth comparison ──────────────────────────────────
        gt_metrics_batch = None
        if gt_compare:
            gt_start = time.time()
            _replay_single = partial(
                _replay_real_msgs_single, sim_init, n_levels=l2_state_n)
            replay_real_vmap = jax.jit(jax.vmap(
                _replay_single, in_axes=(0, 0)))
            real_l2_states = replay_real_vmap(
                sim_states_init, m_seq_raw_eval[:, :n_gen_msgs])

            _compute_div = partial(compute_gt_divergence, tick_size=tick_size)
            gt_metrics_batch = jax.vmap(_compute_div)(
                l2_book_states, real_l2_states)

            tok_accuracy = (
                jnp.array(mgs_tokens).reshape(batch_size, -1) ==
                jnp.array(m_seq_eval)[:, :mgs_tokens.reshape(batch_size, -1).shape[1]]
            ).mean(axis=0)

            gen_changed = ~(l2_book_states[..., 1:, :] == l2_book_states[..., :-1, :]).all(axis=-1)
            validity_rate = gen_changed.mean(axis=-1)

            gt_elapsed = time.time() - gt_start
            mid_div_mean = float(onp.array(gt_metrics_batch['mid_divergence']).mean())
            spread_div_mean = float(onp.array(gt_metrics_batch['spread_divergence']).mean())
            validity_mean = float(onp.array(validity_rate).mean())
            print(f'[GT Compare] mid_div={mid_div_mean:.2f} ticks, '
                  f'spread_div={spread_div_mean:.2f} ticks, '
                  f'validity={validity_mean:.3f}, '
                  f'time={gt_elapsed:.1f}s')

        # only keep actually newly generated messages
        # m_seq_raw_gen = m_seq_raw_gen[-n_gen_msgs:]

        # Detect if msg_seq_raw is encoded (22 cols) vs raw (14 cols)
        msg_is_encoded = (m_seq_raw_eval.shape[-1] == Message_Tokenizer.MSG_LEN)

        # save data for all elements in the batch
        for i, cond_msg, cond_book, real_msg, real_book, gen_msg, gen_book,msg_tok,msg_tok_eval \
            in zip(
                batch_i,
                m_seq_raw_inp, b_seq_pv_inp,
                m_seq_raw_eval, b_seq_pv_eval,
                msgs_decoded, l2_book_states,
                mgs_tokens,m_seq_eval,
            ):

            # get date from filename
            date = ds.get_date(i)
            if overfit_debug:
                jnp.set_printoptions(threshold=sys.maxsize)
                with open(save_folder+f'/tokens/{stock_symbol}_{date}_real_{i}.txt', 'w') as f:
                    print( onp.reshape(msg_tok_eval,(-1,Message_Tokenizer.MSG_LEN)), file=f)

                with open(save_folder+f'/tokens/{stock_symbol}_{date}_gen_{i}.txt', 'w') as f:
                    print( msg_tok, file=f)
                jnp.set_printoptions()

            # Decode encoded messages to 14-col LOBSTER format if needed
            if msg_is_encoded:
                _decode_batch = jax.vmap(encoding.decode_msg, in_axes=(0, None))
                real_msg_dec = onp.array(_decode_batch(jnp.array(real_msg), encoder))
                cond_msg_dec = onp.array(_decode_batch(jnp.array(cond_msg), encoder))
            else:
                real_msg_dec = real_msg
                cond_msg_dec = cond_msg

            # input / cond data
            msg_to_lobster_format(cond_msg_dec).to_csv(
                save_folder + f'/data_cond/{stock_symbol}_{date}_message_real_id_{i}.csv',
                index=False, header=False
            )
            book_to_lobster_format(cond_book).to_csv(
                save_folder + f'/data_cond/{stock_symbol}_{date}_orderbook_real_id_{i}.csv',
                index=False, header=False
            )

            # real data
            msg_to_lobster_format(real_msg_dec).to_csv(
                save_folder + f'/data_real/{stock_symbol}_{date}_message_real_id_{i}.csv',
                index=False, header=False
            )
            book_to_lobster_format(real_book).to_csv(
                save_folder + f'/data_real/{stock_symbol}_{date}_orderbook_real_id_{i}.csv',
                index=False, header=False
            )
            
            # gen data
            msg_to_lobster_format(gen_msg).to_csv(
                save_folder + f'/data_gen/{stock_symbol}_{date}_message_real_id_{i}_gen_id_0.csv',
                index=False, header=False
            )
            book_to_lobster_format(gen_book).to_csv(
                save_folder + f'/data_gen/{stock_symbol}_{date}_orderbook_real_id_{i}_gen_id_0.csv',
                index=False, header=False
            )

            # Save ground-truth comparison metrics
            if gt_compare and gt_metrics_batch is not None:
                gt_dir = save_folder + '/gt_compare/'
                Path(gt_dir).mkdir(exist_ok=True, parents=True)
                batch_idx = batch_i.index(i) if isinstance(batch_i, list) else 0
                gt_data = {
                    k: onp.array(v[batch_idx]) for k, v in gt_metrics_batch.items()
                }
                gt_data['validity_rate'] = float(onp.array(validity_rate[batch_idx]))
                gt_data['tok_accuracy'] = onp.array(tok_accuracy)
                onp.savez_compressed(
                    gt_dir + f'{stock_symbol}_{date}_gt_id_{i}.npz',
                    **gt_data
                )

    # ── GT Compare: per-rank aggregate (only this rank's files) ────
    if gt_compare:
        gt_dir = save_folder + '/gt_compare/'
        gt_files = sorted(glob(gt_dir + '*_gt_id_*.npz'))
        if gt_files:
            all_mid, all_spread, all_validity, all_tok = [], [], [], []
            for gf in gt_files:
                try:
                    gd = onp.load(gf)
                    all_mid.append(gd['mid_divergence'])
                    all_spread.append(gd['spread_divergence'])
                    all_validity.append(float(onp.asarray(gd['validity_rate']).flat[0]))
                    all_tok.append(gd['tok_accuracy'])
                except Exception as e:
                    print(f'  Warning: skipping corrupted {os.path.basename(gf)}: {e}')
            mid_arr = onp.stack(all_mid)          # (N, 500)
            spread_arr = onp.stack(all_spread)
            validity_arr = onp.array(all_validity)
            tok_arr = onp.stack(all_tok)           # (N, n_tokens)

            n_seqs = len(gt_files)
            print(f'\n{"="*60}')
            print(f'GT COMPARE SUMMARY ({n_seqs} sequences)')
            print(f'{"="*60}')
            print(f'  Mid-price div:  {mid_arr.mean():.2f} +/- {mid_arr.mean(1).std():.2f} ticks')
            print(f'  Spread div:     {spread_arr.mean():.2f} +/- {spread_arr.mean(1).std():.2f} ticks')
            print(f'  Validity rate:  {validity_arr.mean():.3f} +/- {validity_arr.std():.3f}')
            print(f'  Token accuracy: {tok_arr.mean():.3f}')
            print(f'\n  Divergence trajectory (mean across {n_seqs} seqs):')
            print(f'  {"Step":<8} {"Mid":>8} {"Spread":>8}')
            for step in [0, 50, 100, 250, 499]:
                s = min(step, mid_arr.shape[1] - 1)
                print(f'  {s:<8} {mid_arr[:,s].mean():>8.2f} {spread_arr[:,s].mean():>8.2f}')

            # Save aggregate
            onp.savez_compressed(
                gt_dir + f'{stock_symbol}_gt_aggregate.npz',
                mid_divergence=mid_arr,
                spread_divergence=spread_arr,
                validity_rate=validity_arr,
                tok_accuracy=tok_arr,
                n_seqs=onp.array(n_seqs),
            )
            print(f'\n  Saved aggregate to {stock_symbol}_gt_aggregate.npz')

def msg_to_lobster_format(
        m_seq: jax.Array,
) -> pd.DataFrame:
    """ 
    message format: [time, event_type, order_id, size, price, direction]
    """
    m_seq_ = onp.array(m_seq)[:, [TIMEs_i, TIMEns_i, EVENT_TYPE_i, ORDER_ID_i, SIZE_i, PRICE_ABS_i, DIRECTION_i]]
    m_seq_ = pd.DataFrame(m_seq_, columns=['time_s', 'time_ns', 'event_type', 'order_id', 'size', 'price', 'direction'])

    # combine time field to single field    
    m_seq_.insert(
        column = 'time',
        loc = 0,
        value = m_seq_['time_s'].astype(str) \
              + '.' \
              + m_seq_['time_ns'].astype(str).str.pad(width=9, side='left', fillchar='0')
    )
    m_seq_.drop(columns=['time_s', 'time_ns'], inplace=True)

    # convert direction {0,1} to {-1,1}
    m_seq_['direction'] = m_seq_['direction'].replace({0: -1})
    return m_seq_

def book_to_lobster_format(
        b_seq: jax.Array,
    ) -> pd.DataFrame:
    """
    """
    b_seq_ = pd.DataFrame(b_seq)

    return b_seq_


transform_L2_state_batch = jax.jit(
    jax.vmap(
        preproc.transform_L2_state_gpu,
        in_axes=(0, None, None)
    ),
    static_argnums=(1, 2)
)