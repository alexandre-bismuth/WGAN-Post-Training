from typing import Iterable, Optional, Tuple, Union
from lob import encoding
from lob.encoding import Message_Tokenizer, Vocab
import pandas as pd
import jax
from jax import nn
# from jax.random import PRNGKeyArray
from jax.experimental import checkify
import chex
import flax
from flax.training.train_state import TrainState
import jax.numpy as np
from functools import partial
import numpy as onp

import logging
logger = logging.getLogger(__name__)
debug = lambda *args: logger.debug(' '.join((str(arg) for arg in args)))
info = lambda *args: logger.info(' '.join((str(arg) for arg in args)))

# TP inference globals — set by run_inference.py when tp_size > 1
_TP_MESH = None   # jax.sharding.Mesh with ('tp',) axis
_TP_NH = None     # total n_heads (for auto-detecting head dim in hidden specs)


def _make_hidden_specs(hidden_state, nh):
    """Create PartitionSpec for each hidden state leaf, sharding along head dim."""
    from jax.sharding import PartitionSpec as P
    def leaf_spec(arr):
        axes = [None] * len(arr.shape)
        for i, d in enumerate(arr.shape):
            if d == nh:
                axes[i] = 'tp'
                break  # first match only
        return P(*axes)
    return jax.tree_util.tree_map(leaf_spec, hidden_state)

from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import repeat_book

v = Vocab()


def syntax_validation_matrix(v = None, block_start_tok = False):
    """ Create a matrix of shape (MSG_LEN, VOCAB_SIZE) where a
        True value indicates that the token is valid for the location
        in the message.
        TODO: rewrite and jit
    """
    if v is None:
        v = Vocab()
    encoder = v.ENCODING

    idx = []
    for i in range(Message_Tokenizer.MSG_LEN):
        field = Message_Tokenizer.get_field_from_idx(i)
        decoder_key = Message_Tokenizer.FIELD_ENC_TYPES[field[0]]
        #for tok, val in v.DECODING[decoder_key].items():
        for tok in encoder[decoder_key][1]:
            idx.append([i, tok])
    idx = tuple(np.array(idx).T)
    mask = np.zeros((Message_Tokenizer.MSG_LEN, len(v)), dtype=bool)
    mask = mask.at[idx].set(True)

    # adjustment for positions only allowing subset of field
    # e.g. +/- at start of price
    i, _ = get_idx_from_field("price")
    mask = update_allowed_tok_slice(mask, i, np.array([1, -1]), encoder['sign'])
    i, _ = get_idx_from_field("price_ref")
    mask = update_allowed_tok_slice(mask, i, np.array([1, -1]), encoder['sign'])

    # adjustments for special tokens (no MSK or HID allowed during generation)
    mask = mask.at[:, v.MASK_TOK].set(False)
    mask = mask.at[:, v.HIDDEN_TOK].set(False)
    # Block START_TOK for transformers (which predict it 30-40% mid-sequence);
    # S5 models never emit it, so leave it unmasked to match upstream behavior.
    if block_start_tok:
        mask = mask.at[:, v.START_TOK].set(False)
    # allow NAN token in ref fields only
    mask = mask.at[:, v.NA_TOK].set(False)
    mask = mask.at[Message_Tokenizer.NEW_MSG_LEN: , v.NA_TOK].set(True)

    return mask

@jax.jit
def get_valid_mask(
        valid_mask_array: jax.Array,
        i: int
    ) -> jax.Array:
    return valid_mask_array[i]


def update_allowed_tok_slice(mask, i, allowed_toks, field_encoder):
    allowed_toks = encoding.encode(allowed_toks, *field_encoder)
    adj_col = np.zeros((mask.shape[1],), dtype=bool).at[allowed_toks].set(True)
    mask = mask.at[i, :].set(adj_col)
    return mask

def is_tok_valid(tok, field, vocab):
    tok = tok.tolist()
    if isinstance(field, str):
        return tok in vocab.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[field]]
    else:
        return [t in vocab.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[f]] 
                for t, f in zip(tok, field)]

def get_masked_idx(seq):
    """ Get the indices of the masked tokens in a given input (batched or not)
    """
    if seq.ndim == 1:
        seq = seq.reshape(-1, Message_Tokenizer.MSG_LEN)
    elif seq.ndim == 2:
        seq = seq.reshape(seq.shape[0], -1, Message_Tokenizer.MSG_LEN)
    return np.argwhere(seq == v.MASK_TOK)

def get_field_from_idx(idx):
    """ Get the field of a given index (or indices) in a message
    """
    return Message_Tokenizer.get_field_from_idx(idx)

def get_idx_from_field(field):
    field_i = Message_Tokenizer.FIELD_I[field]
    return LOBSTER_Dataset._get_tok_slice_i(field_i)

def get_masked_fields(inp_maybe_batched):
    """ Get the fields of the masked tokens in a given input (batched or not)
    """
    mask_pos = get_masked_idx(inp_maybe_batched)
    return get_field_from_idx(mask_pos[..., -1])

def get_valid_toks_for_field(fields):
    """ Get the valid labels for given fields
    """
    return tuple(tuple(
        v.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[field]].keys())
          for field in fields)

def get_valid_toks_for_input(inp_maybe_batched):
    """ Get the valid labels for a given input (batched or not)
    """
    fields = get_masked_fields(inp_maybe_batched)
    return get_valid_toks_for_field(fields)

def get_first_time(m_seq_cond,encoder):
    last_msg=m_seq_cond[-Message_Tokenizer.MSG_LEN:]
    with jax.ensure_compile_time_eval():
        time_s_start_i, time_s_end_i = get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = get_idx_from_field('time_ns')
    time_init_s, time_init_ns = encoding.decode_time(
        last_msg[time_s_start_i:time_ns_end_i],
        encoder
    )
    return (time_init_s, time_init_ns)

def valid_prediction_mass(pred, fields, top_n=None):
    """ for a predicted distribution over tokens get the total mass of the
        syntactically valid labels
        top_n: 
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    assert (len(fields) == pred.shape[0])
    valid_toks = get_valid_toks_for_field(fields)
    dim_0_i = [i for i, tok_list in enumerate(valid_toks) for tok in tok_list]
    dim_1_i = [tok for tok_list in valid_toks for tok in tok_list]
    mask_valid = np.zeros_like(pred)
    mask_valid = mask_valid.at[dim_0_i, dim_1_i].set(1)

    if top_n is not None:
        mask_top_n = mask_n_highest(pred, top_n)
        mask_valid = mask_valid * mask_top_n
        top_n_mass = np.sum(np.exp(pred) * mask_top_n, axis=1)
    else:
        top_n_mass = 1.

    return (np.sum(np.exp(pred) * mask_valid, axis=1)) / top_n_mass

@jax.jit
def mask_n_highest(
        a: jax.Array,
        n: jax.Array
    ) -> jax.Array:
    """ Return a mask for the n highest values in the last axis
        for a given array
    """
    n_th_largest = np.sort(a, axis=-1)[..., -n]
    # add leading dimensions to match pred
    n_th_largest = n_th_largest.reshape((-1,) + (1,)*(a.ndim-1))
    mask_top_n = np.zeros_like(a, dtype=bool)
    mask_top_n = np.where(a >= n_th_largest, True, False)
    return mask_top_n

def pred_rank(pred, labels):
    """ Get the rank of the correct label in the predicted distribution.
        Lower is better (0 is correct prediction).
    """
    correct_mask = np.squeeze(nn.one_hot(labels.astype(int), pred.shape[-1]).astype(bool))
    # ::-1 sorts in descending order (0 is highest rank)
    a = pred.argsort(axis=-1)
    ranks = np.squeeze(a[..., ::-1].argsort(axis=-1))
    return ranks[correct_mask]

@partial(jax.jit, static_argnums=(1,))
def fill_predicted_tok(
        pred_logits: jax.Array,
        top_n: int = 1,
        rng: Optional[jax.dtypes.prng_key] = None,
    ) -> jax.Array:
    """ Set the predicted token in the given sequence
        when top_n=1, the argmax is used, otherwise a random sample
        from the top_n highest scores is used (propotional to the score)
        rng cannot be None when top_n > 1
    """
    if top_n == 1:
        vals = pred_logits.argmax(axis=-1)
    else:
        vals = sample_pred(pred_logits, top_n, rng)
    return vals

@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap, in_axes=(0, None, 0))
def sample_pred(
        pred: jax.Array,
        top_n: int,
        rng: jax.dtypes.prng_key
    ) -> jax.Array:
    """ Sample from the top_n predicted labels
    """
    idx = np.arange(pred.shape[-1]).reshape(pred.shape)
    if top_n > 1 and top_n < pred.shape[-1]:
        mask_top_n = mask_n_highest(pred, top_n)
        p = np.exp(pred) * mask_top_n
    else:
        p = np.exp(pred)
    p = p / p.sum(axis=-1, keepdims=True)
    return jax.random.choice(rng, idx, p=p)

def append_hid_msg(seq):
    """ Append a new empty (HID token) message to a sequence
        removing first message to keep seq_len constant
    """
    l = Message_Tokenizer.MSG_LEN
    return np.concatenate([seq[l:], np.full((Message_Tokenizer.MSG_LEN,), Vocab.HIDDEN_TOK)])

@jax.jit
def get_to_mask_tok(
        seq: jax.Array,
        i: int,
    ) -> jax.Array:
    """ Get a message sequence that is ls-l long 
        and ends with the masked token. 
        ls: length of the input sequence. 
        l: length of a single message
        i: either the position of the masking token from the end (-ve)
            or the position in the message 
        
    """
    start = jax.lax.cond(
        i >= 0,
        lambda x: x,
        lambda x: x + Message_Tokenizer.MSG_LEN,
        i,
    )
    new_seq=jax.lax.dynamic_slice(seq,(start+1,),(seq.shape[0]- Message_Tokenizer.MSG_LEN,))
    new_seq=new_seq.at[-1].set(Vocab.MASK_TOK)
    return new_seq

#@chex.chexify
@jax.jit
#@chex.assert_max_traces(n=1)
def mask_last_msg_in_seq(
        seq: jax.Array,
        i: int,
    ) -> Tuple[jax.Array, jax.Array]:
    
    l = Message_Tokenizer.MSG_LEN
    # slows down execution
    #checkify.check((i >= -l) & (i < l), "i={} must be in [-MSG_LEN, MSG_LEN)", i)
    i = jax.lax.cond(
        i >= 0,
        lambda x, ls: x + ls - l,
        lambda x, ls: x,
        i, seq.shape[0],
    )
    y = seq[i]
    return seq.at[i].set(Vocab.MASK_TOK), y


@jax.jit
#@chex.assert_max_traces(n=1)
def last_token_predict(
        seq: jax.Array,
        i: int,
    ) -> Tuple[jax.Array, jax.Array]:
    
    l = Message_Tokenizer.MSG_LEN
    # slows down execution
    #checkify.check((i >= -l) & (i < l), "i={} must be in [-MSG_LEN, MSG_LEN)", i)
    i = jax.lax.cond(
        i >= 0,
        lambda x, ls: x + ls - l,
        lambda x, ls: x,
        i, seq.shape[0],
    )
    new_seq=jax.lax.dynamic_slice(seq,)
    y = seq[i]
    return seq.at[i].set(Vocab.MASK_TOK)

@partial(jax.jit, static_argnums=(3, 4))
def predict(
        batch_inputs: jax.Array,
        batch_integration_timesteps: jax.Array,
        state: TrainState,
        model: flax.linen.Module,
        batchnorm: bool,
    ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                            *batch_inputs, *batch_integration_timesteps,
                            method="__call_rnn__",
                            )
    else:
        logits = model.apply({"params": state.params},
                             *batch_inputs, *batch_integration_timesteps,
                             method="__call_rnn__",
                             )

    return logits



def _apply_model_impl(
        hidden_state: Tuple,
        m_seq: jax.Array ,
        b_seq: jax.Array ,
        state: TrainState,
        model: flax.linen.Module,
        batchnorm: bool,
        shift_start: bool,
        es=None,
    ):
    batch_inputs = (
        np.expand_dims(m_seq, axis=0),
        np.expand_dims(b_seq, axis=0))
    batch_integration_timesteps = (
        np.ones((1, len(m_seq))),
        np.ones((1, len(m_seq)))
    )
    batch_inputs=repeat_book(*batch_inputs,shift_start)

    dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*(len(hidden_state)-1)
    # EGGROLL ES (S5b): pass the LoRA factor pytree ONLY when present, so the stock
    # call (es=None) stays byte-identical and stock Batch* wrappers (8-arg in_axes)
    # never see a 9th argument. With es, `model` must be BatchPaddedLobPredModelES.
    es_extra = () if es is None else (es,)
    if batchnorm:
        hidden_state,logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                            hidden_state,*batch_inputs, *dones, *batch_integration_timesteps, *es_extra,
                            method="__call_rnn__"
                            )
    else:
        hidden_state,logits = model.apply({"params": state.params},
                             hidden_state,*batch_inputs, *dones, *batch_integration_timesteps, *es_extra,
                             method="__call_rnn__"
                             )

    return hidden_state,logits

# Jitted version for standalone use; TP path uses _apply_model_impl directly
apply_model = jax.jit(_apply_model_impl, static_argnums=(4, 5, 6))


def _apply_model_1tok_impl(
        hidden_state: Tuple,
        m_seq: jax.Array,
        b_seq: jax.Array,
        state: TrainState,
        model: flax.linen.Module,
        batchnorm: bool,
        shift_start: bool,
    ):
    """apply_model variant for 1tok: handles (L, 24) message input and 2-done tuple."""
    batch_inputs = (
        np.expand_dims(m_seq, axis=0),
        np.expand_dims(b_seq, axis=0))
    batch_integration_timesteps = (
        np.ones((1, m_seq.shape[0])),
        np.ones((1, m_seq.shape[0]))
    )
    batch_inputs = repeat_book(*batch_inputs, shift_start)

    # Explicit (1, L) dones — NOT zeros_like(batch_inputs[0]) which would be (1, L, 24)
    L = m_seq.shape[0]
    dones = (np.zeros((1, L), dtype=bool),) * 2  # d_b, d_f (no d_m for 1tok)

    if batchnorm:
        hidden_state, logits = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            hidden_state, *batch_inputs, *dones, *batch_integration_timesteps,
            method="__call_rnn__")
    else:
        hidden_state, logits = model.apply(
            {"params": state.params},
            hidden_state, *batch_inputs, *dones, *batch_integration_timesteps,
            method="__call_rnn__")

    return hidden_state, logits

# Jitted version for standalone use; TP path uses _apply_model_1tok_impl directly
apply_model_1tok = jax.jit(_apply_model_1tok_impl, static_argnums=(4, 5, 6))


def get_first_time_1tok(m_seq_cond, encoder):
    """Extract time from last conditioning message (1tok format: local indices)."""
    from lob.encoding_1tok import local_to_global_jax
    last_msg_local = m_seq_cond[-1]  # (24,) local indices
    last_msg_global = local_to_global_jax(last_msg_local)
    # time fields at positions 10:15 (time_s_0, time_s_1, time_ns_0, time_ns_1, time_ns_2)
    time_toks = last_msg_global[10:15]
    time_s, time_ns = encoding.decode_time(time_toks, encoder)
    return (time_s, time_ns)


@jax.jit
def filter_valid_pred(
        pred: jax.Array,
        valid_mask: jax.Array,
    ):
    """ Filter the predicted distribution to only include valid tokens
    """
    if valid_mask.ndim == 1:
        valid_mask = np.expand_dims(valid_mask, axis=0)
    pred = np.where(
        valid_mask == 0,
        -1e9,
        pred,
    )
    # renormalize (numerically stable log-softmax)
    pred = jax.nn.log_softmax(pred, axis=-1)
    return pred


# TODO: factor out new message creation into separate fn
#       use simulator to update book_seq (separate fn)
#
def pred_next_tok(
        seq,
        state,
        model,
        batchnorm,
        sample_top_n,
        mask_i,
        rng,
        vocab_len,
        book_seq=None,
        new_msg=False,
        valid_mask=None,  # if given, sample only from syntactically valid tokens
    ):
    """ Predict the next token with index i of the last message in the sequence
        if new_msg=True, a new empty message is appended to the sequence
        Returns the updated sequence
    """
    # create masked message for prediction
    if new_msg:
        seq = append_hid_msg(seq)
        # TODO: use simulator to update book_seq
    seq, _ = mask_last_msg_in_seq(seq, mask_i)
    # inference
    integration_timesteps = (np.ones((1, len(seq))), )
    input = (nn.one_hot(
        np.expand_dims(seq, axis=0), vocab_len).astype(float), )
    # append book data to input tuples
    if book_seq is not None:
        input += (book_seq, )
        integration_timesteps += (np.ones((1, len(book_seq))), )
    logits = predict(
        input,
        integration_timesteps, state, model, batchnorm)
    if valid_mask is not None:
        logits = filter_valid_pred(logits, valid_mask)
    # update sequence
    # note: rng arg expects one element per batch element
    seq = fill_predicted_tok(seq, logits, sample_top_n, np.array([rng]))
    return seq


def pred_msg(
        seq: np.ndarray,
        n_messages: int,
        state: TrainState,
        model: flax.linen.Module,
        batchnorm: bool,
        rng: jax.dtypes.prng_key,
        valid_mask_array: Optional[jax.Array] = None,
        sample_top_n: int = 5,
    ) -> np.ndarray:

    valid_mask = None
    l = Message_Tokenizer.MSG_LEN
    for m_i in range(n_messages):
        new_msg = True
        for i in range(l):
            if valid_mask_array is not None:
                valid_mask = valid_mask_array[i]
            seq = pred_next_tok(
                seq,
                state,
                model,
                batchnorm,
                sample_top_n=sample_top_n,
                mask_i=i,
                new_msg=new_msg,
                vocab_len=len(v),
                rng=rng,
                valid_mask=valid_mask,
            )
            new_msg = False
    return seq

def validate_msg(
        msg: np.ndarray,
        tok: Message_Tokenizer,
        vocab: Vocab,
    ) -> bool:
    """ TODO: rewrite this for new encoding
        Validate a message's internal semantics
        Assumes the message is syntactically valid (allowed toks in all places)
        Returns True if valid
    """
    assert len(msg) == Message_Tokenizer.MSG_LEN
    err_count = 0

    msg_dec = tok.decode(msg, vocab).flatten()
    fields = {fname: i for i, fname in enumerate(Message_Tokenizer.FIELDS)}
    
    time = msg_dec[fields['time']]
    event_type = msg_dec[fields['event_type']]
    event_type_new = msg_dec[fields['event_type_new']]
    price = msg_dec[fields['price']]
    direction = msg_dec[fields['direction']]

    # if NA in second half, needs to be all NA
    nas = np.isnan(msg[len(msg)//2:])
    #nas = (msg[len(msg)//2:] == Vocab.NA_TOK)
    err = np.any(nas) and not np.all(nas)
    err_count += err
    if err:
        print("NAs must be in second half of message")

    err = time > 57600000000000  # 16 * 3600 * 1e9
    err_count += err
    if err:
        print("time after opening hours")

    if event_type_new in {2, 3, 4} and not np.isnan(direction):
        direction_new = msg_dec[fields['direction_new']]
        err = direction != direction_new
        err_count += err
        if err:
            print("direction cannot be modified")
    
    return bool(err_count == 0)

def find_orig_msg(
        msg: jax.Array,
        seq: jax.Array,
        comp_cols: Optional[Iterable[int]] = None,
    ) -> Optional[int]:
    """ Finds first msg location in given seq.
        NOTE: could also find earlier msg modifications, might not be the original new message
              but we know at least that the message is in the sequence
        :param msg: message to find (only first/orig half of message)
        :param seq: sequence of messages to search in
        Returns index of first token of msg in seq and None if msg is not found
    """
    occ = find_all_msg_occurances(msg, seq, comp_cols)
    if len(occ) > 0:
        return int(occ.flatten()[0])
    

    
@partial(jax.jit, static_argnums=(2,3))
def find_n_msg_occurances(
        msg: jax.Array,
        seq: jax.Array,
        comp_cols: Tuple[str],
        n_matches: int = 1,
    ) -> jax.Array:
    ''' Returns the indices of the LAST (most recent) n matching messages in the sequence seq. '''
    def get_ref_matches(msg, seq, comp_cols, n_matches):
        comp_cols_ref = \
            [c for c in comp_cols if (c + '_ref' in Message_Tokenizer.FIELDS)]
        comp_i = [idx for c in comp_cols_ref for idx in list(range(*get_idx_from_field(c)))]
        comp_i_ref = [idx for c in comp_cols_ref for idx in list(range(*get_idx_from_field(c + '_ref')))]
        if 'direction' in comp_cols:
            comp_cols_ref += ['direction']  # direction field should be added to ref search
            comp_i += list(range(*get_idx_from_field('direction')))
            comp_i_ref += list(range(*get_idx_from_field('direction')))
        comp_i = sorted(comp_i)
        comp_i_ref = sorted(comp_i_ref)
        ref_matches = np.argwhere(
            (seq[:, comp_i_ref] == msg[comp_i,]).all(axis=1),
            size=n_matches,
            fill_value=-1,
        )
        return ref_matches.flatten()

    l = Message_Tokenizer.MSG_LEN
    seq = seq.reshape((-1, Message_Tokenizer.MSG_LEN))
    comp_i = [idx for c in comp_cols for idx in list(range(*get_idx_from_field(c)))]
    direct_matches = np.argwhere(
        (seq[:, comp_i] == msg[comp_i,]).all(axis=1),
        size=n_matches,
        fill_value=-1,
    )
    matches = jax.lax.cond(
        direct_matches[0] == -1,
        get_ref_matches,  # no direct match found
        lambda *args: direct_matches,  # direct match found
        msg, seq ,comp_cols, n_matches
    )
    return matches

def find_all_msg_occurances_DEPR(
        msg: jax.Array,
        seq: jax.Array,
        comp_cols: Iterable[str],
    ) -> jax.Array:
    """ Finds ALL msg locations in given seq.
        NOTE: could also find earlier msg modifications,
              the original new message might not be included
              but we know at least that the message is in the sequence.
        Returns index of first token of msg in seq and None if msg is not found
    """
    assert msg.ndim == 1
    l = Message_Tokenizer.MSG_LEN
    seq = seq.reshape((-1, Message_Tokenizer.MSG_LEN))#[:, : Message_Tokenizer.NEW_MSG_LEN]

    # indices of columns to compare
    comp_i = [idx for c in comp_cols for idx in list(range(*get_idx_from_field(c)))]

    debug("comp_cols", comp_cols)
    debug("comp_i", comp_i)

    debug('searching for (new)', msg[comp_i,])
    seq_filtered = seq[(seq != -1).all(axis=1)]
    debug('in seq', seq_filtered[:, comp_i])
    debug('non-masked len', len(seq[(seq != -1).all(axis=1)]))

    # filter down to specific columns
    direct_matches = np.argwhere((seq[:, comp_i] == msg[comp_i,]).all(axis=1))
    
    if len(direct_matches.flatten()) > 0:
        debug('found direct matches')
        return direct_matches

    # also search in seq ref part (matching fields)
    
    comp_cols_ref = \
        [c for c in comp_cols if (c + '_ref' in Message_Tokenizer.FIELDS)]
    debug("comp_cols_ref", comp_cols_ref)
    comp_i = [idx for c in comp_cols_ref for idx in list(range(*get_idx_from_field(c)))]
    comp_i_ref = [idx for c in comp_cols_ref for idx in list(range(*get_idx_from_field(c + '_ref')))]
    
    if 'direction' in comp_cols:
        comp_cols_ref += ['direction']  # direction field should be added to ref search
        comp_i += list(range(*get_idx_from_field('direction')))
        comp_i_ref += list(range(*get_idx_from_field('direction')))
    comp_i = sorted(comp_i)
    comp_i_ref = sorted(comp_i_ref)
    debug('ref search for ...')
    debug(msg[comp_i,])
    debug('... in:')
    debug(seq_filtered[:, comp_i_ref])
    
    ref_matches = np.argwhere((seq[:, comp_i_ref] == msg[comp_i,]).all(axis=1))

    if len(ref_matches.flatten()) > 0:
        debug('found ref matches')
    return ref_matches

def find_all_msg_occurances_raw(
        msg: onp.ndarray,
        seq: pd.DataFrame,
    ) -> pd.DataFrame:
    """ Finds ALL msg locations in given seq.
        This version searches in the raw dataframe instead of the encoded jax array.
        Benefit: can find messages that occur before encoded sequece starts
                 and we also get the order ID directly
        Downsides: data needs to be in raw format (e.g. price, time)
        NOTE: could also find earlier msg modifications,
              the original new message might not be included
              but we know at least that the message is in the sequence.
        Returns index of first token of msg in seq and None if msg is not found
    """
    return seq.loc[(seq.drop('order_id', axis=1) == msg).all(axis=1)]


def try_find_msg(
        msg: jax.Array,
        seq: jax.Array,
        seq_mask: Optional[jax.Array] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
    """ 
        Returns match index or None if no match is found; and number of fields removed in match
        If multiple matches are found, the first match is returned.
        seq_mask: filters to messages with correctly matching price level 
                  CAVE: values of 1 in mask are set to -1 in seq if no perfect match is found
    """
    def get_match_idx():
        matches = find_n_msg_occurances(msg, seq, comp_cols)
        idx = int(matches.flatten()[0])
        return idx

    if seq_mask is not None:
        seq = seq.at[seq_mask, :].set(-1)

    # remove fields from matching criteria
    matching_cols = [
        ('event_type', 'direction', 'price', 'size', 'time_s', 'time_ns'),
        ('event_type', 'direction', 'price', 'size'),
    ]

    # TODO: compare while with scan performance
    # idx = jax.lax.while_loop(
    #     lambda cc_and_i : cc_and_i[1] == -1,
    #     lambda cc_and_i: find_n_msg_occurances(seq, msg, cc[0])[0],  # returns new val passed to first arg fn. until cond is False
    #     (matching_cols[0], 1),
    # )

    _, idcs = jax.lax.scan(
        lambda _, comp_cols: find_n_msg_occurances(seq, msg, comp_cols),
        0,  # init for carry (not needed)
        matching_cols
    )
    idcs = idcs.flatten()
    return jnp.where(
        idcs != 1,
        idcs,
        size=1, fill_value=-1
    )
    # TODO: also return n_removed (i.e. arghwere, not only where...)

    # n_removed = 0
    # for comp_cols in matching_cols:
    #     # remove field from matching criteria
    #     matches = find_n_msg_occurances(msg, seq, comp_cols)
    #     if len(matches) > 0:
    #         idx = int(matches.flatten()[0])
    #         debug('found match after removing', n_removed, 'at idx', idx)
    #         return idx, n_removed
    #     n_removed += 1
    # debug('no match found')
    # return None, None
