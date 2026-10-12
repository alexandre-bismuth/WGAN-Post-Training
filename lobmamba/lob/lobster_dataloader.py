""" Datasets for core experimental results """
from pathlib import Path
import random
import re
import sys
from typing import Sequence
import numpy as np
from collections import OrderedDict
import math

#import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

#import torch
#import torchvision
import torch
from torch.utils.data import Dataset, Subset, Sampler
from glob import glob
# Global flag to set a specific platform, must be used at startup.
# jax.config.update('jax_platform_name', 'cpu')

import os
# Encoding is hardcoded to 26tok (lob/encoding.py is byte-identical to
# lob/encoding_26tok.py since efc1552). The previous os.environ["TOKEN_MODE"]
# branch and the --token_mode argparse flag both routed to the same 26tok
# code regardless of value; that lie is removed. To run a different encoding
# (e.g. 24tok), explicitly import from lob.encoding_24tok and rebuild the
# preproc pipeline — there is no runtime selector anymore.
from lob.encoding import Vocab, Message_Tokenizer, encode_msgs
from preproc import transform_L2_state,transform_L2_state_numpy
from s5.dataloaders.base import default_data_path, SequenceDataset
from s5.utils import permutations
default_data_path = Path(__file__).parent.parent.absolute()
default_data_path = default_data_path / "data"


import time
import io

try:
    import zstandard as _zstd_mod
    _zstd_dctx = _zstd_mod.ZstdDecompressor()
except ImportError:
    _zstd_mod = None
    _zstd_dctx = None


def _np_load_zst(path, mmap_mode='r', allow_pickle=False):
    """Load .npy or .npy.zst with format-aware fastpath.

    Prefers raw .npy via mmap_mode='r' (zero-copy, page-on-demand). Falls back
    to in-memory zstd decompress when only .npy.zst exists. The 488-ticker
    SP500 corpus is being progressively decompressed to .npy on Lustre to
    enable mmap; tickers not yet decompressed continue to read via .zst.

    For .zst path, uses per-call ZstdDecompressor() + stream_reader so
    concurrent workers stay safe (the module-level dctx is not thread-safe
    across stream_readers).

    Truncated .zst handling: some files in the corpus are truncated mid-stream
    (compression hit ENOSPC partway). Their .npy header parses cleanly but the
    array data is incomplete, so np.load raises ValueError EOF. We catch that,
    parse the header for the expected shape+dtype, and return a zero-filled
    placeholder. Effect: that single sample carries no signal for one step,
    contributing negligibly to gradient. Beats killing the entire training run.
    """
    # Prefer raw .npy when present (mmap-fastpath). The discover step may
    # return either form depending on whether the ticker has been decompressed.
    npy_path = path[:-4] if path.endswith('.zst') else path
    # Retry-once on .npy lookup to ride out Lustre transient stat failures.
    # Catch any ValueError/EOFError/OSError from corrupted-or-truncated .npy
    # (header-only, mmap-length-mismatch, EOF, magic-byte issue) and treat
    # the file as unreadable → return zero-fill placeholder. The caller's
    # _seqs_cumsum already counted this file as 0 rows via _zst_npy_shape's
    # matching defensive fallback, so reaching here is the rare race case.
    for _attempt in range(2):
        if os.path.exists(npy_path):
            try:
                return np.load(npy_path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
            except (ValueError, EOFError, OSError):
                # Try one more time without mmap (some corruption only trips
                # the mmap path, not the in-memory read).
                try:
                    return np.load(npy_path, mmap_mode=None, allow_pickle=allow_pickle)
                except (ValueError, EOFError, OSError):
                    return np.zeros((0,), dtype=np.int64)
            break
        if _attempt == 0:
            import time as _time
            _time.sleep(0.1)
    # Fall through to .zst decompress path
    if _zstd_mod is None:
        return np.zeros((0,), dtype=np.int64)
    zst_path = path if path.endswith('.zst') else path + '.zst'
    if not os.path.exists(zst_path):
        # Both forms truly absent — return zero-fill placeholder rather than
        # crashing. Caller's _seqs_cumsum will have already counted this file
        # as 0 rows (via _zst_npy_shape's same defensive fallback) so this
        # path should be unreachable in normal operation.
        return np.zeros((0,), dtype=np.int64)
    dctx = _zstd_mod.ZstdDecompressor()
    chunks = []
    with open(zst_path, 'rb') as f:
        reader = dctx.stream_reader(f)
        while True:
            chunk = reader.read(1 << 20)  # 1 MiB chunks
            if not chunk:
                break
            chunks.append(chunk)
    decompressed = b''.join(chunks)
    try:
        return np.load(io.BytesIO(decompressed), allow_pickle=allow_pickle)
    except ValueError as e:
        if 'EOF' not in str(e):
            raise
        import numpy.lib.format as _nfmt
        buf = io.BytesIO(decompressed)
        try:
            version = _nfmt.read_magic(buf)
            if version == (1, 0):
                shape, _fortran, dtype = _nfmt.read_array_header_1_0(buf)
            elif version == (2, 0):
                shape, _fortran, dtype = _nfmt.read_array_header_2_0(buf)
            else:
                raise
        except Exception:
            raise e
        return np.zeros(shape, dtype=dtype)


def _zst_npy_shape(path):
    """Return the .npy shape tuple. For raw .npy uses np.load(mmap_mode='r').
    For .npy.zst streams the first 4 KB to read the header.

    Defensive against transient Lustre stat failures: if BOTH .npy and .npy.zst
    appear missing (or fail to read), returns (0,) so the dataloader treats it
    as a zero-row file and skips it rather than crashing the entire job. A
    real systematic missing file is harmless (drops some training samples);
    a transient Lustre hiccup is recovered next epoch.
    """
    npy_path = path[:-4] if path.endswith('.zst') else path
    # Retry the .npy path once with a brief delay to absorb Lustre stutters.
    # Catch any ValueError/EOFError/OSError from corrupted-or-truncated .npy:
    # 0-byte files raise EOFError; truncated body files raise
    # ValueError("mmap length is greater than file size"); etc. All paths
    # that mean "this .npy is unusable" → fall through to .zst (or empty).
    for _attempt in range(2):
        if os.path.exists(npy_path):
            try:
                arr = np.load(npy_path, mmap_mode='r')
                return arr.shape
            except (ValueError, EOFError, OSError):
                pass  # fall through to .zst path
            break
        if _attempt == 0:
            import time as _time
            _time.sleep(0.1)
    if _zstd_mod is None:
        # No zst fallback available; treat as empty rather than crashing.
        return (0,)
    zst_path = path if path.endswith('.zst') else path + '.zst'
    if not os.path.exists(zst_path):
        # Both forms truly absent (after retry). Skip this file, don't crash.
        print(f"[_zst_npy_shape] WARN: neither {npy_path} nor {zst_path} found, treating as empty",
              flush=True)
        return (0,)
    try:
        dctx = _zstd_mod.ZstdDecompressor()
        with open(zst_path, 'rb') as f:
            reader = dctx.stream_reader(f)
            head = b''
            while len(head) < 4096:
                chunk = reader.read(4096 - len(head))
                if not chunk:
                    break
                head += chunk
    except (OSError, IOError):
        return (0,)
    # Empty-payload files (compressed-empty .npy.zst, ~13 bytes on disk).
    if len(head) < 10:
        return (0,)
    import numpy.lib.format as _nfmt
    buf = io.BytesIO(head)
    try:
        version = _nfmt.read_magic(buf)
        if version == (1, 0):
            shape, _fortran_order, _dtype = _nfmt.read_array_header_1_0(buf)
        elif version == (2, 0):
            shape, _fortran_order, _dtype = _nfmt.read_array_header_2_0(buf)
        else:
            return (0,)
        return shape
    except (ValueError, EOFError):
        return (0,)


def discover_ticker_files(data_root, tickers, date_range=None):
    """Discover .npy files across multiple ticker subdirectories with date filtering.

    Args:
        data_root: Parent directory containing per-ticker subdirs
        tickers: List of ticker symbols
        date_range: Optional (start_date, end_date) inclusive, 'YYYY-MM-DD' format

    Returns:
        dict[ticker -> list[msg_path]], dict[ticker -> list[book_path]]
    """
    date_re = re.compile(r'(\d{4}-\d{2}-\d{2})')
    msg_by_ticker = {}
    book_by_ticker = {}

    def _prefer_npy_over_zst(paths):
        """Given a sorted list of candidate paths (mix of .npy and .npy.zst),
        dedupe by stripping .zst and prefer the .npy form when both exist.
        After SP500 corpus decompression, raw .npy enables mmap fastpath."""
        by_npy = {}  # canonical .npy path -> chosen path (.npy if exists, else .npy.zst)
        for p in paths:
            canonical = p[:-4] if p.endswith('.zst') else p
            existing = by_npy.get(canonical)
            if existing is None:
                by_npy[canonical] = p
            else:
                # Prefer the .npy form (no .zst suffix)
                if existing.endswith('.zst') and not p.endswith('.zst'):
                    by_npy[canonical] = p
        return sorted(by_npy.values())

    for ticker in tickers:
        ticker_dir = Path(data_root) / ticker
        assert ticker_dir.is_dir(), f"Ticker directory not found: {ticker_dir}"

        msg_npy = glob(str(ticker_dir / '**' / '*message*.npy'), recursive=True)
        msg_zst = glob(str(ticker_dir / '**' / '*message*.npy.zst'), recursive=True)
        book_npy = glob(str(ticker_dir / '**' / '*book*.npy'), recursive=True)
        book_zst = glob(str(ticker_dir / '**' / '*book*.npy.zst'), recursive=True)
        # glob '*.npy' also matches '*.npy.zst' on some libc implementations,
        # so explicitly drop .zst from the npy bucket and merge both buckets
        # via _prefer_npy_over_zst.
        msg_npy = [p for p in msg_npy if not p.endswith('.zst')]
        book_npy = [p for p in book_npy if not p.endswith('.zst')]
        msg_files = _prefer_npy_over_zst(msg_npy + msg_zst)
        book_files = _prefer_npy_over_zst(book_npy + book_zst)
        assert len(msg_files) == len(book_files), (
            f"{ticker}: msg files ({len(msg_files)}) != book files ({len(book_files)})")
        assert len(msg_files) > 0, f"No message files found in {ticker_dir}"

        if date_range is not None:
            start_date, end_date = date_range
            filtered_msg = []
            filtered_book = []
            for mf, bf in zip(msg_files, book_files):
                m = date_re.search(Path(mf).name)
                if m:
                    file_date = m.group(1)
                    if start_date <= file_date <= end_date:
                        filtered_msg.append(mf)
                        filtered_book.append(bf)
            msg_files = filtered_msg
            book_files = filtered_book

        # Filter asymmetric pairs: if either side is empty while the partner
        # is non-empty, drop the pair. Empty thresholds differ by format:
        # compressed-empty .npy.zst is ~13 bytes; uncompressed-empty .npy is
        # ~128 bytes (just the npy header for shape=(0,)). Use a per-format
        # threshold so the check works during the corpus's transitional
        # mixed-format state.
        def _is_empty(p):
            try:
                sz = os.path.getsize(p)
            except OSError:
                return True
            return sz <= (20 if p.endswith('.zst') else 200)

        clean_msg, clean_book = [], []
        for mf, bf in zip(msg_files, book_files):
            mf_empty = _is_empty(mf)
            bf_empty = _is_empty(bf)
            if mf_empty != bf_empty:
                # asymmetric: one empty, one non-empty
                continue
            clean_msg.append(mf)
            clean_book.append(bf)
        msg_files = clean_msg
        book_files = clean_book

        msg_by_ticker[ticker] = msg_files
        book_by_ticker[ticker] = book_files

    return msg_by_ticker, book_by_ticker


class LOBSTER_Dataset(Dataset):

    @staticmethod
    def get_masking_fn(*, random_msg_idxs=None, random_fields=None, randomize_message=True):
        """ Get masking function for given fields
            random_msg_idxs: list of message indices to mask
            random_fields: list of fields to mask.
                           NOTE: either this or random_msg_idxs must be given
            randomize_message: if True, select random message to mask, otherwise
                               mask most recent message
        """
        assert (random_msg_idxs is None) != (random_fields is None)
        if random_fields is not None:
            # get corresponding field indices
            random_fields = np.array(random_fields)
            ref = np.array(Message_Tokenizer.FIELDS)
            field_idxs = np.array([np.argwhere(f==ref) for f in random_fields]).flatten()
            random_msg_idxs = [
                idx 
                for f in field_idxs
                for idx in range(*LOBSTER_Dataset._get_tok_slice_i(f))]
        
        def masking_fn(seq, rng):
            seq = seq.copy()
            if randomize_message:
                m_i = rng.integers(0, seq.shape[0])
            else:
                m_i = seq.shape[0] - 1
            if len(random_msg_idxs) == 1:
                tok_i = random_msg_idxs[0]
            else:
                tok_i = rng.choice(random_msg_idxs)
            y = seq[m_i, tok_i]
            seq[m_i, tok_i] = Vocab.MASK_TOK
            return seq, y

        return masking_fn

    @staticmethod
    def random_mask(seq, rng, exclude_time=True):
        """ Select random token in given seq and set to MSK token
            as prediction target
        """
        # mask a random token in the most recent message
        # and HIDe a random uniform number of other tokens randomly
        seq = seq.copy()

        # select random positions to HIDe
        l = Message_Tokenizer.MSG_LEN

        # exclude time from masking if exclude_time == True
        if exclude_time:
            time_field_i = Message_Tokenizer.FIELD_I['time']
            time_start_i, time_end_i = LOBSTER_Dataset._get_tok_slice_i(time_field_i)
            candidate_pos = list(range(time_start_i)) + list(range(time_end_i, l))
            max_hid = l + 1 - (time_end_i - time_start_i)
        else:
            candidate_pos = list(range(l))
            max_hid = l + 1

        # sample uniformly without replacement
        hid_pos = sorted(
            rng.choice(
                #list(range(l)),
                candidate_pos,
                rng.integers(1, max_hid),
                replace=False
            )
        )

        # select one position to MSK from pre-selected HID positions
        msk_pos = rng.choice(hid_pos)
        hid_pos.remove(msk_pos)

        # deterministically hide time if delta_t is not complete (some HID)
        if exclude_time:
            dt_field_i = Message_Tokenizer.FIELDS.index('delta_t')
            dt_idx = list(range(*LOBSTER_Dataset._get_tok_slice_i(dt_field_i)))
            # part of delta_t is hidden
            if any(i in hid_pos for i in dt_idx):
                # --> also HIDe time
                hid_pos.extend(range(time_start_i, time_end_i))

        y = seq[-1, msk_pos]
        seq[-1, msk_pos] = Vocab.MASK_TOK
        seq[-1, hid_pos] = Vocab.HIDDEN_TOK

        return seq, y
    
    @staticmethod
    def no_mask(seq,order_books=None):
        """ Return the whole sequence and a sequence of labels y (which are essentially the sequence)
        TODO: Need to decide whether to put a 0th element in front for the case of generating 1st token...
        """
        seq=seq.copy()
        # ob_seq=np.repeat(order_books, seq.shape[0] // order_books.shape[0], axis=0)
        y= seq
        seq=np.concatenate([[Vocab.START_TOK],
                            seq[:-1]])
        # ob_seq=ob_seq[:-1]
        return (seq,order_books), y

    @staticmethod
    def no_mask_1tok(seq, order_books=None):
        """1-token-per-message masking: shift by 1 message row, prepend START_ROW.

        seq: (n_messages, 24) int32 local per-field indices
        Returns: ((shifted_seq, order_books), labels)
        """
        seq = seq.copy()
        y = seq.copy()
        start_row = np.full((1, seq.shape[1]), Vocab.START_TOK, dtype=seq.dtype)
        seq = np.concatenate([start_row, seq[:-1]], axis=0)
        return (seq, order_books), y

    @staticmethod
    def inference_mask(seq,order_books=None):
        """ Identity function, shouldn't even return the labels.
        """
        seq=np.concatenate([[Vocab.START_TOK],seq])
        return (seq,order_books), np.array(0)

    # @staticmethod
    # def last_position_mask(seq, rng):
    #     """
    #     Selects a field from the latest message where one token is masked with MSK.
    #     Retains tokens to the left of MSK and removes those to the right, labeled as Q.
    #     Deletes the first message from all messages, labeled as P.
    #     Takes tokens to the right of MSK's position in the first message, labeled as O.
    #     Concatenates O, P, and Q in sequence.
    #     """
    #     seq = seq.copy()

    #     # Randomly selects the field to be masked and the hidden field
    #     hidden_fields, msk_field = LOBSTER_Dataset._select_sequential_causal_mask_no_time(rng)

    #     # Gets the start and end indices of the selected masked field
    #     i_start, i_end = LOBSTER_Dataset._get_tok_slice_i(msk_field)

    #     # Randomly selects a token within the chosen field for masking
    #     msk_i = rng.integers(i_start, i_end)
    #     y = seq[-1][msk_i]

    #     # Q: Keeps tokens to the left of MSK
    #     Q = seq[-1, :msk_i]
        
    #     # Inserts MASK_TOK at the position after the selected token for masking
    #     Q = jnp.concatenate([Q, jnp.array([Vocab.MASK_TOK])])

    #     # O: Retrieves tokens to the right of MSK's position in the first message
    #     O = seq[0, msk_i + 1:]

    #     # P: Removes the first message from the sequence
    #     P = seq[1:]

    #     # Concatenates O, flattened P, and Q
    #     new_seq = jnp.concatenate([O] + [P.flatten()] + [Q])

    #     return new_seq, y

    # @staticmethod
    # def last_pos_mask(seq, rng, *args):
    #     """
    #     Generates a mask for the last position in the sequence.
        
    #     Parameters:
    #     seq (list or array): The sequence of positions.
    #     rng (int): The range or length of the sequence.
    #     *args: Additional arguments (e.g., order books).
        
    #     Selects a field from the latest message where one token is masked with MSK.
    #     Retains tokens to the left of MSK and removes those to the right, labeled as Q.
    #     Deletes the first message from all messages, labeled as P.
    #     Takes tokens to the right of MSK's position in the first message, labeled as O.
    #     Concatenates O, P, and Q in sequence.
    #     """
    #     order_books = args[0] if args else None

    #     seq = seq.copy()

    #     # Randomly selects the field to be masked and the hidden field
    #     hidden_fields, msk_field = LOBSTER_Dataset._select_sequential_causal_mask_no_time(rng)

    #     # Gets the start and end indices of the selected masked field
    #     i_start, i_end = LOBSTER_Dataset._get_tok_slice_i(msk_field)

    #     # Randomly selects a token within the chosen field for masking
    #     msk_i = rng.integers(i_start, i_end)
    #     y = seq[-1][msk_i]

    #     # O: Retrieves tokens to the right of MSK's position in the first message
    #     O = seq[0, msk_i + 1:]
    #     # P: Removes the first message from the sequence
    #     P = seq[1:-1]
    #     # Q: Keeps tokens to the left of MSK
    #     Q = seq[-1, :msk_i]
    #     # Inserts MASK_TOK at the position after the selected token for masking
    #     Q = jnp.concatenate([Q, jnp.array([Vocab.MASK_TOK])])
    #     # Concatenates O, flattened P, and Q
    #     new_seq = jnp.concatenate([O] + [P.flatten()] + [Q])

    #     token_index = msk_i  # Token index used for repetition calculation
    #     K = P.shape[1]
    #     # Calculate the repeat counts for each segment

    #     #FIXME: if order_books is None, then this line will FAIL
    #     repeats = jnp.array([K - token_index] + [K] * (len(order_books) - 2) + [token_index])
    #     # order_books is in shape (500,501) # TODO should be shape 501*501 ?
    #     # the repeat should happen in the first dimension and keep the second dimension not changed
    #     new_ob_O = jnp.repeat(order_books[0:1], repeats[0], axis=0)
    #     # Use vmap to apply the function across the first axis of order_books_P
    #     order_books_P = order_books[1:-1]
    #     new_ob_P = jax.vmap(
    #         lambda row: jnp.repeat(row[jnp.newaxis, :], K, axis=0), 
    #         in_axes=(0,),
    #         )(order_books_P)
    #     new_ob_P = new_ob_P.reshape(-1, new_ob_P.shape[-1])
    #     new_ob_Q = jnp.repeat(order_books[-1:], repeats[-1], axis=0)
    #     new_ob = jnp.concatenate([new_ob_O, new_ob_P, new_ob_Q], axis=0)
        
    #     return (new_seq, new_ob), y


    @staticmethod
    def causal_mask(seq, rng):
        """ Select random field (e.g price) in most recent message
            for which one token is MSKd (tokens left of MSK are know,
            right of MSK are NA). MSK token becomes prediction target.
            Random subset of other fields are also set to NA.
            This simulates the causal prediction task, where fields
            can be predicted in arbitrary order.
        """
        seq = seq.copy()
        hidden_fields, msk_field = LOBSTER_Dataset._select_sequential_causal_mask_no_time(rng)

        i_start, i_end = LOBSTER_Dataset._get_tok_slice_i(msk_field)
        msk_i = rng.integers(i_start, i_end)
        # select random token from last message from selected field
        y = seq[-1][msk_i]
        # seq[-1][msk_i] = Vocab.MASK_TOK
        seq = seq.at[-1, msk_i].set(Vocab.MASK_TOK)
        # set tokens after MSK token to HIDDEN for masked field
        if msk_i < (i_end - 1):
            # seq[-1][msk_i + 1: i_end] = Vocab.HIDDEN_TOK
            seq = seq.at[-1, msk_i + 1: i_end].set(Vocab.HIDDEN_TOK)
        # set all hidden_fields to HIDDEN
        for f in hidden_fields:
            # seq[-1][slice(*LOBSTER_Dataset._get_tok_slice_i(f))] = Vocab.HIDDEN_TOK
            seq = seq.at[-1, slice(*LOBSTER_Dataset._get_tok_slice_i(f))].set(Vocab.HIDDEN_TOK)
        return seq, y
    
    @staticmethod
    def _select_random_causal_mask(rng):
        """ Select random subset of fields and one field to mask
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        sel_fields = sorted(rng.choice(
            list(range(n_fields)),
            rng.integers(1, n_fields + 1),
            replace=False
        ))

        msk_field = rng.choice(sel_fields)
        sel_fields = list(set(sel_fields) - {msk_field})
        return sel_fields, msk_field

    @staticmethod
    def _select_unconditional_mask(rng):
        """ Select only one field to mask and all other fields to be hidden
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        sel_fields = list(range(n_fields))
        msk_field = sel_fields.pop(rng.integers(0, n_fields))
        return sel_fields, msk_field
    
    @staticmethod
    def _select_sequential_causal_mask(rng):
        """ Select tokens random field till end for HID
            preceded by one MSK
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        msk_field = rng.integers(0, n_fields)
        #return tuple(range(msk_field)), msk_field
        return tuple(range(msk_field + 1, n_fields)), msk_field

    @staticmethod
    def _select_sequential_causal_mask_no_time(rng):
        """ Select tokens from random field to end for HID
            preceded by one MSK
            TIME field is never MSKd (not predicted)
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        msk_field = rng.integers(0, n_fields - 1)
        i_time_s = Message_Tokenizer.FIELDS.index('time_s')
        i_time_ns = Message_Tokenizer.FIELDS.index('time_ns')
        if msk_field == i_time_s:
            msk_field += 2
        elif msk_field >= i_time_ns:
            msk_field += 1
        # hidden_fields, msk_field
        return tuple(range(msk_field + 1, n_fields)), msk_field

    @staticmethod
    def _get_tok_slice_i(field_i):
        i_start = ([0] + list(Message_Tokenizer.TOK_DELIM))[field_i]
        field_len = Message_Tokenizer.TOK_LENS[field_i]
        return i_start, i_start + field_len

    def __init__(
            self,
            message_files,
            n_messages,
            mask_fn,
            seed=None,
            n_cache_files=0,
            randomize_offset=True,
            *,
            book_files=None,
            use_simple_book=False,
            book_transform=True,
            book_depth=500,
            # if given, also load and return raw sequences
            # -> used for inference (not training!)
            return_raw_msgs=False,
            # for inference, the last message is not masked
            # and hence the book state after the message is
            # already available (shifts by one)
            inference=False,
            limit_seq_per_file=math.inf,
            wide_book_files=None,
            token_mode='24tok',
            book_ablation='real',
            ) -> None:


        assert len(message_files) > 0
        assert not (use_simple_book and book_transform)

        # shift book state by 1 for inference tasks,
        # because the most recent message is not masked (=complete)
        # and the new book state is already available
        self.inference = inference

        self.message_files = message_files #
        if book_files is not None:
            assert len(book_files) == len(message_files)
            self.use_book_data = True
            self.book_files = book_files
            self._book_cache = OrderedDict()
        else:
            self.use_book_data = False
        # Optional wider book files for deeper simulator init
        # Supports both .npy (per-event, e.g. L100) and .npz (per-sequence snapshots, e.g. L-inf)
        if wide_book_files is not None:
            assert len(wide_book_files) == len(message_files)
        self.wide_book_files = wide_book_files
        self._wide_book_cache = OrderedDict()
        self._wide_book_npz_cache = OrderedDict()
        self.use_simple_book = use_simple_book
        self.book_transform = book_transform
        self.book_depth = book_depth
        self.book_ablation = book_ablation
        self.return_raw_msgs = return_raw_msgs
        self.num_days = len(self.message_files)
        self.n_messages = n_messages
        self.token_mode = token_mode

        self.n_cache_files = n_cache_files
        self._message_cache = OrderedDict()
        self.vocab = Vocab()
        self.mask_fn = mask_fn
        # Override mask_fn for 1tok mode
        if self.token_mode == '1tok':
            self.mask_fn = LOBSTER_Dataset.no_mask_1tok
        if self.mask_fn in (LOBSTER_Dataset.no_mask, LOBSTER_Dataset.inference_mask, LOBSTER_Dataset.no_mask_1tok):
            if self.token_mode == '1tok':
                self.seq_len = self.n_messages
            else:
                self.seq_len = self.n_messages * Message_Tokenizer.MSG_LEN
        else:
            raise NotImplementedError("Need to confirm syntax for other mask funcs to ensure backward compat.")
        self.rng = np.random.default_rng(seed)
        self.randomize_offset = randomize_offset
        # Thread-pool the per-file shape lookup: ~5ms each, dominated by Lustre
        # metadata + small read. With 32 threads, 500k files lands in ~2 min
        # vs ~42 min serial. _zst_npy_shape uses module-level _zstd_dctx; a
        # ZstdDecompressor instance is documented thread-safe for stream_reader.
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=32) as _ex:
            self._num_rows_per_file = np.fromiter(
                _ex.map(self._get_num_rows, self.message_files),
                dtype=np.int64, count=len(self.message_files),
            )
            # Also validate book files: a corrupted/truncated book (passes the
            # asymmetric size filter at >200B but fails np.load mmap) would
            # otherwise pair with a fine msg file and crash __getitem__ when
            # preproc.transform_L2_state_numpy hits book[0,0] on the
            # zero-fill placeholder. Drop pairs where book is bad or row
            # counts mismatch by zeroing the msg row count → file excluded
            # from _seqs_cumsum → sampler never selects it.
            if self.use_book_data:
                book_rows = np.fromiter(
                    _ex.map(self._get_num_rows, self.book_files),
                    dtype=np.int64, count=len(self.book_files),
                )
                # SP500 preproc emits book with +N extra rows (initial state +
                # post-last-msg snapshot); __getitem__ slices book to msg len.
                # Only flag book shorter than msg or zero rows — those crash
                # preproc.transform_L2_state_numpy on book[0,0] / out-of-bounds.
                bad = (book_rows == 0) | (book_rows < self._num_rows_per_file)
                n_bad = int(bad.sum())
                if n_bad > 0:
                    print(f"[*] Dropping {n_bad}/{len(book_rows)} pairs (book corrupt/truncated/mismatched)",
                          flush=True)
                    self._num_rows_per_file = np.where(bad, 0, self._num_rows_per_file)
        # Keep offsets in shared memory so persistent DataLoader workers can
        # see per-epoch updates without recreating the DataLoader.
        self.seq_offsets = torch.zeros(len(self.message_files), dtype=torch.int64)
        self.seq_offsets.share_memory_()
        self._reset_offsets()
        self._set_book_dims()
        # Use a conservative fixed length so any runtime offset in
        # [0, n_messages - 1] remains in-bounds without rebuilding dataset state.
        max_offset = (self.n_messages - 1) if self.randomize_offset else 0
        seqs_per_file = np.maximum(
            (self._num_rows_per_file - max_offset) // self.n_messages,
            0,
        )
        if not math.isinf(limit_seq_per_file):
            seqs_per_file = np.minimum(seqs_per_file, int(limit_seq_per_file))
        self._seqs_per_file = seqs_per_file.astype(np.int64)
        # store at which observations files start
        self._seqs_cumsum = np.concatenate(([0], np.cumsum(self._seqs_per_file)))
        # count total number of sequences only once
        self._len = int(self._seqs_cumsum[-1])

    def _set_book_dims(self):
        if self.use_book_data:
            if self.book_transform:
                self.d_book = self.book_depth + 3
            else:
                # Iterate book_files until one loads cleanly; the first might
                # be a corrupt 0-row placeholder from the defensive fallback.
                self.d_book = 0
                for bf in self.book_files:
                    try:
                        b = _np_load_zst(bf, mmap_mode='r', allow_pickle=True)
                        if b.ndim >= 2 and b.shape[0] > 0:
                            self.d_book = int(b.shape[1])
                            break
                    except Exception:
                        continue
                if self.d_book == 0:
                    raise RuntimeError(
                        f"No loadable book file found across {len(self.book_files)} candidates")
            if self.mask_fn in (LOBSTER_Dataset.no_mask, LOBSTER_Dataset.inference_mask, LOBSTER_Dataset.no_mask_1tok):
                self.L_book=self.n_messages
            else:
                raise NotImplementedError("Need to confirm syntax for other mask funcs to ensure backward compat.")
        else:
            self.d_book = 0
            self.L_book = 0
    
    def _reset_offsets(self):
        """ drop a random number of messages from the beginning of every file
            so that sequences don't always contain the same time periods
        """
        if self.randomize_offset:
            new_offsets = np.array(
                [self.rng.integers(0, self.n_messages) for _ in range(len(self.message_files))],
                dtype=np.int64,
            )
        else:
            new_offsets = np.zeros(len(self.message_files), dtype=np.int64)
        self.seq_offsets.copy_(torch.from_numpy(new_offsets))

    @property
    def shape(self):
        return len(self), Message_Tokenizer.MSG_LEN#, len(self.vocab)

    def __len__(self):
        return self._len



    def __getitem__(self, idx):
        #print(idx)
        if hasattr(idx, '__len__'):
            return list(zip(*[self[i] for i in idx]))

        file_idx, seq_idx = self._get_seq_location(idx)
        # print(f"lobster_dataloader.py: File index is {file_idx}, seq idx is {seq_idx} and offset for file is {self.seq_offsets[file_idx]}")
        
        # load sequence from file directly without cache
        if self.n_cache_files == 0:
            X = _np_load_zst(self.message_files[file_idx], mmap_mode='r')
            if self.use_book_data:
                book = _np_load_zst(
                    self.book_files[file_idx],
                    mmap_mode='r'
                )
        else:
            if file_idx not in self._message_cache:
                self._add_to_cache(file_idx)
            #print('fetching from cache')
            X = self._message_cache[file_idx]
            if self.use_book_data:
                # print('fetching book from cache')
                book = self._book_cache[file_idx]

        seq_start = int(self.seq_offsets[file_idx].item()) + seq_idx * self.n_messages
        seq_end = seq_start + self.n_messages
        
        X_raw = np.array(X[seq_start: seq_end])
        # print(X_raw[0])
        # encode message

        X = encode_msgs(X_raw, self.vocab.ENCODING)
        # print(f"lobster_dataloader.py: First loaded message from batch is \n  {X_raw[0]}\n which is \n {X[0]}\nafter encoding.")

        
        
        if self.use_book_data:
            # first message is already dropped, so we can use
            # the book state with the same index (prior to the message)


            book = book[seq_start: seq_end + self.inference].copy()
    
            if self.return_raw_msgs:
                if self.wide_book_files is not None:
                    wb_path = self.wide_book_files[file_idx]
                    if wb_path.endswith('.npz'):
                        # L-infinity NPZ snapshots: indexed by local sequence ID
                        if file_idx in self._wide_book_npz_cache:
                            npz_data = self._wide_book_npz_cache[file_idx]
                        else:
                            npz_data = np.load(wb_path)
                            if self.n_cache_files > 0:
                                if len(self._wide_book_npz_cache) >= self.n_cache_files:
                                    self._wide_book_npz_cache.popitem(last=False)
                                self._wide_book_npz_cache[file_idx] = npz_data
                        local_indices = npz_data['local_seq_indices']
                        pos = np.searchsorted(local_indices, seq_idx)
                        if pos < len(local_indices) and local_indices[pos] == seq_idx:
                            book_l2_init = npz_data['snapshots'][pos].copy()
                        else:
                            # Fallback: pad L10 book to match NPZ snapshot width
                            snap_width = npz_data['snapshots'].shape[1]
                            fallback = book[0, 3:].copy()
                            book_l2_init = np.zeros(snap_width, dtype=fallback.dtype)
                            book_l2_init[:len(fallback)] = fallback
                    else:
                        # Standard .npy wide book (e.g. L100): per-event rows
                        if file_idx in self._wide_book_cache:
                            wide_book = self._wide_book_cache[file_idx]
                        else:
                            wide_book = _np_load_zst(wb_path, mmap_mode='r')
                            if self.n_cache_files > 0:
                                if len(self._wide_book_cache) >= self.n_cache_files:
                                    self._wide_book_cache.popitem(last=False)
                                self._wide_book_cache[file_idx] = wide_book
                        book_l2_init = wide_book[seq_start, 3:].copy()
                else:
                    book_l2_init = book[0, 3:].copy()
            # t0=time.time()
            # tranform from L2 (price volume) representation to fixed volume image 
            if self.book_transform:
                book = transform_L2_state_numpy(book, self.book_depth, 100)
            book=np.array(book)
            # ── Book ablation (applied post-transform) ──
            if self.book_ablation == 'zero':
                book = np.zeros_like(book)
            elif self.book_ablation == 'noise':
                _rng = np.random.default_rng([file_idx, seq_idx])
                book = np.zeros_like(book)
                book[:, 3:] = _rng.standard_normal((book.shape[0], book.shape[1] - 3)) * 0.15
            elif self.book_ablation == 'shuffle':
                _rng = np.random.default_rng([file_idx, seq_idx])
                book = book[_rng.permutation(book.shape[0])]
            # t1=time.time()
            # use raw price, volume series, rather than volume image
            # subtract initial price to start all sequences around 0
            # if self.use_simple_book:
                # CAVE: first column is Delta mid price
                # p_mid_0 = (book[0, 1] + book[0, 3]) / 2
                # book[:, 1::2] = (book[:, 1::2] - p_mid_0)
                # divide volume by 100
                #book[:, 2::2] = book[:, 2::2] / 100
                
            # apply mask and extract prediction target token
            
            
            if self.token_mode == '1tok':
                from lob.encoding_1tok import global_to_local
                X = global_to_local(np.array(X))  # (n_messages, 24) local indices
                X, y = self.mask_fn(X, book)
                X, book = X
                ret_tuple = X, y, book
            else:
                X = X.reshape(-1)
                X, y = self.mask_fn(np.array(X), book)
                X,book=X
                # print(book[0])
                y=y.reshape(-1)
                ret_tuple = X, y, book

        else:
            if self.token_mode == '1tok':
                from lob.encoding_1tok import global_to_local
                X = global_to_local(np.array(X))
                X, y = self.mask_fn(X)
                X, book = X
                ret_tuple = X, y
            else:
                X = X.reshape(-1)
                X, y = self.mask_fn(X)
                X,book=X
                y=y.reshape(-1)
                ret_tuple = X, y

        if self.return_raw_msgs:
            if self.use_book_data:
                ret_tuple += (X_raw, book_l2_init)
            else:
                ret_tuple += (X_raw,)

        return ret_tuple

    def _add_to_cache(self, file_idx):
        if len(self._message_cache) >= self.n_cache_files:
            # remove item in FIFO order
            _ = self._message_cache.popitem(last=False)
            if self.use_book_data:
                _ = self._book_cache.popitem(last=False)
            del _

        Xm = _np_load_zst(self.message_files[file_idx], mmap_mode='r')
        self._message_cache[file_idx] = Xm

        if self.use_book_data:
            Xb = _np_load_zst(self.book_files[file_idx], mmap_mode='r')
            self._book_cache[file_idx] = Xb

    def _get_num_rows(self, file_path):
        # Header-only read: avoid full-file decompression (~20ms × 500k files = hours).
        return _zst_npy_shape(file_path)[0]

    def _get_seq_location(self, idx):
        if idx > len(self) - 1:
            raise IndexError(f'index {idx} out of range for dataset length ({len(self)})')
        file_idx = np.searchsorted(self._seqs_cumsum, idx+1) - 1
        seq_idx = idx - self._seqs_cumsum[file_idx]
        return file_idx, seq_idx

    def get_date(self, idx):
        if hasattr(idx, '__len__'):
            return [self.get_date(i) for i in idx]
        
        file_idx, _ = self._get_seq_location(idx)
        file_name = self.message_files[file_idx].rsplit('/', 1)[1]
        # file name from path -> STOCK_date_xxx
        # date_str = self.message_files[file_idx].rsplit('/', 1)[1].split('_', 2)[1]
        date_str = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", file_name)[0]
        return date_str

class LOBSTER_Sampler(Sampler):
    def __init__(self, dset, n_files_shuffle, batch_size=1, seed=None):
        self.dset = dset
        assert n_files_shuffle > 0
        self.n_files_shuffle = n_files_shuffle
        self.batch_size = batch_size

        self.rng = random.Random(seed)

    def reset(self):
        # LOBSTER_Dataset
        if hasattr(self.dset, "num_days"):
            days = range(self.dset.num_days)
        # LOBSTER_Subset
        elif hasattr(self.dset, "indices_on_day"):
            days = list(self.dset.indices_on_day.keys())
        else:
            raise AttributeError("dataset has neither num_days nor indices_on_day attribute.")
        # days in random order
        self.days_unused = self.rng.sample(
            days,
            len(days)
        )
        self.active_indices = []

    def __iter__(self):
        # reset days and active indices whenever new iterator is created (e.g. new epoch)
        print("lobster_dataloader.py: omitting the reset function. ")
        # self.reset()

        while len(self.days_unused) > 0 or len(self.active_indices) >= self.batch_size:
            batch = []
            # not enough indices available for full batch
            if len(self.active_indices) < self.batch_size:
                batch += list(self.active_indices)
                # get new indices from new days
                self.active_indices = self._get_new_active_indices(self._get_new_days())
            while len(batch) < self.batch_size:
                # TODO: fix pop from empty list error
                batch.append(self.active_indices.pop())
            if self.batch_size == 1:
                batch = batch[0]
            yield batch

    def _get_new_days(self):
        days = []
        for _ in range(self.n_files_shuffle):
            if len(self.days_unused) > 0:
                days.append(self.days_unused.pop())
            else:
                break
        return days

    def _get_new_active_indices(self, days):
        idx = []
        # LOBSTER_Dataset
        if hasattr(self.dset, "_seqs_cumsum"):
            for d in days:
                idx.extend(
                    list(range(
                        self.dset._seqs_cumsum[d],
                        self.dset._seqs_cumsum[d + 1]
                    ))
                )
        elif hasattr(self.dset, "indices_on_day"):
            for d in days:
                idx.extend(self.dset.indices_on_day[d])
        else:
            raise AttributeError("dataset has neither num_days nor indices_on_day attribute.")
        
        self.rng.shuffle(idx)
        return idx

    def __len__(self):
        return len(self.dset)


class LOBSTER_Subset(Subset):
    def __init__(self, dataset: LOBSTER_Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = sorted(indices)

        self.indices_on_day = self.get_indices_by_day(
            self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def get_indices_by_day(self, indices):
        indices_on_day = {}
        day = 0
        i_end = self.dataset._seqs_cumsum[day + 1]

        for i, idx in enumerate(indices):
            while idx >= i_end:
                day += 1
                i_end = self.dataset._seqs_cumsum[day + 1]
            
            if day not in indices_on_day.keys():
                indices_on_day[day] = []
            indices_on_day[day].append(i)
        return indices_on_day


class LOBSTER(SequenceDataset):
    _name_ = "lobster"
    l_output = 0

    _collate_arg_names = ['book_data'] #['book_data'] #['timesteps']

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        x, y, *z = return_value
        if len(z) == 0:
            return x, y, {}
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Custom collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    @property
    def init_defaults(self):
        # NOTE: don't add data_dir here, it's added in the base class
        return {
            #"permute": False,
            #"k_val_segments": 5,  # train/val split is done by picking 5 contiguous folds
            "val_split": 0.01,
            "test_split": 0.1,
            "seed": 42,  # For train/val split
            "mask_fn": LOBSTER_Dataset.random_mask,
            "use_book_data": False,
            "use_simple_book" : False,
            "book_transform": False,
            "book_ablation": "real",
            "n_cache_files": 0,
            "book_depth": 500,
            "test_data_dir": None,
            "return_raw_msgs": False,
            "rand_offset": True,
            "debug_overfit": False,
            # Multi-ticker support
            "tickers": None,
            "data_root": None,
            "train_date_range": None,
            "test_date_range": None,
            "token_mode": "24tok",
        }

    def setup(self):
        self.n_messages = self.msg_seq_len
        self.per_ticker_test_datasets = {}
        self._test_files_by_ticker = {}

        # Adjust collate arg names based on whether book data is used
        if not self.use_book_data:
            self._collate_arg_names = []

        if getattr(self, 'tickers', None) is not None:
            # ── Multi-ticker mode: discover files across ticker subdirectories ──
            assert self.data_root is not None, \
                "data_root required when tickers is set"
            print(f"[*] Multi-ticker mode: {self.tickers}")

            # Discover train files (all tickers, optionally date-filtered)
            train_msg_by_tk, train_book_by_tk = discover_ticker_files(
                self.data_root, self.tickers, self.train_date_range)

            # Discover test files (same tickers, test date range)
            if self.test_date_range is not None:
                test_msg_by_tk, test_book_by_tk = discover_ticker_files(
                    self.data_root, self.tickers, self.test_date_range)
            else:
                test_msg_by_tk = {}
                test_book_by_tk = {}

            self.rng = random.Random(self.seed)

            all_train_msg, all_train_book = [], []
            all_val_msg, all_val_book = [], []
            all_test_msg, all_test_book = [], []

            for ticker in self.tickers:
                t_msg = train_msg_by_tk[ticker]
                t_book = train_book_by_tk[ticker]
                n_val = max(1, int(len(t_msg) * self.val_split)) if self.val_split > 0 else 0

                # Stratified val split per ticker (random days)
                paired = list(zip(t_msg, t_book))
                val_paired = [paired.pop(self.rng.randrange(0, len(paired)))
                              for _ in range(n_val)]
                train_paired = paired

                tr_m, tr_b = zip(*train_paired) if train_paired else ([], [])
                vl_m, vl_b = zip(*val_paired) if val_paired else ([], [])
                all_train_msg.extend(tr_m)
                all_train_book.extend(tr_b)
                all_val_msg.extend(vl_m)
                all_val_book.extend(vl_b)

                # Test files from test date range
                if ticker in test_msg_by_tk:
                    te_msg = test_msg_by_tk[ticker]
                    te_book = test_book_by_tk[ticker]
                    all_test_msg.extend(te_msg)
                    all_test_book.extend(te_book)
                    self._test_files_by_ticker[ticker] = (te_msg, te_book)

                print(f"  {ticker}: {len(list(tr_m))} train, {len(list(vl_m))} val, "
                      f"{len(test_msg_by_tk.get(ticker, []))} test days")

            self.train_files = list(all_train_msg)
            self.val_files = list(all_val_msg)
            self.test_files = list(all_test_msg) if all_test_msg else list(all_val_msg)
            if self.use_book_data:
                self.train_book_files = list(all_train_book)
                self.val_book_files = list(all_val_book)
                self.test_book_files = list(all_test_book) if all_test_book else list(all_val_book)
            else:
                self.train_book_files = None
                self.val_book_files = None
                self.test_book_files = None

        else:
            # ── Single-asset mode: original logic (file discovery + val split) ──
            message_files = sorted(glob(str(self.data_dir) + '/*message*.npy.zst'))
            assert len(message_files) > 0, f'no message files found in {self.data_dir}'
            if self.use_book_data:
                book_files = sorted(glob(str(self.data_dir) + '/*book*.npy.zst'))
                assert len(message_files) == len(book_files)
            else:
                book_files = None

            # Load test files from separate directory if specified
            if self.test_data_dir is not None:
                test_message_files = sorted(glob(str(self.test_data_dir) + '/*message*.npy.zst'))
                assert len(test_message_files) > 0, f'no test message files found in {self.test_data_dir}'
                if self.use_book_data:
                    test_book_files = sorted(glob(str(self.test_data_dir) + '/*book*.npy.zst'))
                    assert len(test_message_files) == len(test_book_files), \
                        f'mismatch between test message files ({len(test_message_files)}) and book files ({len(test_book_files)})'
                else:
                    test_book_files = None
            else:
                test_message_files = None
                test_book_files = None

            if self.debug_overfit:
                self.train_files = message_files[:1]
                self.val_files = message_files[:1]
                self.test_files = message_files[:1]
                if book_files:
                    self.train_book_files = book_files[:1]
                    self.val_book_files = book_files[:1]
                    self.test_book_files = book_files[:1]
                else:
                    self.train_book_files = None
                    self.val_book_files = None
                    self.test_book_files = None
            else:
                if test_message_files is not None:
                    self.test_files = test_message_files
                    self.test_book_files = test_book_files
                    self.train_files = message_files
                    self.train_book_files = book_files
                else:
                    n_test_files = max(1, int(len(message_files) * self.test_split)) if self.test_split > 0 else 0
                    self.train_files = message_files[:len(message_files) - n_test_files]
                    self.test_files = message_files[len(self.train_files):]
                    if book_files:
                        self.train_book_files = book_files[:len(book_files) - n_test_files]
                        self.test_book_files = book_files[len(self.train_book_files):]
                    else:
                        self.train_book_files = None
                        self.test_book_files = None

                self.rng = random.Random(self.seed)

                if book_files or (test_book_files is not None and self.train_book_files is not None):
                    self.train_files = list(zip(self.train_files, self.train_book_files))
                else:
                    self.val_book_files = None

                n_val_files = max(1, int(len(message_files) * self.val_split)) if self.val_split > 0 else 0
                self.val_files = [
                    self.train_files.pop(
                        self.rng.randrange(0, len(self.train_files))
                    ) for _ in range(n_val_files)]
                if book_files or (test_book_files is not None and self.train_book_files is not None):
                    self.train_files, self.train_book_files = zip(*self.train_files)
                    if self.val_files:
                        self.val_files, self.val_book_files = zip(*self.val_files)

        # ── Shared: create Dataset objects from file lists ──
        self.dataset_train = LOBSTER_Dataset(
            self.train_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.seed if self.debug_overfit else self.rng.randint(0, sys.maxsize),
            n_cache_files=self.n_cache_files,
            randomize_offset=self.rand_offset,
            book_files=self.train_book_files,
            use_simple_book=self.use_simple_book,
            book_transform=self.book_transform,
            book_ablation=self.book_ablation,
            book_depth=self.book_depth,
            return_raw_msgs=self.return_raw_msgs,
            token_mode=self.token_mode,
        )
        #self.d_input = self.dataset_train.shape[-1]
        self.d_input = len(self.dataset_train.vocab)
        self.d_output = self.d_input
        # sequence length
        self.L = self.dataset_train.seq_len
        # book sequence lengths and dimension (number of levels + 1)
        self.L_book = self.dataset_train.L_book
        self.d_book = self.dataset_train.d_book



        #self.split_train_val(self.val_split)
        if self.val_split > 0:
            self.dataset_val = LOBSTER_Dataset(
                self.val_files,
                n_messages=self.n_messages,
                mask_fn=self.mask_fn,
                seed=self.seed if self.debug_overfit else self.rng.randint(0, sys.maxsize),
                n_cache_files=self.n_cache_files,
                randomize_offset=False,
                book_files=self.val_book_files,
                use_simple_book=self.use_simple_book,
                book_transform=self.book_transform,
                book_ablation=self.book_ablation,
                book_depth=self.book_depth,
                return_raw_msgs=self.return_raw_msgs,
                token_mode=self.token_mode,
                )
        else:
            self.dataset_val = None

        if self.test_split > 0:
            self.dataset_test = LOBSTER_Dataset(
                self.test_files,
                n_messages=self.n_messages,
                mask_fn=self.mask_fn,
                seed=self.seed if self.debug_overfit else self.rng.randint(0, sys.maxsize),
                n_cache_files=self.n_cache_files,
                randomize_offset=False,
                book_files=self.test_book_files,
                use_simple_book=self.use_simple_book,
                book_transform=self.book_transform,
                book_ablation=self.book_ablation,
                book_depth=self.book_depth,
                return_raw_msgs=self.return_raw_msgs,
                token_mode=self.token_mode,
                )
        else:
            self.dataset_test = None

        # Per-ticker test datasets (multi-ticker mode only).
        # n_cache_files=0 because (a) these are only iterated during eval, and
        # (b) the per-ticker test loaders run with num_workers=0 / persistent=False
        # (see dataloading.py), so any inherited dataset state would COW-expand
        # 488× per worker fork. Keeping this minimal is essential.
        if self._test_files_by_ticker:
            for ticker, (tk_msg, tk_book) in self._test_files_by_ticker.items():
                bk = tk_book if self.use_book_data else None
                self.per_ticker_test_datasets[ticker] = LOBSTER_Dataset(
                    tk_msg,
                    n_messages=self.n_messages,
                    mask_fn=self.mask_fn,
                    seed=self.seed if self.debug_overfit else self.rng.randint(0, sys.maxsize),
                    n_cache_files=0,
                    randomize_offset=False,
                    book_files=bk,
                    use_simple_book=self.use_simple_book,
                    book_transform=self.book_transform,
                    book_ablation=self.book_ablation,
                    book_depth=self.book_depth,
                    return_raw_msgs=self.return_raw_msgs,
                    token_mode=self.token_mode,
                )
            print(f"[*] Per-ticker test datasets: {len(self.per_ticker_test_datasets)} tickers (n_cache=0)")

    def reset_train_offsets(self):
        """ reset the train dataset to a new random offset
            (e.g. every training epoch)
            keeps the same validation set and removes validation
            indices from training set
        """
        # Update train offsets in place so existing DataLoader workers keep
        # running (persistent_workers=True) and pick up the new offsets.
        self.dataset_train._reset_offsets()

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"
