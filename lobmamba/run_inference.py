import argparse
import os
import sys

# --model_code_dir must be parsed BEFORE module imports (s5/, lob/ etc.)
# because `from s5.ssm import *` happens at module level.
if '--model_code_dir' in sys.argv:
    _idx = sys.argv.index('--model_code_dir')
    _model_code_dir = sys.argv[_idx + 1]
    sys.path.insert(0, _model_code_dir)
    print(f"[model_code_dir] {_model_code_dir} → sys.path[0]")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ['XLA_FLAGS'] ='--xla_gpu_deterministic_ops=true'


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

# import torch
# torch.multiprocessing.set_start_method('spawn')

# Add parent folder to path (to run this file from subdirectories)
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(parent_folder_path)

# add git submodule to path to allow imports to work
# AlphaTrade may be a sibling of LOBS5/ (original layout) or inside it (pipeline repo)
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
for candidate in [os.path.join(os.path.abspath(''), submodule_name),
                  os.path.join(parent_folder_path, submodule_name)]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

print(sys.path)
from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

# from argparse import Namespace
from glob import glob
import numpy as onp
import pandas as pd
# from functools import partial
# from typing import Union, Optional
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# from line_profiler import LineProfiler

import jax
import jax.numpy as jnp
from jax.nn import one_hot
# from jax import random
# from jax.scipy.linalg import block_diag
# from flax import jax_utils
# from flax.training import checkpoints
# import orbax

#from lob.lob_seq_model import BatchLobPredModel
# from lob.train_helpers import create_train_state, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
# SSM init handled by init_train_state (supports gdn + mamba3 via ssm_type)
import os as _os
_tok_mode = _os.environ.get("TOKEN_MODE", "24tok")
# Pipeline passes TOKEN_MODE=26 (no suffix) or 26tok; handle both
if _tok_mode in ("26tok", "26"):
    from lob.encoding_26tok import Vocab, Message_Tokenizer
    print(f"[Encoding] Using 26tok encoding (MSG_LEN={Message_Tokenizer.MSG_LEN})")
elif _tok_mode in ("22tok", "22"):
    from lob.encoding_22tok import Vocab, Message_Tokenizer
    print(f"[Encoding] Using 22tok encoding (MSG_LEN={Message_Tokenizer.MSG_LEN})")
else:
    # default lob.encoding IS the 26tok layout (byte-identical to encoding_26tok.py)
    from lob.encoding import Vocab, Message_Tokenizer
    print(f"[Encoding] Using default lob.encoding = 26tok (MSG_LEN={Message_Tokenizer.MSG_LEN})")
# from lobster_dataloader import LOBSTER_Dataset, LOBSTER_Subset, LOBSTER_Sampler, LOBSTER

import preproc
from time import time
# import inference
from lob import inference_no_errcorr as inference
import lob.validation_helpers as valh
from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint
# import lob.encoding as encoding


import lob.evaluation as eval
from preproc import transform_L2_state



##################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, default='GOOG', help='stock to evaluate')
    parser.add_argument('--checkpoint_step', type=int, default=None, help='Which checkpoint step to load')
    parser.add_argument('--test_split', type=float, default=0.1, help='Which test split to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument("--n_sequences", type=int, default=1024, help="Total number of sequences to generate (before rank splitting)")
    parser.add_argument("--n_cond_msgs", type=int, default=500, help="Number of conditioning messages")
    parser.add_argument("--n_gen_msgs", type=int, default=500, help="Number of messages to generate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data directory")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save inference results")
    parser.add_argument("--sample_indices_file", type=str, default=None, help="File with dataset indices for HF-matched mode (one per line)")
    parser.add_argument("--wide_book_dir", type=str, default=None,
                        help="Path to wider L2 book .npy files (e.g. L100) for simulator init")
    parser.add_argument("--wide_levels", type=int, default=10,
                        help="Number of book levels in wide_book_dir data (default: 10 = no change)")
    # Multi-GPU args
    parser.add_argument("--rank", type=int, default=0, help="Rank of this process (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    # Model code override (parsed early in sys.argv for path injection, declared here for --help)
    parser.add_argument("--model_code_dir", type=str, default=None,
                        help="Override s5/lob/preproc modules from this directory (injected at sys.path[0])")
    parser.add_argument("--token_mode", type=str, default='24tok', choices=['24tok', '1tok', '26tok', 'lobert'],
                        help="Token mode: '24tok'/'26tok' (autoregressive) or '1tok'/'lobert' (per-message)")
    parser.add_argument("--gt_compare", action='store_true', default=False,
                        help="Enable ground-truth comparison: token distributions, divergence trajectory, validity rate")
    parser.add_argument("--save_format", type=str, default='csv',
                        help="Output format (csv or npy). Accepted for pipeline compatibility.")

    run_args = parser.parse_args()

    overfit_debug = False

    # Resolve paths: CLI args take priority, fall back to hardcoded defaults
    data_dir = run_args.data_dir
    ckpt_path = run_args.ckpt_path
    save_dir = run_args.save_dir

    if data_dir is None or ckpt_path is None or save_dir is None:
        # Legacy hardcoded paths (only used when CLI args not provided)
        if run_args.stock == 'AMZN':
            data_dir = data_dir or '/home/myuser/processed_data/AMZN/2024_Dec'
            ckpt_path = ckpt_path or '/home/myuser/checkpoints/ruby-aardvark-62_98nov1i7'
            save_dir = save_dir or '/home/myuser/data/evalsequences/s5v2N5/AMZN/2024'
        elif run_args.stock == 'GOOG':
            data_dir = data_dir or '/home/myuser/data/processed_data/GOOG/2023_Jan'
            ckpt_path = ckpt_path or '/home/myuser/data/checkpoints/lobs5_v2/twilight-sound-77_s42sujip'
            save_dir = save_dir or '/home/myuser/data/evalsequences/s5v2N5/GOOG/2023_Jan'
        elif run_args.stock == 'INTC':
            data_dir = data_dir or '/home/myuser/data/processed_data/INTC/2023_Jan'
            ckpt_path = ckpt_path or '/home/myuser/data/checkpoints/lobs5_v2/dazzling-meadow-75_zpp3bf6z'
            save_dir = save_dir or '/home/myuser/data/evalsequences/s5v2N5/INTC/2023_Jan'
        else:
            raise ValueError(f"No default paths for stock '{run_args.stock}' -- provide --data_dir, --ckpt_path, --save_dir")

    ##################################################

    n_gen_msgs = run_args.n_gen_msgs
    n_messages_conditional = run_args.n_cond_msgs
    n_eval_messages = n_gen_msgs  # how many to load from dataset
    token_mode = run_args.token_mode
    if token_mode == '1tok':
        # 1tok: seq lengths still in tokens for dataset loading (same data format)
        # but cond_seq_len is only used for slicing in sample_new
        eval_seq_len = (n_eval_messages-1) * Message_Tokenizer.MSG_LEN
        cond_seq_len = n_messages_conditional * Message_Tokenizer.MSG_LEN
    else:
        eval_seq_len = (n_eval_messages-1) * Message_Tokenizer.MSG_LEN
        cond_seq_len = (n_messages_conditional) * Message_Tokenizer.MSG_LEN
    data_levels = 10
    # TODO: deprecated - remove from functions
    sim_book_levels = 20 # 10  # order book simulator levels
    sim_queue_len = 100  # per price in sim, how many orders in queue

    n_vol_series = 500  # how many book volume series model uses as input

    v = Vocab()
    n_classes = len(v)
    book_dim = 503 #b_enc.shape[1]
    eval_book_seq_len = eval_seq_len

    # Per-rank RNG: different sampling randomness per rank, shared base seed for index selection
    rng = jax.random.key(42 + run_args.rank)
    rng, rng_ = jax.random.split(rng)
    if overfit_debug:
        sample_top_n = 1
    else:
        sample_top_n = -1
    tick_size = 100

    # load train state from disk

    args = load_metadata(ckpt_path)
    tp_size = getattr(args, 'tp_size', 1)
    if tp_size > 1:
        args.num_devices = tp_size
    else:
        args.num_devices = 1
    args.bsz=1
    # init_train_state builds a dummy forward pass of shape (micro_bsz, seq_len, ...).
    # At long generation (seq_len = (n_gen-1) * MSG_LEN ≈ 10^5) this dominates device memory.
    # The training micro_bsz is irrelevant for inference, so force 1.
    args.micro_bsz = 1
    # Force standard optimizer for inference — avoids Muon/optax version issues
    # (we only need params, not the optimizer state)
    args.opt_config = "standard"

    # Hierarchical no-book: route to inference wrapper that accepts book args but discards them
    if getattr(args, 'hierarchical_nobook', False):
        args.no_book_inference_wrapper = True

    new_train_state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=eval_seq_len,
        book_dim=book_dim,
        book_seq_len=eval_book_seq_len,
    )

    # Set up TP mesh for inference if checkpoint was trained with TP
    if tp_size > 1:
        from lob.sharding_utils import create_simple_mesh
        tp_mesh = create_simple_mesh(tp_size, tp_size=tp_size)
        import lob.validation_helpers as valh
        valh._TP_MESH = tp_mesh
        m3_expand = getattr(args, 'mamba3_expand', 2)
        m3_headdim = getattr(args, 'mamba3_headdim', 64)
        d_inner = m3_expand * args.d_model
        valh._TP_NH = max(1, d_inner // m3_headdim)
        # Remove inner @jax.jit from apply_model — outer jit(shard_map(vmap(generate))) handles it
        valh.apply_model = valh._apply_model_impl
        valh.apply_model_1tok = valh._apply_model_1tok_impl
        print(f"[TP] Inference with tp_size={tp_size}, mesh={tp_mesh}, nh={valh._TP_NH}")


    # jax.tree_util.tree_map(lambda x: x.shape,state)
    _step = 0 if run_args.checkpoint_step is None else run_args.checkpoint_step
    try:
        ckpt = load_checkpoint(
            new_train_state,
            ckpt_path,
            step=_step,
            train=False,
            partial_restore=True,
        )
    except (ValueError, TypeError) as _ckpt_err:
        # opt_state tree structure may differ between optax versions.
        # For inference we only need params — read directly via TensorStore.
        print(f"[load_checkpoint] StandardRestore failed: {_ckpt_err}")
        print("[load_checkpoint] TensorStore direct-read fallback (params only)...")
        import json, ast
        import numpy as _onp
        import tensorstore as ts
        import orbax.checkpoint as ocp
        from lob.init_train import deduplicate_trainstate

        _abs_path = os.path.abspath(ckpt_path)
        _mngr = ocp.CheckpointManager(
            _abs_path, item_names=('state', 'metadata'),
            options=ocp.CheckpointManagerOptions(),
        )
        if _step == 0:
            _step = _mngr.latest_step()
        _meta = _mngr.restore(_step, args=ocp.args.Composite(
            metadata=ocp.args.JsonRestore()))['metadata']

        _state_dir = os.path.join(_abs_path, str(_step), 'state')
        _mj = json.loads(open(os.path.join(_state_dir, '_METADATA')).read())
        _use_ocdbt = (os.path.exists(os.path.join(_state_dir, 'ocdbt.base_path'))
                      or os.path.exists(os.path.join(_state_dir, 'manifest.ocdbt')))
        _use_zarr3 = _mj.get('use_zarr3', False)
        _flat = {tuple(ast.literal_eval(k)): v
                 for k, v in _mj['tree_metadata'].items()}

        print(f"  Reading {sum(1 for k in _flat if k[0]=='params')} arrays "
              f"(OCDBT={_use_ocdbt})")

        def _make_tspec(state_dir, name, use_ocdbt, use_zarr3):
            """Build a TensorStore spec for a single array (orbax-version-agnostic)."""
            if use_ocdbt:
                return {'driver': 'zarr3' if use_zarr3 else 'zarr',
                        'kvstore': {'driver': 'ocdbt',
                                    'base': f'file://{state_dir}',
                                    'path': name}}
            else:
                return {'driver': 'zarr3' if use_zarr3 else 'zarr',
                        'kvstore': {'driver': 'file',
                                    'path': os.path.join(state_dir, name)}}

        _raw = {}
        for kp in _flat:
            if kp[0] != 'params':
                continue
            _tspec = _make_tspec(_state_dir, '.'.join(kp), _use_ocdbt, _use_zarr3)
            _raw[kp[1:]] = _onp.asarray(
                ts.open(_tspec, open=True).result().read().result())

        _params = {}
        for kp, arr in _raw.items():
            d = _params
            for key in kp[:-1]:
                d = d.setdefault(key, {})
            d[kp[-1]] = arr
        print(f"  Loaded {len(_raw)} param arrays")

        _dedup = deduplicate_trainstate(new_train_state)
        ckpt = _meta
        ckpt['model'] = _dedup.replace(params=_params)

    state = ckpt['model']

    if 'message_encoder' in state.params:
        print(state.params['message_encoder']['encoder']['embedding'].shape)
    else:
        print(f"[1tok model] field_embedding keys: {list(state.params.get('field_embedding', {}).keys())}")


    import chex
    chex.clear_trace_counter()

    model = model_cls(training=False, step_rescale=1.0)

    ##################################################

    import lob.evaluation as eval

    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    ds = inference.get_dataset(data_dir,
                               n_messages_conditional,
                               n_eval_messages,
                               test_split= run_args.test_split,
                               wide_book_dir=run_args.wide_book_dir,
                            #    day_indeces= [0],
                            #    limit_seq=4
                               )

    print("Dataset length: ", len(ds))

    ##################################################

    import logging
    # logging.basicConfig(filename='ar_debug.log', level=logging.DEBUG)
    _debug_log_path = os.environ.get('GENERATION_DEBUG_LOG', f'/tmp/generation_debug_{os.getpid()}.log')
    fhandler = logging.FileHandler(filename=_debug_log_path, mode='w')
    logger = logging.getLogger()
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(fhandler)
    logger.setLevel(logging.WARNING)
    # logger.setLevel(logging.DEBUG)

    ##################################################

    # Compute this rank's dataset indices
    rank = run_args.rank
    world_size = run_args.world_size
    n_total = run_args.n_sequences

    if run_args.sample_indices_file is not None:
        # HF-matched mode: read indices from file
        with open(run_args.sample_indices_file, 'r') as f:
            all_indices = [int(line.strip()) for line in f if line.strip()]
        n_total = len(all_indices)
    else:
        # Random mode: generate full index set with shared seed 42
        rng_idx = jax.random.key(42)
        all_indices = jax.random.choice(
            rng_idx,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_total,),
            replace=False
        ).tolist()

    # Interleaved split: rank takes every world_size-th index starting from rank
    rank_indices = all_indices[rank::world_size]
    n_samples = len(rank_indices)

    print(f"[Rank {rank}/{world_size}] Processing {n_samples} sequences out of {n_total} total")
    print(f"[Rank {rank}/{world_size}] GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"[Rank {rank}/{world_size}] First indices: {rank_indices[:5]}")

    batch_size = run_args.batch_size

    # Pad rank_indices to be divisible by batch_size
    if n_samples % batch_size != 0:
        n_padded = ((n_samples + batch_size - 1) // batch_size) * batch_size
        n_pad = n_padded - n_samples
        # Tile indices to cover padding (handles n_pad > n_samples, e.g. many ranks / small dataset)
        rank_indices = (rank_indices * (n_padded // n_samples + 1))[:n_padded]
        print(f"[Rank {rank}/{world_size}] Padded {n_pad} indices to fill last batch ({n_padded} total)")

    # m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = inference.sample_new(
    # saves data to disk
    start=time()
    inference.sample_new(
        len(rank_indices),
        batch_size,
        ds,
        rng,
        cond_seq_len,
        n_messages_conditional,
        n_gen_msgs,
        state,
        model,
        args.batchnorm,
        v.ENCODING,
        run_args.stock,
        save_folder=save_dir,
        sample_top_n= sample_top_n,
        args=args,
        conditional= True if n_messages_conditional>0 else False,
        overfit_debug=overfit_debug,
        sample_indices=rank_indices,
        wide_levels=run_args.wide_levels,
        token_mode=token_mode,
        gt_compare=run_args.gt_compare,
    )
    print(f"[Rank {rank}/{world_size}] Generation time for {n_samples} sequences across {batch_size} batch size: {time()-start}")
