from __future__ import annotations
import inspect
import json
import os
from argparse import Namespace
from glob import glob
from functools import partial
from typing import Any, Optional, Tuple, Union
import jax
import jax.numpy as np
from jax import random
import flax
from flax import jax_utils
import orbax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from jax.scipy.linalg import block_diag
from flax.training import checkpoints
from flax import linen as nn
from orbax import checkpoint
from lob.encoding import Vocab
from lob.lob_seq_model import BatchFullLobPredModel, BatchLobPredModel, BatchPaddedLobPredModel,OldBatchPaddedLobPredModel, FullLobPredModel#, ParFullLobPredModel

#from lob.lob_seq_model import BatchLobPredModel
from lob.train_helpers import create_train_state, create_lobs5_learning_rate_schedule, LR_MIN_FRACTION
from s5.gdn import init_GDN_SSM
# from s5.dataloading import make_data_loader
# from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import lob.validation_helpers as valh


def deduplicate_trainstate(
        state: TrainState,
        on_host: bool = False,
    ) -> TrainState:
    """Extract a single copy of state for checkpointing.
    Moves state to host first (safe for any sharding topology).
    If on_host=False (default), places on local GPU 0.
    If on_host=True, returns host numpy to avoid GPU memory pressure
    during restore (Orbax StandardRestore accepts numpy templates).
    """
    host_state = jax.device_get(state)
    if on_host:
        return host_state
    return jax.device_put(host_state, device=jax.local_devices()[0])


def remap_train_state_step(state: TrainState, new_step: int) -> TrainState:
    """Remap state.step and optimizer schedule counts for elastic resume.

    When resuming with a different number of nodes (different global BSZ),
    state.step and optimizer counts must be remapped so the LR schedule
    position matches the correct epoch in the new schedule.

    All optimizer count fields (ScaleByAdamState.count, ScaleByScheduleState.count)
    track the same global step. We replace all scalar int counts matching old_step
    with new_step. Adam bias correction (1/(1-beta^count)) converges for count>100,
    so remapping doesn't affect convergence — only schedule alignment matters.
    """
    import numpy as real_np  # init_train.py aliases jax.numpy as np
    old_step = int(state.step)
    new_step_val = real_np.int32(new_step)

    def remap_leaf(leaf):
        if hasattr(leaf, 'shape') and leaf.shape == () and hasattr(leaf, 'dtype'):
            if leaf.dtype in (real_np.int32, real_np.int64) and int(leaf) == old_step:
                return real_np.int32(new_step)
        return leaf

    new_opt_state = jax.tree_util.tree_map(remap_leaf, state.opt_state)
    return state.replace(step=new_step_val, opt_state=new_opt_state)


def load_args_from_checkpoint(
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> Namespace:

    """Load arguments from checkpoint"""
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    raw_restored = checkpoints.restore_checkpoint(
        checkpoint_path,
        None,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    args = Namespace(**raw_restored['config'])
    return args

def save_checkpoint(
        ckpt_mgr: ocp.CheckpointManager,
        ckpt: dict,
        step: int,
    ) -> bool:
    """Save checkpoint keyed by step (global_step for mid-epoch, or epoch for epoch-end)."""
    return ckpt_mgr.save(
        step,
        # args=ocp.args.PyTreeSave(ckpt)
        args=ocp.args.Composite(
            # train state
            state=ocp.args.StandardSave(ckpt['model']),
            # all other dict elements
            metadata=ocp.args.JsonSave({k: v for k, v in ckpt.items() if k != 'model'}),
        )
    )


# def load_checkpoint(
#         state: TrainState,
#         path: str,
#         config_dict: dict,
#         step: Optional[int] = None,
#     ) -> TrainState:
#     ckpt = {
#         'model': state,
#         'config': config_dict,
#         'metrics': {
#             'loss_train': np.nan,
#             'loss_val': np.nan,
#             'loss_test': np.nan,
#             'acc_val': np.nan,
#             'acc_test': np.nan,
#         }
#     }
#     orbax_checkpointer = checkpoint.PyTreeCheckpointer()
#     restored = checkpoints.restore_checkpoint(
#         path,
#         ckpt,
#         step=step,
#         orbax_checkpointer=orbax_checkpointer
#     )
#     return restored

def load_metadata(
        path: str,
    ) -> Namespace:

    json_path = path + '/metadata/_ROOT_METADATA'
    # load json path to dict
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    # Extract the actual parameters from the nested custom_metadata structure
    if 'custom_metadata' in metadata:
        return Namespace(**metadata['custom_metadata'])
    elif 'custom' in metadata:
        return Namespace(**metadata['custom'])
    else:
        return Namespace(**metadata)

def load_checkpoint(
        state: TrainState,
        path: str,
        # config_dict: dict,
        step: Optional[int] = None,
        train: bool = True,
        mesh=None,
        partial_restore: bool = False,
    ) -> dict[str, Any]:

    mngr = ocp.CheckpointManager(
        os.path.abspath(path),
        item_names=('state', 'metadata'),
        options=ocp.CheckpointManagerOptions(),
        # metadata=ckpt['config']
    )

    if step is None:
        step = mngr.latest_step()

    print(f"[Checkpoint] Loading step={step} from {path} "
          f"(partial_restore={partial_restore})")

    restore_state = deduplicate_trainstate(state, on_host=True)

    try:
        _sr_kwargs = dict(item=restore_state)
        if 'strict' in inspect.signature(ocp.args.StandardRestore).parameters:
            _sr_kwargs['strict'] = not partial_restore
        loaded = mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(**_sr_kwargs),
                metadata=ocp.args.JsonRestore()
            )
        )
    except (ValueError, TypeError, FileNotFoundError) as e:
        if not train:
            # opt_state tree structure may differ between Orbax/optax versions.
            # For inference we only need params — bypass CheckpointManager and
            # restore the state directory directly as a raw dict.
            print(f"[load_checkpoint] StandardRestore failed ({e}), "
                  "falling back to direct restore for inference")

            import numpy as onp
            import tensorstore as ts
            from orbax.checkpoint import type_handlers as _th
            from orbax.checkpoint import utils as _ocp_utils
            from etils import epath as _epath

            # Restore metadata (config dict) via manager — this always works
            meta_loaded = mngr.restore(
                step,
                args=ocp.args.Composite(
                    metadata=ocp.args.JsonRestore()
                )
            )

            state_dir = os.path.join(os.path.abspath(path), str(step), 'state')
            state_dir_ep = _epath.Path(state_dir)
            is_ocdbt = _th.is_ocdbt_checkpoint(state_dir_ep)

            _meta_json = json.loads((state_dir_ep / '_METADATA').read_text())
            _use_zarr3 = _meta_json.get('use_zarr3', False)

            import ast
            _tree_md = _meta_json['tree_metadata']
            flat_abstract = {}
            for key_str, entry in _tree_md.items():
                keypath = tuple(ast.literal_eval(key_str))
                flat_abstract[keypath] = entry

            # Only read 'params' subtree (skip opt_state for inference)
            print(f"[load_checkpoint] Reading {sum(1 for k in flat_abstract if k[0] == 'params')} "
                  f"param arrays via TensorStore (OCDBT={is_ocdbt})")
            raw_params = {}
            _ts_ctx = _th.get_ts_context(use_ocdbt=is_ocdbt)
            for keypath, meta in flat_abstract.items():
                if keypath[0] != 'params':
                    continue
                param_name = '.'.join(keypath)
                _zarr_driver = 'zarr3' if _use_zarr3 else 'zarr'
                if is_ocdbt:
                    tspec = {
                        'driver': _zarr_driver,
                        'kvstore': {
                            'driver': 'ocdbt',
                            'base': f"file://{state_dir}",
                            'path': param_name,
                        },
                    }
                else:
                    tspec = {
                        'driver': _zarr_driver,
                        'kvstore': {
                            'driver': 'file',
                            'path': os.path.join(str(state_dir), param_name),
                        },
                    }
                t = ts.open(
                    ts.Spec(tspec), open=True, context=_ts_ctx
                ).result()
                raw_params[keypath[1:]] = onp.asarray(t.read().result())

            # Rebuild nested params dict from flat
            params = {}
            for keypath, arr in raw_params.items():
                d = params
                for key in keypath[:-1]:
                    d = d.setdefault(key, {})
                d[keypath[-1]] = arr

            print(f"[load_checkpoint] Loaded {len(raw_params)} param arrays")

            restored = restore_state.replace(params=params)

            # Also load batch_stats if present
            batch_stats_keys = [k for k in flat_abstract if k[0] == 'batch_stats']
            if batch_stats_keys:
                raw_bs = {}
                for keypath in batch_stats_keys:
                    param_name = '.'.join(keypath)
                    _zarr_driver = 'zarr3' if _use_zarr3 else 'zarr'
                    if is_ocdbt:
                        tspec = {
                            'driver': _zarr_driver,
                            'kvstore': {
                                'driver': 'ocdbt',
                                'base': f"file://{state_dir}",
                                'path': param_name,
                            },
                        }
                    else:
                        tspec = {
                            'driver': _zarr_driver,
                            'kvstore': {
                                'driver': 'file',
                                'path': os.path.join(str(state_dir), param_name),
                            },
                        }
                    t = ts.open(
                        ts.Spec(tspec), open=True, context=_ts_ctx
                    ).result()
                    raw_bs[keypath[1:]] = onp.asarray(t.read().result())
                batch_stats = {}
                for keypath, arr in raw_bs.items():
                    d = batch_stats
                    for key in keypath[:-1]:
                        d = d.setdefault(key, {})
                    d[keypath[-1]] = arr
                restored = restored.replace(batch_stats=batch_stats)

            loaded = {'state': restored, 'metadata': meta_loaded['metadata']}
        else:
            raise

    ckpt = loaded['metadata']
    # copy train state back to all devices
    if train:
        if mesh is not None:
            # Orbax restores to single device; move to numpy (device-agnostic)
            # then let caller shard to global mesh
            host_state = jax.device_get(loaded['state'])
            ckpt['model'] = host_state
        else:
            ckpt['model'] = jax_utils.replicate(loaded['state'])
    else:
        ckpt['model'] = loaded['state']
    return ckpt


def init_train_state(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        train_size: int = 0,
        print_shapes=False
    ) -> Tuple[TrainState, Union[partial[BatchLobPredModel],
                                  partial[BatchFullLobPredModel],
                                  partial[BatchPaddedLobPredModel],
                                  partial[OldBatchPaddedLobPredModel]]]:

    in_dim = n_classes

    ssm_lr = args.ssm_lr_base

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    padded = False
    retrieval = False

    # SSM type dispatch
    ssm_type = getattr(args, 'ssm_type', 'gdn')

    if ssm_type == 'mamba3':
        from s5.mamba3 import init_Mamba3SSM
        m3_d_state = getattr(args, 'mamba3_d_state', 128)
        m3_expand = getattr(args, 'mamba3_expand', 2)
        m3_headdim = getattr(args, 'mamba3_headdim', 64)
        m3_chunk = getattr(args, 'mamba3_chunk_size', 64)
        m3_rope_frac = getattr(args, 'mamba3_rope_fraction', 0.5)
        m3_use_triton = getattr(args, 'mamba3_use_triton', False)
        m3_use_cuda = getattr(args, 'mamba3_use_cuda', False)
        m3_tp_size = getattr(args, 'tp_size', 1)

        if print_shapes:
            _remat = os.environ.get('REMAT', '0') == '1'
            print(f"[Mamba3] d_state={m3_d_state}, expand={m3_expand}, "
                  f"headdim={m3_headdim}, chunk_size={m3_chunk}, "
                  f"rope_fraction={m3_rope_frac}, use_triton={m3_use_triton}, "
                  f"use_cuda={m3_use_cuda}")
            print(f"[Remat] {'ENABLED' if _remat else 'disabled'} "
                  f"(policy={os.environ.get('REMAT_POLICY', 'none')})")
            if m3_tp_size > 1:
                d_inner = m3_expand * args.d_model
                nh = d_inner // m3_headdim
                print(f"[TP] tp_size={m3_tp_size}, nh_local={nh // m3_tp_size} "
                      f"(total nh={nh})")
            print("book_seq_len", book_seq_len)
            print("book_dim", book_dim)

        ssm_init_fn = init_Mamba3SSM(
            H=args.d_model,
            d_state=m3_d_state,
            expand=m3_expand,
            headdim=m3_headdim,
            chunk_size=m3_chunk,
            rope_fraction=m3_rope_frac,
            use_triton=m3_use_triton,
            use_cuda=m3_use_cuda,
            tp_size=m3_tp_size,
        )
    else:
        # GDN initialization (default)
        gdn_head_dim = getattr(args, 'gdn_head_dim', 128)
        gdn_num_heads = getattr(args, 'gdn_num_heads', None) or max(1, args.d_model // gdn_head_dim)
        gdn_expand_v = getattr(args, 'gdn_expand_v', 2)
        gdn_chunk_size = getattr(args, 'gdn_chunk_size', 64)
        gdn_use_conv = getattr(args, 'gdn_use_conv', True)

        if print_shapes:
            print(f"[GDN] num_heads={gdn_num_heads}, "
                  f"head_dim={gdn_head_dim}, expand_v={gdn_expand_v}, "
                  f"chunk_size={gdn_chunk_size}, use_conv={gdn_use_conv}")
            print("book_seq_len", book_seq_len)
            print("book_dim", book_dim)

        ssm_init_fn = init_GDN_SSM(
            H=args.d_model,
            num_heads=gdn_num_heads,
            head_dim=gdn_head_dim,
            expand_v=gdn_expand_v,
            chunk_size=gdn_chunk_size,
            use_conv=gdn_use_conv,
            use_kda=False,
        )
    
    token_mode = getattr(args, 'token_mode', '24tok')

    if token_mode == '1tok' and args.use_book_data:
        from lob.lob_seq_model import BatchOneTokenPaddedLobPredModel
        model_cls = partial(
            BatchOneTokenPaddedLobPredModel,
            ssm=ssm_init_fn,
            field_vocab_sizes=FIELD_VOCAB_SIZES_WITH_SPECIAL,
            d_model=args.d_model,
            d_book=book_dim,
            n_fused_layers=args.n_layers,
            n_book_pre_layers=args.n_book_pre_layers,
            n_book_post_layers=args.n_book_post_layers,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
        padded = False

    elif args.use_book_data:
        # if args.num_devices > 1:
        #     model_cls = ParFullLobPredModel
        # else:
        #     model_cls = BatchFullLobPredModel


        if args.merging == 'projected':
            model_cls = partial(
                # projecting sequence lengths down has appeared better than padding
                BatchFullLobPredModel,
                #BatchPaddedLobPredModel,
                #model_cls,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,  # 2
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )
        elif args.merging == 'padded': #i.e. 'padded'
            model_cls = partial(
                # projecting sequence lengths down has appeared better than padding
                BatchPaddedLobPredModel,
                #model_cls,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                d_book=book_dim,
                n_message_layers=args.n_message_layers,  # 2
                n_fused_layers=args.n_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
                #args not adding to partial: training & rescale.
            )
        else:
            raise ValueError("Merge method: " + args.merging + " is not valid (check spelling)")

    else:
        if getattr(args, 'hierarchical_nobook', False):
            # Hierarchical no-book: msg_encoder + Dense(d->d) + fused SSM layers
            if getattr(args, 'no_book_inference_wrapper', False):
                from lob.lob_seq_model import BatchHierarchicalNoBookInferenceWrapper
                model_cls = partial(
                    BatchHierarchicalNoBookInferenceWrapper,
                    ssm=ssm_init_fn,
                    d_output=n_classes,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    n_message_layers=getattr(args, 'n_message_layers', 2),
                    activation=args.activation_fn,
                    dropout=args.p_dropout,
                    prenorm=args.prenorm,
                    batchnorm=args.batchnorm,
                    bn_momentum=args.bn_momentum,
                )
            else:
                from lob.lob_seq_model import BatchHierarchicalLobPredModel
                model_cls = partial(
                    BatchHierarchicalLobPredModel,
                    ssm=ssm_init_fn,
                    d_output=n_classes,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    n_message_layers=getattr(args, 'n_message_layers', 2),
                    activation=args.activation_fn,
                    dropout=args.p_dropout,
                    prenorm=args.prenorm,
                    batchnorm=args.batchnorm,
                    bn_momentum=args.bn_momentum,
                )
        else:
            if args.num_devices > 1:
                raise NotImplementedError("Message only model not implemented for multi-device training")

            model_cls = partial(
                BatchLobPredModel,
                ssm=ssm_init_fn,
                d_output=n_classes,
                d_model=args.d_model,
                n_layers=args.n_layers,
                padded=padded,
                activation=args.activation_fn,
                dropout=args.p_dropout,
                mode=args.mode,
                prenorm=args.prenorm,
                batchnorm=args.batchnorm,
                bn_momentum=args.bn_momentum,
            )

    # Create learning rate schedules if train_size is available
    ssm_lr_schedule = None
    lr_schedule = None
    if train_size > 0:
        process_count = getattr(args, 'process_count', jax.process_count())
        grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        # args.micro_bsz is per-GPU BSZ; global BSZ = micro_bsz * num_devices * process_count
        micro_steps_per_epoch = train_size // (args.micro_bsz * args.num_devices * process_count)
        if hasattr(args, 'curtail_epochs') and args.curtail_epochs is not None:
            micro_steps_per_epoch = min(micro_steps_per_epoch, args.curtail_epochs + 1)
        # steps_per_epoch in optimizer updates (= micro_steps // K)
        steps_per_epoch = micro_steps_per_epoch // grad_accum_steps
        total_steps = steps_per_epoch * args.epochs
        warmup_end_step = int(steps_per_epoch * args.warmup_end)

        effective_lr_min = args.lr_min if args.lr_min > 0 else lr * LR_MIN_FRACTION
        effective_ssm_lr_min = args.lr_min if args.lr_min > 0 else ssm_lr * LR_MIN_FRACTION

        if print_shapes:
            print(f"[Schedule] steps_per_epoch: {steps_per_epoch}")
            print(f"[Schedule] total_steps: {total_steps}")
            print(f"[Schedule] warmup_end_step: {warmup_end_step}")
            print(f"[Schedule] Base SSM LR: {ssm_lr}, Base LR: {lr}")
            print(f"[Schedule] lr_min: {effective_lr_min}, ssm_lr_min: {effective_ssm_lr_min}")

        ssm_lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=ssm_lr,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=effective_ssm_lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )
        lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=lr,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=effective_lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )

    # Create Muon kernel LR schedule if using Muon optimizer
    muon_lr_val = getattr(args, 'muon_lr', 0.02)
    muon_wd_val = getattr(args, 'muon_wd', None)
    muon_lr_schedule = None
    if args.opt_config == 'muon' and train_size > 0:
        effective_muon_lr_min = muon_lr_val * LR_MIN_FRACTION
        muon_lr_schedule = create_lobs5_learning_rate_schedule(
            base_lr=muon_lr_val,
            warmup_end_step=warmup_end_step,
            total_steps=total_steps,
            lr_min=effective_muon_lr_min,
            use_cosine_anneal=args.cosine_anneal,
        )
        if print_shapes:
            print(f"[Schedule] Muon kernel LR: {muon_lr_val}, min: {effective_muon_lr_min}")

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=args.use_book_data,
        in_dim=1, # in_dim,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        micro_bsz=args.micro_bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        no_book_inference_wrapper=getattr(args, 'no_book_inference_wrapper', False),
        lr=lr,
        ssm_lr_schedule=ssm_lr_schedule,
        lr_schedule=lr_schedule,
        muon_lr=muon_lr_val,
        muon_wd=muon_wd_val,
        muon_lr_schedule=muon_lr_schedule,
        dt_global=args.dt_global,
        num_devices=args.num_devices,
        model_type="gdn",
        token_mode=token_mode,
    )

    return state, model_cls
