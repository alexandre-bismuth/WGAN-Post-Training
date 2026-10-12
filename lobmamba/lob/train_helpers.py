from functools import partial
import math
import os
import numpy as onp
import jax
import jax.numpy as np
# from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import flax
import optax
from typing import Any, Dict, Optional, Tuple, Union
from lob.encoding import Message_Tokenizer
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import sys
import time
import threading
import signal

import psutil
import os

# Default lr_min as fraction of base LR (when args.lr_min=0).
# 5% matches Chinchilla convention for cosine decay floor.
LR_MIN_FRACTION = 0.05

# Step-level timeout for NCCL hang detection (seconds).
# Normal step ~1s; if D2H transfer exceeds this, likely NCCL deadlock.
STEP_TIMEOUT = int(os.environ.get('STEP_TIMEOUT', '300'))

# Watchdog timeout per training step (seconds). Daemon thread fires os._exit if exceeded.
WATCHDOG_TIMEOUT = int(os.environ.get('WATCHDOG_TIMEOUT', '900'))  # 15 min

# Grace period: first N steps use WATCHDOG_WARMUP_TIMEOUT instead of WATCHDOG_TIMEOUT.
# XLA autotuning + NCCL channel init are one-time costs that make early steps 100x slower.
# 32N: step 0-1 can take >120s each (JIT compile + autotune), causing false watchdog kills.
# After ~50 steps the autotune cache is warm and steps drop to ~1s.
WATCHDOG_WARMUP_STEPS = int(os.environ.get('WATCHDOG_WARMUP_STEPS', '50'))
WATCHDOG_WARMUP_TIMEOUT = int(os.environ.get('WATCHDOG_WARMUP_TIMEOUT', '900'))  # 15 min


class StepWatchdog:
    """Per-step watchdog using threading.Timer.

    Call kick() at the start of each step and stop() at the end of each epoch.
    If kick() is not called again within `timeout` seconds, the process is
    killed with os._exit(1) so SLURM can clean up and the job can be restarted
    from the last checkpoint.

    Unlike the D2H-based watchdog (which itself hangs when NCCL is stuck),
    this runs on a daemon thread that is independent of GPU operations.

    Warmup: the first WATCHDOG_WARMUP_STEPS steps use a longer timeout
    (WATCHDOG_WARMUP_TIMEOUT) because XLA autotuning + NCCL channel init
    make early steps 100x slower than steady state.
    """
    def __init__(self, timeout=WATCHDOG_TIMEOUT,
                 warmup_timeout=WATCHDOG_WARMUP_TIMEOUT,
                 warmup_steps=WATCHDOG_WARMUP_STEPS):
        self.timeout = timeout
        self.warmup_timeout = warmup_timeout
        self.warmup_steps = warmup_steps
        self._timer = None
        self._total_steps = 0

    def kick(self, epoch, batch_idx):
        if self._timer:
            self._timer.cancel()
        effective_timeout = (self.warmup_timeout
                             if self._total_steps < self.warmup_steps
                             else self.timeout)
        self._total_steps += 1
        self._timer = threading.Timer(
            effective_timeout, self._abort, args=(epoch, batch_idx))
        self._timer.daemon = True
        self._timer.start()

    def stop(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _abort(self, epoch, batch_idx):
        msg = (f"FATAL: Step watchdog timeout ({self.timeout}s) "
               f"at epoch {epoch} batch {batch_idx}. Likely NCCL deadlock.")
        print(msg, flush=True)
        # CAVEAT: SIGTERM is caught by JAX's preemption_notifier (preemption_notifier.cc)
        # which treats it as graceful shutdown — the process never actually exits.
        # os._exit(1) bypasses all signal handlers and atexit hooks to guarantee termination.
        os._exit(1)
# from lob.lob_seq_model import LobPredModel


# Absolute-time token positions (time_s + time_ns), taken from the ACTIVE encoding — 26tok:
# [evt:0, dir:1, price:2-4, size:5-6, dt_s:7, dt_ns:8-10, time_s:11-12, time_ns:13-15, ref:16-25]
# -> TIME_START_I=11, TIME_END_I=15. CONVENTION: Message_Tokenizer.TIME_START_I/TIME_END_I are
# INCLUSIVE bounds (the ignore_times CE/accuracy paths below use [(TIME_END_I + 1):]), whereas
# validation_helpers.get_idx_from_field returns EXCLUSIVE ends (time_ns -> (13, 16)). Mixing the
# two conventions produces off-by-one masks.
TIME_START_I = Message_Tokenizer.TIME_START_I
TIME_END_I = Message_Tokenizer.TIME_END_I

# num_devices_global = 2
# global_devices = jax.local_devices()[0: num_devices_global]


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


# ==============================================================================
# Learning Rate Schedule Creation (MaxText-style optax schedules)
# ==============================================================================

def create_lobs5_learning_rate_schedule(
    base_lr: float,
    warmup_end_step: int,
    total_steps: int,
    lr_min: float = 0.0,
    use_cosine_anneal: bool = True,
) -> optax.Schedule:
    """
    Creates an optax Schedule: warmup -> cosine decay (or constant).
    Passed directly to the optimizer, eliminating manual per-step LR updates.
    """
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_end_step
    )

    if use_cosine_anneal:
        cosine_steps = total_steps - warmup_end_step

        def make_cos_schedule(init_lr, final_lr, len_steps):
            """Custom cosine schedule matching LOBS5's original cosine_annealing."""
            len_steps = max(len_steps, 1)  # Guard: CURTAIL_EPOCHS can make cosine_steps=0
            def schedule(step):
                pct = step / len_steps
                pct = np.minimum(pct, 1.0)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * pct))
                lr = (init_lr - final_lr) * cosine_decay + final_lr
                return lr
            return schedule

        cosine_schedule = make_cos_schedule(base_lr, lr_min, cosine_steps)
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_end_step]
        )
    else:
        constant_schedule = optax.constant_schedule(base_lr)
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule],
            boundaries=[warmup_end_step]
        )

    return schedule


def update_learning_rate_per_step(lr_params, state, mesh=None):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # # Update state
    # state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = \
    #     jax_utils.replicate(np.array(lr_val, dtype=np.float32))

    # state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']= \
    #     jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))

    # if opt_config in ["BandCdecay"]:
    #     # In this case we are applying the ssm learning rate to B, even though
    #     # we are also using weight decay on B
    #     state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = \
    #         jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))
    # Multi-host: use globally-replicated JAX arrays to match train_step's
    # in_shardings. numpy arrays become SingleDeviceSharding which causes
    # NCCL deadlock when train_step tries to re-shard across hosts.
    if mesh is not None:
        from jax.sharding import NamedSharding, PartitionSpec as P
        replicated = NamedSharding(mesh, P())
        lr_array = jax.make_array_from_process_local_data(
            replicated, np.array(lr_val, dtype=np.float32))
        ssm_lr_array = jax.make_array_from_process_local_data(
            replicated, np.array(ssm_lr_val, dtype=np.float32))
    else:
        lr_array = np.array(lr_val, dtype=np.float32)
        ssm_lr_array = np.array(ssm_lr_val, dtype=np.float32)
    
    # Update in place by creating new state with updated hyperparams
    # This avoids accumulating replicated tensors while preserving other hyperparameters
    state = state.replace(
        opt_state=state.opt_state._replace(
            inner_states={
                **state.opt_state.inner_states,
                'regular': state.opt_state.inner_states['regular']._replace(
                    inner_state=state.opt_state.inner_states['regular'].inner_state._replace(
                        hyperparams={
                            **state.opt_state.inner_states['regular'].inner_state.hyperparams,
                            'learning_rate': lr_array
                        }
                    )
                ),
                'ssm': state.opt_state.inner_states['ssm']._replace(
                    inner_state=state.opt_state.inner_states['ssm'].inner_state._replace(
                        hyperparams={
                            **state.opt_state.inner_states['ssm'].inner_state.hyperparams,
                            'learning_rate': ssm_lr_array
                        }
                    )
                ),
            }
        )
    )

    if opt_config in ["BandCdecay"]:
        state = state.replace(
            opt_state=state.opt_state._replace(
                inner_states={
                    **state.opt_state.inner_states,
                    'none': state.opt_state.inner_states['none']._replace(
                        inner_state=state.opt_state.inner_states['none'].inner_state._replace(
                            hyperparams={
                                **state.opt_state.inner_states['none'].inner_state.hyperparams,
                                'learning_rate': ssm_lr_array
                            }
                        )
                    ),
                }
            )
        )
    return state, step


_LOG_GRAD_NORMS = bool(int(os.environ.get('LOG_GRAD_NORMS', '0')))

_SSM_KEYS = frozenset({"B", "Lambda_re", "Lambda_im", "log_step", "norm",
                        "B_bias", "C_bias", "dt_bias", "D"})


def _compute_grad_norms(grads):
    """Per-group gradient L2 norms for Muon/SSM/regular parameter groups.

    Returns np.array of shape (6,):
        [global, muon, ssm, regular, in_proj, out_proj]

    Uses tree_map_with_path so predicates resolve at trace time;
    the sum-of-squares arithmetic is traced into the XLA graph.
    """
    def _sq_norm_for(predicate):
        def _leaf_fn(path, leaf):
            keys = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
            name = keys[-1] if keys else ""
            return np.sum(leaf ** 2) if predicate(keys, name) else np.float32(0.0)
        sq_tree = jax.tree_util.tree_map_with_path(_leaf_fn, grads)
        return np.sqrt(sum(jax.tree_util.tree_leaves(sq_tree)))

    return np.array([
        _sq_norm_for(lambda k, n: True),
        _sq_norm_for(lambda k, n: n == "kernel"),
        _sq_norm_for(lambda k, n: n in _SSM_KEYS),
        _sq_norm_for(lambda k, n: n != "kernel" and n not in _SSM_KEYS),
        _sq_norm_for(lambda k, n: n == "kernel" and any("in_proj" in p for p in k)),
        _sq_norm_for(lambda k, n: n == "kernel" and any("out_proj" in p for p in k)),
    ])


@jax.tree_util.register_pytree_node_class
class DiLoCoState:
    """Wrapper around flax TrainState with an outer-loop Nesterov-momentum buffer.

    Used when `diloco_outer='nesterov'` to implement DiLoCo (Douillard et al. 2024):
    each replica runs K local inner steps, then an outer step updates params with
    pseudo-gradient (theta_anchor - theta_local) via Nesterov momentum across nodes.
    Inner optimizer state (Muon/AdamW m/v) lives inside the embedded train_state
    and is carried per-node across outer boundaries (DiLoCo paper default).

    Registered as a JAX pytree (not flax.struct.dataclass) so we have full control
    over `replace(...)` — it transparently delegates inner TrainState fields
    (params, step, opt_state, batch_stats) so legacy code that does
    `state.replace(params=x)` or `state.replace(step=0)` keeps working.
    """

    __slots__ = ('train_state', 'outer_momentum')

    def __init__(self, train_state, outer_momentum):
        # Use object.__setattr__ because we present an immutable-ish facade.
        object.__setattr__(self, 'train_state', train_state)
        object.__setattr__(self, 'outer_momentum', outer_momentum)

    @property
    def params(self):
        return self.train_state.params

    @property
    def step(self):
        return self.train_state.step

    @property
    def opt_state(self):
        return self.train_state.opt_state

    @property
    def apply_fn(self):
        return self.train_state.apply_fn

    @property
    def batch_stats(self):
        return getattr(self.train_state, 'batch_stats', None)

    def apply_gradients(self, **kwargs):
        return DiLoCoState(
            train_state=self.train_state.apply_gradients(**kwargs),
            outer_momentum=self.outer_momentum,
        )

    def replace(self, **kwargs):
        """Split kwargs: DiLoCoState-level fields vs inner TrainState fields."""
        _self_fields = {'train_state', 'outer_momentum'}
        inner_kwargs = {k: v for k, v in kwargs.items() if k not in _self_fields}
        self_kwargs = {k: v for k, v in kwargs.items() if k in _self_fields}

        new_ts = (
            self_kwargs['train_state'] if 'train_state' in self_kwargs
            else (self.train_state.replace(**inner_kwargs) if inner_kwargs
                  else self.train_state)
        )
        new_outer_m = self_kwargs.get('outer_momentum', self.outer_momentum)
        return DiLoCoState(train_state=new_ts, outer_momentum=new_outer_m)

    # JAX pytree protocol
    def tree_flatten(self):
        children = (self.train_state, self.outer_momentum)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        train_state, outer_momentum = children
        return cls(train_state=train_state, outer_momentum=outer_momentum)

    def __repr__(self):
        return (f"DiLoCoState(train_state=<TrainState step={int(self.step)}>,"
                f" outer_momentum=<pytree>)")


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       use_book_data,
                       book_dim,
                       book_seq_len,
                       in_dim=1,
                       micro_bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       no_book_inference_wrapper=False,
                       lr=1e-3,
                       ssm_lr_schedule=None,
                       lr_schedule=None,
                       muon_lr=0.02,
                       muon_wd=None,
                       muon_lr_schedule=None,
                       dt_global=False,
                       num_devices=1,
                       model_type="s5",
                       token_mode="24tok",
                       ):
    """
    Initializes the training state using optax.

    When ssm_lr_schedule/lr_schedule are provided (optax.Schedule functions),
    they are passed directly to the optimizer — no inject_hyperparams needed.
    When None, falls back to inject_hyperparams with scalar LRs (legacy mode).
    """

    # micro_bsz is per-GPU batch size — used directly for dummy data shapes
    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*micro_bsz, seq_len, in_dim)), np.ones(2*micro_bsz))
            integration_timesteps = np.ones((2*micro_bsz, seq_len,))
        else:
            dummy_input = (np.ones((micro_bsz, seq_len, in_dim)), np.ones(micro_bsz))
            integration_timesteps = np.ones((micro_bsz, seq_len,))
    else:
        if use_book_data:
            if token_mode == '1tok':
                from lob.encoding_1tok import N_FIELDS
                dummy_input = (
                    np.ones((micro_bsz, seq_len, N_FIELDS), dtype=np.int32),  # (B, L, 24)
                    np.ones((micro_bsz, seq_len, book_dim)),  # books
                )
            else:
                dummy_input = (
                    np.ones((micro_bsz, seq_len, ), dtype=np.int32),  # messages
                    np.ones((micro_bsz, seq_len, book_dim)),  # books
                )
            integration_timesteps = (
                np.ones((micro_bsz, seq_len, )),
                np.ones((micro_bsz, seq_len, )),
            )
        else:
            if no_book_inference_wrapper:
                # Wrapper model expects 4-arg interface (book discarded internally)
                dummy_input = (
                    np.ones((micro_bsz, seq_len, ), dtype=np.int32),
                    np.ones((micro_bsz, seq_len, book_dim if book_dim > 0 else 503)),
                )
                integration_timesteps = (
                    np.ones((micro_bsz, seq_len, )),
                    np.ones((micro_bsz, seq_len, )),
                )
            elif token_mode == '1tok':
                from lob.encoding_1tok import N_FIELDS
                dummy_input = (np.ones((micro_bsz, seq_len, N_FIELDS), dtype=np.int32), )
                integration_timesteps = (np.ones((micro_bsz, seq_len, )), )
            else:
                dummy_input = (np.ones((micro_bsz, seq_len, ), dtype=np.int32) , )
                integration_timesteps = (np.ones((micro_bsz, seq_len, )), )

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    
    # jax.debug.print("Dummy input shapes (msg,book) ({}, \n {})",dummy_input[0].shape,dummy_input[1].shape)
    #RNN mode and initialisation needs to go in here if we need it. 

    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           *dummy_input, *integration_timesteps,
                           method='__call_ar__' 
                           )
    
    if batchnorm:
        params = variables["params"]#.unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"]#.unfreeze()
        # Note: `unfreeze()` is for using Optax.

    if 'message_encoder' in params:
        print(params['message_encoder']['encoder']['embedding'].shape)

    # Determine whether to use optax schedules (new) or inject_hyperparams (legacy)
    use_schedules = ssm_lr_schedule is not None and lr_schedule is not None
    if use_schedules:
        print("[Optimizer] Using optax schedules (LR managed inside JIT)")
        _ssm_lr = ssm_lr_schedule
        _lr = lr_schedule
    else:
        print("[Optimizer] Using inject_hyperparams (legacy scalar LR)")
        _ssm_lr = ssm_lr
        _lr = lr

    def _make_opt(optimizer_fn, learning_rate, **kwargs):
        """Create optimizer with or without inject_hyperparams."""
        if use_schedules:
            return optimizer_fn(learning_rate=learning_rate, **kwargs)
        else:
            return optax.inject_hyperparams(optimizer_fn)(learning_rate=learning_rate, **kwargs)

    if model_type == "transformer":
        # Transformer: all params are standard Dense/attention weights.
        # Single AdamW optimizer with weight decay — no SSM-specific routing.
        print("configuring transformer optimization (single AdamW, no SSM groups)")
        tx = _make_opt(optax.adamw, _lr, weight_decay=weight_decay)

    elif opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm",
                        "B_bias", "C_bias", "dt_bias", "D"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm",
                        "B_bias", "C_bias", "dt_bias", "D"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": _make_opt(optax.adamw, _ssm_lr, weight_decay=weight_decay),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.adamw(learning_rate=0.0, weight_decay=0.0),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["muon"]:
        """Muon optimizer for 2D kernel weights (Dense layers).
        SSM params use Adam (no weight decay), non-kernel params use AdamW.
        Only 'kernel' leaves are routed to the Muon Newton-Schulz transform.

        Three-tier routing:
          SSM params (B, Lambda_re, Lambda_im, log_step, norm) -> Adam (ssm_lr, no WD)
          2D kernel weights                                     -> Muon NS (muon_lr, muon_wd)
          Everything else (embedding, bias, head, etc.)         -> AdamW (lr, weight_decay)
        """
        _muon_wd = muon_wd if muon_wd is not None else weight_decay
        _muon_lr_sched = muon_lr_schedule if muon_lr_schedule is not None else muon_lr

        print(f"configuring Muon optimization (kernel -> NS, SSM -> Adam, rest -> AdamW)")
        print(f"  Muon kernel LR: {muon_lr}, WD: {_muon_wd}")
        print(f"  AdamW LR: {lr}, WD: {weight_decay}")
        print(f"  SSM LR: {ssm_lr}, WD: 0")

        # scale_by_muon requires explicit weight_dimension_numbers inside multi_transform
        # (default None causes jax.tree.map ValueError: "Expected dict, got None")
        _muon_dim_nums = optax.contrib.MuonDimensionNumbers(
            reduction_axis=0, output_axis=1)
        _wdn_fn = lambda p: jax.tree.map(lambda x: _muon_dim_nums, p)

        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm",
                        "B_bias", "C_bias", "dt_bias", "D"]
                else ("muon" if k == "kernel" else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm",
                        "B_bias", "C_bias", "dt_bias", "D"]
                else ("muon" if k == "kernel" else "regular")
            )

        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "ssm": _make_opt(optax.adam, _ssm_lr),
                "regular": _make_opt(optax.adamw, _lr, weight_decay=weight_decay),
                "muon": optax.chain(
                    optax.contrib.scale_by_muon(
                        nesterov=True,
                        weight_dimension_numbers=_wdn_fn,
                    ),
                    optax.add_decayed_weights(weight_decay=_muon_wd),
                    optax.scale_by_learning_rate(_muon_lr_sched),
                ),
            },
            ssm_fn,
        )

    # Wrap optimizer with gradient clipping to prevent catastrophic divergence.
    # Without clipping, a single outlier batch can cause loss spikes of 67-122x
    # (observed in Jobs 2439364/FP32, 2439874/BF16 at LR=3e-3).
    max_grad_norm = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
    if max_grad_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
        print(f"[*] Gradient clipping enabled: max_norm={max_grad_norm}")

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    #print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # jit+sharding: state replication handled in train.py via create_state_shardings
    if 'message_encoder' in state.params:
        print(f"[*] State params embedding shape: {state.params['message_encoder']['encoder']['embedding'].shape}")

    return state

def get_slices(dims):
    slices = []
    last_i = 0
    for d in dims:
        slices.append(slice(last_i, last_i+d))
        last_i += d
    return slices

# Train and eval steps
# @partial(np.vectorize, signature="(c),()->()")
# def cross_entropy_loss(logits, label):
#     one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[-1])
#     return -np.sum(one_hot_label * logits)

@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    return -np.sum(logits[label])


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss_test(logits, label):
    return -np.sum(logits)

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def _compute_ce_unified(logits, batch_labels, ignore_times):
    """Unified CE computation for both 26tok and 1tok modes.

    26tok: logits is (B, L, d_output), labels is (B, L)
    1tok:  logits is list of 24 (B, L, V_i), labels is (B, L, 24)

    Returns: ce of shape (B, flat_positions) compatible with existing validate().
    """
    if isinstance(logits, list):
        # 1tok mode: per-field CE
        field_ces = []
        for i in range(len(logits)):
            if ignore_times and 10 <= i <= 14:
                continue
            ce_i = cross_entropy_loss(logits[i], batch_labels[:, :, i])
            field_ces.append(ce_i)
        ce = np.stack(field_ces, axis=-1)  # (B, L, n_active_fields)
        return ce.reshape(ce.shape[0], -1)  # (B, L * n_active_fields)
    else:
        # 26tok mode: standard CE with ignore_times reshape
        ce = cross_entropy_loss(logits, batch_labels)
        if ignore_times:
            ce = ce.reshape(ce.shape[0], -1, Message_Tokenizer.MSG_LEN)
            ce_1 = ce[:, :, :Message_Tokenizer.TIME_START_I]
            ce_2 = ce[:, :, (Message_Tokenizer.TIME_END_I + 1):]
            ce = np.concatenate([ce_1, ce_2], axis=2)
            ce = ce.reshape(ce.shape[0], -1)
        return ce


def _compute_acc_unified(logits, batch_labels, ignore_times):
    """Unified accuracy computation for both 26tok and 1tok modes."""
    if isinstance(logits, list):
        field_accs = []
        for i in range(len(logits)):
            if ignore_times and 10 <= i <= 14:
                continue
            acc_i = compute_accuracy(logits[i], batch_labels[:, :, i])
            field_accs.append(acc_i)
        accs = np.stack(field_accs, axis=-1)
        return accs.reshape(accs.shape[0], -1)
    else:
        accs = compute_accuracy(logits, batch_labels)
        if ignore_times:
            accs = accs.reshape(accs.shape[0], -1, Message_Tokenizer.MSG_LEN)
            a_1 = accs[:, :, :Message_Tokenizer.TIME_START_I]
            a_2 = accs[:, :, (Message_Tokenizer.TIME_END_I + 1):]
            accs = np.concatenate([a_1, a_2], axis=2)
            accs = accs.reshape(accs.shape[0], -1)
        return accs

def prep_batch(
        batch: Union[
            Tuple[onp.ndarray, onp.ndarray, Dict[str, onp.ndarray]],
            Tuple[onp.ndarray, onp.ndarray]],
        seq_len: int,
        # in_dim: int,
        num_devices: int,
    ) -> Tuple[Tuple, np.ndarray, Tuple]:

    if len(batch) == 2:
        inputs, targets = batch
        book_data, timestep_msg, timestep_book = None, None, None
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
        book_data = aux_data.get("book_data", None)
        timestep_msg = aux_data.get("timesteps_msg", None)
        timestep_book = aux_data.get("timesteps_book", None)            
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # jit+sharding: no device_reshape needed — sharding handles data distribution
    inputs, labels, integration_times = _prep_batch_par(
        inputs,
        targets,
        seq_len,
        book_data,
        timestep_msg,
        timestep_book,
    )

    return inputs, labels, integration_times

@partial(
    jax.jit,
    static_argnums=(2,),
    # out_axes=(0, 0, 0),
    # devices=global_devices
)
def _prep_batch_par(
        inputs: jax.Array,
        targets: jax.Array,
        seq_len: int,
        # in_dim: int,
        book_data: Optional[jax.Array] = None,
        timestep_msg: Optional[jax.Array] = None,
        timestep_book: Optional[jax.Array] = None,
    ) -> Tuple[Tuple, np.ndarray, Tuple]:
    """
    Take a batch and convert it to a standard x/y format per device
    TODO: document this better for pmapped version
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """

    assert inputs.shape[1] == seq_len, f'inputs: {inputs.shape} seq_len {seq_len}'
    # inputs = one_hot(inputs, in_dim)

    # If there is an aux channel containing the integration times, then add that.
    if timestep_msg is not None:
        #timestep_msg = jax.device_put(timestep_msg, jax.devices()[0])
        integration_timesteps = (np.diff(np.asarray(timestep_msg)), )
    else:
        integration_timesteps = (np.ones((len(inputs), seq_len)), )

    if book_data is not None:
        #book_data = jax.device_put(book_data, jax.devices()[0])
        full_inputs = (inputs.astype(np.int32), book_data)
        if timestep_book is not None:
            #timestep_book = jax.device_put(timestep_book, jax.devices()[0])
            integration_timesteps += (np.diff(timestep_book), )
        else:
            integration_timesteps += (np.ones((len(inputs), seq_len)), )
    else:
        full_inputs = (inputs.astype(np.int32), )

    # CAVE: squeeze very important for training!
    return full_inputs, np.squeeze(targets.astype(np.int32)), integration_timesteps

def print_memory_usage():
    """Print GPU and system memory usage"""
    process = psutil.Process(os.getpid())
    print(f"CPU Memory: {process.memory_info().rss / 1024 ** 3:.2f} GB")
    
    # JAX device memory
    for device in jax.local_devices()[:1]:
        try:
            stats = device.memory_stats()
            if stats:
                print(f"Device {device} Used: {stats['bytes_in_use'] / 1024**2:.2f} MB / {stats['bytes_limit'] / 1024**3:.2f} GB")
        except:
            pass

def print_memory_usage_tofile():
    """Print GPU and system memory usage to a file"""
    process = psutil.Process(os.getpid())
    with open('/tmp/memory_usage.txt', 'a') as f:
        f.write(f"CPU Memory: {process.memory_info().rss / 1024 ** 3:.2f} GB\n")
        
        # JAX device memory
        for device in jax.local_devices()[:1]:
            try:
                stats = device.memory_stats()
                if stats:
                    f.write(f"Device {device} Used: {stats['bytes_in_use'] / 1024**2:.2f} MB / {stats['bytes_limit'] / 1024**3:.2f} GB\n")
            except:
                pass


def train_epoch(
        state,
        rng,
        #model,
        trainloader,
        seq_len,
        # in_dim,
        batchnorm,
        lr_params,
        num_devices,
        debug_loading,
        debug_profiler,
        curtail_epochs,
        init_hiddens,
        epoch,
        ignore_times,
        log_ce_tables,
        mesh=None,
        jit_train_step_fn=None,
        # ── Mid-epoch checkpoint parameters ──
        checkpoint_callback=None,
        checkpoint_every_n_steps="auto",
        job_start_time=None,
        max_job_hours=24.0,
        save_before_timeout_minutes=30,
        resume_from_step=None,
        # ── Mini-epoch validation parameters ──
        validate_callback=None,         # callable(state, epoch, batch_idx) -> bool (should_stop)
        validate_every_n_steps=0,       # trigger validate_callback every N steps (0=disabled)
        # ── Throughput tracking ──
        flops_per_step=None,            # pre-computed FLOPs per training step (from s5.flops)
        num_gpus=None,                  # total GPU count for MFU calculation
    ):

    """
    Training function for an epoch that loops over batches.

    lr_params: If None, LR is managed by optax schedules (no manual update).
               If provided, legacy mode with update_learning_rate_per_step.
    """
    # Store Metrics
    batch_losses = []
    cross_entropies= [] #list of 1xNTok losses

    use_optax_schedules = lr_params is None
    if not use_optax_schedules:
        decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params
    else:
        step = int(state.step)

    epoch_start = time.monotonic()
    watchdog = StepWatchdog()

    # ── Throughput tracking ──
    _last_timing_time = time.monotonic()
    _last_timing_step = 0
    _recent_step_time = 0.0

    # ── Mid-epoch checkpoint: auto vs manual mode ──
    _ckpt_every_str = str(checkpoint_every_n_steps)
    auto_checkpoint_mode = (_ckpt_every_str == "auto")
    if auto_checkpoint_mode:
        _ckpt_every = 0
        AUTO_CKPT_INTERVAL = 900    # 15 min (reduced from 30 min — NCCL deadlocks on 16N+)
        AUTO_WANDB_INTERVAL = 60    # 1 min — dense logging for scaling law curves
        # First checkpoint after ~5 min (not 15 min) to limit data loss on early NCCL deadlocks.
        # Subsequent checkpoints revert to the normal 15-min interval.
        EARLY_FIRST_CKPT_OFFSET = AUTO_CKPT_INTERVAL - 300  # triggers first save at ~5 min
        last_checkpoint_time = time.monotonic() - EARLY_FIRST_CKPT_OFFSET
        last_wandb_log_time = time.monotonic()
    else:
        _ckpt_every = int(_ckpt_every_str) if _ckpt_every_str != "0" else 0

    # resume_from_step: used for tqdm display offset and checkpoint step tracking.
    # Actual batch skipping is now done at sampler level (dataloading.py),
    # so DataLoader never loads skipped batches (zero IO overhead).
    batch_offset = resume_from_step if resume_from_step is not None else 0

    # ── Gradient accumulation / Local Steps (scan) setup ──
    use_grad_accum = isinstance(jit_train_step_fn, tuple)
    use_local_steps_scan = isinstance(jit_train_step_fn, dict) and jit_train_step_fn.get('mode') == 'local_steps_scan'
    if use_grad_accum:
        micro_step_fn, apply_step_fn, grad_K = jit_train_step_fn
        accum_grads = None
        accum_loss = 0.0
        micro_idx = 0
    else:
        grad_K = 1
    if use_local_steps_scan:
        scan_step_fn = jit_train_step_fn['step_fn']
        scan_K = jit_train_step_fn['K']
        scan_batch_buffer = []  # accumulate K batches before calling scan_step_fn

    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    total_steps = len(trainloader) + batch_offset
    for local_idx, batch in enumerate(tqdm(trainloader, initial=batch_offset, total=total_steps)):
        batch_idx = local_idx + batch_offset
        watchdog.kick(epoch, batch_idx)
        if not debug_loading:
            if (step>1) & (step<3) & debug_profiler:
                jax.profiler.start_trace("/tmp/tensorboard")
            inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)

            # jit+sharding: place data on devices with correct sharding
            if mesh is not None:
                from lob.sharding_utils import get_data_shardings_for_batch
                inputs_sh, labels_sh, times_sh = get_data_shardings_for_batch(mesh, has_book_data=(len(inputs) > 1))
                inputs = tuple(jax.make_array_from_process_local_data(sh, inp) for inp, sh in zip(inputs, inputs_sh))
                labels = jax.make_array_from_process_local_data(labels_sh, labels)
                integration_times = tuple(jax.make_array_from_process_local_data(sh, ts) for ts, sh in zip(integration_times, times_sh))

            rng, drop_rng = jax.random.split(rng)

            if use_grad_accum:
                # ── Gradient Accumulation Mode ──
                # micro_step: fwd-bwd + pmean('gpus') only (no cross-node comm)
                grads, micro_loss = micro_step_fn(
                    state, drop_rng, inputs, labels, integration_times,
                    batchnorm, ignore_times,
                )

                if micro_idx == 0:
                    accum_grads = grads
                    accum_loss = float(micro_loss)
                else:
                    accum_grads = jax.tree.map(np.add, accum_grads, grads)
                    accum_loss += float(micro_loss)

                micro_idx += 1

                # Not yet K micro-batches: skip optimizer update + all bookkeeping
                if micro_idx < grad_K:
                    # Still check curtail_epochs (operates on micro-batch index)
                    if (curtail_epochs is not None) and (batch_idx >= curtail_epochs):
                        print(f"[GradAccum] Ending epoch early at micro_batch {batch_idx} "
                              f"(mid-accumulation, {micro_idx}/{grad_K}). "
                              f"Discarding partial accumulation.")
                        break
                    # Check timeout even during accumulation
                    if checkpoint_callback is not None and job_start_time is not None:
                        elapsed_h = (time.monotonic() - job_start_time) / 3600.0
                        remaining_min = (max_job_hours - elapsed_h) * 60
                        if remaining_min <= save_before_timeout_minutes:
                            print(f"[GradAccum] Timeout imminent during accumulation "
                                  f"({micro_idx}/{grad_K}). Saving and exiting.")
                            checkpoint_callback(state, epoch, batch_idx, micro_loss, True)
                            watchdog.stop()
                            loss_mean = sum(batch_losses) / len(batch_losses) if batch_losses else float('nan')
                            return state, loss_mean, None, batch_idx + 1
                    continue

                # K micro-batches done: average accumulated grads and apply
                accum_grads = jax.tree.map(lambda g: g / grad_K, accum_grads)
                accum_loss /= grad_K

                # apply_step: pmean('nodes') + apply_gradients (Slingshot, once per K)
                state = apply_step_fn(state, accum_grads)

                # Reset accumulation state
                loss = accum_loss
                micro_idx = 0
                accum_grads = None
                accum_loss = 0.0

            elif use_local_steps_scan:
                # ── Local Steps (lax.scan) Mode ──
                # Buffer K batches, then call scan_step_fn which does K local
                # steps + one cross-node AllReduce via lax.scan.
                scan_batch_buffer.append((inputs, labels, integration_times))

                if len(scan_batch_buffer) >= scan_K:
                    if batch_idx % 1000 < scan_K:
                        print(f"\n=== Epoch {epoch}, Batch {batch_idx} (scan K={scan_K}) ===")
                        print_memory_usage()

                    # Stack K batches: each tensor gets a leading K dimension
                    inputs_k = tuple(
                        np.stack([b[0][i] for b in scan_batch_buffer])
                        for i in range(len(scan_batch_buffer[0][0]))
                    )
                    labels_k = np.stack([b[1] for b in scan_batch_buffer])
                    times_k = tuple(
                        np.stack([b[2][i] for b in scan_batch_buffer])
                        for i in range(len(scan_batch_buffer[0][2]))
                    )

                    state, loss, ce, grad_norms_or_logits, diloco_stats = scan_step_fn(
                        state, drop_rng, inputs_k, labels_k, times_k,
                        batchnorm, ignore_times,
                    )
                    scan_batch_buffer = []

                    # Log DiLoCo outer-step telemetry to wandb from rank 0
                    # diloco_stats = [‖Δ_avg‖, ‖v_new‖, ‖step_vec‖, ‖update‖]
                    # Non-zero only when diloco_outer='nesterov'.
                    try:
                        if jax.process_index() == 0:
                            _ds = onp.asarray(diloco_stats)
                            if float(_ds[0]) > 0 or float(_ds[1]) > 0:
                                import wandb as _wandb
                                if getattr(_wandb, 'run', None) is not None:
                                    _wandb.log({
                                        'diloco/delta_avg_norm': float(_ds[0]),
                                        'diloco/outer_momentum_norm': float(_ds[1]),
                                        'diloco/outer_step_norm': float(_ds[2]),
                                        'diloco/outer_update_norm': float(_ds[3]),
                                    }, commit=False)
                    except Exception:
                        pass
                else:
                    # Still buffering — skip loss logging for intermediate steps
                    continue

            else:
                # ── Standard Mode or Local Steps ──
                if batch_idx % 1000 == 0:
                    print(f"\n=== Epoch {epoch}, Batch {batch_idx} ===")
                    print_memory_usage()

                if use_local_steps_scan:
                    train_fn = local_step_fn
                else:
                    train_fn = jit_train_step_fn if jit_train_step_fn is not None else train_step
                state, loss, ce, grad_norms_or_logits = train_fn(
                    state, drop_rng, inputs, labels, integration_times,
                    batchnorm, ignore_times,
                )

                # Local Steps: sync params across nodes every K steps (Python-level dispatch)
                if use_local_steps_scan:
                    local_step_counter += 1
                    if local_step_counter >= local_K:
                        state = state.replace(params=sync_params_fn_local(state.params))
                        local_step_counter = 0

            if debug_profiler:
                if not use_grad_accum:
                    loss.block_until_ready()

            # jit+sharding: loss is already a scalar (no device dimension)
            loss_float = float(loss)
            batch_losses.append(loss_float)

            # NaN detection: save emergency checkpoint and abort
            if math.isnan(loss_float):
                print(f"\n[NaN] FATAL: NaN loss detected at epoch {epoch}, "
                      f"batch {batch_idx}, global_step {step}. "
                      f"Saving emergency checkpoint and aborting.")
                if checkpoint_callback is not None:
                    checkpoint_callback(state, epoch, batch_idx, 0.0, save_flag=True)
                watchdog.stop()
                loss_mean = sum(b for b in batch_losses if not math.isnan(b)) / max(1, sum(1 for b in batch_losses if not math.isnan(b)))
                return state, loss_mean, None, batch_idx

            if log_ce_tables and not use_grad_accum:
                cross_entropies.append(ce)

            if use_optax_schedules:
                step += 1
                # D2H watchdog check every 100 optimizer steps
                optimizer_steps_since_start = len(batch_losses)
                if optimizer_steps_since_start % 100 == 0 and optimizer_steps_since_start > 0:
                    watchdog.kick(epoch, batch_idx)
                    t0 = time.monotonic()
                    _device_step = int(state.step)
                    d2h_elapsed = time.monotonic() - t0
                    if d2h_elapsed > STEP_TIMEOUT:
                        print(f"FATAL: D2H transfer took {d2h_elapsed:.1f}s "
                              f"(threshold={STEP_TIMEOUT}s). NCCL collective likely deadlocked. "
                              f"Epoch {epoch}, batch {batch_idx}, global_step {step}")
                        raise TimeoutError(f"NCCL hang detected at epoch {epoch} step {step}")
            else:
                lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
                state, step = update_learning_rate_per_step(lr_params, state, mesh=mesh)

            if (step>20) & (step<=21) & debug_profiler:
                jax.profiler.stop_trace()
                break
            if (curtail_epochs is not None) and (batch_idx>=curtail_epochs):
                print("Ending epoch early at step", step, "due to curtail_epoch arg.")
                break

            # Periodic timing log for hang diagnosis + throughput
            optimizer_steps_since_start = len(batch_losses)
            if optimizer_steps_since_start % 100 == 0 and optimizer_steps_since_start > 0:
                avg_step_time = (time.monotonic() - epoch_start) / (local_idx + 1)
                # Windowed step time (last 100 steps, excludes compilation)
                _now = time.monotonic()
                _steps_delta = optimizer_steps_since_start - _last_timing_step
                _recent_step_time = (_now - _last_timing_time) / max(_steps_delta, 1)
                _last_timing_time = _now
                _last_timing_step = optimizer_steps_since_start

                if use_grad_accum:
                    print(f"[Timing] Epoch {epoch} opt_step {optimizer_steps_since_start} "
                          f"(micro_batch {batch_idx+1}/{total_steps}, K={grad_K}): "
                          f"avg {avg_step_time:.3f} s/micro_step")
                else:
                    timing_msg = (f"[Timing] Epoch {epoch} step {batch_idx+1}/{total_steps}: "
                                  f"avg {avg_step_time:.2f} s/step, "
                                  f"recent {_recent_step_time:.3f} s/step")
                    if flops_per_step and num_gpus and _recent_step_time > 0:
                        from s5.flops import compute_mfu
                        # K factor: each optimizer step processes K micro-batches
                        _K = scan_K if use_local_steps_scan else (grad_K if use_grad_accum else 1)
                        _mfu_info = compute_mfu(flops_per_step * _K, _recent_step_time, num_gpus)
                        timing_msg += (f", MFU {_mfu_info['mfu_pct']:.1f}%"
                                       f" ({_mfu_info['achieved_tflops']:.1f} TFLOPS)")
                    print(timing_msg)

            # ── Mid-epoch checkpoint ──
            if checkpoint_callback is not None:
                should_ckpt = False
                should_wandb = False
                timeout_imminent = False

                if _ckpt_every > 0 and optimizer_steps_since_start % _ckpt_every == 0:
                    should_ckpt = True
                    should_wandb = True

                if auto_checkpoint_mode:
                    now = time.monotonic()
                    if now - last_wandb_log_time >= AUTO_WANDB_INTERVAL:
                        should_wandb = True
                    if now - last_checkpoint_time >= AUTO_CKPT_INTERVAL:
                        should_ckpt = True
                        should_wandb = True

                if job_start_time is not None:
                    elapsed_h = (time.monotonic() - job_start_time) / 3600.0
                    remaining_min = (max_job_hours - elapsed_h) * 60
                    if remaining_min <= save_before_timeout_minutes:
                        should_ckpt = True
                        timeout_imminent = True

                if should_wandb or should_ckpt:
                    watchdog.kick(epoch, batch_idx)
                    # Pass throughput info to callback via extra kwargs
                    _cb_kwargs = {}
                    if flops_per_step and num_gpus and _recent_step_time > 0:
                        from s5.flops import compute_mfu
                        _K = scan_K if use_local_steps_scan else (grad_K if use_grad_accum else 1)
                        _effective_flops = flops_per_step * _K
                        _mfu_info = compute_mfu(_effective_flops, _recent_step_time, num_gpus)
                        _tokens_per_sec = _effective_flops / _recent_step_time / 6  # rough: flops/(6*time) ≈ N*tokens/time
                        _cb_kwargs["throughput"] = {
                            "step_time_s": _recent_step_time,
                            "mfu_pct": _mfu_info["mfu_pct"],
                            "achieved_tflops": _mfu_info["achieved_tflops"],
                            "tokens_per_sec": _tokens_per_sec,
                        }
                    if _LOG_GRAD_NORMS:
                        _cb_kwargs["grad_norms"] = grad_norms_or_logits
                    checkpoint_callback(state, epoch, batch_idx, loss, should_ckpt, **_cb_kwargs)
                    if auto_checkpoint_mode:
                        if should_wandb:
                            last_wandb_log_time = time.monotonic()
                        if should_ckpt:
                            last_checkpoint_time = time.monotonic()

                if timeout_imminent:
                    print(f"[Checkpoint] Timeout imminent! Saved at epoch={epoch}, step={batch_idx}")
                    print(f"[Checkpoint] Resume: RESTORE_STEP={step} RESUME_FROM_STEP={batch_idx+1}")
                    watchdog.stop()
                    loss_mean = sum(batch_losses) / len(batch_losses) if batch_losses else float('nan')
                    return state, loss_mean, None, batch_idx + 1

            # ── Mini-epoch validation ──
            if (validate_every_n_steps > 0 and
                validate_callback is not None and
                optimizer_steps_since_start % validate_every_n_steps == 0 and
                optimizer_steps_since_start > 0):
                watchdog.kick(epoch, batch_idx)
                should_stop = validate_callback(state, epoch, batch_idx)
                if should_stop:
                    watchdog.stop()
                    loss_mean = sum(batch_losses) / len(batch_losses) if batch_losses else float('nan')
                    ce_means = onp.mean(onp.concatenate(cross_entropies, axis=0), axis=0) if log_ce_tables else None
                    return state, loss_mean, ce_means, None

        else:
            continue


    watchdog.stop()
    # Return average loss over batches
    if log_ce_tables:
        ce_means=np.mean(np.concatenate(cross_entropies,axis=0),axis=0)
    else:
        ce_means=None
    # jax.debug.print("CE of epoch by token: {}",ce_means.shape)
    loss_mean = sum(batch_losses) / len(batch_losses)
    return state, loss_mean, ce_means, None


@partial(jax.vmap,in_axes=(0,0,None),out_axes=(0,0))
@partial(jax.jit,static_argnums=(2,))
def repeat_book(msg,book,shift_start):
    #DEFINITION OF START BOOK:
    # print("checking for compile in repeat_book")
    if msg.shape[0]>book.shape[0]:
        book = np.repeat(book, (msg.shape[0]) // book.shape[0], axis=0)
    # if shift_start:
    #     pad=book[:1]
    #     #FIXME: Wrong logic, needs to be the init book state.
    #     # book=np.concatenate([book[:1],book[1:]])
    #     book=np.concatenate([pad,book[:-1]])
    return (msg,book)

def train_step(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 1
        batch_inputs: Tuple[jax.Array, jax.Array], # 2
        batch_labels: jax.Array, # 3
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 4
        batchnorm: bool, # 5
        ignore_times:bool, #6
    ):

    # Print hash values of static arguments
    # print(f"batchnorm hash: {batchnorm.__hash__()}")
    # print(f"ignore_times hash: {ignore_times.__hash__()}")
    # print('checking for compile in train_step')

    if len(batch_inputs) > 1:
        batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)

    def loss_fn(params):
        # print('checking for compile in loss_fn')
        if batchnorm:
            logits, mod_vars = state.apply_fn( 
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
                method='__call_ar__'
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
                method='__call_ar__'
            )


        # jax.debug.print("Shape of Logits: {}",logits.shape)
        # jax.debug.print("Shape of Labels: {}", batch_labels.shape)

        
        ce = _compute_ce_unified(logits, batch_labels, ignore_times)

        ce=np.mean(ce,axis=0)
        # average cross-ent loss
        loss = np.mean(ce)
        return loss, (mod_vars, logits,ce)

    (loss, (mod_vars, logits,ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Per-group gradient norms (pre-clip)
    if _LOG_GRAD_NORMS:
        grad_norms = _compute_grad_norms(grads)
    else:
        grad_norms = np.zeros(6, dtype=np.float32)

    # jit+sharding: no pmean needed — sharding handles cross-device aggregation
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss, ce, grad_norms

@partial(
    jax.jit,
    static_argnums=(5,),
)
def train_step_rnn(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 3
        batch_inputs: Tuple[jax.Array, jax.Array], # 4
        batch_labels: jax.Array, # 5
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 6
        batchnorm: bool, # 7
        init_hiddens: Tuple, 
    ):
    #print('tracing par_loss_and_grad')

    #Never reset the hidden states:

    if len(batch_inputs) > 1:
        batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)


    def loss_fn(params):
        def single_elem_loss(carry,xs):
            shapes=jax.tree_util.tree_map(lambda x: x.shape,xs)
            print("Shapes before using:",shapes)
            batch_inputs,batch_integration_timesteps,batch_labels=xs
            dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*len(hiddens)
            hiddens=carry
            if batchnorm:
                (hiddens,logits), mod_vars = state.apply_fn( 
                    {"params": params, "batch_stats": state.batch_stats},
                    hiddens,
                    *batch_inputs,
                    *dones,
                    *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates", "batch_stats"],
                    method='__call_rnn__'
                )
            else:
                (hiddens,logits), mod_vars = state.apply_fn(
                    {"params": params},
                    hiddens,
                    *batch_inputs,
                    *dones,
                    *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates"],
                    method='__call_rnn__'
                )
            
            
            ce=cross_entropy_loss(logits, batch_labels)
            # jax.debug.print("Shape of CE: {}", ce.shape)
            # average cross-ent loss
            ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
            ce=ce.at[:,:,Message_Tokenizer.TIME_START_I:Message_Tokenizer.TIME_END_I + 1].set(0)
            ce=ce.reshape(ce.shape[0],-1)
            loss = np.mean(ce)
            return (hiddens),(loss,mod_vars)
        # jax.debug.print("Shape of loss: {}", loss.shape)
        xs=(batch_inputs,batch_integration_timesteps,batch_labels)
        xs=jax.tree_util.tree_map(lambda x: np.array(np.split(x,2,axis=1)),xs)
        hiddens,y=jax.lax.scan(single_elem_loss,init_hiddens,xs)
        losses,mod_vars=y
        loss=np.mean(losses)
        mod_vars=jax.tree_util.tree_map(np.mean,mod_vars)
        return loss, mod_vars

    (loss, mod_vars), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss

def train_step_old(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 3
        batch_inputs: Tuple[jax.Array, jax.Array], # 4
        batch_labels: jax.Array, # 5
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 6
        batchnorm: bool, # 7
    ):
    #print('tracing par_loss_and_grad')
    def loss_fn(params):
        if batchnorm:
            logits, mod_vars = state.apply_fn( 
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        # average cross-ent loss
        loss = np.mean(cross_entropy_loss(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)



    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss


def validate(state,
             apply_fn,
             testloader,
             seq_len,
             in_dim,
             batchnorm,
             num_devices,
             epoch,
             curtail_epoch=None,
             ignore_times: bool =False,
             step_rescale=1.0,
             apply_method: str ='__call_ar__',
             init_hiddens=(np.array([0])),
             log_ce_tables : bool =False,
             mesh=None,
             jit_eval_step_fn=None,
             silent=False):
    """Validation function — NCCL-deadlock-free for multi-host.

    Instead of calling process_allgather per batch (which triggers 128-rank
    NCCL AllGather and deadlocks at 32N scale), we read only local shards
    via addressable_shards (zero NCCL communication).  Each host computes
    metrics over its own 1/N_hosts subset; DistributedSampler guarantees
    equal sample counts so local mean ≈ global mean.

    Args:
        silent: If True, suppress tqdm progress bar (used during mini-epoch
                eval to avoid visual "step reset" noise in logs).
    """
    import numpy as onp
    is_multihost = (mesh is not None and jax.process_count() > 1)

    # T5X pattern: assert all hosts have same number of eval batches.
    # Mismatch → one host exits loop while others wait in collective → deadlock.
    if is_multihost:
        num_batches = len(testloader)
        from jax.experimental.multihost_utils import assert_equal
        assert_equal(np.array(num_batches),
                     f"Eval batch count mismatch: rank {jax.process_index()}")

    losses, accuracies, preds = [], [], []
    for batch_idx, batch in enumerate(tqdm(testloader, disable=silent)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, num_devices)

        # jit+sharding: place data on devices (multi-host compatible)
        if mesh is not None:
            from lob.sharding_utils import get_data_shardings_for_batch
            inputs_sh, labels_sh, times_sh = get_data_shardings_for_batch(mesh, has_book_data=(len(inputs) > 1))
            inputs = tuple(jax.make_array_from_process_local_data(sh, inp) for inp, sh in zip(inputs, inputs_sh))
            labels = jax.make_array_from_process_local_data(labels_sh, labels)
            integration_timesteps = tuple(jax.make_array_from_process_local_data(sh, ts) for ts, sh in zip(integration_timesteps, times_sh))

        eval_fn = jit_eval_step_fn if jit_eval_step_fn is not None else eval_step
        loss, acc, pred = eval_fn(
            inputs, labels, integration_timesteps, state, apply_fn, batchnorm,apply_method,init_hiddens,ignore_times)

        # Read loss/acc to numpy — zero NCCL communication path for multi-host.
        # OLD (deadlocks at 32N): process_allgather(loss) triggers 128-rank AllGather per batch.
        # NEW: addressable_shards reads only this host's local GPU data (no collective).
        if is_multihost:
            local_loss = onp.concatenate(
                [onp.asarray(s.data) for s in loss.addressable_shards], axis=0)
            local_acc = onp.concatenate(
                [onp.asarray(s.data) for s in acc.addressable_shards], axis=0)
            losses.append(local_loss)
            accuracies.append(local_acc)
        else:
            losses.append(onp.asarray(loss))
            accuracies.append(onp.asarray(acc))

        if curtail_epoch is not None and batch_idx>=curtail_epoch:
            print(f"Ending epoch early at step {batch_idx} due to curtail_epoch arg.")
            break

    concat_loss=onp.concatenate(losses,axis=0)
    concat_acc=onp.concatenate(accuracies,axis=0)
    print(f"Concat Loss is {concat_loss.shape}")
    print(f"Concat Acc is {concat_acc.shape}")
    if log_ce_tables:
        # Per-position: reshape flat (N, n_orders*tpm) → (N, n_orders, tpm), mean over (0,1)
        tpm_per = (Message_Tokenizer.MSG_LEN - (TIME_END_I - TIME_START_I + 1)
                   if ignore_times else Message_Tokenizer.MSG_LEN)
        ce_means = onp.mean(concat_loss.reshape(concat_loss.shape[0], -1, tpm_per), axis=(0, 1))
        acc_means = onp.mean(concat_acc.reshape(concat_acc.shape[0], -1, tpm_per), axis=(0, 1))
    else:
        ce_means=None
        acc_means=None
    aveloss, aveaccu = onp.mean(concat_loss), onp.mean(onp.asarray(accuracies))

    # Per-field accuracy breakdown (always print when log_ce_tables=True)
    if acc_means is not None and os.environ.get('LOG_PER_FIELD', '') == '1':
        tok_lens = Message_Tokenizer.TOK_LENS
        if ignore_times:
            active_fields = [(i, f) for i, f in enumerate(Message_Tokenizer.FIELDS) if i not in (6, 7)]
        else:
            active_fields = list(enumerate(Message_Tokenizer.FIELDS))
        tok_idx = 0
        field_strs = []
        for field_i, field_name in active_fields:
            n_tok = int(tok_lens[field_i])
            field_acc = float(onp.mean(acc_means[tok_idx:tok_idx + n_tok]))
            field_ce = float(onp.mean(ce_means[tok_idx:tok_idx + n_tok])) if ce_means is not None else 0.0
            field_strs.append(f"{field_name}={field_acc:.4f}(CE={field_ce:.3f})")
            tok_idx += n_tok
        print(f"  [Per-Field Acc] {' | '.join(field_strs)}")

    # Last-order metrics: the final order in each 500-order sequence has the
    # longest context (499 prior orders) and is most comparable to LOBS5's
    # conditional generation setup.
    tpm = (Message_Tokenizer.MSG_LEN - (TIME_END_I - TIME_START_I + 1)
           if ignore_times else Message_Tokenizer.MSG_LEN)
    last_order_losses = concat_loss[:, -tpm:]
    last_order_accs = concat_acc[:, -tpm:]
    last_order_loss = float(onp.mean(last_order_losses))
    last_order_acc = float(onp.mean(last_order_accs))
    last_order_nll = float(last_order_loss * tpm)
    all_orders_nll = float(aveloss * tpm)

    del losses, accuracies
    return aveloss, aveaccu, ce_means, acc_means, last_order_loss, last_order_acc, last_order_nll, all_orders_nll

def eval_step(
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        state,
        #model,
        apply_fn,
        batchnorm,
        apply_method,
        init_hiddens,
        ignore_times,
    ):
    # print("checking for compile in eval_step function")

    if len(batch_inputs) > 1:
        batch_inputs=repeat_book(*batch_inputs,True)

    if apply_method == '__call_ar__':
        # Bug #4 workaround: match train_step's apply_fn signature so the
        # produced HLO module looks similar enough that XLA's TritonFusionAnalysis
        # populates dim_orders_ for the same instructions. Without rngs+mutable,
        # eval's HLO differs from train's, leading to a `dim_orders_.at(hlo)`
        # cache miss in xla/service/gpu/triton_fusion_analysis.cc:178/191
        # which crashes with `IndexError: absl::raw_hash_map::at` at the first
        # validate() call (see j3432283 step 5009 crash).
        # Static dummy rng — the model's eval mode keeps dropout deterministic,
        # so the key value is unused but the signature shape matches train.
        _eval_rng = jax.random.key(0)
        if batchnorm:
            logits, _ = apply_fn({"params": state.params, "batch_stats": state.batch_stats},
                                *batch_inputs, *batch_integration_timesteps,
                                rngs={"dropout": _eval_rng},
                                mutable=["intermediates"],
                                method=apply_method,
                                )
        else:
            logits, _ = apply_fn({"params": state.params},
                                *batch_inputs, *batch_integration_timesteps,
                                rngs={"dropout": _eval_rng},
                                mutable=["intermediates"],
                                method=apply_method,
                                )
    elif apply_method == '__call_rnn__':
        dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*3

        if batchnorm:
            hiddens,logits=apply_fn(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        init_hiddens,
                        *batch_inputs,
                        *dones,
                        *batch_integration_timesteps,
                        method='__call_rnn__'
                    )
        else:
            hiddens,logits=apply_fn(
                        {"params": state.params},
                        init_hiddens,
                        *batch_inputs,
                        *dones,
                        *batch_integration_timesteps,
                        method='__call_rnn__'
                    )
    elif apply_method == 'scan_rnn':
        dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*3
        hiddens,logits=eval_rnn_scan(apply_fn,
                                     init_hiddens,
                                     state,
                                     batch_inputs,
                                     dones,
                                     batch_integration_timesteps,
                                     batchnorm)



    losses = _compute_ce_unified(logits, batch_labels, ignore_times)
    accs = _compute_acc_unified(logits, batch_labels, ignore_times)

    return losses, accs, logits


def eval_rnn_scan(apply_fn,hiddens,state,batch_inputs,batch_dones,batch_inttimes,batchnorm):
    def apply_fn_scan(carry,x):
        (hiddens,state)=carry
        (batch_inputs,batch_dones,batch_inttimes)=x
        if batchnorm:
            hiddens,logits=apply_fn(
                    {"params": state.params, "batch_stats": state.batch_stats},
                    hiddens,
                    *batch_inputs,
                    *batch_dones,
                    *batch_inttimes,
                    method='__call_rnn__'
                )
        else:
            hiddens,logits=apply_fn(
                    {"params": state.params},
                    hiddens,
                    *batch_inputs,
                    *batch_dones,
                    *batch_inttimes,
                    method='__call_rnn__'
                )
        return (hiddens,state),logits
    #FIXME : Poor practice, but just for debugging purposes. 
    Ntoks=11000

    init=(hiddens,state)
    xs=(batch_inputs,batch_dones,batch_inttimes)
    
    xs=jax.tree_util.tree_map(partial(swap_leading,Ntoks),xs)
    
    carry_out,logits=jax.lax.scan(apply_fn_scan,init,xs)
    (hiddens,state)
    logits=np.concatenate(logits,axis=-2)
    return hiddens,logits
    
def swap_leading(targetsize,x):
    x=np.expand_dims(x,0)
    x=np.swapaxes(x,0,x.shape.index(targetsize))
    return x


# ============================================================================
# JIT-compiled step functions (replacement for @pmap decorators)
# ============================================================================

def create_jit_train_step(mesh, state, has_book_data=True, hierarchical=False,
                          batchnorm=False, ignore_times=True, local_steps_k=0,
                          grad_accum_steps=1,
                          diloco_outer='none',
                          diloco_outer_lr=0.7,
                          diloco_outer_momentum=0.9):
    """Create JIT-compiled train_step with explicit sharding.

    hierarchical=True: uses shard_map + explicit pmean('gpus') + pmean('nodes')
    for hierarchical AllReduce on 2D mesh. Requires mesh with ('nodes','gpus') axes.

    local_steps_k>0: Local Steps mode — each node trains independently for K steps
    (only intra-node pmean('gpus') per step), then params averaged via pmean('nodes')
    every K steps. Requires hierarchical=True. K=0 disables (standard AllReduce).

    diloco_outer='nesterov': replace naive params-pmean with DiLoCo pseudo-gradient
    + Nesterov outer step (requires state to be a DiLoCoState). 'none' keeps the
    legacy naive averaging behavior.

    grad_accum_steps>1: Gradient accumulation — returns (micro_step_fn, apply_step_fn, K)
    tuple. micro_step_fn does fwd-bwd + pmean('gpus') only. apply_step_fn does
    pmean('nodes') + apply_gradients. Requires hierarchical=True.
    """
    # TP mode always uses shard_map (same code path as hierarchical)
    use_shard_map = hierarchical or 'tp' in mesh.axis_names
    if use_shard_map and grad_accum_steps > 1:
        return _create_hierarchical_grad_accum_fns(mesh, has_book_data, batchnorm,
                                                    ignore_times, grad_accum_steps)
    if use_shard_map:
        return _create_hierarchical_train_step(mesh, has_book_data, batchnorm,
                                               ignore_times, local_steps_k,
                                               diloco_outer=diloco_outer,
                                               diloco_outer_lr=diloco_outer_lr,
                                               diloco_outer_momentum=diloco_outer_momentum)

    from lob.sharding_utils import create_state_shardings, get_data_shardings_for_batch

    state_shardings = create_state_shardings(state, mesh)
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(mesh, has_book_data=has_book_data)

    in_shardings = (
        state_shardings,    # state - replicated
        None,               # rng
        inputs_shardings,   # batch_inputs - sharded
        labels_sharding,    # batch_labels - sharded
        timesteps_shardings,# batch_integration_timesteps - sharded
    )
    out_shardings = (
        state_shardings,    # state
        None,               # loss
        None,               # ce
        None,               # logits
    )

    jit_train_step = jax.jit(
        train_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(5, 6),  # batchnorm, ignore_times
        donate_argnums=(0,),    # donate state for memory reuse
    )
    print("[JIT] Created JIT-compiled train_step with sharding")
    return jit_train_step


def _create_hierarchical_grad_accum_fns(mesh, has_book_data, batchnorm, ignore_times,
                                         grad_accum_steps):
    """Create split shard_map functions for gradient accumulation.

    Returns (micro_step_fn, apply_step_fn, K) where:
    - micro_step_fn: fwd-bwd + pmean('gpus') only (NVLink, fast)
      Signature: micro_step_fn(state, rng, inputs, labels, times) -> (grads, loss)
    - apply_step_fn: pmean('nodes') + apply_gradients (Slingshot, once per K)
      Signature: apply_step_fn(state, accum_grads) -> state
    - K: number of micro-batches per optimizer update
    """
    from jax.experimental.shard_map import shard_map
    from lob.sharding_utils import _get_batch_axis

    # Resolve mesh axes from the actual mesh rather than hardcoding 'nodes'/'gpus'.
    # Patterns:
    #   - hierarchical DP (no TP): mesh = ('nodes','gpus')
    #       batch sharded across both, intra-node pmean on 'gpus' (NVLink),
    #       cross-node pmean on 'nodes' (Slingshot).
    #   - TP+DP (multi-node):       mesh = ('dp','tp')
    #       batch sharded across 'dp', no intra-group data sync needed
    #       (TP devices share the same data shard), apply pmean('dp').
    #   - TP only (single node):    mesh = ('tp',) — no DP, grad_accum is
    #       degenerate; we still produce a valid spec but no pmeans run.
    use_tp = 'tp' in mesh.axis_names
    if use_tp:
        batch_axis = _get_batch_axis(mesh)   # 'dp', 'nodes', or None
        intra_axis = None                    # TP shares data → no intra pmean
        dp_axis = ('dp' if 'dp' in mesh.axis_names
                   else 'nodes' if 'nodes' in mesh.axis_names else None)
    else:
        batch_axis = ('nodes', 'gpus')
        intra_axis = 'gpus'
        dp_axis = 'nodes'

    if has_book_data:
        in_data = (P(batch_axis, None), P(batch_axis, None))
        in_times = (P(batch_axis, None), P(batch_axis, None))
    else:
        in_data = (P(batch_axis, None),)
        in_times = (P(batch_axis, None),)

    # ── micro_step: fwd-bwd + intra-axis pmean (NVLink only) ──
    micro_in_specs = (
        P(),            # state — replicated (needed for apply_fn + params)
        P(),            # rng — replicated
        in_data,        # batch_inputs — sharded tuple
        P(batch_axis),  # batch_labels — sharded
        in_times,       # batch_integration_timesteps — sharded tuple
    )
    # grads: per-node averaged (same within node, different across nodes)
    # loss: per-node averaged scalar
    micro_out_specs = (P(), P())

    def micro_body(state, rng, batch_inputs, batch_labels,
                   batch_integration_timesteps):
        """Forward-backward + intra-node gradient average (NVLink only)."""
        if len(batch_inputs) > 1:
            batch_inputs = repeat_book(*batch_inputs, True)

        def loss_fn(params):
            if batchnorm:
                logits, mod_vars = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    *batch_inputs, *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates", "batch_stats"],
                    method='__call_ar__'
                )
            else:
                logits, mod_vars = state.apply_fn(
                    {"params": params},
                    *batch_inputs, *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates"],
                    method='__call_ar__'
                )

            ce = _compute_ce_unified(logits, batch_labels, ignore_times)

            ce = np.mean(ce, axis=0)
            loss = np.mean(ce)
            return loss, (mod_vars, logits, ce)

        (loss, (mod_vars, logits, ce)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)

        # Level 1: intra-node sync over fast interconnect (NVLink for hierarchical
        # DP). Skipped under TP since TP devices share the same data shard.
        if intra_axis is not None:
            grads = jax.lax.pmean(grads, axis_name=intra_axis)
            loss = jax.lax.pmean(loss, axis_name=intra_axis)

        return grads, loss

    mapped_micro = shard_map(micro_body, mesh=mesh,
                             in_specs=micro_in_specs, out_specs=micro_out_specs,
                             check_rep=False)
    jitted_micro = jax.jit(mapped_micro)

    # API-compatible wrapper: train_epoch passes batchnorm, ignore_times as args 6-7
    def micro_step_fn(state, rng, batch_inputs, batch_labels,
                      batch_integration_timesteps, _batchnorm, _ignore_times):
        return jitted_micro(state, rng, batch_inputs, batch_labels,
                            batch_integration_timesteps)

    # ── apply_step: pmean('nodes') + apply_gradients ──
    apply_in_specs = (
        P(),  # state — replicated
        P(),  # accum_grads — per-node (different across nodes, same within node)
    )
    apply_out_specs = P()  # state — replicated after pmean + apply

    def apply_body(state, accum_grads):
        """Cross-DP-group gradient average + parameter update."""
        # Level 2: cross-data-parallel-group sync (Slingshot for hierarchical,
        # also Slingshot for cross-node TP+DP). No-op when there's no DP axis
        # (single-node TP with grad_accum is degenerate but valid).
        if dp_axis is not None:
            accum_grads = jax.lax.pmean(accum_grads, axis_name=dp_axis)
        state = state.apply_gradients(grads=accum_grads)
        return state

    mapped_apply = shard_map(apply_body, mesh=mesh,
                             in_specs=apply_in_specs, out_specs=apply_out_specs,
                             check_rep=False)
    jitted_apply = jax.jit(mapped_apply, donate_argnums=(0,))

    print(f"[JIT] Created hierarchical grad_accum shard_map functions "
          f"(K={grad_accum_steps}: micro_step=pmean('{intra_axis or 'none'}'), "
          f"apply_step=pmean('{dp_axis or 'none'}')+apply_gradients)")

    return micro_step_fn, jitted_apply, grad_accum_steps


def _create_hierarchical_train_step(mesh, has_book_data, batchnorm, ignore_times,
                                     local_steps_k=0,
                                     diloco_outer='none',
                                     diloco_outer_lr=0.7,
                                     diloco_outer_momentum=0.9):
    """Create shard_map-based train_step with hierarchical AllReduce.

    Supports mesh layouts:
    - ('nodes', 'gpus'): Pure DP. pmean('gpus') intra-node, pmean('nodes') inter-node.
    - ('nodes', 'tp'):   TP+DP (intra-node TP). psum('tp'), pmean('nodes').
    - ('dp', 'tp'):      TP+DP (cross-node TP). psum('tp'), pmean('dp').
    - ('tp',):           Pure TP (no DP). psum('tp') only.

    When local_steps_k > 0: "Local Steps" mode. Each DP group runs K local
    optimizer steps, then syncs via pmean(dp_axis) every K steps.

    Outer-step variants:
    - diloco_outer='none': naive FedAvg — pmean(params) directly.
    - diloco_outer='nesterov': DiLoCo (Douillard et al. 2024) — pmean of
      pseudo-gradient (theta_anchor - theta_local) + Nesterov-momentum outer
      step. Requires state to be a DiLoCoState with outer_momentum buffer.
    """
    from jax.experimental.shard_map import shard_map
    from lob.sharding_utils import _get_batch_axis

    use_tp = 'tp' in mesh.axis_names
    if use_tp:
        batch_axis = _get_batch_axis(mesh)  # 'dp', 'nodes', or None
        intra_axis = 'tp'
        # DP axis for cross-group gradient sync
        dp_axis = 'dp' if 'dp' in mesh.axis_names else \
                  'nodes' if 'nodes' in mesh.axis_names else None
    else:
        batch_axis = ('nodes', 'gpus')
        intra_axis = 'gpus'
        dp_axis = 'nodes'

    # in_specs must match pytree structure of each argument
    if has_book_data:
        in_data = (P(batch_axis, None), P(batch_axis, None))
        in_times = (P(batch_axis, None), P(batch_axis, None))
    else:
        in_data = (P(batch_axis, None),)
        in_times = (P(batch_axis, None),)

    in_specs = (
        P(),            # state — replicated
        P(),            # rng — replicated
        in_data,        # batch_inputs — sharded tuple
        P(batch_axis),  # batch_labels — sharded
        in_times,       # batch_integration_timesteps — sharded tuple
    )
    # state, loss, ce are replicated after pmean; logits replaced with dummy scalar
    out_specs = (P(), P(), P(), P())

    def sharded_step(state, rng, batch_inputs, batch_labels,
                     batch_integration_timesteps):
        """Per-shard train step with hierarchical gradient reduction."""
        if len(batch_inputs) > 1:
            batch_inputs = repeat_book(*batch_inputs, True)

        def loss_fn(params):
            if batchnorm:
                logits, mod_vars = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    *batch_inputs, *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates", "batch_stats"],
                    method='__call_ar__'
                )
            else:
                logits, mod_vars = state.apply_fn(
                    {"params": params},
                    *batch_inputs, *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates"],
                    method='__call_ar__'
                )

            ce = _compute_ce_unified(logits, batch_labels, ignore_times)

            ce = np.mean(ce, axis=0)
            loss = np.mean(ce)
            return loss, (mod_vars, logits, ce)

        (loss, (mod_vars, logits, ce)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)

        # ── Hierarchical AllReduce ──
        # Level 1: intra-node via NVLink (478 GB/s)
        # DP mode ('gpus'): pmean averages replicated gradients
        # TP mode ('tp'):   psum aggregates partial gradients from head shards
        if use_tp:
            grads = jax.lax.psum(grads, axis_name=intra_axis)
        else:
            grads = jax.lax.pmean(grads, axis_name=intra_axis)
        loss = jax.lax.pmean(loss, axis_name=intra_axis)
        ce = jax.lax.pmean(ce, axis_name=intra_axis)

        # Per-group gradient norms (pre-clip, post intra-node pmean)
        if _LOG_GRAD_NORMS:
            grad_norms = _compute_grad_norms(grads)
        else:
            grad_norms = np.zeros(6, dtype=np.float32)

        if local_steps_k > 0:
            # ── Local Steps mode ──
            # Each node applies its own intra-node grads.
            # NO cross-node communication here; param sync after K steps.
            if batchnorm:
                state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
            else:
                state = state.apply_gradients(grads=grads)
        else:
            # ── Standard hierarchical AllReduce ──
            # Level 2: cross-group DP sync (grads only, loss/ce skipped)
            if dp_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=dp_axis)

            if batchnorm:
                state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
            else:
                state = state.apply_gradients(grads=grads)

        return state, loss, ce, grad_norms

    mapped_fn = shard_map(sharded_step, mesh=mesh,
                          in_specs=in_specs, out_specs=out_specs,
                          check_rep=False)
    jitted_fn = jax.jit(mapped_fn, donate_argnums=(0,))

    # API-compatible wrapper: train_epoch passes batchnorm, ignore_times as args 6-7
    def compatible_fn(state, rng, batch_inputs, batch_labels,
                      batch_integration_timesteps, _batchnorm, _ignore_times):
        return jitted_fn(state, rng, batch_inputs, batch_labels,
                         batch_integration_timesteps)

    if local_steps_k > 0:
        # ── Option 2: lax.scan over K local steps + single AllReduce ──
        # The inner sharded_step does fwd+bwd+pmean('gpus')+apply_gradients only.
        # We wrap K calls in lax.scan, then do ONE pmean(params, 'nodes') at the end.
        # This truly skips cross-node communication for K-1 out of K steps.
        # Benchmarked 30% faster than K=1 on 55M/16N (j2899272 vs j2899158).

        def sharded_k_steps(state, rng, batch_inputs_k, batch_labels_k,
                            batch_integration_timesteps_k):
            """K local steps via lax.scan + single cross-node outer step.

            batch_*_k tensors have leading dim K (stacked K batches).

            For diloco_outer='nesterov': state is DiLoCoState. theta_anchor
            is snapshot from state.train_state.params before the scan;
            pseudo-gradient (theta_anchor - theta_local) is pmean'd across
            dp_axis, then a Nesterov-momentum outer step updates params.
            """
            # Snapshot theta_anchor BEFORE inner scan (DiLoCo pseudo-grad reference)
            if diloco_outer == 'nesterov':
                theta_anchor = state.train_state.params

            def scan_body(carry, batch_slice):
                state, rng = carry
                # Advance RNG per step to avoid correlated dropout
                rng, step_rng = jax.random.split(rng)

                # Unpack the batch slice
                if has_book_data:
                    inputs = (batch_slice[0], batch_slice[1])
                    labels = batch_slice[2]
                    times = (batch_slice[3], batch_slice[4])
                else:
                    inputs = (batch_slice[0],)
                    labels = batch_slice[1]
                    times = (batch_slice[2],)

                state, loss, ce, grad_norms = sharded_step(
                    state, step_rng, inputs, labels, times)
                return (state, rng), (loss, grad_norms)

            # Stack batch data for scan: each element has leading dim K
            if has_book_data:
                scan_data = (
                    batch_inputs_k[0],    # [K, B, seq, ...]
                    batch_inputs_k[1],    # [K, B, seq, ...]
                    batch_labels_k,       # [K, B, ...]
                    batch_integration_timesteps_k[0],  # [K, B, ...]
                    batch_integration_timesteps_k[1],  # [K, B, ...]
                )
            else:
                scan_data = (
                    batch_inputs_k[0],    # [K, B, seq, ...]
                    batch_labels_k,       # [K, B, ...]
                    batch_integration_timesteps_k[0],  # [K, B, ...]
                )

            (state, _rng), (losses, all_grad_norms) = jax.lax.scan(
                scan_body, (state, rng), scan_data)

            # ── Cross-group DP outer step ──
            # Telemetry placeholders — emitted every outer step when diloco enabled,
            # zeros otherwise (preserves the 4-value return signature).
            diloco_stats = np.zeros(4, dtype=np.float32)
            if dp_axis is not None:
                if diloco_outer == 'nesterov':
                    # DiLoCo: pseudo-gradient + Nesterov outer momentum
                    # Δ_local = θ_anchor − θ_local (local outer update direction)
                    theta_local = state.train_state.params
                    delta_local = jax.tree.map(
                        lambda a, l: a - l, theta_anchor, theta_local)
                    # Single inter-node AllReduce on pseudo-gradient
                    delta_avg = jax.lax.pmean(delta_local, axis_name=dp_axis)
                    # Nesterov momentum buffer (PyTorch SGD(nesterov=True) convention)
                    v_new = jax.tree.map(
                        lambda m, d: diloco_outer_momentum * m + d,
                        state.outer_momentum, delta_avg)
                    # Nesterov lookahead step: Δ + β·v_new
                    step_vec = jax.tree.map(
                        lambda d, m: d + diloco_outer_momentum * m,
                        delta_avg, v_new)
                    # θ_new = θ_anchor − lr_out · step_vec
                    theta_new = jax.tree.map(
                        lambda a, s: a - diloco_outer_lr * s,
                        theta_anchor, step_vec)
                    new_ts = state.train_state.replace(params=theta_new)
                    state = state.replace(train_state=new_ts,
                                          outer_momentum=v_new)

                    # Outer-step telemetry (to diagnose Nesterov overshoots)
                    _tree_sq_norm = lambda tree: sum(
                        np.sum(np.square(leaf)) for leaf in
                        jax.tree_util.tree_leaves(tree))
                    diloco_stats = np.stack([
                        np.sqrt(_tree_sq_norm(delta_avg)),                  # ‖Δ_avg‖
                        np.sqrt(_tree_sq_norm(v_new)),                      # ‖v_new‖
                        np.sqrt(_tree_sq_norm(step_vec)),                   # ‖step_vec‖
                        diloco_outer_lr * np.sqrt(_tree_sq_norm(step_vec)), # ‖update‖
                    ]).astype(np.float32)
                else:
                    # Naive FedAvg: pmean params directly
                    state = state.replace(
                        params=jax.lax.pmean(state.params, axis_name=dp_axis))

            # Return mean loss across K steps, grad_norms from last step,
            # and per-outer DiLoCo telemetry (zeros when diloco_outer='none').
            return state, np.mean(losses), np.mean(losses), all_grad_norms[-1], diloco_stats

        # Shard specs: same as single step but batch dims have extra leading K
        if has_book_data:
            in_data_k = (P(None, batch_axis, None), P(None, batch_axis, None))
            in_times_k = (P(None, batch_axis, None), P(None, batch_axis, None))
        else:
            in_data_k = (P(None, batch_axis, None),)
            in_times_k = (P(None, batch_axis, None),)

        in_specs_k = (
            P(),                    # state — replicated
            P(),                    # rng — replicated
            in_data_k,              # batch_inputs_k — [K, sharded, ...]
            P(None, batch_axis),    # batch_labels_k — [K, sharded]
            in_times_k,             # batch_integration_timesteps_k — [K, sharded, ...]
        )
        out_specs_k = (P(), P(), P(), P(), P())

        mapped_k_fn = shard_map(sharded_k_steps, mesh=mesh,
                                in_specs=in_specs_k, out_specs=out_specs_k,
                                check_rep=False)
        jitted_k_fn = jax.jit(mapped_k_fn, donate_argnums=(0,))

        def k_steps_compatible(state, rng, batch_inputs_k, batch_labels_k,
                               batch_integration_timesteps_k, _batchnorm, _ignore_times):
            return jitted_k_fn(state, rng, batch_inputs_k, batch_labels_k,
                               batch_integration_timesteps_k)

        _outer_desc = (
            f"nesterov(lr={diloco_outer_lr},β={diloco_outer_momentum})"
            if diloco_outer == 'nesterov' else "naive-avg"
        )
        print(f"[JIT] Created hierarchical shard_map train_step — Local Steps (lax.scan) "
              f"(K={local_steps_k}: {local_steps_k} local pmean('gpus') steps + "
              f"1 outer [{_outer_desc}] per group)")
        return {
            'step_fn': k_steps_compatible,
            'K': local_steps_k,
            'mode': 'local_steps_scan',
        }

        # ── Option 1 (commented out): Python-level dispatch (51c3ef3f) ──
        # Two separate JIT functions, Python loop dispatches sync every K steps.
        # Functionally equivalent to Option 2 but ~30% slower due to K Python→XLA
        # dispatch calls per group vs 1 for lax.scan. Kept for reference.
        #
        # def sync_params_body(params):
        #     return jax.lax.pmean(params, axis_name='nodes')
        #
        # sync_mapped = shard_map(sync_params_body, mesh=mesh,
        #                         in_specs=P(), out_specs=P(),
        #                         check_rep=False)
        # sync_params_fn = jax.jit(sync_mapped)
        #
        # return {
        #     'step_fn': compatible_fn,
        #     'sync_fn': sync_params_fn,
        #     'K': local_steps_k,
        #     'mode': 'local_steps',
        # }
    else:
        print(f"[JIT] Created hierarchical shard_map train_step "
              f"(2D mesh, pmean('gpus') + pmean('nodes'))")
        return compatible_fn


def create_jit_eval_step(mesh, state, has_book_data=True):
    """Create JIT-compiled eval_step with explicit sharding."""
    from lob.sharding_utils import create_state_shardings, get_data_shardings_for_batch

    state_shardings = create_state_shardings(state, mesh)
    inputs_shardings, labels_sharding, timesteps_shardings = get_data_shardings_for_batch(mesh, has_book_data=has_book_data)

    in_shardings = (
        inputs_shardings,   # batch_inputs
        labels_sharding,    # batch_labels
        timesteps_shardings,# batch_integration_timesteps
        state_shardings,    # state
        None,               # init_hiddens
    )
    out_shardings = (None, None, None)

    jit_eval_step = jax.jit(
        eval_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=(4, 5, 6, 8),  # apply_fn, batchnorm, apply_method, ignore_times
    )
    print("[JIT] Created JIT-compiled eval_step with sharding")
    return jit_eval_step

