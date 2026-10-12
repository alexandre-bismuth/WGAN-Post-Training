"""Pytest wrappers over the stage-gate modules.

Each gate module (``s2_head_sigma0``, ``s3_es_rollout``, ``s5b_proj_gates``) has a
``__main__`` that runs its checks and exits non-zero on failure. We run them as
subprocesses in a CPU-only, thread-limited environment so the suite is
login-node safe. With no flags the modules run their CPU checks; the GPU-only
checks (``--rollout`` / ``--use_checkpoint`` / ``--run``) are exercised by the
``gpu``-marked tests, which are skipped unless ``--run-gpu`` is passed.
"""
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAMBA = os.path.join(_ROOT, "lobmamba")


def _cpu_env():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Keep the thread footprint tiny: these are correctness gates, not perf, and the login node's
    # per-user thread limit (RLIMIT_NPROC) is shared — an unbounded XLA/Eigen threadpool trips
    # `pthread_create: Resource temporarily unavailable`. Single-threaded CPU is plenty here.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = ("--xla_force_host_platform_device_count=1 "
                        "--xla_cpu_multi_thread_eigen=false")
    env["PYTHONPATH"] = os.pathsep.join([_ROOT, _MAMBA, env.get("PYTHONPATH", "")])
    return env


def _run(module, *args, timeout=1800):
    cmd = [sys.executable, "-u", "-m", module, *args]
    # Cap CPU affinity to a small core set when `taskset` is available: on a 144-core login node JAX
    # sizes its CPU/LLVM-JIT threadpools to nproc, and a conv compile (the learned-critic TCN gate) can
    # spawn enough pthreads to trip the per-user RLIMIT_NPROC -> 'pthread_create failed' aborts. Harmless
    # for the lighter pure-jnp gates; essential for the conv gate. (Belt-and-suspenders with the
    # OMP/MKL/eigen thread caps in _cpu_env.)
    import shutil
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", "0-7"] + cmd
    r = subprocess.run(cmd, cwd=_ROOT, env=_cpu_env(),
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:  # surface the tail of the gate's own output on failure
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
    assert r.returncode == 0, f"{module} exited {r.returncode}"


# ---- CPU gates (login-node safe; run by default) ----------------------------
@pytest.mark.cpu
def test_s2_head_sigma0_cpu():
    _run("eggroll_gan.tests.s2_head_sigma0")


@pytest.mark.cpu
def test_s3_es_rollout_cpu():
    _run("eggroll_gan.tests.s3_es_rollout")


@pytest.mark.cpu
def test_s5b_proj_gates_cpu():
    _run("eggroll_gan.tests.s5b_proj_gates")


@pytest.mark.cpu
def test_muon_gate_cpu():
    _run("eggroll_gan.tests.muon_gate")


@pytest.mark.cpu
def test_learned_critic_cpu():
    # learned TCN critic: descriptor sequence map, encoder forward (tcn/cnn × attn/mean/last),
    # spectral-norm activity, d-step real-vs-fake separation, and R1 shrinking the input-grad norm.
    _run("eggroll_gan.critic.learned_critic")


@pytest.mark.cpu
def test_raw_message_features_cpu():
    # raw decoded-message featuriser (--critic_input raw): field channel map, NA-sentinel gating,
    # standardiser contract, and TCN real-vs-fake separation from raw fields alone.
    _run("eggroll_gan.critic.raw_message_features")


@pytest.mark.cpu
def test_r1_gate_cpu():
    # critic R1 gradient penalty: gamma=0 bit-identical-path, gamma>0 active+finite, and R1 shrinks
    # the critic input-gradient norm (the anti-hacking effect). Model-free, login-node safe.
    _run("eggroll_gan.tests.r1_gate")


@pytest.mark.cpu
def test_eval_monitor_cpu():
    _run("eggroll_gan.eval.eval_monitor")


@pytest.mark.cpu
def test_soup_cpu():
    _run("eggroll_gan.eval.soup", "--selftest")


@pytest.mark.cpu
def test_calib_report_cpu():
    _run("eggroll_gan.eval.calib_report", "--selftest")


# ---- GPU gates (need a CUDA device; skipped unless --run-gpu) ----------------
@pytest.mark.gpu
def test_s3_es_rollout_gpu():
    _run("eggroll_gan.tests.s3_es_rollout", "--rollout")


@pytest.mark.gpu
def test_s5b_proj_gates_gpu():
    _run("eggroll_gan.tests.s5b_proj_gates", "--rollout")
