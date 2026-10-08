"""Pytest configuration for the EGGROLL-GAN stage-gate suite.

The CPU gates run as subprocesses (see test_gates.py); GPU-only gates are marked
``gpu`` and skipped unless ``--run-gpu`` is passed.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False,
                     help="also run the GPU-only stage gates (need a CUDA device)")


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu: login-node-safe CPU stage gate")
    config.addinivalue_line("markers", "gpu: requires a GPU (skipped unless --run-gpu)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
