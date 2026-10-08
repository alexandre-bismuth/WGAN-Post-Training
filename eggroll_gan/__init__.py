"""EGGROLL-GAN: adversarial post-training of a pretrained Mamba3 LOB generator.

The frozen Mamba3-78M generator is updated by EGGROLL zero-order evolution
strategies (low-rank LoRA on the in/out projections); a WGAN critic (frozen
Mamba3 backbone + spectral-normed head) is trained by backprop. No gradient
flows into the generator. See ``config.py`` and the top-level ``README.md``.

Subpackages: ``data``, ``critic``, ``es``, ``training``, ``eval``, ``baselines``, ``tests``.
"""

from . import _paths  # noqa: F401  -- put the vendored lobmamba package on sys.path (must run first)
from . import config  # noqa: F401

__all__ = ["config"]
