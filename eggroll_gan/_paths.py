"""Put the vendored Mamba3 model package (``lobmamba``) on ``sys.path`` so the
first-party modules can ``import lob`` / ``s5`` / ``preproc`` / ``utils`` as
top-level packages.

Imported once by ``eggroll_gan/__init__.py`` before anything else, so every
``import eggroll_gan...`` (and ``python -m eggroll_gan...``) sets the path before
any module reaches its top-level ``from lob ...`` import.
"""
import sys

from .config import MAMBA_ROOT

if MAMBA_ROOT not in sys.path:
    sys.path.insert(0, MAMBA_ROOT)
