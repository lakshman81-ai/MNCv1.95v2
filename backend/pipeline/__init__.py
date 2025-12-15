"""Pipeline package initializer.

This module aliases the top‑level stage and utility modules into the
``backend.pipeline`` namespace.  When running under a flat module
layout (where modules like ``stage_a.py`` live at the project root),
importing ``backend.pipeline.stage_a`` will resolve to the
corresponding root‑level module.  This enables tests written
against a namespaced package structure to work without modifying
import paths.
"""

from __future__ import annotations

import sys
import os
from importlib import import_module

# Determine the project root (two levels up from this file)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

# List of top‑level modules that correspond to pipeline stages and utilities.
# We intentionally omit 'models' here because a local ``backend/pipeline/models.py``
# file is provided.  Importing ``backend.pipeline.models`` should resolve to
# that local file rather than aliasing the top‑level ``models`` module.  If
# you add new modules at the project root that need aliasing, include them
# here.
_MODULES = [
    'stage_a',
    'stage_b',
    'stage_c',
    'stage_d',
    'detectors',
    'config',
    # 'models' is intentionally excluded to allow the local models module to be used
]

# Alias root‑level modules into the backend.pipeline namespace
for _mod_name in _MODULES:
    # Skip aliasing if a local file exists (e.g., backend/pipeline/stage_a.py)
    _local_path = os.path.join(os.path.dirname(__file__), f"{_mod_name}.py")
    if os.path.exists(_local_path):
        # A local module is present; do not alias the top-level module.
        continue
    try:
        _mod = import_module(_mod_name)
        sys.modules[f'backend.pipeline.{_mod_name}'] = _mod
    except Exception:
        # If the module cannot be imported, skip aliasing; tests may handle
        # missing optional dependencies separately.
        pass

# Re‑export common entry points from stage modules (for convenience)
try:
    from stage_a import load_and_preprocess  # type: ignore
    from stage_b import extract_features  # type: ignore
    from stage_c import apply_theory  # type: ignore
    from stage_d import quantize_and_render  # type: ignore
    from transcribe import transcribe  # type: ignore
except Exception:
    # If these imports fail, the test suite will surface an error
    pass

# Define what is available when importing backend.pipeline
__all__ = [
    'load_and_preprocess',
    'extract_features',
    'apply_theory',
    'quantize_and_render',
    'transcribe',
]
