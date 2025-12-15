import sys
import types
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so "backend" can be imported during tests
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Provide a lightweight torch stub so tests that import torch can run
# without the heavy dependency being installed. The stub implements only
# the minimal surface used by the test suite and Stage A mocks.
class _Tensor:
    def __init__(self, array):
        self._array = np.array(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def unsqueeze(self, axis):
        return _Tensor(np.expand_dims(self._array, axis))

    def __getitem__(self, idx):
        return _Tensor(self._array[idx])


def _tensor(data, dtype=None):
    return _Tensor(np.array(data, dtype=dtype or np.float32))


def _zeros(shape, dtype=np.float32):
    return _Tensor(np.zeros(shape, dtype=dtype))


torch_stub = types.SimpleNamespace(
    tensor=_tensor,
    zeros=_zeros,
    float32=np.float32,
)

# Install stub if real torch is unavailable
if "torch" not in sys.modules:
    sys.modules["torch"] = torch_stub
