"""
Opaque model handles — framework-independent boundary types.

Runtime tensors / PreTrainedModel / PeftModel instances must never appear in
domain modules. Ports accept and return these opaque aliases so application
code cannot depend on HuggingFace or Torch types.
"""

from __future__ import annotations

from typing import NewType

# Opaque runtime handles owned by infrastructure backends.
BaseModelHandle = NewType("BaseModelHandle", object)
TrainableModelHandle = NewType("TrainableModelHandle", object)
