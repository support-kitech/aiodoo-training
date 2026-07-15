"""Opaque optimizer / scheduler handles — no framework types."""

from __future__ import annotations

from typing import NewType

OptimizerHandle = NewType("OptimizerHandle", object)
SchedulerHandle = NewType("SchedulerHandle", object)
