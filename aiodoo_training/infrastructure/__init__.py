"""
Infrastructure adapters (HuggingFace, PEFT, Accelerate, resources, storage, trackers).

This package is the only allowed home for third-party ML library imports.
Domain, ports, and pipeline code must never import from here directly.
Resource planners that probe hardware also live here (CPU planner first).
"""
