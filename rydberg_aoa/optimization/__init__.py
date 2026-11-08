"""
Optimization module for two-stage level selection.

Legacy package shim: the concrete optimizer implementations now live in
``rydberg_aoa.level_finder`` (single-stage ARC architecture). This package
retains no imports so that importing it can never raise ImportError
(RF-001 removed the dead ``level_optimizer`` import and stale ``__all__``).
"""

__all__ = []

