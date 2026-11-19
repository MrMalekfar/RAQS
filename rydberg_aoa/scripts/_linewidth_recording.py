"""Record the solver calls of one unmodified pipeline run (analysis only).

Used by ``compare_linewidth_models``.  The pipeline code is not changed:
for the duration of one ``with`` block the two names the pipeline looks up
at call time (``steady_state.simulate_spectrum`` and the validation hook of
``laboratory_execution.LaboratoryWorkerTask``) are wrapped by recorders that
forward every call unchanged, and are restored afterwards even on failure.
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def record_solver_calls():
    from ..simulation import laboratory_execution, steady_state
    calls = {"spectrum": [], "tasks": []}
    original_spectrum = steady_state.simulate_spectrum
    task_class = laboratory_execution.LaboratoryWorkerTask
    original_post_init = task_class.__post_init__

    def recording_spectrum(*args, **kwargs):
        calls["spectrum"].append((args, dict(kwargs)))
        return original_spectrum(*args, **kwargs)

    def recording_post_init(self):
        original_post_init(self)
        calls["tasks"].append(self)

    # The task class itself must stay the module attribute (tasks are pickled
    # for the spawn pool), so only its validation hook is wrapped.
    steady_state.simulate_spectrum = recording_spectrum
    task_class.__post_init__ = recording_post_init
    try:
        yield calls
    finally:
        steady_state.simulate_spectrum = original_spectrum
        task_class.__post_init__ = original_post_init


def published_rf_orientation():
    from ..config import atomic_config
    return atomic_config.get_rf_orientation()
