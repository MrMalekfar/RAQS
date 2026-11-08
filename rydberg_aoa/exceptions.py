"""F05 — dedicated physics exception hierarchy (fail-fast contract).

Non-negotiable rule
-------------------
For every PHYSICS-DERIVED quantity, if the designated calculation path
fails, returns invalid data, or returns a physically unusable value, the
production code MUST raise one of the exceptions below and MUST NOT
continue the simulation.

Forbidden substitutions (F05):

* ``0`` / ``0.0``
* ``None``
* resonance (zero detuning)
* a cached / previous value
* an approximate analytical value
* a dipole-based alternative
* a hard-coded constant
* a default parameter
* ``warnings.warn(...)`` + continue

Placement note (why not ``rydberg_aoa/physics/exceptions.py``)
--------------------------------------------------------------
``rydberg_aoa/physics/__init__.py`` eagerly imports ``.detuning``,
``.hamiltonian``, ``.lindblad`` and ``.susceptibility``; those in turn
import ``rydberg_aoa.config.laser_rf_config`` and
``rydberg_aoa.config.atomic_config``.  The ARC layer
(``rydberg_aoa/arc/rabi_frequency.py``, ``.../dipole.py``) and the config
layer must be able to raise these exceptions while *they themselves* are
still being imported, so the exception module must have an import-cheap
parent package.  ``rydberg_aoa/__init__.py`` imports nothing, therefore the
hierarchy lives at the package root.  Only one small module is created — no
large exception framework.

Backward compatibility
----------------------
:class:`PhysicsError` derives from :class:`RuntimeError` so that pre-existing
``except RuntimeError`` / ``pytest.raises(RuntimeError)`` contracts (RF-003
detuning cache-miss contract, ARC decay-rate contract, dipole-matrix
self-heal contract, …) keep behaving identically after F05.
"""

__all__ = [
    "PhysicsError",
    "PhysicsConfigurationError",
    "PhysicsCalculationError",
]


class PhysicsError(RuntimeError):
    """Base class for fail-fast physics failures.

    Derives from :class:`RuntimeError` so that historical ``RuntimeError``
    contracts remain satisfied while callers can still catch the dedicated
    hierarchy.
    """


class PhysicsConfigurationError(PhysicsError):
    """A REQUIRED physics input/configuration value is missing or invalid.

    Raised when a required parameter, level, atom or configuration key that
    the model needs cannot be supplied — for example a missing detuning
    entry, a non-positive laser power/waist, or an unavailable ARC atom.
    Never use this to substitute a default value.
    """


class PhysicsCalculationError(PhysicsError):
    """A designated physics CALCULATION failed or returned unusable data.

    Raised when the designated calculation path (ARC Rabi frequency, dipole
    matrix element, transition frequency, detuning, …) raises, returns
    ``None``, returns NaN/Inf, or returns a physically unusable value such as
    a zero for a transition the model explicitly drives.

    Messages must carry enough context to diagnose the failure: the quantity,
    the transition, the relevant quantum states, the input parameters and —
    when applicable — the original exception (use ``raise ... from exc``).
    """
