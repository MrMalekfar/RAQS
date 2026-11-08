"""Runtime DOA estimator registry.

Estimator selection is resolved at call time so configuration can change the
active method without re-import (there is deliberately no import-time
snapshot of the method name).  The dedicated post-processing configuration
``rydberg_aoa.config.doa_config.DOA_METHOD`` is the configuration owner and
is read LAZILY at every call; the legacy ``parameters.DOA_METHOD`` (when
present) is retained only as a fallback for backward compatibility, and the
package-level default below applies last.  An explicit ``name`` argument
always takes precedence over any configuration value.
"""

from __future__ import annotations

from .covariance_estimators import (
    BartlettEstimator,
    MUSICEstimator,
    MVDREstimator,
    SnapshotDOAEstimator,
)
from .estimators import DOAEstimator, PhaseBeamformingEstimator
from .structured_estimators import ESPRITEstimator, RootMUSICEstimator

DEFAULT_DOA_METHOD = "phase_beamforming"

_ESTIMATOR_CLASSES = {
    "phase_beamforming": PhaseBeamformingEstimator,
    "bartlett": BartlettEstimator,
    "mvdr": MVDREstimator,
    "capon": MVDREstimator,
    "music": MUSICEstimator,
    "esprit": ESPRITEstimator,
    "root_music": RootMUSICEstimator,
}


def register_estimator(name, estimator_cls):
    """Register an estimator class for future algorithms."""
    _ESTIMATOR_CLASSES[name] = estimator_cls


def _resolve_configured_method():
    """Resolve the configured default method lazily at call time.

    Priority (DOA-INTEGRATE-001):
        1. config.doa_config.DOA_METHOD   (dedicated DOA post-processing owner)
        2. parameters.DOA_METHOD          (legacy fallback, if present)
        3. DEFAULT_DOA_METHOD             (package default)
    """
    try:
        from ..config import doa_config
        value = getattr(doa_config, "DOA_METHOD", None)
        if value is not None:
            return value
    except ImportError:
        pass
    try:
        from ..config import parameters as params
        value = getattr(params, "DOA_METHOD", None)
        if value is not None:
            return value
    except ImportError:
        pass
    return DEFAULT_DOA_METHOD


def get_doa_estimator(name=None, **kwargs):
    """Instantiate the named estimator.

    Args:
        name: estimator name; when None the active configuration value
            (config.doa_config.DOA_METHOD, read lazily at every call) is
            used.  An explicit name ALWAYS takes precedence.
        kwargs: forwarded to the estimator constructor (for example
            scan_azimuth_deg / elevation_deg for every estimator,
            n_sources for MUSIC/ESPRIT/Root-MUSIC, loading_rel for
            MVDR/Capon).

    Returns:
        A DOAEstimator instance.

    Raises:
        ValueError: for an unknown estimator name.
    """
    if name is None:
        name = _resolve_configured_method()
    if name not in _ESTIMATOR_CLASSES:
        available = ", ".join(sorted(_ESTIMATOR_CLASSES))
        raise ValueError(
            f"Unknown DOA estimator {name!r}. Available estimators: {available}"
        )
    return _ESTIMATOR_CLASSES[name](**kwargs)