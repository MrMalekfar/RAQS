"""LAB-002: explicit RF publication and consistent lab-input snapshots.

LO/signal declarations are authoritative when explicitly passed to the setter.
Reading an inconsistent legacy configuration raises; it never reconciles it.
No atomic initialization, resonance tuning or scan runs here.
"""

from dataclasses import asdict

from .laboratory import AppliedRF, AppliedLaser, ExperimentalLabRequest, _real
from ..exceptions import PhysicsConfigurationError


def _require_pair(lo, signal):
    if not isinstance(lo, AppliedRF) or not isinstance(signal, AppliedRF):
        raise PhysicsConfigurationError('LAB-002: LO/signal require explicit AppliedRF declarations.')
    if lo.polarization != signal.polarization:
        raise PhysicsConfigurationError('LAB-002: LO/signal must drive the same RF polarization channel.')


def read_configured_rf_fields():
    """Read applied LO/signal and reject disagreement among compatibility views."""
    from . import parameters as params
    from .laser_rf_config import LO_RF_PARAMS, RF_PARAMS, SIGNAL_RF_PARAMS
    try:
        lo, signal = AppliedRF(**LO_RF_PARAMS), AppliedRF(**SIGNAL_RF_PARAMS)
        selection = AppliedRF(**RF_PARAMS)
    except TypeError as exc:
        raise PhysicsConfigurationError('LAB-002: missing or unexpected RF configuration field.') from exc
    _require_pair(lo, signal)
    if selection != lo:
        raise PhysicsConfigurationError('LAB-002: RF_PARAMS disagrees with LO_RF_PARAMS; selection carrier is ambiguous.')
    expected = {'RF_LO_FREQUENCY': lo.frequency_hz, 'RF_LO_E_FIELD': lo.electric_field_vm,
                'RF_SIGNAL_FREQUENCY': signal.frequency_hz, 'RF_SIGNAL_E_FIELD': signal.electric_field_vm,
                'RF_FREQUENCY': signal.frequency_hz,
                'IF_FREQ': (signal.frequency_hz - lo.frequency_hz) / 1e6}
    for key, value in expected.items():
        actual = _real(getattr(params, key), key)
        if actual != value:
            raise PhysicsConfigurationError(f'LAB-002: {key}={actual} disagrees with applied RF fields ({value}); no silent reconciliation.')
    return lo, signal


def configure_rf_fields(*, lo, signal):
    """Publish an explicit pair to all existing RF views, preserving dict aliases.

    Validate everything before writes. Once physics is initialized, a changed
    pair is refused: RF publication is not a cache rebuild or force-init API.
    Before initialization, changed inputs invalidate the four derived caches.
    Spacing, record duration and other independent user inputs stay unchanged.
    """
    _require_pair(lo, signal)
    from . import atomic_config as ac, parameters as params
    from .laser_rf_config import LO_RF_PARAMS, RF_PARAMS, SIGNAL_RF_PARAMS
    lo_values, signal_values = asdict(lo), asdict(signal)
    scalars = {'RF_LO_FREQUENCY': lo.frequency_hz, 'RF_LO_E_FIELD': lo.electric_field_vm,
               'RF_SIGNAL_FREQUENCY': signal.frequency_hz, 'RF_SIGNAL_E_FIELD': signal.electric_field_vm,
               'RF_FREQUENCY': signal.frequency_hz,
               'IF_FREQ': (signal.frequency_hz - lo.frequency_hz) / 1e6}
    # Check every existing view before equality comparisons (and before any
    # short-circuit can bypass a malformed later view).
    try:
        for mapping in (LO_RF_PARAMS, RF_PARAMS, SIGNAL_RF_PARAMS):
            AppliedRF(**mapping)
        for key in scalars:
            _real(getattr(params, key), key)
    except (PhysicsConfigurationError, TypeError) as exc:
        raise PhysicsConfigurationError('LAB-002: existing RF views are malformed; no partial publication.') from exc
    try:
        changed = (LO_RF_PARAMS != lo_values or RF_PARAMS != lo_values or
                   SIGNAL_RF_PARAMS != signal_values or
                   any(getattr(params, key) != value for key, value in scalars.items()))
    except (TypeError, ValueError) as exc:
        raise PhysicsConfigurationError('LAB-002: existing RF views are malformed; no partial publication.') from exc
    if ac.is_atomic_physics_initialized() and changed:
        raise PhysicsConfigurationError('LAB-002: RF inputs cannot change while atomic physics is published; explicitly reset/reinitialize first.')
    if not changed:
        return
    for mapping, values in ((LO_RF_PARAMS, lo_values), (RF_PARAMS, lo_values), (SIGNAL_RF_PARAMS, signal_values)):
        mapping.clear()
        mapping.update(values)
    for key, value in scalars.items():
        setattr(params, key, value)
    for key in ('_ATOMIC_LEVELS_CACHE', '_DETUNING_CACHE', '_DIPOLE_MATRIX_CACHE', '_OPTIMIZATION_RESULTS_CACHE'):
        getattr(ac, key).clear()


def build_configured_lab_request(*, preparation, bounds, atom):
    """Capture explicit preparation/hardware and coherent applied config.

    Requires the actual atom to validate F and the HFS→FS nuclear projection.
    Does not install the request in the legacy initialization pipeline; future
    lab-selection stages must consume it explicitly rather than ignore fields.
    """
    from .laser_rf_config import LASER_PARAMS
    lo, signal = read_configured_rf_fields()
    try:
        probe = AppliedLaser(**LASER_PARAMS['probe'])
        coupling = AppliedLaser(**LASER_PARAMS['coupling'])
    except (KeyError, TypeError) as exc:
        raise PhysicsConfigurationError('LAB-002: incomplete applied laser configuration.') from exc
    request = ExperimentalLabRequest(probe, coupling, lo, signal, preparation, bounds)
    request.validate_for_atom(atom)
    return request
