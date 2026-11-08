"""Pure relative RF phase assembly and declared LO-reference restoration.

No field, rate, atomic channel or estimator sign is changed. Restoration takes
already RF-orientation-normalized data; it must not be applied twice.
"""
import numpy as np
from .snapshots import _finite_numeric_array
from ..config.lo_phase_reference import SpatialLOReference
from ..config.laboratory import _integer,_real
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError


def lo_phase_offsets(reference,n_elements):
    if not isinstance(reference,SpatialLOReference):
        raise PhysicsConfigurationError('BBR-019: explicit typed LO reference required; no assumed map.')
    count=_integer(n_elements,'BBR-019 element count',minimum=1)
    if reference.model=='uniform':return np.zeros(count)
    if len(reference.phase_offsets_rad)!=count:
        raise PhysicsConfigurationError('BBR-019: phase map count does not match data rows.')
    return np.asarray(reference.phase_offsets_rad,dtype=float)


def relative_drive_phases(signal_phases_rad,reference,phase_at_reference_rad):
    """phi_rel(m)=phi_rel(0)+phi_sig(m)-phi_sig(0)-delta_phi_LO(m)."""
    try:values=_finite_numeric_array(signal_phases_rad,'BBR-019 signal phases',real=True)
    except ValueError as exc:raise PhysicsConfigurationError('BBR-019: invalid signal phase array.') from exc
    if values.ndim!=1 or not len(values):
        raise PhysicsConfigurationError('BBR-019: real one-dimensional signal phases required.')
    if not np.isfinite(values).all():
        raise PhysicsConfigurationError('BBR-019: nonfinite signal phase; no replacement.')
    phase=_real(phase_at_reference_rad,'BBR-019 relative phase at sensor 0 [rad]')
    with np.errstate(over='ignore',invalid='ignore'):
        result=phase+values-values[0]-lo_phase_offsets(reference,len(values))
    if not np.isfinite(result).all():raise PhysicsCalculationError('BBR-019: relative phase overflow; no wrapping repair.')
    return result


def restore_signal_reference(normalized_if,reference):
    """Multiply normalized IF vector or MxK snapshots by exp(+i delta_phi_LO).

    Input must already have the canonical RF-DIR polarity. This operation
    restores the signal spatial reference and is separate from conjugation.
    """
    try:values=_finite_numeric_array(normalized_if,'BBR-019 normalized IF')
    except ValueError as exc:raise PhysicsConfigurationError('BBR-019: invalid IF array for calibration.') from exc
    if values.ndim not in (1,2) or not values.size:
        raise PhysicsConfigurationError('BBR-019: numeric response vector or sensor-by-snapshot matrix required.')
    if not np.isfinite(values).all():raise PhysicsCalculationError('BBR-019: nonfinite IF data; no phase calibration applied.')
    factors=np.exp(1j*lo_phase_offsets(reference,values.shape[0]))
    result=values*(factors if values.ndim==1 else factors[:,None])
    if not np.isfinite(result).all():raise PhysicsCalculationError('BBR-019: nonfinite calibrated IF; no clipping.')
    return result
