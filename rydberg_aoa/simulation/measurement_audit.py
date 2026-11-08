"""Non-mutating PIN calibration receipts; no Hamiltonian or solver changes."""
import math
from scipy.constants import epsilon_0, hbar, c
from ..config.laboratory import _real
from ..exceptions import PhysicsCalculationError


def audit_pin_context(profile, medium, context, probe_rabi, probe_dipole):
    """Check the same declared Gaussian channel drives Rabi and susceptibility.

    Power in W, 1/e² intensity radius in m, dipole in C m, Rabi rad/us.
    The relative tolerance is algebraic consistency, not an ODE tolerance.
    No measured input, context, or physical result is modified.
    """
    try:
        omega = _real(probe_rabi, 'NOISE-018 probe Rabi [rad/us]', positive=True)*1e6
        mu = _real(probe_dipole, 'NOISE-018 probe dipole [C m]', positive=True)
        power = profile.probe.power_mw/1000
        radius = profile.probe.beam_waist_mm/1000
        field = math.sqrt(4*power/(math.pi*radius**2*c*epsilon_0))
        expected_omega = mu*field/hbar
        expected_chi = -2*medium.number_density_m3*mu**2/(epsilon_0*hbar*omega)
        expected_od = math.pi*medium.cell_length_m/context.probe_wavelength_m
        pairs = {'probe_rabi_rad_per_s': (omega, expected_omega),
                 'susceptibility_factor': (context.susceptibility_factor, expected_chi),
                 'od_factor': (context.od_factor, expected_od),
                 'incident_probe_power_w': (context.incident_probe_power_w, power),
                 'probe_wavelength_m': (context.probe_wavelength_m, profile.probe.wavelength_nm*1e-9)}
        for name, (actual, expected) in pairs.items():
            actual = _real(actual, 'NOISE-018 executed '+name)
            if not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=0.):
                raise PhysicsCalculationError(f'NOISE-018: inconsistent PIN calibration {name}: actual={actual!r}, expected={expected!r}; no repair.')
        return {'status': 'same_declared_probe_channel_verified', 'probe_dipole_c_m': mu,
                'gaussian_peak_field_v_per_m': field, 'rabi_rad_per_s': omega,
                'forward_context': dict(vars(context)), 'algebraic_relative_tolerance': 1e-10,
                'scope': 'unit/channel consistency, not experimental calibration or ODE convergence'}
    except PhysicsCalculationError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError('NOISE-018: PIN calibration audit failed; no substituted context.') from exc


def check_fixed_readout(first, current):
    """Comparisons must retain the actual optical/detector transfer parameters."""
    a, b = first.get('readout_noise'), current.get('readout_noise')
    if a is None and b is None:
        return  # non-PIN interfaces and synthetic non-readout test seams
    if a is None or b is None:
        raise PhysicsCalculationError('NOISE-018: readout receipt disappeared between compared jobs.')
    if a.get('model') != 'pin_boxcar_sample_hold' and b.get('model') != 'pin_boxcar_sample_hold':
        return
    for key in ('model', 'units', 'detector', 'forward_context', 'integration_time_s'):
        if key not in a or key not in b or a[key] != b[key]:
            raise PhysicsCalculationError(f'NOISE-018: fixed readout {key} changed; no valid comparison.')
