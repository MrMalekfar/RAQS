"""Opt-in internal-core transient and transmission IF; no legacy fallback."""
import numpy as np

from ..config.magnetic_core import CoreTemporalDiagnostic
from ..config.run_numerics import TemporalODE
from ..arc._scan_validation import real_scalar
from ..arc.rf_orientation import RFTransitionOrientation
from ..physics.magnetic_core import MagneticCoreModel
from ..utils.density_validation import validate_density_matrix
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError,PhysicsError


def run_magnetic_core_temporal(model,diagnostic,ode,omega_signal_rad_per_us,signed_if_mhz,*,absorption_prefactors=None,transit_rate_per_us=0.):
    """Declared initial state; outputs are certified before optical observables.

    The weighted coherence matches the established four-state observable gauge.
    The RF signal uses the model's LO polarization and prepared-reference gauge.
    Its product with |d_ref| is the positive-frequency dipole component [C m],
    not a real-field peak amplitude or a laboratory-calibrated detector voltage.
    """
    import qutip as qt
    from .time_domain import signal_coefficient
    if (not isinstance(model,MagneticCoreModel) or not isinstance(diagnostic,CoreTemporalDiagnostic) or
            (not isinstance(ode,TemporalODE) and not (isinstance(diagnostic,CoreTemporalDiagnostic) and diagnostic.backend=='charge_block_periodic' and ode is None))):
        raise PhysicsConfigurationError('BBR-014: typed model, diagnostic grid and explicit ODE controls required.')
    if not isinstance(model.rf_orientation,RFTransitionOrientation):
        raise PhysicsConfigurationError('BBR-014: explicit atomic RF orientation required; no legacy default.')
    omega=real_scalar(omega_signal_rad_per_us,'BBR-014 signal Rabi [rad/us]')
    frequency=real_scalar(signed_if_mhz,'BBR-014 signed laboratory IF [MHz]')
    reference=real_scalar(model.probe_reference_dipole_c_m,'BBR-014 reference probe dipole [C m]')
    if reference<=0:raise PhysicsConfigurationError('BBR-014: positive reference probe dipole required.')
    if omega<0:raise PhysicsConfigurationError('BBR-014: nonnegative signal amplitude required; phase is separate.')
    try:times=np.linspace(0.,diagnostic.duration_us,diagnostic.sample_count)
    except (ValueError,OverflowError,MemoryError) as exc:
        raise PhysicsConfigurationError('BBR-014: cannot construct requested diagnostic grid.') from exc
    if not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise PhysicsConfigurationError('BBR-014: diagnostic grid must be finite and strictly increasing.')
    transit=real_scalar(transit_rate_per_us,'BBR-015 transit [1/us]')
    if transit<0 or (transit and diagnostic.backend=='qutip'):
        raise PhysicsConfigurationError('BBR-015: nonnegative transit requires explicit charge-block runtime.')
    if diagnostic.extract_if:
        from ..utils.numerical import _if_fit_grid
        if absorption_prefactors is None:
            raise PhysicsConfigurationError('BBR-015: transmission calibration required for IF extraction.')
        try:
            _,retained,_=_if_fit_grid(times,frequency,diagnostic.transient_delay_us)
        except ValueError as exc:
            raise PhysicsConfigurationError('BBR-015: invalid IF acquisition grid.') from exc
        if (retained[-1]-retained[0])*abs(frequency)<2:
            raise PhysicsConfigurationError('BBR-015: retain at least two IF periods.')
    if diagnostic.backend=='charge_block_periodic':
        from .magnetic_core_periodic import run_periodic_core
        try:
            return run_periodic_core(model,diagnostic,times,omega,frequency,transit,absorption_prefactors)
        except PhysicsError:raise
        except Exception as exc:
            raise PhysicsCalculationError('BBR-017: periodic solve failed; no retry or alternative backend.') from exc
    n=len(model.basis);initial=qt.Qobj(model.initial_density())
    validate_density_matrix(initial,context='BBR-014 prepared initial state',expected_dim=n)
    operator=qt.Qobj(model.rf_signal_operator);observable=qt.Qobj(model.probe_coherence_operator)
    amplitude=complex(omega*np.exp(1j*diagnostic.signal_phase_rad))
    def coefficient(t,args=None):return signal_coefficient(t,amplitude,frequency,model.rf_orientation)
    def partner(t,args=None):return np.conj(coefficient(t,args))
    h=[qt.Qobj(model.hamiltonian),[operator,coefficient],[operator.dag(),partner]]
    count=[0];worst=dict(trace_error=0.,hermiticity_error=0.,min_eigenvalue=0.)
    def measure(t,state):
        index=count[0]
        if index>=len(times) or not np.isfinite(t) or t!=times[index]:
            raise PhysicsCalculationError(f'BBR-014: unexpected certification timestamp/index {t!r}/{index}.')
        metrics=validate_density_matrix(state,context=f'BBR-014 output {index} at {t} us',expected_dim=n)
        worst['trace_error']=max(worst['trace_error'],metrics['trace_error'])
        worst['hermiticity_error']=max(worst['hermiticity_error'],metrics['hermiticity_error'])
        worst['min_eigenvalue']=min(worst['min_eigenvalue'],metrics['min_eigenvalue'])
        count[0]+=1
        return qt.expect(observable,state if isinstance(state,qt.Qobj) else qt.Qobj(state))
    backend_report=None
    if diagnostic.backend=='charge_block_bdf':
        from .magnetic_core_blocks import integrate_charge_blocks
        values,backend_report=integrate_charge_blocks(model,diagnostic,ode,times,coefficient,transit,measure)
    else:
        try:
            result=qt.mesolve(h,initial,times,[qt.Qobj(x) for x in model.collapse_operators],
                             e_ops=[measure],options={**ode.options(),'store_states':False})
        except PhysicsError:raise
        except Exception as exc:
            raise PhysicsCalculationError('BBR-014: internal-core temporal solve failed; no retry or substitute.') from exc
        try:values=np.asarray(result.expect[0])
        except (AttributeError,IndexError,TypeError,ValueError) as exc:
            raise PhysicsCalculationError('BBR-014: missing or malformed solver observable result.') from exc
    if count[0]!=len(times):
        raise PhysicsCalculationError(f'BBR-014: certified {count[0]} of {len(times)} requested outputs.')
    if values.shape!=times.shape or values.dtype.kind not in 'iufc' or not np.isfinite(values).all():
        raise PhysicsCalculationError('BBR-014: invalid weighted optical coherence output.')
    transmission=None
    if absorption_prefactors is not None:
        from ..physics.susceptibility.calculate_transmission_from_rho21 import calculate_transmission_from_rho21
        transmission=[float(calculate_transmission_from_rho21(x,absorption_prefactors=absorption_prefactors)) for x in values]
    with np.errstate(over='ignore',invalid='ignore'):dipole=reference*values
    if not np.isfinite(dipole).all():
        raise PhysicsCalculationError('BBR-014: nonfinite positive-frequency dipole; no clipping.')
    if_result=None
    if diagnostic.extract_if:
        from ..utils.numerical import extract_if_component
        try:
            amp,_,_=extract_if_component(transmission,times,frequency,diagnostic.transient_delay_us)
        except ValueError as exc:
            raise PhysicsCalculationError('BBR-015: transmission IF fit failed.') from exc
        if_result=dict(real=float(amp.real),imag=float(amp.imag),magnitude=float(abs(amp)),
            phase_rad=float(np.angle(amp)) if omega>0 and amp!=0 else None,
            phase_defined=bool(omega>0 and amp!=0),phase_qualified=False,
            phase_scope='raw sinusoidal-fit candidate; numerical convergence and detectability not certified',
            phase_epoch_us=float(retained[0]),signed_fit_frequency_mhz=frequency,
            units='dimensionless transmission peak amplitude',
            observable='Beer-Lambert transmission of ARC-weighted coherence',
            zero_signal_semantics='residual fit retained; phase undefined when applied signal is OFF')
    return dict(scope='internal four-manifold transient only; no outside-FS loss/return, Doppler or noise',
        backend_report=backend_report,if_result=if_result,
        initial_state=diagnostic.initial_state,
        dimension=n,times_us=times.tolist(),coherence_real=values.real.tolist(),coherence_imag=values.imag.tolist(),
        positive_frequency_dipole_real_c_m=dipole.real.tolist(),positive_frequency_dipole_imag_c_m=dipole.imag.tolist(),
        transmission=transmission,certified_outputs=count[0],density_metrics=worst,
        ode_options=ode.options(),normalize_output=False,
        signed_if_mhz=frequency,signal_amplitude_rad_per_us=omega,signal_phase_rad=diagnostic.signal_phase_rad,
        signal_drive_included=omega>0,IF_or_AoA_qualified=False,IF_extraction_executed=diagnostic.extract_if)
