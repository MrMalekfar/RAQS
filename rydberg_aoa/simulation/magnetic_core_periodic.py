"""Explicit periodic core solver promoted from the verified BBR-016 reference.

Only Fourier harmonics are truncated; Hamiltonian and bath physics are unchanged.
"""
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError
from ..config.laboratory import _integer, _real
from .magnetic_core_blocks import block_generators,reconstruct_density
from ..utils.density_validation import validate_density_matrix
from ..physics.susceptibility.calculate_transmission_from_rho21 import calculate_transmission_from_rho21
from ..utils.numerical import extract_if_component,if_fit_noise_weights


def trace_norm(matrix):
    return float(np.sum(np.linalg.svd(matrix,compute_uv=False)))


def solve_periodic(model, rate, omega, frequency, phase, order):
    """(ik*nu-L0)rho_k-Lplus*rho_(k-1)-Lminus*rho_(k+1)=0.

    Replace one redundant population equation per harmonic with Tr rho_k=delta_k0.
    Evaluate residuals against the ORIGINAL equations including omitted edges.
    """
    order=_integer(order,'BBR-016 harmonic order',minimum=0)
    rate=_real(rate,'BBR-016 prepared transit [1/us]',positive=True)
    omega=_real(omega,'BBR-016 signal Rabi',nonnegative=True)
    frequency=_real(frequency,'BBR-016 laboratory IF [MHz]')
    phase=_real(phase,'BBR-016 signal phase [rad]')
    if frequency==0:raise PhysicsConfigurationError('BBR-016: nonzero IF required.')
    full,base,sig,dag=block_generators(model,rate)
    nu=2*np.pi*model.rf_orientation.direction*frequency
    amplitude=omega*np.exp(1j*phase);plus=amplitude*sig;minus=np.conj(amplitude)*dag
    size=base.shape[0];modes=np.arange(-order,order+1);identity=sparse.eye(size,format='csr')
    matrix=sparse.bmat([[1j*k*nu*identity-base if i==j else -plus if j==i-1 else
        -minus if j==i+1 else None for j in range(len(modes))] for i,k in enumerate(modes)],format='csr')
    original=matrix;matrix=matrix.tolil(copy=True)
    n=len(model.basis);diagonal=np.flatnonzero(model.coordinates%(n+1)==0)
    if len(diagonal)!=n:raise PhysicsCalculationError('BBR-016: missing density populations.')
    rhs=np.zeros(matrix.shape[0],complex)
    for i,k in enumerate(modes):
        row=i*size+int(diagonal[0]);matrix[row,:]=0.;matrix[row,i*size+diagonal]=1.;rhs[row]=float(k==0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',MatrixRankWarning)
            vector=spsolve(matrix.tocsc(),rhs)
    except Exception as exc:
        raise PhysicsCalculationError('BBR-016: Fourier solve failed; no regularization or substitute.') from exc
    if not np.isfinite(vector).all():raise PhysicsCalculationError('BBR-016: nonfinite Fourier solution.')
    coefficients=vector.reshape((len(modes),size))
    residue=(original@vector).reshape(coefficients.shape)
    # The two omitted harmonics are part of the ODE defect, not discarded evidence.
    residue=np.vstack((-minus@coefficients[0],residue,-plus@coefficients[-1]))
    full_residues=[reconstruct_density(model,x) for x in residue]
    residual_bound=sum(trace_norm(x) for x in full_residues)
    trace_defect_bound=sum(abs(np.trace(x)) for x in full_residues)
    matrices=[reconstruct_density(model,x) for x in coefficients]
    trace_error=max(abs(np.trace(x)-float(k==0)) for k,x in zip(modes,matrices))
    pairing_error=max(np.max(abs(x-matrices[-i-1].conj().T)) for i,x in enumerate(matrices))
    if not np.isfinite(residual_bound) or trace_error>1e-9 or pairing_error>1e-9:
        raise PhysicsCalculationError('BBR-016: Fourier trace/Hermitian pairing contract failed; no repair.')
    metrics=dict(harmonic_order=order,angular_frequency_rad_per_us=float(nu),
        original_equation_residual_norm=float(np.linalg.norm(original@vector)),
        uniform_trace_norm_ode_defect_bound_per_us=float(residual_bound),
        uniform_trace_defect_bound_per_us=float(trace_defect_bound),
        harmonic_trace_error=float(trace_error),harmonic_pairing_error=float(pairing_error),
        reduced_dimension=size,linear_system_dimension=matrix.shape[0])
    return coefficients,metrics


def sample_periodic(model,coefficients,frequency,times):
    """Certify every requested density before the weighted optical observable."""
    coefficients=np.asarray(coefficients);order=(len(coefficients)-1)//2
    nu=2*np.pi*model.rf_orientation.direction*frequency
    modes=np.arange(-order,order+1)
    vectors=np.exp(1j*np.outer(times,nu*modes))@coefficients
    observable=model.probe_coherence_operator
    coherence=[];worst=dict(trace_error=0.,hermiticity_error=0.,min_eigenvalue=float('inf'))
    for i,vector in enumerate(vectors):
        rho=reconstruct_density(model,vector)
        checked=validate_density_matrix(rho,context=f'BBR-016 periodic output {i}',expected_dim=len(model.basis))
        worst['trace_error']=max(worst['trace_error'],checked['trace_error'])
        worst['hermiticity_error']=max(worst['hermiticity_error'],checked['hermiticity_error'])
        worst['min_eigenvalue']=min(worst['min_eigenvalue'],checked['min_eigenvalue'])
        coherence.append(np.trace(observable@rho))
    return np.asarray(coherence),worst


def evaluate_reference(model,coefficients,frequency,times,prefactors,delay,initial):
    """Certify requested densities, use existing transmission/IF and bound ODE defect.

    Bound for e=rho_exact-rho_Fourier from prepared-reset contraction:
    ||e(t)||1 <= exp(-Gamma*t)||e(0)||1 + (1-exp(-Gamma*t))*
    (|Tr e(0)| + t*Rtrace + Rnorm/Gamma).
    Rnorm/Rtrace are uniform Fourier-sum bounds, including truncated edges.
    It uses measured floating-point residues, not interval arithmetic.
    """
    coherence,worst=sample_periodic(model,coefficients,frequency,times)
    transmission=np.array([calculate_transmission_from_rho21(x,absorption_prefactors=prefactors) for x in coherence])
    amplitude,_,_=extract_if_component(transmission,times,frequency,delay)
    difference=initial-reconstruct_density(model,np.sum(coefficients,axis=0))
    return dict(real=float(amplitude.real),imag=float(amplitude.imag),magnitude=float(abs(amplitude)),
        raw_phase_rad=float(np.angle(amplitude)) if amplitude!=0 else None,
        density_metrics=worst,certified_outputs=len(times),initial_difference_trace_norm=trace_norm(difference),
        initial_trace_error=float(abs(np.trace(difference)))),coherence,transmission


def if_error_bound(model,summary,metrics,times,transmission,frequency,delay,rate,prefactors):
    """Propagate the uniform density defect bound through exp and exact LS weights."""
    indices,weights,_=if_fit_noise_weights(times,frequency,delay)
    retained=np.asarray(times)[indices];decay=np.exp(-rate*retained)
    bound=decay*summary['initial_difference_trace_norm']+(1-decay)*(summary['initial_trace_error']+
        retained*metrics['uniform_trace_defect_bound_per_us']+
        metrics['uniform_trace_norm_ode_defect_bound_per_us']/rate)
    operator_norm=float(np.linalg.norm(model.probe_coherence_operator.toarray(),2))
    exponent_bound=2*abs(prefactors[0])*abs(prefactors[1])*operator_norm*bound
    with np.errstate(over='raise',invalid='raise'):
        try:delta=np.abs(transmission[indices])*np.expm1(exponent_bound)
        except FloatingPointError as exc:raise PhysicsCalculationError('BBR-016: IF error bound overflow; no clipping.') from exc
    estimate=float(np.dot(np.abs(weights),delta))
    if not np.isfinite(estimate):raise PhysicsCalculationError('BBR-016: nonfinite IF bound.')
    return dict(complex_IF_absolute_error_estimate=estimate,
        complex_IF_relative_error_estimate=estimate/summary['magnitude'] if summary['magnitude'] else None,
        maximum_retained_density_error_estimate=float(max(bound)),
        qualification='computed-residual contraction estimate, not an interval-certified bound',
        phase_epoch_us=float(retained[0]))


def run_periodic_core(model,diagnostic,times,omega,frequency,transit,prefactors):
    """Fixed-order periodic IF with explicit error gates; never integrates an ODE.

    Only the retained late-time window is returned. LO steady is the initial
    reference for a contraction estimate, not the returned periodic t=0 state.
    """
    from dataclasses import asdict
    from .magnetic_core_blocks import lo_steady_state
    if transit<=0:
        raise PhysicsConfigurationError('BBR-017: positive prepared transit required for periodic/transient qualification.')
    controls=diagnostic.periodic
    retained=times[times>=diagnostic.transient_delay_us]
    full,base,*_=block_generators(model,transit)
    initial,steady_metrics=lo_steady_state(model,full,base)
    initial=reconstruct_density(model,initial)
    records=[]
    for order in (controls.harmonic_order-1,controls.harmonic_order):
        coefficients,metrics=solve_periodic(model,transit,omega,frequency,diagnostic.signal_phase_rad,order)
        summary,values,transmission=evaluate_reference(model,coefficients,frequency,retained,
            prefactors,diagnostic.transient_delay_us,initial)
        records.append(dict(summary=summary,metrics=metrics))
    estimate=if_error_bound(model,summary,metrics,retained,transmission,frequency,
        diagnostic.transient_delay_us,transit,prefactors)
    transient_metrics={**metrics,'uniform_trace_norm_ode_defect_bound_per_us':0.,
                       'uniform_trace_defect_bound_per_us':0.}
    transient_estimate=if_error_bound(model,summary,transient_metrics,retained,transmission,frequency,
        diagnostic.transient_delay_us,transit,prefactors)
    refined=np.linspace(retained[0],retained[-1],2*len(retained)-1)
    fine,_,_=evaluate_reference(model,coefficients,frequency,refined,prefactors,
        diagnostic.transient_delay_us,initial)
    amp=complex(summary['real'],summary['imag'])
    previous=complex(records[0]['summary']['real'],records[0]['summary']['imag'])
    finer=complex(fine['real'],fine['imag'])
    checks=dict(defect_and_transient=estimate['complex_IF_absolute_error_estimate'],
                cutoff_difference=float(abs(amp-previous)),sample_refinement_difference=float(abs(amp-finer)))
    for name,value in checks.items():
        if not np.isfinite(value) or value<0 or value>controls.if_max_abs_error:
            raise PhysicsCalculationError(f'BBR-017: periodic IF absolute error budget exceeded: {name}={value!r}, budget={controls.if_max_abs_error!r}.')
        if omega>0 and (abs(amp)==0 or value/abs(amp)>controls.if_max_rel_error):
            raise PhysicsCalculationError(f'BBR-017: periodic IF relative error budget exceeded or unresolved IF: {name}={value!r}, amplitude={abs(amp)!r}, budget={controls.if_max_rel_error!r}.')
    dipole=model.probe_reference_dipole_c_m*values
    if not np.isfinite(dipole).all():
        raise PhysicsCalculationError('BBR-017: nonfinite periodic dipole; no clipping.')
    if_result=dict(real=float(amp.real),imag=float(amp.imag),magnitude=float(abs(amp)),
        phase_rad=float(np.angle(amp)) if omega>0 and amp!=0 else None,
        phase_defined=bool(omega>0 and amp!=0),phase_qualified=False,
        numerical_if_qualified=True,phase_numerically_qualified=bool(omega>0 and amp!=0),
        phase_scope='numerically checked periodic late-window phase; no detectability or AoA qualification',
        phase_epoch_us=float(retained[0]),signed_fit_frequency_mhz=frequency,
        units='dimensionless transmission peak amplitude',
        observable='Beer-Lambert transmission of ARC-weighted coherence',
        error_checks_absolute=checks,error_checks_relative={k:v/abs(amp) if omega>0 and amp!=0 else None for k,v in checks.items()},
        error_estimate=estimate,initial_condition_only_error_estimate=transient_estimate,
        zero_signal_semantics='residual fit retained; physical phase undefined when signal is OFF')
    return dict(scope='internal four-manifold periodic steady response; no outside-FS loss/return, Doppler or noise',
        backend_report=dict(backend='sparse_fourier',controls=asdict(controls),automatic_retry=False,
            fourier_orders=records,sample_refinement=dict(samples=len(refined),summary=fine),
            steady_state_metrics=steady_metrics,transit_rate_per_us=transit,
            initial_reference='certified LO-only stationary density at t=0',
            returned_state_kind='periodic steady; not the initial transient',
            qualification='floating-point residual/contraction estimate and two resolution checks; not interval arithmetic'),
        initial_state=diagnostic.initial_state,transient_states_returned=False,
        requested_sample_count=len(times),dimension=len(model.basis),times_us=retained.tolist(),
        coherence_real=values.real.tolist(),coherence_imag=values.imag.tolist(),
        positive_frequency_dipole_real_c_m=dipole.real.tolist(),positive_frequency_dipole_imag_c_m=dipole.imag.tolist(),
        transmission=transmission.tolist(),certified_outputs=len(retained),density_metrics=summary['density_metrics'],
        ode_options=None,normalize_output=False,if_result=if_result,signed_if_mhz=frequency,
        signal_amplitude_rad_per_us=omega,signal_phase_rad=diagnostic.signal_phase_rad,
        signal_drive_included=omega>0,IF_extraction_executed=True,IF_or_AoA_qualified=False)
