"""Explicit sparse charge-block BDF runtime; no retries or state repair."""
import warnings
import numpy as np
from scipy import sparse
from scipy.integrate import BDF
from scipy.sparse.linalg import spsolve, MatrixRankWarning

from ..config.run_numerics import BDFOrderControls
from ..physics.lindblad.create_transit_reset_collapse_ops import create_prepared_transit_reset_ops
from ..utils.validation import validate_steady_state
from ..exceptions import PhysicsError, PhysicsCalculationError, PhysicsConfigurationError


def reconstruct_density(model, vector):
    """Restore column-stacked charge blocks without renormalization."""
    n=len(model.basis)
    full=np.zeros(n*n,complex)
    full[model.coordinates]=vector
    return full.reshape((n,n),order='F')


def block_generators(model, transit_rate):
    """Original LO generator and exact projected signal commutators [1/us]."""
    reset=create_prepared_transit_reset_ops(transit_rate,len(model.basis),model.active_indices[0])
    full,base=model.liouvillians(reset)
    identity=sparse.eye(len(model.basis),format='csr')
    def commutator(op):
        generator=-1j*(sparse.kron(identity,op,format='csr')-sparse.kron(op.T,identity,format='csr'))
        return model.project_generator(generator)
    return full,base,commutator(model.rf_signal_operator),commutator(model.rf_signal_operator.getH())


def lo_steady_state(model, full, reduced):
    """Sparse trace-constrained solve, certified against ORIGINAL full L."""
    n=len(model.basis)
    diagonal=np.flatnonzero(model.coordinates % (n+1) == 0)
    if len(diagonal)!=n:
        raise PhysicsCalculationError('BBR-015: charge reduction omits populations.')
    row=int(diagonal[0]);matrix=reduced.tolil(copy=True)
    matrix[row,:]=0.;matrix[row,diagonal]=1.
    rhs=np.zeros(len(model.coordinates),complex);rhs[row]=1.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',MatrixRankWarning)
            state=spsolve(matrix.tocsc(),rhs)
    except Exception as exc:
        raise PhysicsCalculationError('BBR-015: LO stationary solve failed; no regularization.') from exc
    density=reconstruct_density(model,state)
    metrics=validate_steady_state(density,full,context='BBR-015 LO stationary state',backend='charge_block_sparse')
    return state,metrics


def integrate_charge_blocks(model, diagnostic, ode, times, coefficient, transit_rate, measure):
    """SciPy BDF with exact sparse Jacobian and certified requested outputs.

    nsteps limits accepted internal steps needed to reach each output point.
    Dense output is the BDF interpolant, never a replacement density matrix.
    """
    if isinstance(ode,BDFOrderControls):
        raise PhysicsConfigurationError('BBR-015: SciPy BDF does not expose max_order; control cannot be ignored.')
    if diagnostic.initial_state=='lo_steady' and transit_rate<=0:
        raise PhysicsConfigurationError('BBR-015: this LO stationary path requires positive prepared-reset transit.')
    full,base,signal,partner=block_generators(model,transit_rate)
    steady=None
    if diagnostic.initial_state=='lo_steady':
        initial,steady=lo_steady_state(model,full,base)
    else:
        initial=model.reduce_density(model.initial_density())
    cache=[None,None]
    def generator(t):
        if cache[0]!=t:
            value=coefficient(t)
            cache[:]=[t,(base+value*signal+np.conj(value)*partner).tocsc()]
        return cache[1]
    values=[measure(times[0],reconstruct_density(model,initial))]
    steps=0
    try:
        solver=BDF(lambda t,y:generator(t)@y,times[0],initial,times[-1],
                   rtol=ode.rtol,atol=ode.atol,jac=lambda t,y:generator(t),
                   max_step=float(np.min(np.diff(times))))
        for t in times[1:]:
            local_steps=0
            while solver.t<t:
                if local_steps>=ode.nsteps:
                    raise PhysicsCalculationError(f'BBR-015: nsteps exceeded before output t={t} us.')
                message=solver.step();local_steps+=1;steps+=1
                if solver.status=='failed':
                    raise PhysicsCalculationError(f'BBR-015: BDF failed at {solver.t} us: {message}')
            vector=solver.y if solver.t==t else solver.dense_output()(t)
            values.append(measure(t,reconstruct_density(model,vector)))
    except PhysicsError:raise
    except Exception as exc:
        raise PhysicsCalculationError('BBR-015: sparse BDF failed; no retry or substitute.') from exc
    return np.asarray(values),dict(backend='scipy.integrate.BDF',reduced_dimension=len(initial),
        full_dimension=full.shape[0],accepted_steps=steps,nfev=solver.nfev,njev=solver.njev,nlu=solver.nlu,
        initial_state=diagnostic.initial_state,steady_state_metrics=steady,
        transit_rate_per_us=transit_rate,transit_target_index=model.active_indices[0],
        max_order=5,max_step_us=float(np.min(np.diff(times))),output_sampling='BDF dense output at requested times',normalize_output=False)
