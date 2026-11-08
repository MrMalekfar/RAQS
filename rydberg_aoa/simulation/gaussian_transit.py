"""Ballistic atomic crossing of coaxial Gaussian fields, without reset.

Positions/waists are meters, velocities m/s, integration time microseconds.
Input generators are in 1/us on a real Hermitian density-coordinate basis.
This solves one trajectory; it does not supply an ensemble or flux weight.
"""
import numpy as np
from numbers import Real
from scipy import sparse
from scipy.integrate import solve_ivp
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError


def gaussian_crossing(base,probe,coupling,initial,*,waists_m,endpoint_m,velocity_m_per_s,
                      boundary_waists,probe_log_attenuation,rtol,atol,max_step_us,
                      exit_flux_row=None):
    """Evolve from a far-field thermal state to t=0 at the chosen position.

    No transit decay or replenishment is added here. The supplied base must
    exclude them. A requested first-exit counter integrates actual loss and
    does not represent a physical reservoir. All solver failures are visible.
    """
    arrays=[sparse.csr_matrix(x) for x in (base,probe,coupling)]
    initial=np.asarray(initial);w=np.asarray(waists_m);r=np.asarray(endpoint_m);v=np.asarray(velocity_m_per_s)
    n=initial.size
    if (initial.ndim!=1 or initial.dtype.kind not in 'iuf' or n==0 or
        any(a.shape!=(n,n) or np.iscomplexobj(a.data) or not np.isfinite(a.data).all() for a in arrays) or
        any(a.shape!=(2,) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in (w,r,v)) or
        not np.isfinite(initial).all() or np.any(w<=0) or np.linalg.norm(v)<=0):
        raise PhysicsConfigurationError('Finite real generators, positions, waists and nonzero transverse velocity required')
    scalars=(boundary_waists,probe_log_attenuation,rtol,atol,max_step_us)
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in scalars):
        raise PhysicsConfigurationError('Real scalar Gaussian integration controls required')
    controls=np.asarray(scalars)
    if not np.isfinite(controls).all() or boundary_waists<=0 or probe_log_attenuation>0 or min(rtol,atol,max_step_us)<=0:
        raise PhysicsConfigurationError('Invalid Gaussian boundary, attenuation or integration tolerances')
    start=-(np.linalg.norm(r)+boundary_waists*max(w))/np.linalg.norm(v)*1e6
    use_counter=exit_flux_row is not None
    if use_counter:
        flux=np.asarray(exit_flux_row)
        if flux.shape!=(n,) or flux.dtype.kind not in 'iuf' or not np.isfinite(flux).all():
            raise PhysicsConfigurationError('Finite real first-exit flux row required')
        arrays=[sparse.bmat([[a,None],[sparse.csr_matrix(flux[None,:] if i==0 else (1,n)),sparse.csr_matrix((1,1))]],format='csr')
                for i,a in enumerate(arrays)]
        initial=np.r_[initial,0.]
    cache_time=None;cache_operator=None
    def operator(t):
        nonlocal cache_time,cache_operator
        if cache_time!=t:
            position=r+v*(t*1e-6);r2=position@position
            factors=np.exp(-r2/(w*w));factors[0]*=np.exp(probe_log_attenuation/2)
            cache_operator=arrays[0]+factors[0]*arrays[1]+factors[1]*arrays[2]
            cache_time=t
        return cache_operator
    result=solve_ivp(lambda t,y:operator(t)@y,(start,0.),initial,method='BDF',
                     jac=lambda t,y:operator(t),rtol=rtol,atol=atol,max_step=max_step_us,dense_output=False)
    if not result.success or not np.isfinite(result.y).all():
        raise PhysicsCalculationError('Gaussian ballistic integration failed: '+result.message)
    return result
