"""Local ballistic Gaussian response with axial Doppler and finite endcaps.

The prescribed optical attenuation is constant along each history. This is
a local kinetic control, not the nonlocal propagation solution for y(z).
"""
import numpy as np
from numbers import Real
from scipy import sparse
from scipy.integrate import solve_ivp
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError


def kinetic_crossing(base,probe,coupling,axial,initial,*,waists_m,endpoint_m,
                     velocity_m_per_s,cell_length_m,boundary_waists,
                     probe_log_attenuation,rtol,atol,max_step_us,exit_flux_row):
    """Full time evolution, with meters, m/s, microseconds and generators 1/us.

    ``axial`` is the Doppler generator per m/s, not a detuning in MHz.
    The initial state enters at the earlier physical boundary encountered
    backwards: a Gaussian far boundary or either cell endcap. Base excludes
    transit reset. The appended counter counts radiative first exits only.
    """
    arrays=[sparse.csr_matrix(a) for a in (base,probe,coupling,axial)]
    initial=np.asarray(initial);w=np.asarray(waists_m);r=np.asarray(endpoint_m);v=np.asarray(velocity_m_per_s)
    flux=np.asarray(exit_flux_row);n=initial.size
    controls=(cell_length_m,boundary_waists,probe_log_attenuation,rtol,atol,max_step_us)
    if (initial.ndim!=1 or initial.dtype.kind not in 'iuf' or not n or not np.isfinite(initial).all()
            or any(a.shape!=(n,n) or np.iscomplexobj(a.data) or not np.isfinite(a.data).all() for a in arrays)
            or any(a.shape!=shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all()
                   for a,shape in ((w,(2,)),(r,(3,)),(v,(3,)),(flux,(n,))))):
        raise PhysicsConfigurationError('Finite real kinetic generators, state, flux, waists, 3D position and velocity required')
    if (any(isinstance(a,(bool,np.bool_)) or not isinstance(a,Real) or not np.isfinite(a) for a in controls)
            or min(cell_length_m,boundary_waists,rtol,atol,max_step_us)<=0 or probe_log_attenuation>0
            or np.any(w<=0) or not 0<r[2]<cell_length_m or np.linalg.norm(v)==0):
        raise PhysicsConfigurationError('Invalid kinetic geometry or integration controls')
    transverse=np.linalg.norm(v[:2]);far=-np.inf if transverse==0 else -(np.linalg.norm(r[:2])+boundary_waists*max(w))/transverse*1e6
    wall=-np.inf if v[2]==0 else -(r[2] if v[2]>0 else cell_length_m-r[2])/abs(v[2])*1e6
    start=max(far,wall)
    arrays[0]=arrays[0]+v[2]*arrays.pop()
    arrays=[sparse.bmat([[a,None],[sparse.csr_matrix(flux[None,:] if i==0 else (1,n)),sparse.csr_matrix((1,1))]],format='csr')
            for i,a in enumerate(arrays)]
    def operator(t):
        position=r[:2]+v[:2]*t*1e-6;r2=position@position
        factors=np.exp(-r2/(w*w));factors[0]*=np.exp(probe_log_attenuation/2)
        return arrays[0]+factors[0]*arrays[1]+factors[1]*arrays[2]
    result=solve_ivp(lambda t,y:operator(t)@y,(start,0.),np.r_[initial,0.],method='BDF',
        jac=lambda t,y:operator(t),rtol=rtol,atol=atol,max_step=max_step_us,
        t_eval=np.linspace(start,0.,33),dense_output=False)
    if not result.success or not np.isfinite(result.y).all():
        raise PhysicsCalculationError('Kinetic Gaussian integration failed: '+result.message)
    result.boundary_kind='endcap' if wall>far else 'Gaussian_far'
    result.entry_time_us=start
    result.output_sampling='33 fixed times; adaptive integration steps unchanged'
    return result
