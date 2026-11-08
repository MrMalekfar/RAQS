"""First-return bounds for a declared finite, passive population network."""
import numpy as np
from ..arc._scan_validation import real_scalar
from ..exceptions import PhysicsConfigurationError,PhysicsCalculationError


def first_return_before_reset(halo_generator, to_core, to_unknown, reset_per_us):
    """Bound return before independent exponential transit, rates in µs^-1.

    Q uses column orientation and includes the loss to core/unknown on its
    diagonal. Unknown exits give no return in the lower control and immediate
    return in the upper control. Neither control makes the unknown population
    a physical dark state. Assumes passive Markov rates; no coherent fields.
    """
    gamma=real_scalar(reset_per_us,'first-return transit rate [1/us]')
    q=np.asarray(halo_generator);core=np.asarray(to_core);unknown=np.asarray(to_unknown)
    n=core.size
    if (gamma<=0 or q.shape!=(n,n) or core.ndim!=1 or n==0 or unknown.shape!=(n,) or
            not all(a.dtype.kind in 'iuf' and np.isfinite(a).all() for a in (q,core,unknown))):
        raise PhysicsConfigurationError('Invalid finite first-return rate budget.')
    off=q.copy();np.fill_diagonal(off,0.)
    scale=max(1.,np.linalg.norm(q,ord=1))
    if (np.any(off<0) or np.any(np.diag(q)>0) or np.any(core<0) or np.any(unknown<0) or
            np.max(abs(q.sum(axis=0)+core+unknown))>1e-12*scale):
        raise PhysicsConfigurationError('First-return generator must include all declared exit rates.')
    system=gamma*np.eye(n)-q.T
    lower=np.linalg.solve(system,core);upper=np.linalg.solve(system,core+unknown)
    if np.any(lower<-1e-12) or np.any(upper>1+1e-12) or np.any(upper<lower-1e-12):
        raise PhysicsCalculationError('Invalid first-return probability; no clipping or renormalization.')
    return lower,upper
