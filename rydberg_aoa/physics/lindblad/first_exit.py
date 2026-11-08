"""Core part of a first-exit quantum instrument; not a dark-state reservoir."""
import numpy as np
from ...exceptions import PhysicsConfigurationError


def first_exit_loss_diagonal(outgoing_per_us):
    """Return column-vec diagonal of -{Gamma,rho}/2, in µs^-1.

    Gamma is the nonnegative, explicitly supplied per-state sum of omitted
    event rates. This CP, trace-decreasing evolution stops trajectories at
    their first exit. Return paths require explicit additional states; the
    lost probability must not be renormalized into the retained core.
    """
    rates=np.asarray(outgoing_per_us)
    if rates.ndim!=1 or not rates.size or rates.dtype.kind not in 'iuf' or not np.isfinite(rates).all() or np.any(rates<0):
        raise PhysicsConfigurationError('First-exit loss requires finite nonnegative event rates [1/us].')
    return (-.5*rates[:,None]-.5*rates[None,:]).reshape(-1,order='F')
