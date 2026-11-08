"""Ballistic entry-cohort survival for the hard-circle transit approximation."""
import numpy as np
from scipy.special import gammainc
from ..arc._scan_validation import real_scalar,integer_scalar
from ..exceptions import PhysicsConfigurationError


def hard_circle_entry_survival(times_us, *, radius_m, transverse_sigma_m_per_s, order=128):
    """Return survival after entry, not the residual lifetime of atoms already inside.

    A circular hard boundary, uniform incident flux and independent 2-D Maxwell
    velocities give uniform impact parameter b/R on [-1,1]. Entry flux weights
    the Rayleigh speed density by v, so v/sigma has the Maxwell(shape=3) CDF.
    Residence time is 2*sqrt(R^2-b^2)/v. This is not Gaussian-beam OBEs or a
    Markov reset: the non-exponential distribution is kept explicitly.
    """
    radius=real_scalar(radius_m,'transit hard radius [m]')
    sigma=real_scalar(transverse_sigma_m_per_s,'transverse thermal sigma [m/s]')
    order=integer_scalar(order,'impact-parameter quadrature order',minimum=8)
    times=np.asarray(times_us)
    if (radius<=0 or sigma<=0 or times.ndim!=1 or times.dtype.kind not in 'iuf' or
            not np.isfinite(times).all() or np.any(times<0)):
        raise PhysicsConfigurationError('Positive geometry/speed and finite nonnegative times [us] required.')
    nodes,weights=np.polynomial.legendre.leggauss(order)
    chord=2*radius*np.sqrt(1-nodes*nodes)
    result=np.ones(len(times));positive=times>0
    ratio=chord[:,None]/(sigma*times[positive][None,:]*1e-6)
    result[positive]=(weights/2)@gammainc(1.5,.5*ratio*ratio)
    return result
