"""Hyperfine Hamiltonian in an explicitly enumerated |mJ,mI> basis."""
from numbers import Real
import numpy as np
from ..exceptions import PhysicsConfigurationError


def product_hyperfine_hz(states,nuclear_spin,a_hz,b_hz):
    """Return A I.J + B quadrupole in Hz, without changing the basis or rates.

    States must form one complete fine-structure product manifold. No mF is
    converted to mJ. The caller supplies measured or explicitly modeled A/B.
    """
    if not states or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x)
                         for x in (nuclear_spin,a_hz,b_hz)):
        raise PhysicsConfigurationError('Explicit finite spin and hyperfine coefficients [Hz] required')
    spin=float(nuclear_spin);j=states[0].j
    if spin<0 or abs(2*spin-round(2*spin))>1e-12 or any(s.fs!=states[0].fs or s.f is not None for s in states):
        raise PhysicsConfigurationError('One complete fine-structure product manifold required')
    expected={(m,i) for m in np.arange(-j,j+1,1.) for i in np.arange(-spin,spin+1,1.)}
    lookup={(s.mj,s.mi):z for z,s in enumerate(states)}
    if len(lookup)!=len(states) or set(lookup)!=expected:
        raise PhysicsConfigurationError('Complete explicit mJ,mI basis required')
    dot=np.zeros((len(states),len(states)))
    for z,s in enumerate(states):
        dot[z,z]=s.mj*s.mi
        for sign in (-1,1):
            target=(s.mj+sign,s.mi-sign)
            if target not in lookup:continue
            dot[lookup[target],z]=.5*np.sqrt((j*(j+1)-s.mj*(s.mj+sign))*(spin*(spin+1)-s.mi*(s.mi-sign)))
    result=a_hz*dot
    if b_hz!=0:
        if min(spin,j)<1:
            raise PhysicsConfigurationError('Quadrupole hyperfine coefficient requires I,J >= 1')
        result+=b_hz*(3*dot@dot+1.5*dot-spin*(spin+1)*j*(j+1)*np.eye(len(states)))/(2*spin*(2*spin-1)*j*(2*j-1))
    return result
