"""
Calculate transmission from rho21 coherence using susceptibility model.
"""

import numpy as np
import logging
from ...config import parameters as params
from ...config import atomic_config as acfg
#from ...arc.dipole_matrix import get_probe_dipole
from ...arc.dipole_matrix import calculate_dipole_matrix
from ...arc.dipole import calculate_dipole_matrix_element
from .absorption_const import calculate_absorption_constant
logger = logging.getLogger(__name__)


def calculate_transmission_from_rho21(rho_21, *, absorption_prefactors=None):
    """
    Calculates transmission from rho21 coherence using the susceptibility model.

    Args:
        rho_21: Complex coherence value (rho_21)
        absorption_prefactors: optional explicit (od_factor, K_chi) derived
            from the applied medium. Bypasses configuration/cache lookup.

    Returns:
        float: Transmission value T = exp(-2 * absorption_factor * Im{chi})
    """
    # Absorption factor (pi * d / lambda_p)
    if absorption_prefactors is None:
        od_factor, K_chi = calculate_absorption_constant()
    else:
        from ...exceptions import PhysicsCalculationError
        def scalar(value, name, allow_complex=False):
            kinds = (int, float, np.integer, np.floating)
            if allow_complex:
                kinds += (complex, np.complexfloating)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, kinds):
                raise PhysicsCalculationError(f'LAB-007: {name} must be a numeric scalar; got {value!r}.')
            try:
                converted = complex(value) if allow_complex else float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise PhysicsCalculationError(f'LAB-007: {name} cannot be converted.') from exc
            if not np.isfinite(converted):
                raise PhysicsCalculationError(f'LAB-007: nonfinite {name}; no substitute.')
            return converted
        try:
            od_factor, K_chi = absorption_prefactors
        except (TypeError, ValueError) as exc:
            raise PhysicsCalculationError('LAB-007: exactly two transmission prefactors required.') from exc
        od_factor = scalar(od_factor, 'optical-depth factor')
        K_chi = scalar(K_chi, 'susceptibility factor')
        rho_21 = scalar(rho_21, 'coherence', allow_complex=True)
        if od_factor < 0:
            raise PhysicsCalculationError('LAB-007: negative optical-depth factor; no absolute-value repair.')

    # Normalize rho_21 by Omega_P to get the required term in chi formula: rho_21/Omega_P
    #rho_21_normalized = rho_21 / acfg.OMEGA_P

    # Susceptibility: chi = K_chi * rho_21_normalized
    chi = K_chi * rho_21

    # Transmission: T = exp(-2 * od_factor * Im{chi})
    with np.errstate(**({} if absorption_prefactors is None else dict(over='ignore', invalid='ignore'))):
        transmission = np.exp(-2.0 * od_factor * np.imag(chi))
    if absorption_prefactors is not None and not np.isfinite(transmission):
        raise PhysicsCalculationError('LAB-007: nonfinite spectrum transmission; no clipping.')
    # Ensure transmission is between 0 and 1
    #return np.clip(transmission, 0.0, 1.0)
    return transmission
