# analytical_solutions.py
"""
Contains the EXACT steady-state coherence (rho_21) based on the full 
polynomial solution (Appendix A).
"""

import numpy as np
from ..config import parameters as params, atomic_config as acfg

def calculate_coefficients(Delta_P, Delta_C, Delta_L, Omega_C, Omega_P, gamma_2):
    """
    Calculates the exact algebraic coefficients A_i, B_i, C_i from Appendix A.
    All inputs are expected to be in rad/us.
    """
    
    # ----------------------------------------------------------------------
    # Simplified variables
    # ----------------------------------------------------------------------
    DP, DC, DL = Delta_P, Delta_C, Delta_L
    OC_sq, OP_sq = Omega_C**2, Omega_P**2
    g2_sq = gamma_2**2
    
    # ----------------------------------------------------------------------
    # A-Coefficients (Real Part Numerator) - Re-checked
    # ----------------------------------------------------------------------
    A1 = -2 * DP

    A2 = (
        16 * DC**2 * DP + 32 * DC * DP**2 + 16 * DL * DC * DP 
        - 2 * DC * OC_sq + 16 * DP**3 + 16 * DL * DP**2 
        - 2 * DP * OC_sq - 2 * DL * OC_sq
    )

    A3 = (
        -32 * DC**4 * DP - 64 * DC**3 * DL * DP - 128 * DC**3 * DP**2 
        + 8 * DC**3 * OC_sq - 32 * DC**2 * DL**2 * DP - 192 * DC**2 * DL * DP**2 
        + 16 * DC**2 * DL * OC_sq - 192 * DC**2 * DP**3 + 24 * DC**2 * DP * OC_sq 
        - 64 * DC * DL**2 * DP**2 + 8 * DC * DL**2 * OC_sq - 192 * DC * DL * DP**3 
        + 32 * DC * DL * DP * OC_sq - 128 * DC * DP**4 + 24 * DC * DP**2 * OC_sq 
        - 32 * DL**2 * DP**3 + 8 * DL**2 * DP * OC_sq - 64 * DL * DP**4 
        + 16 * DL * DP**2 * OC_sq - 32 * DP**5 + 8 * DP**3 * OC_sq
    )

    # ----------------------------------------------------------------------
    # B-Coefficients (Imaginary Part Numerator Multiplier) - Re-checked
    # ----------------------------------------------------------------------
    B1 = gamma_2

    # Note: B2 has a common factor of 8*gamma_2
    B2_term = (-DC**2 - 2*DC*DP - DL*DC - DP**2 - DL*DP)
    B2 = 8 * gamma_2 * B2_term #--+ 2 * gamma_2 * OC_sq
    
    B3 = (
         16 * gamma_2 * DC**4 + 32 * gamma_2 * DC**3 * DL + 64 * gamma_2 * DC**3 * DP + 16 * gamma_2 * DC**2 * DL**2 
        + 96 * gamma_2 * DC**2 * DL * DP + 96 * gamma_2 * DC**2 * DP**2 
        + 32 * gamma_2 * DC * DL**2 * DP + 96 * gamma_2 * DC * DL * DP**2 
        + 64 * gamma_2 * DC * DP**3 + 16 * gamma_2 * DL**2 * DP**2
        + 32 * gamma_2 * DL * DP**3 + 16 * gamma_2 * DP**4
    )

    # ----------------------------------------------------------------------
    # C-Coefficients (Denominator) - Re-checked
    # ----------------------------------------------------------------------
    C1 = 4 * DP**2 + 2 * OP_sq + g2_sq

    C2 = (
        -32 * DC**2 * DP**2 - 16 * DC**2 * OP_sq - 8 * DC**2 * g2_sq 
        - 64 * DC * DP**3 - 32 * DL * DC * DP**2 + 8 * DC * DP * OC_sq 
        - 32 * DC * DP * OP_sq - 16 * DC * DP * g2_sq - 16 * DL * DC * OP_sq 
        - 8 * DL * DC * g2_sq - 32 * DP**4 - 32 * DL * DP**3 + 8 * DP**2 * OC_sq 
        - 16 * DP**2 * OP_sq - 8 * DP**2 * g2_sq + 8 * DL * DP * OC_sq 
        - 16 * DL * DP * OP_sq - 8 * DL * DP * g2_sq + 2 * OC_sq * OP_sq 
        + 2 * OP_sq**2
    )

    C3 = ( 
        # DC^4 terms
        64 * DC**4 * DP**2 + 32 * DC**4 * OP_sq + 16 * DC**4 * g2_sq
        # DC^3 DL terms
        + 128 * DC**3 * DL * DP**2 + 64 * DC**3 * DL * OP_sq + 32 * DC**3 * DL * g2_sq
        # DC^3 DP terms
        + 256 * DC**3 * DP**3 - 32 * DC**3 * DP * OC_sq + 128 * DC**3 * DP * OP_sq + 64 * DC**3 * DP * g2_sq
        # DC^2 DL^2 terms
        + 64 * DC**2 * DL**2 * DP**2 + 32 * DC**2 * DL**2 * OP_sq + 16 * DC**2 * DL**2 * g2_sq
        # DC^2 DL DP terms
        + 384 * DC**2 * DL * DP**3 - 64 * DC**2 * DL * DP * OC_sq + 192 * DC**2 * DL * DP * OP_sq + 96 * DC**2 * DL * DP * g2_sq
        # DC^2 DP terms
        + 384 * DC**2 * DP**4 - 96 * DC**2 * DP**2 * OC_sq + 192 * DC**2 * DP**2 * OP_sq + 96 * DC**2 * DP**2 * g2_sq
        # DC^2 Omega terms
        + 4 * DC**2 * OC_sq**2 + 8 * DC**2 * OC_sq * OP_sq + 8 * DC**2 * OP_sq**2
        # DC DL^2 DP terms
        + 128 * DC * DL**2 * DP**3 - 32 * DC * DL**2 * DP * OC_sq + 64 * DC * DL**2 * DP * OP_sq + 32 * DC * DL**2 * DP * g2_sq
        # DC DL DP terms
        + 384 * DC * DL * DP**4 - 128 * DC * DL * DP**2 * OC_sq + 192 * DC * DL * DP**2 * OP_sq + 96 * DC * DL * DP**2 * g2_sq
        # DC DL Omega terms
        + 8 * DC * DL * OC_sq**2 + 16 * DC * DL * OC_sq * OP_sq + 8 * DC * DL * OP_sq**2
        # DC DP terms
        + 256 * DC * DP**5 - 96 * DC * DP**3 * OC_sq + 128 * DC * DP**3 * OP_sq + 64 * DC * DP**3 * g2_sq
        # DC DP Omega terms
        + 8 * DC * DP * OC_sq**2 + 16 * DC * DP * OC_sq * OP_sq + 16 * DC * DP * OP_sq**2
        # DL^2 DP terms
        + 64 * DL**2 * DP**4 - 32 * DL**2 * DP**2 * OC_sq + 32 * DL**2 * DP**2 * OP_sq + 16 * DL**2 * DP**2 * g2_sq
        # DL^2 Omega terms
        + 4 * DL**2 * OC_sq**2 + 8 * DL**2 * OC_sq * OP_sq + 4 * DL**2 * OP_sq**2
        # DL DP terms
        + 128 * DL * DP**5 - 64 * DL * DP**3 * OC_sq + 64 * DL * DP**3 * OP_sq + 32 * DL * DP**3 * g2_sq
        # DL DP Omega terms
        + 8 * DL * DP * OC_sq**2 + 16 * DL * DP * OC_sq * OP_sq + 8 * DL * DP * OP_sq**2
        # DP terms
        + 64 * DP**6 - 32 * DP**4 * OC_sq + 32 * DP**4 * OP_sq + 16 * DP**4 * g2_sq
        # DP Omega terms
        + 4 * DP**2 * OC_sq**2 + 8 * DP**2 * OC_sq * OP_sq + 8 * DP**2 * OP_sq**2
    )
    
    return A1, A2, A3, B1, B2, B3, C1, C2, C3


def calculate_analytical_rho21(delta_p_single, cfg, Omega_rf_complex, gamma_2_eff=None,
                             rf_orientation=None, *, detunings_rad_per_us=None,
                             omegas_rad_per_us=None):
    """Closed-form steady-state probe coherence (Appendix A polynomial).

    Args:
        delta_p_single: probe scan offset [rad/us].
        cfg: atomic configuration module (detunings/Rabi source).
        Omega_rf_complex: RF Rabi frequency [rad/us] (complex allowed).
        gamma_2_eff: effective probe decay [1/us]; None -> cfg.GAMMA_21.
        detunings_rad_per_us: optional explicit (probe, coupling, RF) centers.
        omegas_rad_per_us: optional explicit (probe, coupling) Rabi inputs.
        rf_orientation: AN-RF-001 - RFTransitionOrientation | None.  For an
            explicit downward RF edge the laboratory Delta_rf enters the R2
            diagonal/denominator as ``-Delta_rf`` (``Delta_p + Delta_c -
            Delta_rf``), matching the RF-DIR-001C numerical Hamiltonian.
            ``None`` and upward edges keep the historical ``+Delta_rf``
            evaluation bit-identically.

    Units: detunings and Rabi frequencies [rad/us]; decay/dephasing [1/us].
    """
    # ... (function body remains identical) ...
    # Get detunings from atomic config
    DELTA_P_get, DELTA_C_get, DELTA_RF_get = (acfg.get_cached_detunings()
        if detunings_rad_per_us is None else detunings_rad_per_us)
    # Corrected: Account for calculated detunings - when delta_p_single = DELTA_P,
    # all detunings should be zero for resonance (matching Hamiltonian convention)
    Delta_P = delta_p_single + DELTA_P_get  # Zero-centered scan
    Delta_C = DELTA_C_get                   # Fixed relative detuning
    Delta_L = DELTA_RF_get                  # Fixed relative detuning
    # AN-RF-001: energy-aware RF sign.  None/upward keep the historical
    # +Delta_rf evaluation bit-identically; only an explicit downward edge
    # evaluates the derivation at -Delta_rf.
    if rf_orientation is not None and rf_orientation.is_downward:
        Delta_L = -Delta_L
    Omega_P, Omega_C = ((acfg.OMEGA_P, acfg.OMEGA_C)
                       if omegas_rad_per_us is None else omegas_rad_per_us)
    if gamma_2_eff is None:
        gamma_2_eff = acfg.GAMMA_21
    gamma_2 = gamma_2_eff
    
    A1, A2, A3, B1, B2, B3, C1, C2, C3 = calculate_coefficients(
        Delta_P, Delta_C, Delta_L, Omega_C, Omega_P, gamma_2
    )
    
    Omega_RF_sq = Omega_rf_complex**2
    Omega_RF_pow4 = Omega_RF_sq**2
    
    # Numerator Polynomial (P_real - j * Q_imag_mult)
    P_real = A1 * Omega_RF_pow4 + A2 * Omega_RF_sq + A3
    Q_imag_mult = B1 * Omega_RF_pow4 + B2 * Omega_RF_sq + B3
    Numerator = P_real - 1j * Q_imag_mult
    
    # Denominator Polynomial (Real)
    Denominator = C1 * Omega_RF_pow4 + C2 * Omega_RF_sq + C3
    
    # Final Coherence
    rho_21 = Omega_P * (Numerator / Denominator)

    return rho_21
