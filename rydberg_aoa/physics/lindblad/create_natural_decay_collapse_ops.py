import numpy as np
import qutip as qt
from ...config import atomic_config as acfg

def create_natural_decay_collapse_ops():
    """
    Create natural decay collapse operators (Spontaneous Emission).
    
    Physics:
        For a transition |j> -> |i> with rate Gamma, the Lindblad operator is:
        C = sqrt(Gamma) * |i><j|
        This causes population transfer from j to i and decoherence of rho_ij.

    Returns:
        list: List of qutip.Qobj representing spontaneous emission.
    """
    # Natural decay rates (Gamma)
    # Ensure these are rates (Hz or rad/s depending on your units), not amplitudes
    natural_line = [acfg.GAMMA_21, acfg.GAMMA_32, acfg.GAMMA_43]

    c_ops = []
    dim = 4  # 4-level system

    # Loop through transitions. 
    # Index 0 corresponds to |2>->|1>
    # Index 1 corresponds to |3>->|2>
    # Index 2 corresponds to |4>->|3>
    for k, gamma in enumerate(natural_line):
        if gamma > 0:
            upper_level = k + 1  # e.g., |2> (index 1)
            lower_level = k      # e.g., |1> (index 0)
            
            # Create the lowering operator |lower><upper|
            # qutip.basis(dim, n) creates a state vector |n>
            op = qt.basis(dim, lower_level) * qt.basis(dim, upper_level).dag()
            
            # The collapse operator includes the square root of the rate
            c_op = np.sqrt(gamma) * op
            c_ops.append(c_op)

    return c_ops