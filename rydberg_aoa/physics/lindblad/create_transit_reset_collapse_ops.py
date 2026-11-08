"""
Transit loss/repopulation (reset) collapse operators for the 4-level model.

Physical model (DPH-002B)
-------------------------
Atoms leaving the finite interaction region are replaced by fresh atoms in
the participating ground state |1>. For

    C_j = sqrt(Gamma_t) · |1><j| ,   j = 1..4 ,

the summed Lindblad dissipator is exactly

    sum_j D[C_j](rho) = Gamma_t · ( |1><1| · Tr(rho) − rho ) ,

i.e. every coherence decays at Gamma_t, every excited population relaxes at
Gamma_t, the ground state is replenished, and the trace is preserved exactly.
The j = 1 term is deliberate: it completes the anticommutator sum
C_j†C_j = Gamma_t·Σ_j |j><j| = Gamma_t·1 and makes the identity above exact.
"""

import numpy as np  # np.sqrt for collapse-operator normalization
import qutip as qt  # qt.basis for Fock states


def create_transit_reset_collapse_ops(gamma_t):
    """
    Build the four reset collapse operators C_j = sqrt(Gamma_t)|1><j|.

    Args:
        gamma_t: effective Markovian transit rate [1/μs]. Non-positive or
                 None values disable the channel (empty list), preserving the
                 zero-transit invariant of the pre-DPH-002B model.

    Returns:
        list[qt.Qobj]: [sqrt(Gamma_t)|1><1|, sqrt(Gamma_t)|1><2|,
                        sqrt(Gamma_t)|1><3|, sqrt(Gamma_t)|1><4|]
                       (empty list when the channel is disabled).
    """
    if gamma_t is None or gamma_t <= 0:
        return []
    s = [qt.basis(4, i) for i in range(4)]
    # j = 1 term included deliberately (see module docstring).
    return [np.sqrt(gamma_t) * (s[0] * s[j].dag()) for j in range(4)]

def create_prepared_transit_reset_ops(gamma_t, dimension, prepared_index):
    """C_j=sqrt(Gamma_t)|prepared><j| for N states; Gamma_t in 1/us.

    Exactly the established reset dissipator, including its self-jump.
    No extra pure-dephasing rate or population reservoir is introduced.
    """
    from ...config.laboratory import _real, _integer
    from ...exceptions import PhysicsConfigurationError
    rate = _real(gamma_t, 'BBR-015 transit [1/us]', nonnegative=True)
    size = _integer(dimension, 'BBR-015 transit dimension', minimum=1)
    target = _integer(prepared_index, 'BBR-015 prepared index', minimum=0)
    if target >= size:
        raise PhysicsConfigurationError('BBR-015: prepared index outside basis.')
    if rate == 0:
        return []
    if size == 4 and target == 0:
        return create_transit_reset_collapse_ops(rate)
    state = qt.basis(size, target)
    return [np.sqrt(rate)*(state*qt.basis(size,j).dag()) for j in range(size)]
