"""Independent white phase diffusion of the probe and coupling laser fields."""
import math
from scipy import sparse
from ...arc._scan_validation import real_scalar
from ...exceptions import PhysicsConfigurationError


def create_laser_phase_diffusion_ops(basis, *, probe_fwhm_mhz, coupling_fwhm_mhz):
    """Return sparse collapse operators for Lorentzian laser FWHM in MHz.

    Time is µs. A laser of cyclic-frequency FWHM w has phase-increment variance
    2*pi*w*t, so a coherence spanning one optical phase decays at pi*w /µs.
    In the laser-following frame the generators are Qp=P_e+P_r1+P_r2 and
    Qc=P_r1+P_r2; each collapse is sqrt(2*pi*w)*Q. Optical noise therefore
    preserves coherences within a manifold and between the two Rydberg manifolds.
    Assumes independent Markov phase noise, with no amplitude noise, correlated
    laser noise, RF linewidth or transit contribution. Zero widths add no ops.
    """
    roles=tuple(state.role for state in basis)
    if not roles or any(role not in ('ground','excited','rydberg1','rydberg2') for role in roles):
        raise PhysicsConfigurationError('Laser phase diffusion requires the declared four-role basis.')
    operators=[]
    for name,value,selected in (
        ('probe',probe_fwhm_mhz,('excited','rydberg1','rydberg2')),
        ('coupling',coupling_fwhm_mhz,('rydberg1','rydberg2'))):
        width=real_scalar(value,f'{name} laser Lorentzian FWHM [MHz]')
        if width<0:raise PhysicsConfigurationError('Laser FWHM must be nonnegative.')
        if width:
            scale=math.sqrt(2*math.pi*width)
            if not math.isfinite(scale):raise PhysicsConfigurationError('Laser diffusion rate overflow.')
            operators.append(sparse.diags([scale if role in selected else 0. for role in roles],format='csr'))
    return tuple(operators)
