"""Column-vec radiative gain and loss on explicitly closed density spaces."""
import numpy as np
from scipy import sparse
from ...exceptions import PhysicsConfigurationError, PhysicsCalculationError


def restricted_transfer(jump, source_coordinates, target_coordinates):
    """Return vec(J rho J†), retaining collective magnetic amplitudes."""
    jump = sparse.csr_matrix(jump)
    if min(jump.shape) == 0 or not np.isfinite(jump.data).all():
        raise PhysicsConfigurationError('Finite nonempty radiative jump required')
    full = sparse.kron(jump.conjugate(), jump, format='csr')[:, source_coordinates]
    retained = full[target_coordinates]
    outside = np.ones(full.shape[0], dtype=bool)
    outside[target_coordinates] = False
    if sparse.linalg.norm(full[outside]) > 1e-12 * max(1., sparse.linalg.norm(full)):
        raise PhysicsCalculationError('Radiative transfer leaves the retained density subspace')
    return retained


def restricted_loss(outgoing, coordinates):
    """Anticommutator -{Gamma,rho}/2 [1/us], including nondiagonal Gamma."""
    outgoing = sparse.csr_matrix(outgoing)
    if outgoing.shape[0] != outgoing.shape[1] or not np.isfinite(outgoing.data).all():
        raise PhysicsConfigurationError('Finite square outgoing operator required')
    if sparse.linalg.norm(outgoing-outgoing.getH()) > 1e-12:
        raise PhysicsConfigurationError('Outgoing operator must be Hermitian')
    if np.linalg.eigvalsh(outgoing.toarray()).min() < -1e-12:
        raise PhysicsConfigurationError('Outgoing operator must be positive semidefinite')
    identity = sparse.eye(outgoing.shape[0], format='csr')
    full = -.5*(sparse.kron(identity, outgoing, format='csr')
                 + sparse.kron(outgoing.T, identity, format='csr'))
    columns = full[:, coordinates]
    retained = columns[coordinates]
    outside = np.ones(full.shape[0], dtype=bool)
    outside[coordinates] = False
    if sparse.linalg.norm(columns[outside]) > 1e-12*max(1., sparse.linalg.norm(columns)):
        raise PhysicsCalculationError('Outgoing anticommutator leaves the retained density subspace')
    return retained
