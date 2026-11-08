"""Exact steady-state elimination of undriven radiative density blocks.

All generators use column-stacked density coordinates and rates in 1/us.
The halo has no incoming source and no velocity-dependent Hamiltonian.
Elimination retains its coherences; it is not population reinjection.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


from ..physics.lindblad.radiative_blocks import restricted_loss, restricted_transfer
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError


class SteadyHalo:
    """Schur correction for [A B; C E][core,halo] = [rhs,0].

    ``recover`` also acts on a Gaussian-averaged core because E and C are
    velocity independent. No trace normalization is performed.
    """
    def __init__(self, core_to_halo, halo_to_core, halo_generator):
        self.c = sparse.csr_matrix(core_to_halo)
        self.b = sparse.csr_matrix(halo_to_core)
        self.e = sparse.csc_matrix(halo_generator)
        n, m = self.c.shape
        if self.e.shape != (n, n) or self.b.shape != (m, n) or n == 0:
            raise PhysicsConfigurationError('Inconsistent nonempty core/halo blocks')
        if not all(np.isfinite(x.data).all() for x in (self.c, self.b, self.e)):
            raise PhysicsConfigurationError('Nonfinite core/halo generator')
        try:
            lu = splu(self.e)
        except RuntimeError as exc:
            raise PhysicsCalculationError('Undriven halo generator is singular') from exc
        columns = np.unique(self.c.nonzero()[1])
        response = lu.solve(-self.c[:, columns].toarray())
        self.response = sparse.csr_matrix((n, m), dtype=response.dtype).tolil()
        self.response[:, columns] = response
        self.response = self.response.tocsr()
        error = sparse.linalg.norm(self.e@self.response+self.c)
        if error > 1e-10*max(1., sparse.linalg.norm(self.c)):
            raise PhysicsCalculationError('Halo elimination residual exceeds tolerance')
        self.elimination_residual = float(error)
        self.correction = (self.b@self.response).tocsr()

    def recover(self, core):
        return self.response@core
