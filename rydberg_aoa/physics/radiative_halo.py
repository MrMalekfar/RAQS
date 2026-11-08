"""Radiative coupling of a driven core to undriven angular manifolds.

Rates and Hamiltonian shifts use 1/us and Hz respectively. External loss is
kept as a positive operator; it is never replaced by trace renormalization.
"""
import numpy as np
from numbers import Real
from scipy import sparse
from .lindblad.radiative_blocks import restricted_loss, restricted_transfer
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError


def radiative_halo_blocks(model, domain, parts, shifts, outgoing, jumps, gamma):
    """Return the invariant density generator and trace budgets.

    ``domain`` starts with the core FS manifolds; subsequent entries are
    undriven. ``jumps`` contains (source_fs,target_fs,rectangular_jump).
    ``outgoing`` includes the complete declared parent, including unknown
    destinations. Reset removes every state at gamma but injects only core.
    """
    if isinstance(gamma,(bool,np.bool_)) or not isinstance(gamma,Real) or not np.isfinite(gamma) or gamma <= 0:
        raise PhysicsConfigurationError('Positive finite renewal rate required')
    core_fs = tuple(dict.fromkeys(s.fs for s in model.basis))
    if tuple(domain[:len(core_fs)]) != core_fs or len(set(domain)) != len(domain):
        raise PhysicsConfigurationError('Domain must start with the unique driven core manifolds')
    halo_fs = domain[len(core_fs):]
    states = [tuple(model.basis)]+[parts[fs] for fs in halo_fs]
    coords = [model.coordinates]
    for block in states[1:]:
        m = np.array([s.total_m for s in block])
        coords.append(np.flatnonzero((m[:,None]==m[None,:]).reshape(-1,order='F')))
    sizes = [len(c) for c in coords]
    where = {}
    for fs in domain:
        block = 0 if fs in core_fs else 1+halo_fs.index(fs)
        indexes = np.array([i for i,s in enumerate(states[block]) if s.fs == fs])
        where[fs] = block,indexes
    total = [np.zeros((len(s),len(s)),complex) for s in states]
    retained = [np.zeros_like(x) for x in total]
    for fs,(block,ind) in where.items(): total[block][np.ix_(ind,ind)] += outgoing[fs]
    blocks = [[sparse.csr_matrix((a,b),dtype=complex) for b in sizes] for a in sizes]
    for source,target,j in jumps:
        if source not in domain or target not in domain: continue
        a,si = where[source]; b,ti = where[target]
        coo = sparse.coo_matrix(j)
        embedded = sparse.csr_matrix((coo.data,(ti[coo.row],si[coo.col])),shape=(len(states[b]),len(states[a])))
        blocks[b][a] += restricted_transfer(embedded,coords[a],coords[b])
        retained[a] += (embedded.getH()@embedded).toarray()
    unknown = [a-b for a,b in zip(total,retained)]
    for i in range(len(states)):
        if np.linalg.eigvalsh(unknown[i]).min() < -1e-11:
            raise PhysicsCalculationError('Retained jumps exceed the declared total outgoing operator')
        blocks[i][i] += restricted_loss(total[i],coords[i])
        if i:
            energy = np.array([-2*np.pi*shifts[s.fs][s.f]/1e6 for s in states[i]])
            commutator = (-1j*(energy[:,None]-energy[None,:])).reshape(-1,order='F')[coords[i]]
            blocks[i][i] += sparse.diags(commutator)-gamma*sparse.eye(sizes[i])
    return sparse.bmat(blocks,format='csr'),states,coords,unknown
