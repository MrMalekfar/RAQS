"""Radiative density blocks with explicitly driven, grouped halo manifolds."""
from numbers import Real
import numpy as np
from scipy import sparse
from .lindblad.radiative_blocks import restricted_loss, restricted_transfer
from ..exceptions import PhysicsConfigurationError, PhysicsCalculationError


def coherent_halo_blocks(model, domain, parts, shifts, outgoing, jumps, gamma,
                         groups, hamiltonians):
    """Build core/halo blocks retaining coherences inside each FS group.

    ``groups`` partitions all halo FS manifolds, preserving its declared order.
    ``hamiltonians`` supplies one additional Hermitian rad/us matrix per group
    in concatenated ``parts`` order, including rotating-frame diagonal offsets.
    HFS shifts are Hz; outgoing operators and gamma are event rates in 1/us.
    Core Hamiltonian/reset source remain the caller's responsibility. Halo
    reset removal is included here, exactly once; no source is added there.
    Returns (generator, states, column-vec coordinates, unknown-loss operators).
    """
    if isinstance(gamma,(bool,np.bool_)) or not isinstance(gamma,Real) or not np.isfinite(gamma) or gamma<=0:
        raise PhysicsConfigurationError('Positive finite renewal rate required')
    domain=tuple(domain);core_fs=tuple(dict.fromkeys(s.fs for s in model.basis))
    groups=tuple(tuple(group) for group in groups)
    flat=tuple(fs for group in groups for fs in group)
    halo=domain[len(core_fs):]
    if (domain[:len(core_fs)]!=core_fs or len(set(domain))!=len(domain)
            or any(not g for g in groups) or len(set(flat))!=len(flat)
            or set(flat)!=set(halo) or len(hamiltonians)!=len(groups)):
        raise PhysicsConfigurationError('Groups must partition the unique halo after the driven core')
    states=[tuple(model.basis)]+[tuple(s for fs in group for s in parts[fs]) for group in groups]
    coords=[model.coordinates]
    for block in states[1:]:
        m=np.array([s.total_m for s in block])
        coords.append(np.flatnonzero((m[:,None]==m[None,:]).reshape(-1,order='F')))
    sizes=[len(c) for c in coords]
    where={fs:(i,np.array([j for j,s in enumerate(block) if s.fs==fs]))
           for i,block in enumerate(states) for fs in dict.fromkeys(s.fs for s in block)}
    total=[np.zeros((len(s),len(s)),complex) for s in states]
    retained=[np.zeros_like(x) for x in total]
    for fs,(i,ind) in where.items():total[i][np.ix_(ind,ind)]+=outgoing[fs]
    blocks=[[sparse.csr_matrix((a,b),dtype=complex) for b in sizes] for a in sizes]
    for source,target,jump in jumps:
        if source not in domain or target not in domain:continue
        a,si=where[source];b,ti=where[target];coo=sparse.coo_matrix(jump)
        op=sparse.csr_matrix((coo.data,(ti[coo.row],si[coo.col])),shape=(len(states[b]),len(states[a])))
        blocks[b][a]+=restricted_transfer(op,coords[a],coords[b])
        retained[a]+=(op.getH()@op).toarray()
    unknown=[a-b for a,b in zip(total,retained)]
    for i,block in enumerate(states):
        if np.linalg.eigvalsh(unknown[i]).min() < -1e-11:
            raise PhysicsCalculationError('Retained jumps exceed total outgoing operator')
        blocks[i][i]+=restricted_loss(total[i],coords[i])
        if not i:continue
        n=len(block);extra=sparse.csr_matrix(hamiltonians[i-1],dtype=complex)
        if (extra.shape!=(n,n) or not np.isfinite(extra.data).all()
                or sparse.linalg.norm(extra-extra.getH())>1e-12):
            raise PhysicsConfigurationError('Each halo Hamiltonian must be finite, Hermitian and correctly shaped')
        energy=np.array([-2*np.pi*shifts[s.fs][s.f]/1e6 for s in block])
        h=extra+sparse.diags(energy);eye=sparse.eye(n,format='csr')
        full=-1j*(sparse.kron(eye,h,format='csr')-sparse.kron(h.T,eye,format='csr'))
        columns=full[:,coords[i]];restricted=columns[coords[i]]
        outside=np.ones(n*n,bool);outside[coords[i]]=False
        if sparse.linalg.norm(columns[outside])>1e-12*max(1.,sparse.linalg.norm(columns)):
            raise PhysicsCalculationError('Halo Hamiltonian leaves the retained total-M density subspace')
        blocks[i][i]+=restricted-gamma*sparse.eye(sizes[i])
    return sparse.bmat(blocks,format='csr'),states,coords,unknown
