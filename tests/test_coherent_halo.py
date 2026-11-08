"""Independent full-density and analytic driven-two-level controls."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from rydberg_aoa.physics.coherent_halo import coherent_halo_blocks
from rydberg_aoa.physics.radiative_halo import radiative_halo_blocks
from rydberg_aoa.exceptions import PhysicsConfigurationError,PhysicsCalculationError


def fixture_data():
    fs=((1,0,.5),(2,1,.5),(3,2,1.5))
    states=[SimpleNamespace(fs=f,total_m=0.,f=None) for f in fs]
    model=SimpleNamespace(basis=(states[0],),coordinates=np.array([0]))
    parts={f:(s,) for f,s in zip(fs,states)}
    shifts={f:{None:0.} for f in fs};outgoing={f:np.array([[.4]]) for f in fs}
    jumps=[(fs[0],fs[1],sparse.csr_matrix([[np.sqrt(.3)]])),
           (fs[1],fs[0],sparse.csr_matrix([[np.sqrt(.2)]])),
           (fs[2],fs[0],sparse.csr_matrix([[np.sqrt(.1)]]))]
    return model,fs,parts,shifts,outgoing,jumps,.5


def test_full_liouville_oracle_and_cross_fs_coherence():
    args=fixture_data();fs=args[1];ham=np.array([[0.,-.7],[-.7,2.3]])
    generator,states,coords,unknown=coherent_halo_blocks(*args,(fs[1:],),(ham,))
    # Full 3x3 density oracle: construct every J and commutator independently.
    h=np.zeros((3,3));h[1:,1:]=ham;eye=np.eye(3)
    full=-1j*(np.kron(eye,h)-np.kron(h.T,eye))-.4*np.eye(9)
    for source,target,j in args[5]:
        op=np.zeros((3,3));op[fs.index(target),fs.index(source)]=j.toarray()[0,0]
        full+=np.kron(op,op.conj())
    full-=args[6]*np.diag([0.,1.,1.,1.,1.,1.,1.,1.,1.])
    selected=np.array([0,4,5,7,8]);assert np.max(abs(generator.toarray()-full[np.ix_(selected,selected)]))<1e-14
    # Supply the core reset loss and physical renewal source.
    generator=generator-sparse.diags([.5,0.,0.,0.,0.]);rhs=np.array([-.5,0.,0.,0.,0.])
    rho=spsolve(generator.tocsc(),rhs);halo=rho[1:].reshape((2,2),order='F')
    assert abs(halo[0,1])>1e-3
    assert np.linalg.eigvalsh(halo).min()>0
    trace=rho[0].real+np.trace(halo).real
    flux=(unknown[0][0,0]*rho[0]+np.trace(unknown[1]@halo)).real
    assert abs(.5*(1-trace)-flux)<1e-14


def test_zero_drive_matches_existing_single_fs_builder():
    args=fixture_data();fs=args[1]
    old,*_=radiative_halo_blocks(*args)
    new,*_=coherent_halo_blocks(*args,((fs[1],),(fs[2],)),(np.zeros((1,1)),np.zeros((1,1))))
    assert np.max(abs((new-old).toarray()))<1e-14


@pytest.mark.parametrize('h',[np.eye(3),np.array([[0,1],[0,0]]),np.full((2,2),np.nan)])
def test_rejects_invalid_hamiltonian(h):
    args=fixture_data()
    with pytest.raises(PhysicsConfigurationError):coherent_halo_blocks(*args,(args[1][1:],),(h,))


def test_rejects_m_changing_drive_and_duplicate_group():
    args=fixture_data();fs=args[1]
    args[2][fs[2]][0].total_m=1.
    with pytest.raises(PhysicsCalculationError):
        coherent_halo_blocks(*args,(fs[1:],),(np.array([[0,1],[1,0]]),))
    with pytest.raises(PhysicsConfigurationError):
        coherent_halo_blocks(*args,((fs[1],fs[1]),),(np.zeros((2,2)),))
