"""Independent full-Lindblad checks of radiative halo elimination."""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from rydberg_aoa.exceptions import PhysicsConfigurationError, PhysicsCalculationError
from rydberg_aoa.simulation.steady_halo import (
    SteadyHalo, restricted_loss, restricted_transfer,
)


def test_coherent_return_matches_full_three_level_renewal():
    # Two coherently driven core levels and one undriven returning halo level.
    h = np.array([[0., .7, 0.], [.7, .2, 0.], [0., 0., -.3]])
    j = np.zeros((3, 3)); j[2, 1] = np.sqrt(.4)
    k = np.zeros((3, 3)); k[0, 2] = np.sqrt(.6)
    identity = sparse.eye(3)
    full = -1j*(sparse.kron(identity, h)-sparse.kron(h.T, identity))
    for op in (j, k):
        gamma = op.T@op
        full += sparse.kron(op, op)-.5*(sparse.kron(identity, gamma)+sparse.kron(gamma.T, identity))
    full = full.tocsr()-.23*sparse.eye(9)
    rhs = np.zeros(9); rhs[0] = -.23
    expected = spsolve(full.tocsc(), rhs)
    cc = np.array([0, 1, 3, 4]); hh = np.array([8])
    halo = SteadyHalo(full[hh][:, cc], full[cc][:, hh], full[hh][:, hh])
    core = spsolve((full[cc][:, cc]+halo.correction).tocsc(), rhs[cc])
    np.testing.assert_allclose(core, expected[cc], atol=1e-13)
    np.testing.assert_allclose(halo.recover(core), expected[hh], atol=1e-13)
    assert abs(expected[[0, 4, 8]].sum()-1) < 1e-13


def test_collective_jump_preserves_returned_coherence():
    jump = np.array([[1., .2j], [.3j, .7]])
    rho = np.array([[.6, .1j], [-.1j, .4]])
    transfer = restricted_transfer(jump, np.arange(4), np.arange(4))
    actual = (transfer@rho.reshape(-1, order='F')).reshape((2, 2), order='F')
    np.testing.assert_allclose(actual, jump@rho@jump.conj().T, atol=1e-15)
    gamma = jump.conj().T@jump
    loss = restricted_loss(gamma, np.arange(4))
    np.testing.assert_allclose(loss@rho.reshape(-1, order='F'),
                              (-.5*(gamma@rho+rho@gamma)).reshape(-1, order='F'))


def test_reject_nonclosed_population_projection():
    with pytest.raises(PhysicsCalculationError, match='leaves'):
        restricted_transfer(np.ones((2, 1)), np.array([0]), np.array([0, 3]))


def test_reject_negative_outgoing_rate():
    with pytest.raises(PhysicsConfigurationError, match='positive'):
        restricted_loss(np.diag([1., -.1]), np.arange(4))


def test_reject_invalid_halo_shapes():
    with pytest.raises(PhysicsConfigurationError, match='Inconsistent'):
        SteadyHalo(sparse.eye(2), sparse.eye(3), sparse.eye(2))


def test_radiative_blocks_match_full_density_operator():
    from types import SimpleNamespace
    import qutip as qt
    from rydberg_aoa.physics.radiative_halo import radiative_halo_blocks
    a,b=(6,0,.5),(7,1,.5)
    states=lambda fs:tuple(SimpleNamespace(fs=fs,total_m=0.,f=None) for _ in range(2))
    model=SimpleNamespace(basis=states(a),coordinates=np.arange(4))
    j=np.array([[.3,.1j],[.2j,.4]])
    back=np.array([[.5,.1j],[.3j,.2]])
    outgoing={a:j.conj().T@j,b:back.conj().T@back}
    blocks,parts,coords,unknown=radiative_halo_blocks(model,(a,b),{b:states(b)},
        {b:{None:0.}},outgoing,[(a,b,j),(b,a,back)],.23)
    jumps=[]
    for si,ti,op in ((0,2,j),(2,0,back)):
        full=np.zeros((4,4),complex);full[ti:ti+2,si:si+2]=op;jumps.append(qt.Qobj(full))
    expected=qt.liouvillian(qt.Qobj(np.zeros((4,4))),jumps).full()-.23*np.eye(16)
    retained=np.array([0,1,4,5,10,11,14,15])
    actual=blocks.toarray();actual[:4,:4]-=.23*np.eye(4)
    np.testing.assert_allclose(actual,expected[np.ix_(retained,retained)],atol=1e-14)
    assert all(np.linalg.norm(g)<1e-14 for g in unknown)
    rhs=np.zeros(8);rhs[0]=-.23
    rho=spsolve(sparse.csc_matrix(actual),rhs)
    assert abs(rho[[0,3,4,7]].sum()-1)<1e-13
    assert np.linalg.norm(rho[[1,2,5,6]])>1e-3


def test_halo_budget_rejects_missing_outgoing():
    from types import SimpleNamespace
    from rydberg_aoa.physics.radiative_halo import radiative_halo_blocks
    a,b=(6,0,.5),(7,1,.5)
    state=lambda fs:(SimpleNamespace(fs=fs,total_m=0.,f=None),)
    model=SimpleNamespace(basis=state(a),coordinates=np.array([0]))
    with pytest.raises(PhysicsCalculationError,match='exceed'):
        radiative_halo_blocks(model,(a,b),{b:state(b)},{b:{None:0.}},
            {a:np.zeros((1,1)),b:np.zeros((1,1))},[(a,b,np.ones((1,1)))],.2)


@pytest.mark.parametrize('gamma',[True,np.bool_(False),'0.2',None,0.,-.2,np.nan,np.inf])
def test_invalid_renewal_rate_is_a_configuration_error(gamma):
    from types import SimpleNamespace
    from rydberg_aoa.physics.radiative_halo import radiative_halo_blocks
    model=SimpleNamespace(basis=(),coordinates=np.array([],int))
    with pytest.raises(PhysicsConfigurationError,match='renewal'):
        radiative_halo_blocks(model,(),{},{},{},[],gamma)


def test_singular_halo_preserves_calculation_error_cause():
    with pytest.raises(PhysicsCalculationError,match='singular') as caught:
        SteadyHalo(sparse.eye(2),sparse.eye(2),sparse.csr_matrix((2,2)))
    assert isinstance(caught.value.__cause__,RuntimeError)


def test_nonfinite_jump_cannot_enter_a_physical_generator():
    with pytest.raises(PhysicsConfigurationError,match='Finite'):
        restricted_transfer(np.array([[np.nan]]),np.array([0]),np.array([0]))
