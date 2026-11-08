"""Independent trajectory and symmetry checks; no fitted physical inputs."""
import numpy as np
import pytest
from scipy import sparse
from scipy.special import erf
from types import SimpleNamespace
from rydberg_aoa.physics.magnetic_core import CoreState
from rydberg_aoa.physics.magnetic_reflection import reflection_density_basis
from rydberg_aoa.simulation.gaussian_transit import gaussian_crossing
from rydberg_aoa.exceptions import PhysicsConfigurationError


def test_gaussian_time_integral_has_correct_meters_seconds_microseconds():
    w=4e-4;speed=200.;rate=.3
    result=gaussian_crossing(sparse.csr_matrix((1,1)),sparse.csr_matrix([[-rate]]),
        sparse.csr_matrix((1,1)),np.array([1.]),waists_m=[w,w],endpoint_m=[0.,0.],
        velocity_m_per_s=[speed,0.],boundary_waists=4.,probe_log_attenuation=0.,
        rtol=1e-10,atol=1e-12,max_step_us=.02)
    integral=w/speed*1e6*np.sqrt(np.pi)/2*erf(4.)
    assert result.y[0,-1]==pytest.approx(np.exp(-rate*integral),rel=2e-9)


def test_ballistic_counter_matches_independent_constant_loss_solution():
    rate=.12
    result=gaussian_crossing(sparse.csr_matrix([[-rate]]),sparse.csr_matrix((1,1)),
        sparse.csr_matrix((1,1)),np.array([1.]),waists_m=[4e-4,4e-4],endpoint_m=[0.,0.],
        velocity_m_per_s=[200.,0.],boundary_waists=2.,probe_log_attenuation=0.,
        rtol=1e-10,atol=1e-12,max_step_us=.02,exit_flux_row=np.array([rate]))
    expected=np.exp(-rate*4.)
    np.testing.assert_allclose(result.y[:,-1],[expected,1-expected],atol=2e-10)


def test_reflection_invariant_density_matches_independent_spin_one_rotation():
    states=(CoreState('g',6,0,.5,f=0.,mf=0.),)+tuple(
        CoreState('e',6,1,.5,f=1.,mf=m) for m in (-1.,0.,1.))
    m=np.array([s.mf for s in states]);coords=np.flatnonzero((m[:,None]==m[None,:]).reshape(-1,order='F'))
    p=reflection_density_basis(states,coords,.5)
    # Spin-1 Ry(pi), multiplied by odd orbital parity on the excited manifold.
    u=np.diag([1.,0.,1.,0.]);u[1,3]=u[3,1]=-1.
    reflection=sparse.kron(u,u).tocsr()[coords][:,coords].toarray()
    np.testing.assert_allclose((p@p.T).toarray(),(np.eye(len(coords))+reflection)/2,atol=1e-14)


def test_zero_transverse_velocity_cannot_be_hidden_as_finite_crossing():
    with pytest.raises(PhysicsConfigurationError,match='nonzero'):
        gaussian_crossing(sparse.csr_matrix((1,1)),sparse.csr_matrix((1,1)),sparse.csr_matrix((1,1)),
            np.array([1.]),waists_m=[.001,.001],endpoint_m=[0.,0.],velocity_m_per_s=[0.,0.],
            boundary_waists=4.,probe_log_attenuation=0.,rtol=1e-6,atol=1e-10,max_step_us=.1)


@pytest.mark.parametrize('spin',[True,np.nan,1/3])
def test_invalid_nuclear_spin_is_not_a_reflection_phase(spin):
    state=CoreState('g',6,0,.5,f=0.,mf=0.)
    with pytest.raises(PhysicsConfigurationError,match='nuclear spin'):
        reflection_density_basis((state,),np.array([0]),spin)


def test_duplicate_density_coordinates_are_rejected():
    state=CoreState('g',6,0,.5,f=0.,mf=0.)
    with pytest.raises(PhysicsConfigurationError,match='Unique'):
        reflection_density_basis((state,),np.array([0,0]),.5)


@pytest.mark.parametrize('rtol',[True,'1e-6',np.nan,0.,-1.])
def test_invalid_scalar_tolerance_has_physics_context(rtol):
    with pytest.raises(PhysicsConfigurationError,match='integration'):
        gaussian_crossing(sparse.csr_matrix((1,1)),sparse.csr_matrix((1,1)),sparse.csr_matrix((1,1)),
            np.array([1.]),waists_m=[.001,.001],endpoint_m=[0.,0.],velocity_m_per_s=[1.,0.],
            boundary_waists=4.,probe_log_attenuation=0.,rtol=rtol,atol=1e-10,max_step_us=.1)
