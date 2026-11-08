import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm
from scipy.special import erf
from rydberg_aoa.simulation.kinetic_gaussian import kinetic_crossing
from rydberg_aoa.simulation.kinetic_quadrature import transverse_maxwell_rule,axial_maxwell_rule,rayleigh_speed_rule
from rydberg_aoa.exceptions import PhysicsConfigurationError


def test_volume_measure_has_cartesian_maxwell_moments_without_flux_bias():
    velocity,weight=transverse_maxwell_rule(137.,5,12)
    assert abs(weight.sum()-1)<1e-14
    np.testing.assert_allclose(weight@velocity,[0,0],atol=1e-12)
    np.testing.assert_allclose(weight@(velocity**2),[137.**2]*2,atol=1e-10)
    assert abs(weight@(np.linalg.norm(velocity,axis=1)**4)-8*137.**4)<1e-5


@pytest.mark.parametrize('order',[1,2,4,8,12,16])
def test_direct_speed_gauss_rule_matches_independent_rayleigh_moments(order):
    from scipy.special import gamma
    v,w=rayleigh_speed_rule(137.,order)
    for power in range(2*order):
        expected=2**(power/2)*gamma(1+power/2)
        assert w@((v/137.)**power)==pytest.approx(expected,rel=3e-12,abs=1e-12)


def test_axial_cdf_measure_preserves_raw_mass_and_resolves_narrow_velocity_response():
    edges=np.array([-6,-3,-1,-.1,-.01,0,.01,.1,1,3,6])*137
    v,w=axial_maxwell_rule(137.,edges,24)
    assert np.all(w>0) and abs(w.sum()-erf(6/np.sqrt(2)))<1e-14
    assert abs(w@v)<1e-10
    # Independent adaptive integral in ordinary velocity, with a narrow Lorentzian.
    from scipy.integrate import quad
    target=quad(lambda x:np.exp(-x*x/(2*137**2))/(137*np.sqrt(2*np.pi))/(1+(x/.1)**2),
                -822,822,points=edges[1:-1].tolist(),epsabs=1e-11)[0]
    assert abs(w@(1/(1+(v/.1)**2))-target)<1e-8


@pytest.mark.parametrize('vz,z,entry_us',[(100.,.001,-10.),(-100.,.0015,-5.)])
def test_endcap_and_axial_generator_match_exact_constant_matrix(vz,z,entry_us):
    base=np.array([[-.3,0],[0,-.3]]);axial=np.array([[0,-.007],[.007,0]])
    result=kinetic_crossing(base,sparse.csr_matrix((2,2)),sparse.csr_matrix((2,2)),axial,np.array([1.,0.]),
        waists_m=[.001,.001],endpoint_m=[0.,0.,z],velocity_m_per_s=[0.,0.,vz],cell_length_m=.002,
        boundary_waists=5.,probe_log_attenuation=0.,rtol=1e-10,atol=1e-12,max_step_us=.02,
        exit_flux_row=np.zeros(2))
    assert result.boundary_kind=='endcap' and result.entry_time_us==pytest.approx(entry_us)
    np.testing.assert_allclose(result.y[:2,-1],expm(-entry_us*(base+vz*axial))@[1.,0.],atol=1e-8)


@pytest.mark.parametrize('z,vz,length,entry_us,kind',[
    (.0008,50.,.002,-16.,'endcap'),
    (.005,5.,.01,-44.47213595499958,'Gaussian_far')])
def test_oblique_gaussian_history_matches_closed_form(z,vz,length,entry_us,kind):
    r=np.array([.0002,-.0001]);v=np.array([40.,-30.])*1e-6
    waists=np.array([.0003,.0005]);attenuation=-.8
    result=kinetic_crossing([[-.03]],[[-.4]],[[-.2]],[[-.0002]],np.ones(1),
        waists_m=waists,endpoint_m=[*r,z],velocity_m_per_s=[40.,-30.,vz],cell_length_m=length,
        boundary_waists=4.,probe_log_attenuation=attenuation,rtol=1e-10,atol=1e-12,max_step_us=.03,
        exit_flux_row=np.zeros(1))
    integrals=[]
    for width in waists:
        a=v@v/width**2;b=2*(r@v)/width**2;c=r@r/width**2
        integrals.append(np.exp(-c+b*b/(4*a))*np.sqrt(np.pi)/(2*np.sqrt(a))*
                         (erf(b/(2*np.sqrt(a)))-erf(np.sqrt(a)*entry_us+b/(2*np.sqrt(a)))))
    expected=np.exp((-entry_us)*(-.03-.0002*vz)-.4*np.exp(attenuation/2)*integrals[0]-.2*integrals[1])
    assert result.boundary_kind==kind and result.entry_time_us==pytest.approx(entry_us)
    np.testing.assert_allclose(result.y[0,-1],expected,rtol=2e-7,atol=1e-10)


@pytest.mark.parametrize('sigma',[0,-1,np.nan,True])
def test_invalid_velocity_measure_is_visible(sigma):
    with pytest.raises(PhysicsConfigurationError):transverse_maxwell_rule(sigma,2,4)
