"""Opt-in internal magnetic core: optical HFS + Rydberg FS product states.

All states in four working FS manifolds are retained. Rydberg hyperfine is
deliberately omitted by this named approximation, not replaced after ARC failure.
No outside-FS loss/return, nonradiative channels or detector model is implied.
H uses the project's positive laser-minus-atom detuning convention (rad/us).
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.constants import c, h, hbar, k, epsilon_0, elementary_charge, physical_constants
from scipy import sparse

from ..arc._scan_validation import arc_real_scalar, real_scalar
from ..arc.prepared_decay import calculate_prepared_radiative_graph, ROLES
from ..config.magnetic_core import MagneticCoreOptions
from ..exceptions import PhysicsCalculationError, PhysicsConfigurationError, PhysicsError


def _call(atom, method, *args):
    try:
        return getattr(atom, method)(*args)
    except PhysicsError:
        raise
    except Exception as exc:
        raise PhysicsCalculationError(f'BBR-003: ARC {method}{args} failed; no substitute.') from exc


def _number(atom, method, *args):
    return arc_real_scalar(_call(atom, method, *args), f'BBR-003 ARC {method}{args}')


def _projections(j):
    return tuple(-j+i for i in range(int(2*j)+1))


def planck_occupation(frequency_hz, temperature_k):
    """Stable Planck occupation; no threshold or artificial zero replacement."""
    nu = real_scalar(frequency_hz, 'BBR-003 transition frequency [Hz]')
    t = real_scalar(temperature_k, 'BBR-003 radiation temperature [K]')
    if nu <= 0 or t < 0:
        raise PhysicsConfigurationError('BBR-003: positive frequency and nonnegative temperature required.')
    if t == 0:
        return 0.
    try:
        x = h*nu/(k*t)
    except (ZeroDivisionError, OverflowError) as exc:
        raise PhysicsCalculationError('BBR-003: Planck exponent is not representable.') from exc
    if not math.isfinite(x) or x <= 0:
        raise PhysicsCalculationError('BBR-003: invalid Planck exponent.')
    if x > 50:
        value = math.exp(-x)
        if value == 0:
            raise PhysicsCalculationError('BBR-003: Planck underflow; no zero substitute.')
        return value/(1-value)
    value = 1/math.expm1(x)
    if not math.isfinite(value) or value <= 0:
        raise PhysicsCalculationError('BBR-003: invalid Planck occupation.')
    return value


@dataclass(frozen=True)
class CoreState:
    role: str
    n: int
    l: int
    j: float
    f: float | None = None
    mf: float | None = None
    mj: float | None = None
    mi: float | None = None

    @property
    def fs(self):
        return self.n, self.l, self.j

    @property
    def total_m(self):
        return self.mf if self.f is not None else self.mj+self.mi


@dataclass
class MagneticCoreModel:
    basis: tuple
    charges: np.ndarray
    active_indices: tuple
    hamiltonian: sparse.csr_matrix
    collapse_operators: tuple
    coordinates: np.ndarray
    report: dict
    probe_coherence_operator: sparse.csr_matrix
    rf_signal_operator: sparse.csr_matrix
    probe_reference_dipole_c_m: float
    rf_orientation: object

    def initial_density(self):
        rho = np.zeros((len(self.basis), len(self.basis)), complex)
        rho[self.active_indices[0], self.active_indices[0]] = 1.
        return rho

    def liouvillians(self, extra_collapse_operators=()):
        """Full sparse reference and exact charge-block generator (both 1/us)."""
        import qutip as qt
        full = qt.liouvillian(qt.Qobj(self.hamiltonian),
                              [qt.Qobj(op) for op in (*self.collapse_operators, *extra_collapse_operators)]).data.as_scipy().tocsr()
        return full, self.project_generator(full)

    def project_generator(self, full):
        """Restrict a sparse generator only after checking charge closure."""
        if full.shape != (len(self.basis)**2,len(self.basis)**2) or not np.isfinite(full.data).all():
            raise PhysicsCalculationError('BBR-015: invalid sparse generator shape or nonfinite coefficients.')
        reduced = full[self.coordinates][:, self.coordinates].tocsr()
        # Check discarded rows without densifying the N^2 reference.
        projection = sparse.csr_matrix((np.ones(len(self.coordinates)),
            (self.coordinates, np.arange(len(self.coordinates)))), shape=(full.shape[0], len(self.coordinates)))
        leakage = full@projection-projection@reduced
        norm = float(sparse.linalg.norm(leakage))
        if norm > 1e-10*max(1., float(sparse.linalg.norm(full@projection))):
            raise PhysicsCalculationError('BBR-003: discarded coherences are dynamically generated.')
        return reduced

    def reduce_density(self, rho):
        from ..utils.density_validation import validate_density_matrix
        rho = np.asarray(rho)
        validate_density_matrix(rho, context='BBR-003 initial density', expected_dim=len(self.basis))
        vector = rho.reshape(-1, order='F')
        selected = np.zeros_like(vector); selected[self.coordinates] = vector[self.coordinates]
        if not np.array_equal(selected, vector):
            raise PhysicsConfigurationError('BBR-003: initial inter-charge coherence cannot be discarded.')
        return vector[self.coordinates].copy()


def build_magnetic_core(atom, levels, fields, rabis_rad_us, nuclear_projection, options):
    """Build an internal core from resolved inputs; no configuration/cache writes.

    Optical shifts/radiative Bohr frequencies use ARC low-state HFS in Hz.
    Rydberg states are |n,l,j,mJ,mI>, with no Rydberg HFS or Zeeman term.
    Isotropic photon jumps are grouped by resolved HFS line and q, retaining
    degenerate magnetic amplitudes coherently within each jump.
    """
    if not isinstance(options, MagneticCoreOptions):
        raise PhysicsConfigurationError('BBR-003: explicit MagneticCoreOptions required.')
    from ..config.laboratory import AppliedLaser, AppliedRF
    try:
        for name in ('probe','coupling'):AppliedLaser(**fields[name])
        AppliedRF(**fields['lo'])
    except (KeyError, TypeError) as exc:
        raise PhysicsConfigurationError('BBR-003: complete explicit fields required.') from exc
    identity = calculate_prepared_radiative_graph(atom, levels, nuclear_projection=nuclear_projection)
    spin = identity.nuclear_spin
    canonical = {r:dict(zip(('n','l','j','f','mf','mj'),s)) for r,s in zip(ROLES,identity.states)}
    if len(rabis_rad_us) != 3:
        raise PhysicsConfigurationError('BBR-003: three explicit reference Rabi frequencies required.')
    rabis = tuple(real_scalar(x, 'BBR-003 reference Rabi [rad/us]') for x in rabis_rad_us)
    if any(x < 0 for x in rabis):
        raise PhysicsConfigurationError('BBR-003: reference Rabi magnitudes must be nonnegative.')
    qp,qc,qr = (fields[name]['polarization'] for name in ('probe','coupling','lo'))
    if any(type(q) is not int or q not in (-1,0,1) for q in (qp,qc,qr)):
        raise PhysicsConfigurationError('BBR-003: pure integer field polarisations required.')
    if canonical['excited']['mf']+qc-canonical['rydberg1']['mj'] != nuclear_projection:
        raise PhysicsConfigurationError('BBR-003: mI differs from the declared reference channel.')
    offsets = dict(ground=0, excited=qp, rydberg1=qp+qc, rydberg2=qp+qc+qr)
    basis = []; shifts = {}; coefficients = {}
    for role in ROLES:
        s = canonical[role]; fs = (s['n'],s['l'],s['j'])
        if role in ('ground','excited'):
            ab = _call(atom, 'getHFSCoefficients', *fs)
            if not isinstance(ab,(tuple,list)) or len(ab)!=2:
                raise PhysicsCalculationError('BBR-003: missing/invalid ARC HFS coefficients.')
            a,b = (arc_real_scalar(x, 'BBR-003 HFS coefficient [Hz]') for x in ab)
            coefficients[role] = dict(A_hz=a,B_hz=b)
            for f in tuple(abs(spin-s['j'])+i for i in range(int(2*min(spin,s['j']))+1)):
                shifts[(role,f)] = _number(atom,'getHFSEnergyShift',s['j'],f,a,b)
                basis.extend(CoreState(role,*fs,f=f,mf=m) for m in _projections(f))
        else:
            shifts[(role,None)] = 0.  # Named FS-product approximation; no ARC failure is caught here.
            basis.extend(CoreState(role,*fs,mj=m,mi=mi) for m in _projections(s['j']) for mi in _projections(spin))
    basis = tuple(basis); n = len(basis)
    active = tuple(next(i for i,s in enumerate(basis) if s.role==role and
        all(getattr(s,key)==canonical[role].get(key) for key in ('n','l','j','f','mf','mj')) and
        (s.f is not None or s.mi==nuclear_projection)) for role in ROLES)
    charges = np.array([s.total_m-offsets[s.role] for s in basis])
    if len(set(charges[list(active)])) != 1:
        raise PhysicsConfigurationError('BBR-003: reference polarisations do not connect the declared states.')
    def frequency(sr,tr):
        return _number(atom,'getTransitionFrequency',*canonical_fs[sr],*canonical_fs[tr])
    canonical_fs = {r:(canonical[r]['n'],canonical[r]['l'],canonical[r]['j']) for r in ROLES}
    refshift = {r:shifts[(r,canonical[r].get('f'))] for r in ROLES}
    fp,fc = [c/(real_scalar(fields[r]['wavelength_nm'],'BBR-003 wavelength [nm]')*1e-9)+
             real_scalar(fields[r]['detuning_mhz'],'BBR-003 laser offset [MHz]')*1e6 for r in ('probe','coupling')]
    signed = frequency('rydberg1','rydberg2')
    if signed == 0:
        raise PhysicsCalculationError('BBR-003: degenerate RF manifolds are unsupported.')
    dp = 2*np.pi*(fp-(frequency('ground','excited')+refshift['excited']-refshift['ground']))/1e6
    dc = 2*np.pi*(fc-(frequency('excited','rydberg1')-refshift['excited']))/1e6
    dr = 2*np.pi*(real_scalar(fields['lo']['frequency_hz'],'BBR-003 LO [Hz]')-abs(signed))/1e6
    base = dict(ground=0.,excited=dp,rydberg1=dp+dc,rydberg2=dp+dc+np.sign(signed)*dr)
    diagonal = np.array([base[s.role]-2*np.pi*(shifts[(s.role,s.f)]-refshift[s.role])/1e6 for s in basis])
    cache = {}
    def dipole(i,j,q):
        key=(i,j,q)
        if key in cache:return cache[key]
        a,b=basis[i],basis[j]
        if b.total_m-a.total_m != q or abs(a.l-b.l)!=1 or abs(a.j-b.j)>1:
            return 0.
        if a.f is None and b.f is None and a.mi!=b.mi:return 0.
        if a.f is not None and b.f is not None:
            value=_number(atom,'getDipoleMatrixElementHFS',*a.fs,a.f,a.mf,*b.fs,b.f,b.mf,q)
        elif a.f is not None:
            value=_number(atom,'getDipoleMatrixElementHFStoFS',*a.fs,a.f,a.mf,*b.fs,b.mj,q)
        elif b.f is not None:
            value=(-1.)**q*_number(atom,'getDipoleMatrixElementHFStoFS',*b.fs,b.f,b.mf,*a.fs,a.mj,-q)
        else:value=_number(atom,'getDipoleMatrixElement',*a.fs,a.mj,*b.fs,b.mj,q)
        cache[key]=value;return value
    interaction=np.zeros((n,n),complex)
    probe_observable=np.zeros((n,n),complex)
    rf_signal=np.zeros((n,n),complex)
    for ri,rj,q,omega in [(0,1,qp,rabis[0]),(1,2,qc,rabis[1]),(2,3,qr,rabis[2])]:
        reference=dipole(active[ri],active[rj],q)
        if reference==0:raise PhysicsConfigurationError('BBR-003: declared driven reference is dark.')
        for i,a in enumerate(basis):
            if a.role!=ROLES[ri]:continue
            for j,b in enumerate(basis):
                if b.role==ROLES[rj]:
                    ratio=dipole(i,j,q)/reference
                    interaction[i,j]=-.5*omega*ratio
                    interaction[j,i]=np.conj(interaction[i,j])
                    if ri==0:probe_observable[j,i]=ratio
                    if ri==2:rf_signal[j,i]=-.5*ratio
    hamiltonian=sparse.csr_matrix(interaction+np.diag(diagonal))
    if sparse.linalg.norm(hamiltonian-hamiltonian.getH())>1e-12:
        raise PhysicsCalculationError('BBR-003: non-Hermitian core Hamiltonian.')
    jumps=[];lines=[]
    multiplets=tuple(dict.fromkeys((s.role,s.f) for s in basis))
    ea0=elementary_charge*physical_constants['Bohr radius'][0]
    for sr,fs in multiplets:
        for tr,ft in multiplets:
            if ROLES.index(sr)>=ROLES.index(tr):continue
            sa,ta=canonical_fs[sr],canonical_fs[tr]
            if abs(sa[1]-ta[1])!=1 or abs(sa[2]-ta[2])>1:continue
            signed_line=frequency(sr,tr)+shifts[(tr,ft)]-shifts[(sr,fs)]
            if signed_line==0:raise PhysicsCalculationError('BBR-003: zero E1 Bohr frequency.')
            upper,lower=((tr,ft),(sr,fs)) if signed_line>0 else ((sr,fs),(tr,ft))
            nu=abs(signed_line);occupation=planck_occupation(nu,options.effective_temperature_k)
            prefactor=(2*np.pi*nu)**3*ea0**2/(3*np.pi*epsilon_0*hbar*c**3)/1e6
            if not math.isfinite(prefactor) or prefactor<=0:
                raise PhysicsCalculationError('BBR-003: invalid Einstein prefactor.')
            for q in (-1,0,1):
                down=np.zeros((n,n),complex)
                for i,a in enumerate(basis):
                    if (a.role,a.f)!=upper:continue
                    for j,b in enumerate(basis):
                        if (b.role,b.f)==lower:down[j,i]=dipole(i,j,q)
                if not np.any(down):continue
                jumps.append(sparse.csr_matrix(math.sqrt(prefactor*(1+occupation))*down))
                if occupation>0:jumps.append(sparse.csr_matrix(math.sqrt(prefactor*occupation)*down.conj().T))
                lines.append(dict(upper=upper,lower=lower,q=q,frequency_hz=nu,occupation=occupation,
                    spontaneous_outgoing_per_us=(prefactor*np.diag(down.conj().T@down).real).tolist()))
    coordinates=np.flatnonzero((charges[:,None]==charges[None,:]).reshape(-1,order='F'))
    report=dict(model=options.model,hilbert_dimension=n,full_density_coordinates=n*n,
        block_density_coordinates=len(coordinates),prepared_ground_index=active[0],reference_indices=active,
        bbr_enabled=options.bbr_enabled,configured_radiation_temperature_k=options.radiation_temperature_k,
        effective_radiation_temperature_k=options.effective_temperature_k,optical_hfs_coefficients=coefficients,
        detunings_reference_rad_per_us=[float(dp),float(dc),float(dr)],
        rydberg_hfs_model='omitted_by_named_fs_product_approximation',zeeman_included=False,
        internal_e1_lines=lines,closed_physical_space=False,outside_fs_rates_or_return_added=False,
        nonradiative_channels_included=False,signal_drive_included=False,IF_or_AoA_qualified=False,
        energy_convention='optical_hfs_rydberg_fs_product',reference_rabis_rad_per_us=rabis,
        radiative_secular_approximation='resolved optical HFS lines; collective degenerate magnetic q jumps',
        applied_frequencies_hz=dict(probe=fp,coupling=fc,lo=fields['lo']['frequency_hz']))
    from ..arc.rf_orientation import resolve_rf_orientation
    probe_reference=abs(dipole(active[0],active[1],qp))*ea0
    report.update(probe_observable='dipole-weighted raising operator, normalized to the prepared reference',
                  probe_reference_dipole_c_m=probe_reference,rf_signal_operator_available=True,
                  rf_operator_polarization=qr,temporal_diagnostic_available=True)
    return MagneticCoreModel(basis,charges,active,hamiltonian,tuple(jumps),coordinates,report,
        sparse.csr_matrix(probe_observable),sparse.csr_matrix(rf_signal),probe_reference,resolve_rf_orientation(signed))
