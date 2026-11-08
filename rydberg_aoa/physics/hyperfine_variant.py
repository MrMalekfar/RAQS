"""Explicit coherent hyperfine variant of an existing magnetic-core model."""
from dataclasses import replace
import copy
import numpy as np
from scipy import sparse
from .product_hyperfine import product_hyperfine_hz
from ..exceptions import PhysicsConfigurationError


def with_product_hyperfine(model,manifold_fs,*,nuclear_spin,a_hz,b_hz,provenance):
    """Return a new model with -2pi H_HFS/1e6 in its existing product basis.

    Frequencies remain referenced to the original FS centroid. Dipoles,
    fields, basis, rates and the original model are unchanged. This adds
    coherent HFS only; it does not resolve new radiative bath frequencies.
    Supply the coefficients and their independent origin explicitly.
    """
    fs=tuple(manifold_fs)
    if not isinstance(provenance,dict) or not provenance.get('source') or not provenance.get('data_origin'):
        raise PhysicsConfigurationError('Explicit hyperfine source and data_origin required')
    components=model.report.get('product_hyperfine_components',[])
    if any(tuple(c['fs'])==fs for c in components):
        raise PhysicsConfigurationError(f'Hyperfine Hamiltonian already added for {fs}')
    indices=np.array([i for i,s in enumerate(model.basis) if s.fs==fs])
    states=tuple(model.basis[i] for i in indices)
    hz=product_hyperfine_hz(states,nuclear_spin,a_hz,b_hz)
    rows,cols=np.nonzero(hz)
    addition=sparse.csr_matrix((-2*np.pi*hz[rows,cols]/1e6,(indices[rows],indices[cols])),shape=model.hamiltonian.shape)
    report=copy.deepcopy(model.report)
    report['product_hyperfine_components']=[*copy.deepcopy(components),dict(fs=fs,nuclear_spin=nuclear_spin,
        A_hz=a_hz,B_hz=b_hz,provenance=copy.deepcopy(provenance),HFS_span_hz=float(np.ptp(np.linalg.eigvalsh(hz))))]
    report['rydberg_hfs_model']='explicit_product_basis_components; other manifolds retain their declared approximation'
    report['added_HFS_changes_radiative_bath']=False
    report['hyperfine_frequency_reference']='original fine-structure centroid; no RF retuning'
    return replace(model,hamiltonian=model.hamiltonian+addition,report=report)
