"""Candidate density reflection for real pi-polarized E1 drive.

The phase is the pi spin rotation times orbital parity. The caller MUST
verify generator and source closure; this is not a general polarization
reduction. Coordinates use column stacking and may be block ordered.
"""
import numpy as np
from numbers import Real
from scipy import sparse
from ..exceptions import PhysicsConfigurationError


def reflection_density_basis(states, coordinates, nuclear_spin):
    coordinates=np.asarray(coordinates)
    if (not states or isinstance(nuclear_spin,(bool,np.bool_)) or not isinstance(nuclear_spin,Real) or
        not np.isfinite(nuclear_spin) or nuclear_spin<0 or abs(2*nuclear_spin-round(2*nuclear_spin))>1e-12):
        raise PhysicsConfigurationError('Nonempty states and a finite integer/half-integer nuclear spin required')
    if (coordinates.ndim!=1 or not coordinates.size or coordinates.dtype.kind not in 'iu' or
        len(np.unique(coordinates))!=len(coordinates) or coordinates.min()<0 or coordinates.max()>=len(states)**2):
        raise PhysicsConfigurationError('Unique in-range density coordinates required')
    def key(s,sign=1):
        return (s.fs,s.f,None if s.mf is None else sign*s.mf,
                None if s.mj is None else sign*s.mj,None if s.mi is None else sign*s.mi)
    lookup={key(s):i for i,s in enumerate(states)}
    if len(lookup)!=len(states):
        raise PhysicsConfigurationError('Reflection basis requires unique atomic states')
    try:mirror=np.array([lookup[key(s,-1)] for s in states])
    except KeyError as exc:
        raise PhysicsConfigurationError('Atomic basis is not closed under magnetic reflection') from exc
    exponents=np.array([(s.f-s.mf if s.f is not None else s.j-s.mj+nuclear_spin-s.mi)+s.l
                        for s in states])
    if np.max(abs(exponents-np.rint(exponents)))>1e-12:
        raise PhysicsConfigurationError('Integer spin-plus-parity reflection phase required')
    phase=(-1.)**np.rint(exponents).astype(int)
    n=len(states);index={int(a):i for i,a in enumerate(coordinates)}
    try:partners=np.array([index[int(mirror[a%n]+n*mirror[a//n])] for a in coordinates])
    except KeyError as exc:
        raise PhysicsConfigurationError('Density coordinates are not reflection closed') from exc
    signs=np.array([phase[a%n]*phase[a//n] for a in coordinates])
    if not np.array_equal(partners[partners],np.arange(len(partners))) or not np.all(signs*signs[partners]==1):
        raise PhysicsConfigurationError('Density reflection is not an involution')
    rows=[];columns=[];values=[];column=0
    for i,j in enumerate(partners):
        if j<i:continue
        if i==j:
            if signs[i]<0:continue
            rows.append(i);columns.append(column);values.append(1.)
        else:
            rows.extend((i,j));columns.extend((column,column))
            values.extend((1/np.sqrt(2),signs[i]/np.sqrt(2)))
        column+=1
    return sparse.csr_matrix((values,(rows,columns)),shape=(len(coordinates),column))
