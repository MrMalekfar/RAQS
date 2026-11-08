"""Array geometry abstraction for direction-of-arrival estimation.

Canonical storage is a Cartesian position array of shape (M, 3) in metres.
The element/path count M is always inferred from the array and is never
hard-coded by the estimator layer.  The abstraction supports uniform-linear
(ULA), rectangular (URA), circular (UCA) and fully arbitrary non-uniform
geometries, so future estimator algorithms can be gated on geometry
capability without any estimator rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

_GEOMETRY_TOL = 1.0e-9


@dataclass(frozen=True, eq=False)
class ShiftInvariantPair:
    """Minimal structured-array descriptor for shift-invariance methods.

    For a ULA with M elements in index order:

        indices_1     = (0, 1, ..., M-2)
        indices_2     = (1, 2, ..., M-1)
        displacement_m = positions[1] - positions[0] (metres, shape (3,)).

    The displacement is the ACTUAL Cartesian translation between the two
    maximally overlapping subarrays (never a hard-coded x-axis step), so a
    translated or arbitrarily rotated ULA carries its true geometry.
    """

    indices_1: Tuple[int, ...]
    indices_2: Tuple[int, ...]
    displacement_m: np.ndarray = field(compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "indices_1", tuple(int(i) for i in self.indices_1)
        )
        object.__setattr__(
            self, "indices_2", tuple(int(i) for i in self.indices_2)
        )
        object.__setattr__(
            self, "displacement_m",
            np.asarray(self.displacement_m, dtype=float).reshape(3,),
        )


class ArrayGeometry:
    """Immutable Cartesian description of a receiver/path array."""

    __slots__ = ("_positions_m",)

    def __init__(self, positions_m):
        arr = np.asarray(positions_m, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                "positions_m must be a 2-D array of shape (M, 3); "
                f"got shape {arr.shape}"
            )
        if arr.shape[1] != 3:
            raise ValueError(
                "positions_m must have exactly 3 columns (x, y, z) in metres; "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("positions_m must contain at least one element (M >= 1)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("positions_m contains non-finite values")
        self._positions_m = arr.copy()

    # ------------------------------------------------------------------ #
    # canonical accessors
    # ------------------------------------------------------------------ #
    @property
    def positions_m(self):
        """Read-only (M, 3) Cartesian positions in metres."""
        return self._positions_m

    @property
    def m(self):
        """Number of elements/paths.  Always inferred from the positions."""
        return int(self._positions_m.shape[0])

    @classmethod
    def from_ula(cls, n_elements, spacing_m, offset_xyz=None):
        """Helper constructor for a uniform linear array along +x.

        Args:
            n_elements: number of elements/paths (M, >= 1).
            spacing_m: uniform inter-element spacing in metres (> 0).
            offset_xyz: optional common (dx, dy, dz) offset in metres.

        Returns:
            ArrayGeometry with positions [(i * spacing + dx, dy, dz)].
        """
        n = int(n_elements)
        if n < 1:
            raise ValueError("n_elements must be a positive integer (M >= 1)")
        spacing = float(spacing_m)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"spacing_m must be a positive finite value; got {spacing_m!r}")
        off = (0.0, 0.0, 0.0) if offset_xyz is None else tuple(offset_xyz)
        if len(off) != 3:
            raise ValueError("offset_xyz must be a 3-tuple (dx, dy, dz) in metres")
        pos = np.zeros((n, 3), dtype=float)
        pos[:, 0] = np.arange(n, dtype=float) * spacing
        pos[:, 0] += float(off[0])
        pos[:, 1] = float(off[1])
        pos[:, 2] = float(off[2])
        return cls(pos)

    @classmethod
    def from_xy(cls, positions_xy):
        """Adapter for legacy 2-D (M, 2) position arrays (z set to 0)."""
        arr = np.asarray(positions_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"positions_xy must have shape (M, 2); got {arr.shape}")
        return cls(np.column_stack([arr, np.zeros(arr.shape[0], dtype=float)]))

    # ------------------------------------------------------------------ #
    # structured-array descriptors (DOA-ALG-002, search-free estimators)
    # ------------------------------------------------------------------ #
    @property
    def shift_invariant_pair(self) -> Optional[ShiftInvariantPair]:
        """Maximally overlapping translated subarray pair, or None.

        Currently exposed ONLY for uniform linear arrays (M >= 2) whose
        consecutive edge vectors are identical (collinear, uniform spacing
        AND a common orientation/order).  The displacement is the actual
        Cartesian step ``positions[1] - positions[0]`` in metres, so a
        translated or arbitrarily rotated ULA keeps its true geometry and
        no estimator may hard-code an x-axis step.

        Arbitrary non-uniform arrays, geometries with unequal translated
        subarray spacing, and arrays with fewer than two elements expose
        no pair (None); ESPRIT/Root-MUSIC estimators must fail loudly on
        such geometries instead of silently falling back.
        """
        if self.m < 2:
            return None
        if not self._is_ula():
            return None
        edges = np.diff(self._positions_m, axis=0)
        disp = np.asarray(edges[0], dtype=float).reshape(3,)
        if not np.all(np.isfinite(disp)):
            return None
        if float(np.linalg.norm(disp)) <= 0.0:
            return None
        return ShiftInvariantPair(
            indices_1=tuple(range(self.m - 1)),
            indices_2=tuple(range(1, self.m)),
            displacement_m=disp,
        )

    @property
    def ula_parameters(self) -> Optional[dict]:
        """ULA spacing/axis descriptor, or None when not a valid ULA.

        Returns ``{"spacing_m", "axis", "displacement_m", "n_elements"}``
        with ``axis`` the unit Cartesian direction of the array line.  Used
        by Root-MUSIC (ULA only); supports arbitrarily rotated ULAs via the
        detected axis instead of an assumed x-axis orientation.
        """
        pair = self.shift_invariant_pair
        if pair is None:
            return None
        disp = np.asarray(pair.displacement_m, dtype=float).reshape(3,)
        spacing = float(np.linalg.norm(disp))
        if not np.isfinite(spacing) or spacing <= 0.0:
            return None
        return {
            "spacing_m": spacing,
            "axis": disp / spacing,
            "displacement_m": disp.copy(),
            "n_elements": self.m,
        }

    # ------------------------------------------------------------------ #
    # structural classification (future algorithm/geometry gating)
    # ------------------------------------------------------------------ #
    @property
    def capabilities(self):
        """Capability-contract flags evaluated from the actual positions.

        Keys follow the DOA-ARCH-001 contract:
            arbitrary        - concrete geometry usable by a generic estimator
            shift_invariant  - a valid ShiftInvariantPair exists (currently a
                               ULA with M >= 2 and identical consecutive edge
                               vectors); a common translation of all positions
                               leaves the AoA invariant under steering maths
            ula              - collinear positions with uniform spacing
            ura              - planar rectangular lattice
            uca              - planar ring of equal radius (circular array)
        """
        return {
            "arbitrary": self.m >= 1,
            "shift_invariant": self.shift_invariant_pair is not None,
            "ula": self._is_ula(),
            "ura": self._is_ura(),
            "uca": self._is_uca(),
        }

    def _tol(self):
        extent = np.max(
            np.linalg.norm(self._positions_m - self._positions_m.mean(axis=0), axis=1)
        )
        return _GEOMETRY_TOL * max(1.0, float(extent))

    def _is_ula(self):
        if self.m < 2:
            return False
        edges = np.diff(self._positions_m, axis=0)
        tol = self._tol()
        norms = np.linalg.norm(edges, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            return False
        if not np.allclose(norms, norms[0], rtol=0.0, atol=tol):
            return False
        # Identical consecutive edge VECTORS (not only collinear norms):
        # an out-of-order ULA layout ([0, d, 3d, 2d]) or an unequal
        # translated-subarray spacing must not expose shift invariance.
        for edge in edges[1:]:
            if not np.allclose(edge, edges[0], rtol=0.0, atol=tol):
                return False
        return True

    def _is_planar(self, centered):
        _, singular, _ = np.linalg.svd(centered, full_matrices=False)
        singular = np.maximum(singular, 0.0)
        return bool(singular.shape[0] < 3 or singular[2] <= self._tol())

    def _is_ura(self):
        if self.m < 4:
            return False
        centered = self._positions_m - self._positions_m.mean(axis=0)
        if not self._is_planar(centered):
            return False
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        plane = centered @ vt[:2].T
        tol = self._tol()
        u = np.unique(np.round(plane[:, 0] / tol))
        v = np.unique(np.round(plane[:, 1] / tol))
        return bool(len(u) >= 2 and len(v) >= 2 and len(u) * len(v) == self.m)

    def _is_uca(self):
        if self.m < 3:
            return False
        centered = self._positions_m - self._positions_m.mean(axis=0)
        if not self._is_planar(centered):
            return False
        radii = np.linalg.norm(centered, axis=1)
        return bool(np.allclose(radii, radii[0], rtol=0.0, atol=self._tol()))