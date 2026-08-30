import numpy as np
import sys
from typing import List, Tuple, Optional
from collections import defaultdict

# ---------------------------------------------------------------------------
# SU(3) constants
# ---------------------------------------------------------------------------
_sqrt3 = np.sqrt(3.0)
_half = 0.5
_inv_sqrt2 = 1.0 / np.sqrt(2.0)

_czero = 0.0 + 0.0j
_cone = 1.0 + 0.0j
_cimag = 0.0 + 1.0j

# ---------------------------------------------------------------------------
# Gell-Mann basis generators
# ---------------------------------------------------------------------------

def _init_generators_GM():
    """
    Build SU(3) generators in Gell-Mann basis.
    Returns T_q(3,3,8), T_aq(3,3,8), T_g(8,8,8).
    Matches init_generators_GM in color_wf.f90.
    """
    lam = np.zeros((8, 3, 3), dtype=complex)
    lam[0] = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
    lam[1] = np.array([[0, -1j, 0],
                       [1j, 0, 0],
                       [0, 0, 0]], dtype=complex)
    lam[2] = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]], dtype=complex)
    lam[3] = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [1, 0, 0]], dtype=complex)
    lam[4] = np.array([[0, 0, -1j],
                       [0, 0, 0],
                       [1j, 0, 0]], dtype=complex)
    lam[5] = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]], dtype=complex)
    lam[6] = np.array([[0, 0, 0],
                       [0, 0, -1j],
                       [0, 1j, 0]], dtype=complex)
    lam[7] = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -2]], dtype=complex) / _sqrt3

    # Fortran indexing: T_q(c,c2,a) with c,c2 = 1..3, a = 1..8
    # Python shape: (3,3,8)
    T_q = np.zeros((3, 3, 8), dtype=complex)
    T_aq = np.zeros((3, 3, 8), dtype=complex)
    for a in range(8):
        T_q[:, :, a] = lam[a] * _half
        T_aq[:, :, a] = -lam[a].T * _half

    # f_abc (antisymmetric structure constants)
    f = np.zeros((8, 8, 8), dtype=complex)
    f[0, 1, 2] = _cone
    f[0, 3, 6] = _half * _cone
    f[0, 4, 5] = -_half * _cone
    f[1, 3, 5] = _half * _cone
    f[1, 4, 6] = _half * _cone
    f[2, 3, 4] = _half * _cone
    f[2, 5, 6] = -_half * _cone
    f[3, 4, 7] = _sqrt3 * _half * _cone
    f[5, 6, 7] = _sqrt3 * _half * _cone

    # Enforce full antisymmetry
    for i in range(8):
        for j in range(8):
            for a in range(8):
                if abs(f[i, j, a]) > 0.0:
                    f[j, a, i] = f[i, j, a]
                    f[a, i, j] = f[i, j, a]
                    f[j, i, a] = -f[i, j, a]
                    f[i, a, j] = -f[i, j, a]
                    f[a, j, i] = -f[i, j, a]

    # T_g(i,j,a) = -i * f(a,i,j)  with i,j = 1..8, a = 1..8
    # Fortran shape: T_g(8,8,8)
    T_g = np.zeros((8, 8, 8), dtype=complex)
    for a in range(8):
        for i in range(8):
            for j in range(8):
                T_g[i, j, a] = -1j * f[a, i, j]

    return T_q, T_aq, T_g


# ---------------------------------------------------------------------------
# State-space helpers
# ---------------------------------------------------------------------------

def _decode(idx: int, dims: List[int]) -> List[int]:
    """Decode a linear index into color indices for each particle."""
    colors = []
    temp = idx
    for d in dims:
        colors.append(temp % d)
        temp //= d
    return colors


def _encode(colors: List[int], dims: List[int]) -> int:
    """Encode color indices into a linear state index."""
    idx = 0
    mult = 1
    for c, d in zip(colors, dims):
        idx += c * mult
        mult *= d
    return idx


def _build_stride(dims: List[int]) -> List[int]:
    """Build stride array for index arithmetic."""
    stride = [1]
    for d in dims[:-1]:
        stride.append(stride[-1] * d)
    return stride


# ---------------------------------------------------------------------------
# Theoretical singlet counting (matches Fortran exactly)
# ---------------------------------------------------------------------------

def _count_singlets(n: int, m: int, k: int) -> int:
    """
    Count the number of SU(3) singlets using the same algorithm as Fortran.
    """
    if (n - m) % 3 != 0:
        return 0

    max_dim = n + m + 2 * k + 5

    # Initialize multi(0,0) = 1
    multi = defaultdict(int)
    multi[(0, 0)] = 1

    # Multiply by quarks
    for _ in range(n):
        new_multi = defaultdict(int)
        for (p, q), c in multi.items():
            if c == 0:
                continue
            if p + 1 <= max_dim:
                new_multi[(p + 1, q)] += c
            if p > 0 and q + 1 <= max_dim:
                new_multi[(p - 1, q + 1)] += c
            if q > 0:
                new_multi[(p, q - 1)] += c
        multi = new_multi

    # Multiply by antiquarks
    for _ in range(m):
        new_multi = defaultdict(int)
        for (p, q), c in multi.items():
            if c == 0:
                continue
            if q + 1 <= max_dim:
                new_multi[(p, q + 1)] += c
            if q > 0 and p + 1 <= max_dim:
                new_multi[(p + 1, q - 1)] += c
            if p > 0:
                new_multi[(p - 1, q)] += c
        multi = new_multi

    # Multiply by gluons
    for _ in range(k):
        # First quark-like step: T = multi * quark
        T = defaultdict(int)
        for (p, q), c in multi.items():
            if c == 0:
                continue
            if p + 1 <= max_dim:
                T[(p + 1, q)] += c
            if p > 0 and q + 1 <= max_dim:
                T[(p - 1, q + 1)] += c
            if q > 0:
                T[(p, q - 1)] += c

        # Then antiquark-like step: B = T * antiquark
        B = defaultdict(int)
        for (p, q), c in T.items():
            if c == 0:
                continue
            if q + 1 <= max_dim:
                B[(p, q + 1)] += c
            if q > 0 and p + 1 <= max_dim:
                B[(p + 1, q - 1)] += c
            if p > 0:
                B[(p - 1, q)] += c

        # B = B - multi (gluon = quark*antiquark - singlet)
        new_multi = defaultdict(int)
        for (p, q), c in B.items():
            new_multi[(p, q)] = c
        for (p, q), c in multi.items():
            new_multi[(p, q)] -= c

        multi = new_multi

    return multi.get((0, 0), 0)


# ---------------------------------------------------------------------------
# C2 matrix construction (full space, no zero-weight reduction)
# ---------------------------------------------------------------------------

def _build_c2_matrix(n: int, m: int, k: int,
                     T_q: np.ndarray, T_aq: np.ndarray, T_g: np.ndarray) -> np.ndarray:
    """
    Build the full C2 = Σ_a T^a T^a matrix in the complete state space.

    Parameters:
        n, m, k : number of quarks, antiquarks, gluons
        T_q, T_aq, T_g : generators from _init_generators_GM()
            T_q shape: (3,3,8), T_aq shape: (3,3,8), T_g shape: (8,8,8)

    Returns:
        C2 : (nstates, nstates) Hermitian complex matrix
    """
    npart = n + m + k
    dims = [3] * n + [3] * m + [8] * k
    nstates = int(np.prod(dims))
    stride = _build_stride(dims)
    tol = 1.0e-14

    C2 = np.zeros((nstates, nstates), dtype=complex)

    for a in range(8):
        # Build total generator T^a as a dense matrix
        Ta = np.zeros((nstates, nstates), dtype=complex)

        for idx in range(nstates):
            colors = _decode(idx, dims)

            # --- diagonal part ---
            diag = _czero
            for p in range(n):
                c = colors[p]
                diag += T_q[c, c, a]
            for p in range(n, n + m):
                c = colors[p]
                diag += T_aq[c, c, a]
            for p in range(n + m, npart):
                g = colors[p]
                diag += T_g[g, g, a]

            if abs(diag) > tol:
                Ta[idx, idx] = diag

            # --- quark off-diagonal ---
            for p in range(n):
                c = colors[p]
                for c2 in range(3):
                    if c2 == c:
                        continue
                    val = T_q[c2, c, a]
                    if abs(val) > tol:
                        idx2 = idx + (c2 - c) * stride[p]
                        Ta[idx2, idx] += val

            # --- antiquark off-diagonal ---
            for p in range(n, n + m):
                c = colors[p]
                for c2 in range(3):
                    if c2 == c:
                        continue
                    val = T_aq[c2, c, a]
                    if abs(val) > tol:
                        idx2 = idx + (c2 - c) * stride[p]
                        Ta[idx2, idx] += val

            # --- gluon off-diagonal ---
            for p in range(n + m, npart):
                g = colors[p]
                for g2 in range(8):
                    if g2 == g:
                        continue
                    val = T_g[g2, g, a]
                    if abs(val) > tol:
                        idx2 = idx + (g2 - g) * stride[p]
                        Ta[idx2, idx] += val

        C2 += Ta @ Ta

    return C2

eigv, eigvec = np.linalg.eigh(_build_c2_matrix(0, 0, 3, ))
print(eigvec)