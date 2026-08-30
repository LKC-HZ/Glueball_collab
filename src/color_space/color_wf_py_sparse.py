#!/usr/bin/env python3
"""
color_wf_py.py — Pure Python color singlet wavefunction generator.
Direct full-space diagonalization in Gell-Mann basis (no zero-weight reduction).

Replicates the logic of color_wf.f90, but:
  • works in the FULL state space (no Cartan zero-weight subspace reduction)
  • directly builds the C2 matrix and diagonalizes it with numpy.linalg.eigh
  • keeps the same realification pipeline (Gram-Schmidt → M-diag → P12)
  • produces stdout identical to the Fortran program so colorwf_runner.py
    can parse it without any changes.

Usage:
    python color_wf_py.py n m k
    # or imported:
    from color_wf_py import run_color_wf_py
    result_str = run_color_wf_py(n=0, m=0, k=2)
"""

import numpy as np
import sys
from typing import List, Tuple, Optional
from collections import defaultdict
from scipy import sparse
from scipy.sparse.linalg import eigsh

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
                     T_q: np.ndarray, T_aq: np.ndarray, T_g: np.ndarray):
    """
    Build the full C2 = Σ_a T^a T^a matrix in the complete state space.
    SPARSE version using scipy.sparse to avoid O(N^2) memory blow-up.

    Parameters:
        n, m, k : number of quarks, antiquarks, gluons
        T_q, T_aq, T_g : generators from _init_generators_GM()

    Returns:
        C2 : scipy.sparse.csr_matrix (Hermitian)
    """
    npart = n + m + k
    dims = [3] * n + [3] * m + [8] * k
    nstates = int(np.prod(dims))
    stride = _build_stride(dims)
    tol = 1.0e-14

    C2 = None

    for a in range(8):
        ta_rows, ta_cols, ta_data = [], [], []

        for idx in range(nstates):
            colors = _decode(idx, dims)

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
                ta_rows.append(idx)
                ta_cols.append(idx)
                ta_data.append(diag)

            for p in range(n):
                c = colors[p]
                for c2 in range(3):
                    if c2 == c:
                        continue
                    val = T_q[c2, c, a]
                    if abs(val) > tol:
                        idx2 = idx + (c2 - c) * stride[p]
                        ta_rows.append(idx2)
                        ta_cols.append(idx)
                        ta_data.append(val)

            for p in range(n, n + m):
                c = colors[p]
                for c2 in range(3):
                    if c2 == c:
                        continue
                    val = T_aq[c2, c, a]
                    if abs(val) > tol:
                        idx2 = idx + (c2 - c) * stride[p]
                        ta_rows.append(idx2)
                        ta_cols.append(idx)
                        ta_data.append(val)

            for p in range(n + m, npart):
                g = colors[p]
                for g2 in range(8):
                    if g2 == g:
                        continue
                    val = T_g[g2, g, a]
                    if abs(val) > tol:
                        idx2 = idx + (g2 - g) * stride[p]
                        ta_rows.append(idx2)
                        ta_cols.append(idx)
                        ta_data.append(val)

        Ta = sparse.csr_matrix((ta_data, (ta_rows, ta_cols)),
                                shape=(nstates, nstates))
        C2_a = Ta.conj().T @ Ta

        if C2 is None:
            C2 = C2_a
        else:
            C2 = C2 + C2_a

    C2 = (C2 + C2.conj().T) * 0.5
    return C2


# ---------------------------------------------------------------------------
# Realification pipeline (after nullspace extraction)
# ---------------------------------------------------------------------------

def _realify_nullspace(nullspace: np.ndarray, n: int, m: int, k: int,
                       dims: List[int], nstates: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process the nullspace basis:
      - Pure systems (only quarks or only gluons): real Gram-Schmidt -> M-diag -> P12
      - Mixed systems (quarks + gluons): complex Gram-Schmidt -> M-diag -> P12
        (skips forced zeroing of imaginary parts)

    Returns:
        (basis, aeigval_or_None)
    """
    tol = 1.0e-10
    nullity = nullspace.shape[1]
    tmp_basis = nullspace.copy()

    is_pure = (k == 0) or (n == 0 and m == 0)

    if is_pure:
        # ---- Pure system: realify ----
        # Step 1: Zero out residual imaginary parts
        for j in range(nullity):
            for i in range(nstates):
                tmp_basis[i, j] = complex(tmp_basis[i, j].real, 0.0)

        # Step 2: Real Gram-Schmidt
        kk = 0
        for cand in range(nullity):
            tmpv = tmp_basis[:, cand].copy()
            for i in range(kk):
                proj_val = np.real(np.vdot(tmp_basis[:, i], tmpv))
                tmpv -= proj_val * tmp_basis[:, i]
            norm_val = np.sqrt(np.real(np.vdot(tmpv, tmpv)))
            if norm_val > tol:
                tmp_basis[:, kk] = tmpv / norm_val
                kk += 1

        if kk != nullity:
            raise RuntimeError(f"GM Gram-Schmidt rank {kk} != {nullity}")

        def _diag(rmat):
            return _jacobi_diag_sym(np.real(rmat))
    else:
        # ---- Mixed system: keep complex ----
        # Step 2: Complex Gram-Schmidt
        kk = 0
        for cand in range(nullity):
            tmpv = tmp_basis[:, cand].copy()
            for i in range(kk):
                proj_val = np.vdot(tmp_basis[:, i], tmpv)
                tmpv -= proj_val * tmp_basis[:, i]
            norm_val = np.sqrt(np.real(np.vdot(tmpv, tmpv)))
            if norm_val > tol:
                tmp_basis[:, kk] = tmpv / norm_val
                kk += 1

        if kk != nullity:
            raise RuntimeError(f"Complex Gram-Schmidt rank {kk} != {nullity}")

        def _diag(rmat):
            eigval, eigvec = np.linalg.eigh(rmat)
            return eigval, eigvec

    # ---- Step 3: M-diagonalisation ----
    aeigval = None
    if n == m and nullity > 1:
        psi_M = tmp_basis.copy()
        rmat = np.zeros((nullity, nullity), dtype=complex)
        for j in range(nullity):
            tmpv = _apply_M_full(psi_M[:, j], n, m, k, dims, nstates)
            for i in range(nullity):
                rmat[i, j] = np.vdot(psi_M[:, i], tmpv)
        aeigval, aeigvec = _diag(rmat)
        for kk in range(nullity):
            tmp_basis[:, kk] = _czero
            for j in range(nullity):
                tmp_basis[:, kk] += aeigvec[j, kk] * psi_M[:, j]

    # ---- Step 4: P12 interchange for d/f labelling ----
    p1_swap = 0
    p2_swap = 0
    if k >= 2:
        p1_swap = n + m + 1
        p2_swap = n + m + 2
    elif n >= 2:
        p1_swap = 1
        p2_swap = 2
    elif m >= 2:
        p1_swap = n + 1
        p2_swap = n + 2

    if p1_swap > 0:
        rmat = np.zeros((nullity, nullity), dtype=complex)
        for j in range(nullity):
            for kk in range(nullity):
                rmat[j, kk] = _swap_overlap(tmp_basis[:, j], tmp_basis[:, kk],
                                            p1_swap - 1, p2_swap - 1, dims, nstates)
        aeigval, aeigvec = _diag(rmat)
        psi_M = tmp_basis.copy()
        for kk in range(nullity):
            tmp_basis[:, kk] = _czero
            for j in range(nullity):
                tmp_basis[:, kk] += aeigvec[j, kk] * psi_M[:, j]
        # Sort: +1 (d-type) first, -1 (f-type) last
        for i in range(nullity - 1):
            kk = i
            for j in range(i + 1, nullity):
                if aeigval[j] > aeigval[kk]:
                    kk = j
            if kk != i:
                aeigval[i], aeigval[kk] = aeigval[kk], aeigval[i]
                tmp_basis[:, [i, kk]] = tmp_basis[:, [kk, i]]
        # Second round for refinement
        rmat = np.zeros((nullity, nullity), dtype=complex)
        for j in range(nullity):
            for kk in range(nullity):
                rmat[j, kk] = _swap_overlap(tmp_basis[:, j], tmp_basis[:, kk],
                                            p1_swap - 1, p2_swap - 1, dims, nstates)
        aeigval, aeigvec = _diag(rmat)
        psi_M = tmp_basis.copy()
        for kk in range(nullity):
            tmp_basis[:, kk] = _czero
            for j in range(nullity):
                tmp_basis[:, kk] += aeigvec[j, kk] * psi_M[:, j]
        # Sort again
        for i in range(nullity - 1):
            kk = i
            for j in range(i + 1, nullity):
                if aeigval[j] > aeigval[kk]:
                    kk = j
            if kk != i:
                aeigval[i], aeigval[kk] = aeigval[kk], aeigval[i]
                tmp_basis[:, [i, kk]] = tmp_basis[:, [kk, i]]

    # ---- Final: normalise and clean numerical noise ----
    for j in range(nullity):
        tmp_basis[:, j] /= np.sqrt(np.sum(np.abs(tmp_basis[:, j]) ** 2))
        if is_pure:
            for i in range(nstates):
                if abs(tmp_basis[i, j].imag) < 1e-10 * max(tol, abs(tmp_basis[i, j].real)):
                    tmp_basis[i, j] = complex(tmp_basis[i, j].real, 0.0)

    return tmp_basis, aeigval


# ---------------------------------------------------------------------------
# M operator (total weight inversion)
# ---------------------------------------------------------------------------

# In GM basis, M = U U^T where U is CW→GM transform.
# M swaps λ1↔λ2, λ4↔λ5, λ6↔λ7 and leaves λ3, λ8 unchanged.
# 0-based GM indexing: 0↔1, 3↔4, 5↔6, 2 and 7 fixed.
_M_PERM_GM = np.array([1, 0, 2, 4, 3, 6, 5, 7])


def _apply_M_full(psi: np.ndarray, n: int, m: int, k: int,
                  dims: List[int], nstates: int) -> np.ndarray:
    """Apply the M operator (weight inversion) to a full-space GM vector."""
    psi_out = np.zeros(nstates, dtype=complex)
    for idx in range(nstates):
        if abs(psi[idx]) < 1.0e-30:
            continue
        colors = _decode(idx, dims)
        mapped = colors.copy()
        # Swap quark ↔ antiquark color indices
        for ii in range(min(n, m)):
            mapped[ii] = colors[n + ii]
            mapped[n + ii] = colors[ii]
        # Apply M_perm to each gluon slot (GM basis)
        for p in range(n + m, n + m + k):
            g = colors[p]
            mapped[p] = _M_PERM_GM[g]
        idx2 = _encode(mapped, dims)
        psi_out[idx2] = psi[idx]
    return psi_out


# ---------------------------------------------------------------------------
# Swap overlap (P12)
# ---------------------------------------------------------------------------

def _swap_overlap(v1: np.ndarray, v2: np.ndarray,
                  p1: int, p2: int, dims: List[int], nstates: int) -> complex:
    """
    Compute <P12 v1 | v2> where P12 swaps particle indices p1, p2.
    p1, p2 are 0-based particle indices.
    """
    ov = _czero
    for idx in range(nstates):
        if abs(v2[idx]) < 1.0e-14:
            continue
        colors = _decode(idx, dims)
        colors[p1], colors[p2] = colors[p2], colors[p1]
        idx2 = _encode(colors, dims)
        colors[p1], colors[p2] = colors[p2], colors[p1]  # restore
        ov += np.conj(v1[idx2]) * v2[idx]
    return ov


# ---------------------------------------------------------------------------
# Jacobi diagonalisation of real symmetric matrix
# ---------------------------------------------------------------------------

def _jacobi_diag_sym(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jacobi diagonalisation of a real symmetric matrix.
    Returns (eigval, eigvec).
    """
    n = A.shape[0]
    A = A.copy()
    eigvec = np.eye(n, dtype=float)
    conv_tol = 1.0e-14
    max_iter = 100

    for _iter in range(max_iter):
        # Find largest off-diagonal element
        off = 0.0
        p = 0
        q = 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(A[i, j]) > off:
                    off = abs(A[i, j])
                    p = i
                    q = j
        if off < conv_tol:
            break

        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]
        theta = 0.5 * np.arctan2(2.0 * apq, aqq - app)
        c = np.cos(theta)
        s = np.sin(theta)

        # Diagonal block
        A[p, p] = c * c * app + s * s * aqq - 2.0 * c * s * apq
        A[q, q] = s * s * app + c * c * aqq + 2.0 * c * s * apq
        A[p, q] = 0.0
        A[q, p] = 0.0

        # Off-diagonal rows/cols
        for i in range(n):
            if i != p and i != q:
                aip = A[i, p]
                aiq = A[i, q]
                A[i, p] = c * aip - s * aiq
                A[p, i] = A[i, p]
                A[i, q] = s * aip + c * aiq
                A[q, i] = A[i, q]

        # Eigenvectors
        for i in range(n):
            vip = eigvec[i, p]
            viq = eigvec[i, q]
            eigvec[i, p] = c * vip - s * viq
            eigvec[i, q] = s * vip + c * viq

    eigval = np.diag(A).copy()
    return eigval, eigvec


# ---------------------------------------------------------------------------
# Output formatting (matches Fortran color_wf.f90 GM section)
# ---------------------------------------------------------------------------

def _format_output(nullspace_basis: np.ndarray, n: int, m: int, k: int,
                   dims: List[int], nstates: int, nsing: int,
                   aeigval: Optional[np.ndarray] = None) -> str:
    """Format the singlet wavefunctions to match Fortran output."""
    lines = []
    tol = 1.0e-10
    npart = n + m + k
    nullity = nullspace_basis.shape[1]

    # Header diagnostics
    lines.append(f" Theoretical singlets:  {nsing}")
    lines.append(f" Full dimension:        {nstates}")
    lines.append(f" Zero-weight dimension: {nstates}")
    lines.append(f" Converged eigenvalues: {nullity}")
    lines.append(f" Actual number of singlets:  {nullity}")
    lines.append("")
    lines.append("===== Gell-Mann basis representation =====")
    lines.append("")

    # Output in Gell-Mann basis directly (no CW transformation)
    for j in range(nullity):
        sum_re = np.sum(nullspace_basis[:, j] ** 2)
        sum_im = sum_re
        #sum_im = np.sum(nullspace_basis[:, j].imag ** 2)

        if aeigval is not None:
            if aeigval[j] >= 0.0:
                lines.append(f"--- Singlet {j + 1} (GM; d-type) ---")
            else:
                lines.append(f"--- Singlet {j + 1} (GM; f-type) ---")
        else:
            if sum_re >= sum_im:
                lines.append(f"--- Singlet {j + 1} (GM) ---")
            else:
                lines.append(f"--- Singlet {j + 1} (GM; imag) ---")

        for idx in range(nstates):
            coeff = nullspace_basis[idx, j]
            if abs(coeff) > tol:
                colors = _decode(idx, dims)
                def _trim(s):
                    return s.rstrip(" ")
                label = ""
                # Quarks
                if n > 0:
                    label = ",".join(f"q{colors[p] + 1}" for p in range(n))
                # Antiquarks
                if m > 0:
                    if n > 0:
                        label = _trim(label) + " | "
                    label = _trim(label) + ",".join(f"qb{colors[p] + 1}" for p in range(n, n + m))
                # Gluons (GM labels)
                if k > 0:
                    if n > 0 or m > 0:
                        label = _trim(label) + " | "
                    label = _trim(label) + ",".join(f"lam{colors[p] + 1}" for p in range(n + m, npart))
                lines.append(f"{label} : {coeff.real:+.8f} {coeff.imag:+.8f}i")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_color_wf_py(n: int, m: int, k: int) -> str:
    """
    Compute color singlets for (n quarks, m antiquarks, k gluons).
    Returns stdout-formatted string compatible with colorwf_runner.parse_color_wf_output().
    """
    if n < 0 or m < 0 or k < 0 or (n - m) % 3 != 0:
        raise ValueError("Invalid input or triality")

    nsing = _count_singlets(n, m, k)

    npart = n + m + k
    dims = [3] * n + [3] * m + [8] * k
    nstates = int(np.prod(dims))

    if nstates > 500000:
        raise ValueError("State space too large for direct diagonalization")

    if nsing == 0:
        return " Theoretical singlets:  0\nNo singlets\n"

    # Build generators
    T_q, T_aq, T_g = _init_generators_GM()

    # Build C2 matrix (sparse) and diagonalize
    C2 = _build_c2_matrix(n, m, k, T_q, T_aq, T_g)

    # Find smallest eigenvalues using sparse eigensolver
    k_eig = min(nsing + 20, nstates - 2)
    if k_eig < 1:
        k_eig = 1

    try:
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', tol=1e-7, maxiter=5000)
    except Exception:
        # Fallback for singular matrices: use a tiny positive sigma
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', sigma=1e-10, tol=1e-7, maxiter=5000)
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM')
    except Exception:
        # Fallback for singular matrices: use a tiny positive sigma
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', sigma=1e-10)

    # Extract nullspace (zero eigenvalues)
    tol = 1.0e-10
    null_mask = np.abs(eigvals) < tol
    nullity = int(np.sum(null_mask))

    if nullity == 0:
        return f" Theoretical singlets:  {nsing}\nNo singlet states\n"

    if nullity != nsing:
        pass  # Could raise warning

    nullspace = eigvecs[:, null_mask]

    # Realification pipeline
    basis, aeigval = _realify_nullspace(nullspace, n, m, k, dims, nstates)

    # Format output
    return _format_output(basis, n, m, k, dims, nstates, nsing, aeigval)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pure Python color singlet generator")
    parser.add_argument("n", type=int, help="Number of quarks")
    parser.add_argument("m", type=int, help="Number of antiquarks")
    parser.add_argument("k", type=int, help="Number of gluons")
    args = parser.parse_args()

    out = run_color_wf_py(args.n, args.m, args.k)
    print(out, end="")
