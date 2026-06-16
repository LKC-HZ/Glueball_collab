"""
su3constants.py — SU(3) group constants and utilities for color-space calculations.

This module provides the Gell-Mann matrices, the antisymmetric (fabc) and 
symmetric (dabc) structure constants, the SU(3) generators in the fundamental 
(Ta) and adjoint (Fa) representations, and the quadratic Casimir invariants 
(CF, CA, TF).  It is designed for reuse in BLFQ / glueball / ColorFactor 
workflows and any other QCD color-algebra script.

------------------------------------------------------------------------------
USAGE IN OTHER PROGRAMS
------------------------------------------------------------------------------

    from su3constants import SU3Constants

    su3 = SU3Constants()

    # 1. Gell-Mann matrices (lambda_a, a = 1..8)
    lam1 = su3.gellmann[0]          # 3×3 complex matrix

    # 2. Fundamental generators  Ta = lambda_a / 2
    T1 = su3.T[0]                   # 3×3 complex matrix

    # 3. Antisymmetric structure constants f_abc  (a,b,c = 1..8)
    f123 = su3.fabc[0, 1, 2]       # = 1.0  (real)
    f_full = su3.fabc               # 8×8×8 real numpy array

    # 4. Symmetric structure constants d_abc  (a,b,c = 1..8)
    d118 = su3.dabc[0, 0, 7]       # = 1/sqrt(3)
    d_full = su3.dabc              # 8×8×8 real numpy array

    # 5. Adjoint representation generators (Fa)_{bc} = -i f_{abc}
    F1 = su3.F[0]                   # 8×8 complex matrix

    # 6. Casimir invariants
    print(su3.CF)                   # 4/3
    print(su3.CA)                   # 3
    print(su3.TF)                   # 1/2

    # 7. Group-theory checks
    su3.run_all_checks()            # prints a battery of sanity checks

------------------------------------------------------------------------------
QUICK REFERENCE — FORMULAS IMPLEMENTED
------------------------------------------------------------------------------

    [T^a, T^b]      = i f^{abc} T^c
    {T^a, T^b}      = 1/3 delta^{ab} + d^{abc} T^c
    Tr(T^a T^b)     = 1/2 delta^{ab}
    (F^a)_{bc}      = -i f^{abc}
    T^a T^a         = C_F · I_3   (summed over a)
    F^a F^a         = C_A · I_8   (summed over a)

    For SU(N) generally:  CF = (N^2-1)/(2N),  CA = N,  TF = 1/2.

    All indices are 1-based in physics notation but stored 0-based in NumPy.
"""

import numpy as np
from typing import Dict, Tuple, List


class SU3Constants:
    """Container for SU(3) color algebra constants and helper methods."""

    # ------------------------------------------------------------------
    # SU(3) dimension data
    # ------------------------------------------------------------------
    Nc: int = 3                      # number of colors
    Nc2_minus_1: int = 8             # number of generators
    CF: float = 4.0 / 3.0            # quadratic Casimir of fundamental
    CA: float = 3.0                  # quadratic Casimir of adjoint
    TF: float = 0.5                  # Dynkin index of fundamental

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self):
        self._build_gellmann()
        self._build_generators()
        self._build_fabc()
        self._build_dabc()
        self._build_adjoint_generators()
        self._build_casimir_matrices()

    # ------------------------------------------------------------------
    # 1. Gell-Mann matrices
    # ------------------------------------------------------------------
    def _build_gellmann(self) -> None:
        l1 = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
        l2 = np.array([[0, -1j, 0],
                       [1j, 0, 0],
                       [0, 0, 0]], dtype=complex)
        l3 = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]], dtype=complex)
        l4 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [1, 0, 0]], dtype=complex)
        l5 = np.array([[0, 0, -1j],
                       [0, 0, 0],
                       [1j, 0, 0]], dtype=complex)
        l6 = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]], dtype=complex)
        l7 = np.array([[0, 0, 0],
                       [0, 0, -1j],
                       [0, 1j, 0]], dtype=complex)
        l8 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -2]], dtype=complex) / np.sqrt(3)
        self.gellmann: List[np.ndarray] = [l1, l2, l3, l4, l5, l6, l7, l8]

    # ------------------------------------------------------------------
    # 2. Fundamental generators  Ta = λa / 2
    # ------------------------------------------------------------------
    def _build_generators(self) -> None:
        self.T: List[np.ndarray] = [lam / 2.0 for lam in self.gellmann]

    # ------------------------------------------------------------------
    # 3. Antisymmetric structure constants fabc
    # ------------------------------------------------------------------
    def _build_fabc(self) -> None:
        # Dictionary with 1-based keys for non-zero entries
        fabc_nonzero: Dict[Tuple[int, int, int], complex] = {}
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    trace_val = np.trace(
                        self._comm(self.T[a], self.T[b]) @ self.T[c]
                    )
                    if abs(trace_val) > 1e-14:
                        fabc_nonzero[(a + 1, b + 1, c + 1)] = -2j * trace_val
        self.fabc_nonzero_dict: Dict[Tuple[int, int, int], complex] = fabc_nonzero

        # Full 8×8×8 NumPy array (0-based indexing)
        fabc = np.zeros((8, 8, 8), dtype=complex)
        for (a, b, c), val in fabc_nonzero.items():
            fabc[a - 1, b - 1, c - 1] = val
        # Enforce antisymmetry: f_{bac} = -f_{abc}, f_{acb} = -f_{abc}, f_{cba} = -f_{abc}
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    fabc[b, a, c] = -fabc[a, b, c]
                    fabc[a, c, b] = -fabc[a, b, c]
                    fabc[c, b, a] = -fabc[a, b, c]
        self.fabc: np.ndarray = fabc.real  # fabc is purely real for SU(3)

    # ------------------------------------------------------------------
    # 4. Symmetric structure constants dabc
    # ------------------------------------------------------------------
    def _build_dabc(self) -> None:
        # d^{abc} = 2 * Tr({T^a, T^b} T^c)
        dabc_nonzero: Dict[Tuple[int, int, int], complex] = {}
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    trace_val = np.trace(
                        self._anti_comm(self.T[a], self.T[b]) @ self.T[c]
                    )
                    if abs(trace_val) > 1e-14:
                        dabc_nonzero[(a + 1, b + 1, c + 1)] = 2.0 * trace_val
        self.dabc_nonzero_dict: Dict[Tuple[int, int, int], complex] = dabc_nonzero

        dabc = np.zeros((8, 8, 8), dtype=complex)
        for (a, b, c), val in dabc_nonzero.items():
            dabc[a - 1, b - 1, c - 1] = val
        # Enforce full symmetry
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    dabc[b, a, c] = dabc[a, b, c]
                    dabc[a, c, b] = dabc[a, b, c]
                    dabc[c, b, a] = dabc[a, b, c]
                    dabc[b, c, a] = dabc[a, b, c]
                    dabc[c, a, b] = dabc[a, b, c]
        self.dabc: np.ndarray = dabc.real  # dabc is purely real for SU(3)

    # ------------------------------------------------------------------
    # 5. Adjoint representation generators  (F^a)_{bc} = -i f^{abc}
    # ------------------------------------------------------------------
    def _build_adjoint_generators(self) -> None:
        self.F: List[np.ndarray] = [
            -1j * self.fabc[a, :, :] for a in range(8)
        ]

    # ------------------------------------------------------------------
    # 6. Quadratic Casimir matrices (for quick tests)
    # ------------------------------------------------------------------
    def _build_casimir_matrices(self) -> None:
        # Sum_a T^a T^a  should equal CF * I_3
        self.TT_sum_fund = sum(T @ T for T in self.T)
        # Sum_a F^a F^a  should equal CA * I_8
        self.FF_sum_adj = sum(F @ F for F in self.F)

    # ------------------------------------------------------------------
    # Algebra methods
    # ------------------------------------------------------------------
    @staticmethod
    def _comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix commutator [A, B]."""
        return A @ B - B @ A

    @staticmethod
    def _anti_comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix anticommutator {A, B}."""
        return A @ B + B @ A

    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Public alias for the commutator."""
        return A @ B - B @ A

    @staticmethod
    def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Public alias for the anticommutator."""
        return A @ B + B @ A

    # ------------------------------------------------------------------
    # Accessor helpers (1-based physics indexing)
    # ------------------------------------------------------------------
    def f(self, a: int, b: int, c: int) -> float:
        """Return f_{abc} using 1-based indices (a,b,c = 1..8)."""
        return float(self.fabc[a - 1, b - 1, c - 1])

    def d(self, a: int, b: int, c: int) -> float:
        """Return d_{abc} using 1-based indices (a,b,c = 1..8)."""
        return float(self.dabc[a - 1, b - 1, c - 1])

    def generator(self, a: int) -> np.ndarray:
        """Return T^a using 1-based index (a = 1..8)."""
        return self.T[a - 1]

    def gellmann_matrix(self, a: int) -> np.ndarray:
        """Return λ_a using 1-based index (a = 1..8)."""
        return self.gellmann[a - 1]

    def adjoint_generator(self, a: int) -> np.ndarray:
        """Return F^a using 1-based index (a = 1..8)."""
        return self.F[a - 1]

    # ------------------------------------------------------------------
    # Structure-constant derived identities
    # ------------------------------------------------------------------
    def jacobi_identity(self, a: int, b: int, c: int, d: int) -> complex:
        """
        Evaluate the Jacobi identity for four indices (1-based):
            f_{abe} f_{ecd} + f_{cbe} f_{ead} + f_{dbe} f_{ace} = 0
        Returns the sum (should be ~0 numerically).
        """
        total = 0.0 + 0.0j
        for e in range(1, 9):
            total += (
                self.f(a, b, e) * self.f(e, c, d)
                + self.f(c, b, e) * self.f(e, a, d)
                + self.f(d, b, e) * self.f(e, a, c)
            )
        return total

    def f_times_d_identity(self, a: int, b: int, c: int, d: int) -> complex:
        """
        Mixed identity (1-based indices):
            f_{abe} d_{ecd} + f_{cbe} d_{ead} + f_{dbe} d_{ace} = 0
        Returns the sum (should be ~0 numerically).
        """
        total = 0.0 + 0.0j
        for e in range(1, 9):
            total += (
                self.f(a, b, e) * self.d(e, c, d)
                + self.f(c, b, e) * self.d(e, a, d)
                + self.f(d, b, e) * self.d(e, a, c)
            )
        return total

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    def _is_hermitian(self, M: np.ndarray, tol: float = 1e-12) -> bool:
        return np.allclose(M, M.conj().T, atol=tol)

    def _is_anti_hermitian(self, M: np.ndarray, tol: float = 1e-12) -> bool:
        return np.allclose(M, -M.conj().T, atol=tol)

    def run_all_checks(self) -> None:
        """Print a battery of numerical sanity checks."""
        print("=" * 60)
        print("SU(3) Constants — Numerical Sanity Checks")
        print("=" * 60)

        # 1. Gell-Mann Hermiticity
        print("\n[1] Gell-Mann matrices Hermitian?")
        for a in range(8):
            ok = self._is_hermitian(self.gellmann[a])
            print(f"    λ_{a+1}: {'PASS' if ok else 'FAIL'}")

        # 2. T^a Hermiticity
        print("\n[2] Fundamental generators T^a Hermitian?")
        for a in range(8):
            ok = self._is_hermitian(self.T[a])
            print(f"    T_{a+1}: {'PASS' if ok else 'FAIL'}")

        # 3. Tr(T^a T^b) = 1/2 delta^{ab}
        print("\n[3] Tr(T^a T^b) = 1/2 delta^{ab} ?")
        trTT = np.zeros((8, 8), dtype=complex)
        for a in range(8):
            for b in range(8):
                trTT[a, b] = np.trace(self.T[a] @ self.T[b])
        expected = 0.5 * np.eye(8, dtype=complex)
        ok = np.allclose(trTT, expected, atol=1e-12)
        print(f"    Max deviation: {np.max(np.abs(trTT - expected)):.2e}  →  {'PASS' if ok else 'FAIL'}")

        # 4. [T^a, T^b] = i f^{abc} T^c
        print("\n[4] [T^a, T^b] = i f^{abc} T^c ?")
        max_dev = 0.0
        for a in range(8):
            for b in range(8):
                lhs = self._comm(self.T[a], self.T[b])
                rhs = sum(1j * self.fabc[a, b, c] * self.T[c] for c in range(8))
                dev = np.max(np.abs(lhs - rhs))
                if dev > max_dev:
                    max_dev = dev
        ok = max_dev < 1e-12
        print(f"    Max deviation: {max_dev:.2e}  →  {'PASS' if ok else 'FAIL'}")

        # 5. {T^a, T^b} = 1/3 delta^{ab} + d^{abc} T^c
        print("\n[5] {T^a, T^b} = 1/3 delta^{ab} + d^{abc} T^c ?")
        max_dev = 0.0
        for a in range(8):
            for b in range(8):
                lhs = self._anti_comm(self.T[a], self.T[b])
                rhs = (1.0 / 3.0) * np.eye(3, dtype=complex) if a == b else np.zeros((3, 3), dtype=complex)
                rhs += sum(self.dabc[a, b, c] * self.T[c] for c in range(8))
                dev = np.max(np.abs(lhs - rhs))
                if dev > max_dev:
                    max_dev = dev
        ok = max_dev < 1e-12
        print(f"    Max deviation: {max_dev:.2e}  →  {'PASS' if ok else 'FAIL'}")

        # 6. F^a Hermitian
        print("\n[6] Adjoint generators F^a Hermitian?")
        for a in range(8):
            ok = self._is_hermitian(self.F[a])
            print(f"    F_{a+1}: {'PASS' if ok else 'FAIL'}")

        # 7. [F^a, F^b] = i f^{abc} F^c
        print("\n[7] [F^a, F^b] = i f^{abc} F^c ?")
        max_dev = 0.0
        for a in range(8):
            for b in range(8):
                lhs = self._comm(self.F[a], self.F[b])
                rhs = sum(1j * self.fabc[a, b, c] * self.F[c] for c in range(8))
                dev = np.max(np.abs(lhs - rhs))
                if dev > max_dev:
                    max_dev = dev
        ok = max_dev < 1e-12
        print(f"    Max deviation: {max_dev:.2e}  →  {'PASS' if ok else 'FAIL'}")

        # 8. Casimir sums
        print("\n[8] Quadratic Casimir matrices:")
        ok_fund = np.allclose(self.TT_sum_fund, self.CF * np.eye(3), atol=1e-12)
        ok_adj = np.allclose(self.FF_sum_adj, self.CA * np.eye(8), atol=1e-12)
        print(f"    Σ_a T^a T^a = CF·I_3  →  {'PASS' if ok_fund else 'FAIL'}")
        print(f"    Σ_a F^a F^a = CA·I_8  →  {'PASS' if ok_adj else 'FAIL'}")

        # 9. Jacobi random spot-check
        print("\n[9] Jacobi identity spot-check (a=1,b=2,c=3,d=4):")
        val = self.jacobi_identity(1, 2, 3, 4)
        print(f"    Value: {val:.2e}  →  {'PASS' if abs(val) < 1e-12 else 'FAIL'}")

        print("\n" + "=" * 60)


# --------------------------------------------------------------------------
# Legacy alias for backward compatibility with older scripts
# --------------------------------------------------------------------------
groupconstants = SU3Constants


# --------------------------------------------------------------------------
# CHECK
# --------------------------------------------------------------------------
if __name__ == "__main__":
    su3 = SU3Constants()
    su3.run_all_checks()

    # Quick demo of the 1-based accessors
    print("\nSample values:")
    print(f"f_{123} = {su3.f(1, 2, 3)}")
    print(f"d_{118} = {su3.d(1, 1, 8):.6f}")
    print(f"d_{146} = {su3.d(1, 4, 6):.6f}")
