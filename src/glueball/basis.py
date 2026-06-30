from dataclasses import dataclass
import itertools
from pathlib import Path

# ===================================================================
# Light-front single-particle state for a gluon.
# Quantum numbers:
#   k : longitudinal momentum fraction (DLCQ integer, discrete k+)
#   s : helicity / spin projection (±1 for massless gluon)
#   n : transverse 2D harmonic oscillator radial quantum number
#   m : transverse 2D HO angular momentum projection (magnetic)

# CHECKED. 
# ===================================================================

@dataclass(frozen=True)
class SingleParticleState:
    k: int
    s: int
    n: int
    m: int


# ===================================================================
# Multi-particle state for n gluons.
# The color sector is tagged by a singlet index (1-based) obtained
# from the external Fortran color_wf code.
# ===================================================================

@dataclass(frozen=True)
class NParticleState:
    particles: tuple       # tuple of SingleParticleState, length = n_gluons
    color_singlet: int     # 1-based singlet index from color_wf


# ===================================================================
# 1. Longitudinal momentum decomposition (DLCQ)
# ===================================================================
# Generate all ordered compositions of k0 into n positive integers.
# Each composition corresponds to a valid longitudinal momentum
# assignment for n gluons with total P+ = k0.
# Constraint: k1 + k2 + ... + kn = k0,  each ki >= 1
# ===================================================================
def k_basis(k0, n):
    def k_decomp(k0, n):
        if n == 1:
            yield (k0,)
        else:
            for i in range(1, k0 - n + 2):
                for r in k_decomp(k0 - i, n - 1):
                    yield (i,) + r
    return list(k_decomp(k0, n))


# ===================================================================
# 2. Color singlet labels
# ===================================================================
# The actual singlet wavefunctions are computed by the Fortran
# color_wf program.  Here we only enumerate the singlet indices
# as external labels (1, 2, ..., num_singlets).
# ===================================================================
def color_list(color_singlet_number):
    return list(range(1, color_singlet_number + 1))


# ===================================================================
# 3. Spin / helicity combinations
# ===================================================================
# Each gluon carries helicity ±1.  For n gluons we enumerate all
# 2^n combinations via direct product against itself.
# ===================================================================
def spin_list(n):
    return list(itertools.product([-1, 1], repeat=n))


# ===================================================================
# 4. Transverse 2D HO basis
# ===================================================================
# Single gluon:  N_perp = 2n + |m| + 1 <= Nmax
# n gluons:      sum_i N_perp_i <= Nmax  (BLFQ truncation)
#
# Strategy: generate all valid (n, m) for one gluon, then take the
# n-fold Cartesian product and filter by the total Nmax constraint.
# ===================================================================

def _ho_single_candidates(Nmax):
    candidates = []
    max_n = Nmax // 2
    for n in range(max_n + 1):
        max_abs_m = Nmax - 2 * n - 1
        if max_abs_m < 0:
            continue
        for m in range(-max_abs_m, max_abs_m + 1):
            if 2 * n + abs(m) + 1 <= Nmax:
                candidates.append((n, m))
    return candidates


def hmosc_basis(n, Nmax):
    """
    All valid transverse HO combinations for n gluons.
    Returns a list of tuples: ((n1,m1), (n2,m2), ..., (nn,mn))
    """
    single = _ho_single_candidates(Nmax)
    valid = []
    for combo in itertools.product(single, repeat=n):
        total_N = sum(2 * ni + abs(mi) + 1 for ni, mi in combo)
        if total_N <= Nmax:
            valid.append(combo)
    return valid


# ===================================================================
# 5. Sector basis builder
# ===================================================================
# Cross-product of four independent quantum numbers:
#   k-tuples  ×  color IDs  ×  spin tuples  ×  HO combos
# followed by the angular-momentum projection constraint:
#   sum_i (m_i + s_i) = Mj // 2
# ===================================================================
def build_sector_basis(n, K, Nmax, Mj, color_singlet_number):
    """
    Build the Fock basis for an n-gluon sector.

    Returns:
        list of NParticleState
    """
    k_tuples = k_basis(K, n)
    colors   = color_list(color_singlet_number)
    spins    = spin_list(n)
    ho_combos = hmosc_basis(n, Nmax)

    basis = []

    for k_tuple in k_tuples:
        for color_id in colors:
            for s_tuple in spins:
                # target_m = Mj//2 - sum(s_i)
                # so that sum(m_i) + sum(s_i) = Mj//2
                # (WE ARE PERFORMING ANGULAR MOMENTUM CONSERVATION HERE)
                target_m = Mj // 2 - sum(s_tuple)

                for ho_tuple in ho_combos:
                    m_sum = sum(mi for ni, mi in ho_tuple)
                    if m_sum != target_m:
                        continue

                    particles = []
                    for i in range(n):
                        particles.append(SingleParticleState(
                            k=k_tuple[i],
                            s=s_tuple[i],
                            n=ho_tuple[i][0],
                            m=ho_tuple[i][1]
                        ))

                    basis.append(NParticleState(
                        particles=tuple(particles),
                        color_singlet=color_id
                    ))

    return basis


# ===================================================================
# 6. Dump basis to .dat file
# ===================================================================
# Outputs a human-readable listing of all basis states for inspection
# and for cross-checking against Fortran or other codes.
# ===================================================================

def dump_basis_to_dat(basis, filename):
    """
    Write a basis list to a formatted .dat file.

    Parameters:
        basis    : list of NParticleState
        filename : str, output path (default: "basis.dat")
    """
    n_gluons = len(basis[0].particles) if basis else 0

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Basis dump: {len(basis)} states\n")
        f.write(f"# Number of gluons per state: {n_gluons}\n")
        f.write(f"# Columns per particle: k, s, n, m\n")
        f.write(f"# color_singlet: singlet index from color_wf\n")
        f.write("#" + "=" * 60 + "\n")

        for idx, state in enumerate(basis):
            f.write(f"state {idx:6d}  color_singlet={state.color_singlet}\n")
            for i, p in enumerate(state.particles):
                f.write(
                    f"    g{i+1} : k={p.k:3d}  s={p.s:3d}  n={p.n:3d}  m={p.m:4d}\n"
                )
            f.write("\n")

    print(f"Basis dumped to {filename}: {len(basis)} states")

file_dir = Path(__file__).parent


if __name__ == "__main__":
    n = int(input('basis.py test: n = '))
    K = int(input('basis.py test: K = '))
    Nmax = int(input('basis.py test: Nmax = '))
    Mj = int(input('basis.py test: Mj = '))
    color_singlet_number = int(input('basis.py test: color_singlet_number = '))
    basis = build_sector_basis(n, K, Nmax, Mj, color_singlet_number)
    #print(basis)
    dump_basis_to_dat(basis, filename = file_dir/'output'/'basis.py_test'/f'basis_for_n_{n}_K_{K}_Nmax_{Nmax}_Mj_{Mj}_color_singlet_number_{color_singlet_number}')