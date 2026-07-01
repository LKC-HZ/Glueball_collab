from dataclasses import dataclass
from math import sqrt, isnan
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import os
from gCF.colorwf_runner import run_color_wf
from basis import *
from hami import *
from Ctool import *

# =========================================================
# 1. Physics parameters 
# =========================================================

@dataclass(frozen=True)
class PhysicsParams:
    couplings: float
    b: float
    mass_g: float
    mass_gg: float
    # 与纵向截断 Kmax 对齐的总动量 P+（hami 中 bee 等用）；与 __main__ 里 Kmax 同一数值即可
    p_plus: float

def build_hamiltonian(Nmax, K, params: PhysicsParams):
    Mj_s = 2 #
    # Num. of color singlet
    g_color_num = 1

    gg_result = run_color_wf(0, 0, 2)
    gg_color_num = gg_result.actual_singlets

    ggg_result = run_color_wf(0, 0, 3)
    ggg_color_num = ggg_result.actual_singlets

    gggg_result = run_color_wf(0, 0, 4)
    gggg_color_num = gggg_result.actual_singlets

    g_basis = build_sector_basis(1, K, Nmax, Mj_s, g_color_num)

    gg_basis = build_sector_basis(2, K, Nmax, Mj_s, gg_color_num)

    ggg_basis = build_sector_basis(3, K, Nmax, Mj_s, ggg_color_num)

    #gggg_basis = build_sector_basis(4, K, Nmax, Mj_s, gggg_color_num)


    sector_basis = {
        "g": g_basis,
        "gg": gg_basis,
        "ggg": ggg_basis
    }
    # -------------------------
    # global basis
    # -------------------------
    global_basis, index_map = build_global_basis(sector_basis)
    # -------------------------
    # Hamiltonian
    # -------------------------
    H_dict, dim = build_sparse_hamiltonian(global_basis, params)
    H = to_sparse_matrix(H_dict, dim)
    # H=0

    return H

def eig(Nmax, kt, b, coupling, p_plus):
    params = PhysicsParams(
            couplings=coupling,
            b=b,
            mass_g=0.0,
            mass_gg=0.0,
            p_plus=p_plus,
        )

    H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)

    vals, vecs = eigsh(H, k=3, which='SA')

    return vals[0] * kt

if __name__ == "__main__":
    #print(os.getcwd())
    
    Nmax = 6
    Kmax = 7
    p_plus = float(Kmax)
    b = 1.0
    coupling = 1.0
    output_file = "D:/Star's Personal File Repository (INSP 16)/Work/Research/3_Physics Research/2026 Glueball/Program/allspaces/newblfq orig/newblfq/Output/renom.dat"

    with open(output_file, "w") as f:

        for k in range(2, Kmax + 1):
            for n in range(3, Nmax + 1):

                mass = eig(
                    Nmax=n,
                    kt=k,
                    b=b,
                    coupling=coupling,
                    p_plus=p_plus,
                )

                # 只写三列：n k value
                f.write(f"{n:6d} {k:6d} {mass:.10e}\n")
    