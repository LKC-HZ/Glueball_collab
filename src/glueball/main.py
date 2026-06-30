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

    gggg_basis = build_sector_basis(4, K, Nmax, Mj_s, gggg_color_num)


    sector_basis = {
        "g": g_basis,
        "gg": gg_basis,
        "ggg": ggg_basis,
        "gggg": gggg_basis
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










































'''
def renorm(Nmax, kt, b, coupling, p_plus, loop_max=30, tol=1e-10, max_retry=5):
    """
    若单次对角化/迭代出现 nan，最多重试 max_retry 次（与 scan 原先逻辑一致）。
    """

    def _once():
        # -------------------------
        # 初始构造
        # -------------------------
        params = PhysicsParams(
            couplings=coupling,
            b=b,
            mass_g=0.0,
            mass_gg=0.0,
            p_plus=p_plus,
        )

        H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)

        vals, vecs = eigsh(H, k=3, which='SA')
        eigenv1 = vals[0] * kt

        renomass2 = np.sqrt(-eigenv1)
        inputmass = renomass2

        params = PhysicsParams(
            couplings=coupling,
            b=b,
            mass_g=inputmass,
            mass_gg=inputmass,
            p_plus=p_plus,
        )

        H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)
        vals, vecs = eigsh(H, k=3, which='SA')

        eigenv2 = vals[0] * kt

        renomass1 = 0.0

        for _ in range(loop_max):

            if abs(eigenv2) < tol:
                break

            renomass3 = np.sqrt(
                renomass1**2 +
                (renomass2**2 - renomass1**2) / (eigenv2 - eigenv1) * (-eigenv1)
            )

            inputmass = renomass3

            params = PhysicsParams(
                couplings=coupling,
                b=b,
                mass_g=inputmass,
                mass_gg=inputmass,
                p_plus=p_plus,
            )

            H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)
            vals, vecs = eigsh(H, k=3, which='SA')

            eigenv1 = eigenv2
            eigenv2 = vals[0] * kt

            renomass1 = renomass2
            renomass2 = renomass3

        return renomass2
'''
        









































def scan_and_plot(
    coupling_range,
    b_range,
    Nmax=4,
    kt=5,
    savefile="scan.dat",
    plotfile=None,
    p_plus=None,
):
    """
    扫描 renorm 并画：
        横轴 = coupling
        纵轴 = renorm mass
        不同 b = 不同折线
    """

    coupling_min, coupling_max, coupling_step = coupling_range
    b_min, b_max, b_step = b_range

    couplings = np.arange(
        coupling_min,
        coupling_max + coupling_step,
        coupling_step
    )

    bs = np.arange(
        b_min,
        b_max + b_step,
        b_step
    )

    result = np.zeros((len(bs), len(couplings)))

    if p_plus is None:
        p_plus = float(kt)

    # =========================================
    # 扫描
    # =========================================
    with open(savefile, "w") as f:

        for i, b in enumerate(bs):
            for j, coupling in enumerate(couplings):

                value = renorm(
                    Nmax=Nmax,
                    kt=kt,
                    b=b,
                    coupling=coupling,
                    p_plus=p_plus,
                )

                result[i, j] = value

                f.write(
                    f"{b:15.8f} "
                    f"{coupling:15.8f} "
                    f"{value:20.10e}\n"
                )

                print(
                    f"b={b:.3f}, "
                    f"coupling={coupling:.3f}, "
                    f"renorm={value:.6e}"
                )

    # =========================================
    # 画图
    # =========================================
    # plt.figure(figsize=(8, 6))

    # for i, b in enumerate(bs):

    #     plt.plot(
    #         couplings,
    #         result[i],
    #         marker='o',
    #         linewidth=2,
    #         markersize=5,
    #         label=fr"$b={b:.2f}$"
    #     )

    # plt.xlabel(r"Coupling_g", fontsize=14)
    # plt.ylabel(r"Smallest deltam", fontsize=14)

    # plt.title(
    #     fr"$N_{{\max}}={Nmax},\ K_{{\max}}={kt}$",
    #     fontsize=15
    # )

    # plt.grid(alpha=0.3)

    # plt.legend(
    #     fontsize=10,
    #     frameon=False,
    #     ncol=2
    # )

    # plt.tight_layout()

    # if plotfile is not None:
    #     plt.savefig(plotfile, dpi=300)

    # plt.show()

    plt.figure(figsize=(8, 6))

    for j, g in enumerate(couplings):

        plt.plot(
            bs,
            result[:, j],   # 注意这里转置
            marker='o',
            linewidth=2,
            markersize=5,
            label=fr"$g={g:.2f}$"
        )

    plt.xlabel(r"$b$", fontsize=14)
    plt.ylabel(r"Smallest deltam", fontsize=14)

    plt.title(
        fr"$n={Nmax},\ k={kt}$",
        fontsize=15
    )

    plt.grid(alpha=0.3)

    plt.legend(
        fontsize=10,
        frameon=False,
        ncol=2
    )

    plt.tight_layout()

    if plotfile is not None:
        plt.savefig(plotfile, dpi=300)

    plt.show()

    return bs, couplings, result











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
    
#     scan_and_plot(
#     coupling_range=(0.5, 20.0, 2.0),
#     b_range=(0.1, 3.0, 0.4),
#     savefile="Output/scan.dat",
#     plotfile="Output/scan.png"
# )

    

