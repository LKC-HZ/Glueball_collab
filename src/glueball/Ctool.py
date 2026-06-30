from scipy.sparse import coo_matrix


def to_sparse_matrix(H_dict, dim):

    rows = []
    cols = []
    data = []

    for (i, j), val in H_dict.items():
        rows.append(i)
        cols.append(j)
        data.append(val)

    return coo_matrix((data, (rows, cols)), shape=(dim, dim))




def dump_hamiltonian_dense(H, filename="hamiltonian.dat"):
    """
    将稀疏/稠密 Hamiltonian 输出为完整方阵 dat 文件
    """

    # 转成 dense（关键一步）
    H_dense = H.toarray()

    dim = H_dense.shape[0]

    with open(filename, "w") as f:
        # 写维度
        f.write(f"# dimension = {dim}\n")

        for i in range(dim):
            for j in range(dim):
                val = H_dense[i, j]
                # 写成科学计数法，方便 Fortran / Python 读
                f.write(f"{i:6d} {j:6d} {val.real:20.12e} \n")

    print(f"saved to {filename}")


def dump_gg_basis(
    basis,
    filename="Output/gg_basis.dat"
):
    """
    输出 gg basis 到 dat 文件
    """

    with open(filename, "w") as f:

        f.write(
            "# idx  color   "
            "k1 k2 s1 s2    "
            "n1 m1 n2 m2\n"
        )

        for idx, state in enumerate(basis):

            g1, g2 = state.particles

            f.write(
                f"{idx:6d} "
                f"{state.color_state:6d}   "
                f"{g1.k:3d} {g2.k:3d} {g1.s:3d} {g2.s:4d}   "
                f"{g1.n:3d} {g1.m:3d} {g2.n:3d} {g2.m:4d}\n"
            )

    print(f"gg basis saved to {filename}")
    
