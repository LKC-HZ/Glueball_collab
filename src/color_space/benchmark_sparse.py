"""Benchmark C2 build time and eigsh convergence for different k."""
from color_wf_py_sparse import _build_c2_matrix, _init_generators_GM, _count_singlets
import time
import numpy as np

T_q, T_aq, T_g = _init_generators_GM()

for k in [2, 3, 4, 5]:
    nstates = 8 ** k
    nsing = _count_singlets(0, 0, k)
    print(f"\n=== k={k}g: nstates={nstates}, nsing={nsing} ===")
    
    t0 = time.perf_counter()
    C2 = _build_c2_matrix(0, 0, k, T_q, T_aq, T_g)
    t1 = time.perf_counter()
    print(f"  C2 build time: {t1-t0:.2f}s")
    print(f"  C2 nnz: {C2.nnz}, shape: {C2.shape}")
    print(f"  C2 memory: {C2.data.nbytes / 1e6:.1f} MB")
    
    from scipy.sparse.linalg import eigsh
    k_eig = min(nsing + 20, nstates - 2)
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', tol=1e-7, maxiter=5000)
        t1 = time.perf_counter()
        print(f"  eigsh time: {t1-t0:.2f}s")
        print(f"  Smallest eigvals: {eigvals[:5]}")
        null_mask = np.abs(eigvals) < 1e-10
        print(f"  Nullity found: {int(np.sum(null_mask))}")
    except Exception as e:
        t1 = time.perf_counter()
        print(f"  eigsh FAILED after {t1-t0:.2f}s: {e}")
