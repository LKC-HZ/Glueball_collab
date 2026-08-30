"""Profile run_color_wf_py step by step for 5g."""
from color_wf_py_sparse import (
    _build_c2_matrix, _init_generators_GM, _count_singlets,
    _realify_nullspace, _format_output
)
import time
import numpy as np
from scipy.sparse.linalg import eigsh

n, m, k = 0, 0, 5

nsing = _count_singlets(n, m, k)
dims = [8] * k
nstates = int(np.prod(dims))
print(f"k={k}g, nstates={nstates}, nsing={nsing}")

T_q, T_aq, T_g = _init_generators_GM()

t0 = time.perf_counter()
C2 = _build_c2_matrix(n, m, k, T_q, T_aq, T_g)
t1 = time.perf_counter()
print(f"1. Build C2: {t1-t0:.2f}s")

k_eig = min(nsing + 20, nstates - 2)
t0 = time.perf_counter()
eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', tol=1e-7, maxiter=5000)
t1 = time.perf_counter()
print(f"2. eigsh: {t1-t0:.2f}s")

tol = 1.0e-10
null_mask = np.abs(eigvals) < tol
nullity = int(np.sum(null_mask))
print(f"   nullity={nullity}")

nullspace = eigvecs[:, null_mask]

t0 = time.perf_counter()
basis, aeigval = _realify_nullspace(nullspace, n, m, k, dims, nstates)
t1 = time.perf_counter()
print(f"3. realify_nullspace: {t1-t0:.2f}s")

t0 = time.perf_counter()
out = _format_output(basis, n, m, k, dims, nstates, nsing, aeigval)
t1 = time.perf_counter()
print(f"4. format_output: {t1-t0:.2f}s")

print(f"\nTotal singlet count: {basis.shape[1]}")
