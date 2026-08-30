"""Profile _realify_nullspace sub-steps for 5g."""
from color_wf_py_sparse import (
    _build_c2_matrix, _init_generators_GM, _count_singlets,
    _realify_nullspace, _apply_M_full, _swap_overlap,
    _decode, _encode, _build_stride
)
import time
import numpy as np
from scipy.sparse.linalg import eigsh

n, m, k = 0, 0, 5

nsing = _count_singlets(n, m, k)
dims = [8] * k
nstates = int(np.prod(dims))

T_q, T_aq, T_g = _init_generators_GM()
C2 = _build_c2_matrix(n, m, k, T_q, T_aq, T_g)
k_eig = min(nsing + 20, nstates - 2)
eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', tol=1e-7, maxiter=5000)
tol = 1.0e-10
null_mask = np.abs(eigvals) < tol
nullspace = eigvecs[:, null_mask]
nullity = nullspace.shape[1]
print(f"k={k}g, nullity={nullity}, nstates={nstates}")

# --- Profile _apply_M_full ---
t0 = time.perf_counter()
for _ in range(nullity):
    _apply_M_full(nullspace[:, 0], n, m, k, dims, nstates)
t1 = time.perf_counter()
print(f"_apply_M_full x{nullity}: {t1-t0:.2f}s (per call: {(t1-t0)/nullity*1000:.1f}ms)")

# --- Profile _swap_overlap ---
t0 = time.perf_counter()
count = 0
for i in range(min(nullity, 5)):
    for j in range(min(nullity, 5)):
        _swap_overlap(nullspace[:, i], nullspace[:, j], 0, 1, dims, nstates)
        count += 1
t1 = time.perf_counter()
print(f"_swap_overlap x{count}: {t1-t0:.2f}s (per call: {(t1-t0)/count*1000:.1f}ms)")

# --- Profile decode/encode ---
t0 = time.perf_counter()
stride = _build_stride(dims)
for idx in range(nstates):
    colors = _decode(idx, dims)
    idx2 = _encode(colors, dims)
t1 = time.perf_counter()
print(f"decode+encode x{nstates}: {t1-t0:.2f}s (per call: {(t1-t0)/nstates*1e6:.1f}us)")

# Extrapolation
print(f"\n--- Extrapolation ---")
for kk in [5, 6, 7, 8, 10, 15, 20]:
    ns = 8 ** kk
    # _realify_nullspace has ~nullity^2 * 2 rounds of _swap_overlap + nullity _apply_M_full
    # Assume nullity grows roughly as 2^k or similar
    nsing_k = _count_singlets(0, 0, kk)
    # Time scales as nullity^2 * nstates * per_state_time
    # From 5g: per_state_time for _swap_overlap ~ 0.6us, _apply_M_full ~ 0.6us
    est_swap = (nsing_k ** 2 * 2) * ns * 0.6e-6
    est_m = nsing_k * ns * 0.6e-6
    est_total = est_swap + est_m
    print(f"k={kk}g: nstates={ns:.2e}, nsing={nsing_k}, est_realify={est_total:.1f}s")
