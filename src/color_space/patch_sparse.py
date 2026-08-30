import re

with open('color_wf_py.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. Add scipy imports
old_import = "import numpy as np\nimport sys\nfrom typing import List, Tuple, Optional\nfrom collections import defaultdict"
new_import = old_import + "\nfrom scipy import sparse\nfrom scipy.sparse.linalg import eigsh"
src = src.replace(old_import, new_import)

# 2. Replace _build_c2_matrix function
old_func = '''def _build_c2_matrix(n: int, m: int, k: int,
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

    return C2'''

new_func = '''def _build_c2_matrix(n: int, m: int, k: int,
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
    return C2'''

src = src.replace(old_func, new_func)

# 3. Replace diagonalization in run_color_wf_py
old_diag = '''    # Build C2 matrix and diagonalize
    C2 = _build_c2_matrix(n, m, k, T_q, T_aq, T_g)
    eigvals, eigvecs = np.linalg.eigh(C2)

    # Extract nullspace (zero eigenvalues)
    tol = 1.0e-10
    null_mask = np.abs(eigvals) < tol
    nullity = int(np.sum(null_mask))

    if nullity == 0:
        return f" Theoretical singlets:  {nsing}\\nNo singlet states\\n"

    if nullity != nsing:
        pass  # Could raise warning

    nullspace = eigvecs[:, null_mask]

    # Realification pipeline
    basis, aeigval = _realify_nullspace(nullspace, n, m, k, dims, nstates)

    # Format output
    return _format_output(basis, n, m, k, dims, nstates, nsing, aeigval)'''

new_diag = '''    # Build C2 matrix (sparse) and diagonalize
    C2 = _build_c2_matrix(n, m, k, T_q, T_aq, T_g)

    # Find smallest eigenvalues using sparse eigensolver
    k_eig = min(nsing + 20, nstates - 2)
    if k_eig < 1:
        k_eig = 1

    try:
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM')
    except Exception:
        # Fallback for singular matrices: use a tiny positive sigma
        eigvals, eigvecs = eigsh(C2, k=k_eig, which='SM', sigma=1e-10)

    # Extract nullspace (zero eigenvalues)
    tol = 1.0e-10
    null_mask = np.abs(eigvals) < tol
    nullity = int(np.sum(null_mask))

    if nullity == 0:
        return f" Theoretical singlets:  {nsing}\\nNo singlet states\\n"

    if nullity != nsing:
        pass  # Could raise warning

    nullspace = eigvecs[:, null_mask]

    # Realification pipeline
    basis, aeigval = _realify_nullspace(nullspace, n, m, k, dims, nstates)

    # Format output
    return _format_output(basis, n, m, k, dims, nstates, nsing, aeigval)'''

src = src.replace(old_diag, new_diag)

with open('color_wf_py_sparse.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("color_wf_py_sparse.py written successfully")
