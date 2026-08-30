import numpy as np
from functools import lru_cache
from pathlib import Path
from colorwf_runner import run_color_wf, get_singlet_array, get_singlet_coefficients, parse_state_label
from su3constants import SU3Constants

'''
/
'''


su3 = SU3Constants()

# cached wrapper for color_wf calls to avoid repeated Fortran invocations
@lru_cache(maxsize=None)
def _cached_run_color_wf(n, m, k):
    return run_color_wf(n, m, k)


def three_gluon_vertex(): # fabc
    '''
    print("f_abc:", su3.fabc)
    '''
    return 1j * su3.fabc


def four_gluon_vertex(): # 4-gluon vertex

    f = np.sqrt(2) * su3.fabc
    V = (np.einsum('abe,cde->abcd', f, f))
    return V


def quark_gluon_vertex(): # q -> q + g
    return np.stack([su3.T[a] for a in range(8)])


def antiquark_gluon_vertex(): # qb -> qb + g
    return np.stack([-su3.T[a].T for a in range(8)])


def gluon_splitting_vertex(): # g -> q + qb
    return np.stack([su3.T[a] for a in range(8)])


def build_wavefunction(coeffs, labels, n_q, n_qb, n_g):
    """
    Build a numpy wavefunction array from color_wf GM output.
    Shape: (3,)*n_q + (3,)*n_qb + (8,)*n_g
    """
    shape = (3,)*n_q + (3,)*n_qb + (8,)*n_g
    wf = np.zeros(shape, dtype=complex)
    for coeff, label in zip(coeffs, labels):
        q_colors, qb_colors, g_labels = parse_state_label(label, n_q, n_qb, n_g)
        indices = [c-1 for c in q_colors] + [c-1 for c in qb_colors]
        for glbl in g_labels:
            if glbl.startswith('lam'):
                adj_idx = int(glbl[3:]) - 1
            elif glbl.startswith('g_C'):
                adj_idx = int(glbl[3:]) - 1
            else:
                adj_idx = int(''.join(filter(str.isdigit, glbl))) - 1
            indices.append(adj_idx)
        wf[tuple(indices)] = coeff
    return wf


def _ptype(elem):
    """Extract particle type.  Bare int defaults to 'g'."""
    return elem[1] if isinstance(elem, (tuple, list)) else 'g'


def _pid(elem):
    """Extract numeric ID.  Bare int returns itself."""
    return elem[0] if isinstance(elem, (tuple, list)) else elem


def gluon_contraction_by_plan(inc_gl_n, out_gl_n, plan, singlet_in, singlet_out,
                              inc_qrk_n=0, out_qrk_n=0,
                              inc_antiqrk_n=0, out_antiqrk_n=0):
    """
    A general color factor contraction based on a vertex plan.
    Supports mixed quark / antiquark / gluon systems.
    Spectator delta_ab is handled index-free (no explicit eye(8)).
    
    Parameters:
        inc_gl_n, out_gl_n : int
            Number of incoming / outgoing gluons.
        plan : dict
            A single plan dict from allvertices.generate_vertex_plans().
        singlet_in, singlet_out : int
            Singlet indices for color_wf.
        inc_qrk_n, out_qrk_n, inc_antiqrk_n, out_antiqrk_n : int (default 0)
            Number of incoming / outgoing (anti)quarks.
    
    Returns:
        dict : {'plan': plan, 'contraction_result': scalar, ...}
    """

    # --- counts ---
    inc_n_q = inc_qrk_n
    inc_n_qb = inc_antiqrk_n
    inc_n_g = inc_gl_n
    out_n_q = out_qrk_n
    out_n_qb = out_antiqrk_n
    out_n_g = out_gl_n

    # --- wavefunctions ---
    incoming_result = _cached_run_color_wf(inc_n_q, inc_n_qb, inc_n_g)
    gm_coeffs_inc, gm_labels_inc = get_singlet_array(incoming_result, basis="GM", singlet_index=singlet_in)

    outgoing_result = _cached_run_color_wf(out_n_q, out_n_qb, out_n_g)
    gm_coeffs_out, gm_labels_out = get_singlet_array(outgoing_result, basis="GM", singlet_index=singlet_out)

    # Pure gluon: use original hard-coded label parsing
    if inc_n_q == 0 and inc_n_qb == 0:
        incoming_wf = np.zeros((8,) * inc_n_g, dtype=complex)
        for k in range(len(gm_labels_inc)):
            indices = []
            for i in range(inc_n_g):
                pos = 3 + 5 * i
                index = int(gm_labels_inc[k][pos:pos+1])
                indices.append(index - 1)
            incoming_wf[tuple(indices)] = gm_coeffs_inc[k]
    else:
        incoming_wf = build_wavefunction(gm_coeffs_inc, gm_labels_inc, inc_n_q, inc_n_qb, inc_n_g)

    if out_n_q == 0 and out_n_qb == 0:
        outgoing_wf = np.zeros((8,) * out_n_g, dtype=complex)
        for k in range(len(gm_labels_out)):
            indices = []
            for i in range(out_n_g):
                pos = 3 + 5 * i
                index = int(gm_labels_out[k][pos:pos+1])
                indices.append(index - 1)
            outgoing_wf[tuple(indices)] = gm_coeffs_out[k]
    else:
        outgoing_wf = build_wavefunction(gm_coeffs_out, gm_labels_out, out_n_q, out_n_qb, out_n_g)

    # --- delta-free spectator map ---
    spectator_map = {}
    for v in plan['vertices']:
        if v['type'] == 'spectator':
            spectator_map[v['outgoing'][0]] = v['incoming'][0]

    # --- einsum index mapping ---
    def idx(elem, incoming):
        pid = _pid(elem)
        if incoming:
            return pid
        else:
            if elem in spectator_map:
                return _pid(spectator_map[elem])
            return 25 + pid

    # --- operands list for np.einsum ---
    operands = []

    # incoming wavefunction: quarks -> antiquarks -> gluons
    inc_idx = []
    for i in range(1, inc_n_q + 1): inc_idx.append(idx((i, 'q'), True))
    for i in range(1, inc_n_qb + 1): inc_idx.append(idx((inc_n_q + i, 'qb'), True))
    for i in range(1, inc_n_g + 1): inc_idx.append(idx((inc_n_q + inc_n_qb + i, 'g'), True))
    operands.extend([incoming_wf, inc_idx])

    # outgoing wavefunction
    out_idx = []
    for i in range(1, out_n_q + 1): out_idx.append(idx((i, 'q'), False))
    for i in range(1, out_n_qb + 1): out_idx.append(idx((out_n_q + i, 'qb'), False))
    for i in range(1, out_n_g + 1): out_idx.append(idx((out_n_q + out_n_qb + i, 'g'), False))
    operands.extend([outgoing_wf, out_idx])

    # vertices
    for v in plan['vertices']:
        inc_ids = v['incoming']
        out_ids = v['outgoing']
        vtype = v['type']

        if vtype == 'spectator':
            continue  # delta-free: indices already merged via spectator_map

        # Build index list: incoming first, then outgoing
        v_idx = []
        for elem in inc_ids:
            v_idx.append(idx(elem, True))
        for elem in out_ids:
            v_idx.append(idx(elem, False))

        if vtype == '3-gluon':
            v_tensor = three_gluon_vertex()
        elif vtype == '4-gluon':
            v_tensor = four_gluon_vertex()
        elif vtype == 'quark-gluon':
            v_tensor = quark_gluon_vertex()
            # inc=[q], out=[q, g]; T^a shape (8,3,3) = (adj, out, in)
            v_idx = [v_idx[2], v_idx[1], v_idx[0]]
        elif vtype == 'antiquark-gluon':
            v_tensor = antiquark_gluon_vertex()
            # inc=[qb], out=[qb, g]; same reordering
            v_idx = [v_idx[2], v_idx[1], v_idx[0]]
        elif vtype == 'gluon-splitting':
            v_tensor = gluon_splitting_vertex()
            # inc=[g], out=[q, qb]; T^a shape (8,3,3) = (adj, q, qb)
            v_tensor = three_gluon_vertex()
        elif vtype == '4-gluon vertex':
            v_tensor = four_gluon_vertex()
        elif vtype == 'quark-gluon vertex':
            v_tensor = quark_gluon_vertex()
            # inc=[q], out=[q, g]; T^a shape (8,3,3) = (adj, out, in)
            v_idx = [v_idx[2], v_idx[1], v_idx[0]]
        elif vtype == 'antiquark-gluon vertex':
            v_tensor = antiquark_gluon_vertex()
            # inc=[qb], out=[qb, g]; same reordering
            v_idx = [v_idx[2], v_idx[1], v_idx[0]]
        elif vtype == 'gluon-splitting vertex':
            v_tensor = gluon_splitting_vertex()
            # inc=[g], out=[q, qb]; T^a shape (8,3,3) = (adj, q, qb)
        else:
            raise ValueError(f"Unknown vertex type: {vtype}")

        operands.extend([v_tensor, v_idx])

    # Result is a scalar
    operands.append([])
    result = np.einsum(*operands, optimize='greedy')

    def _get_singlet_type(result_obj, singlet_index):
        for s in result_obj.gm_singlets:
            if s.index == singlet_index:
                return s.state_type
        return "unknown"

    type_in = _get_singlet_type(incoming_result, singlet_in)
    type_out = _get_singlet_type(outgoing_result, singlet_out)

    return {
        'plan': plan,
        'singlet_in_index': singlet_in,
        'singlet_out_index': singlet_out,
        'singlet_in_type': type_in,
        'singlet_out_type': type_out,
        'contraction_result': result
    }

'''
result = run_color_wf(n=0, m=0, k=2)
gm_coeffs, gm_labels = get_singlet_array(result, basis="GM", singlet_index=1)
print(gm_coeffs)
print(gm_labels)
num = int(gm_labels[0][3:4])
print(num)
'''
