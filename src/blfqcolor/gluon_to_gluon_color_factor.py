import numpy as np
from functools import lru_cache
from pathlib import Path
from color_wf_py import run_color_wf_py
from colorwf_runner import ColorWfResult, parse_color_wf_output, get_singlet_array, get_singlet_coefficients, parse_state_label
from su3constants import SU3Constants


su3 = SU3Constants()

# cached wrapper for color_wf calls — now uses pure Python backend
@lru_cache(maxsize=None)
def _cached_run_color_wf(n, m, k):
    stdout = run_color_wf_py(n, m, k)
    result = ColorWfResult(n=n, m=m, k=k)
    parse_color_wf_output(stdout, result)
    return result


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
                # fallback: extract trailing digits
                adj_idx = int(''.join(filter(str.isdigit, glbl))) - 1
            indices.append(adj_idx)
        wf[tuple(indices)] = coeff
    return wf


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

    # --- type maps: ID -> 'q' / 'qb' / 'g' ---
    inc_types = {}
    for i in range(1, inc_n_q + 1): inc_types[i] = 'q'
    for i in range(1, inc_n_qb + 1): inc_types[inc_n_q + i] = 'qb'
    for i in range(1, inc_n_g + 1): inc_types[inc_n_q + inc_n_qb + i] = 'g'

    out_types = {}
    for i in range(1, out_n_q + 1): out_types[i] = 'q'
    for i in range(1, out_n_qb + 1): out_types[out_n_q + i] = 'qb'
    for i in range(1, out_n_g + 1): out_types[out_n_q + out_n_qb + i] = 'g'

    # --- wavefunctions ---
    incoming_result = _cached_run_color_wf(inc_n_q, inc_n_qb, inc_n_g)
    gm_coeffs_inc, gm_labels_inc = get_singlet_array(incoming_result, basis="GM", singlet_index=singlet_in)
    incoming_wf = build_wavefunction(gm_coeffs_inc, gm_labels_inc, inc_n_q, inc_n_qb, inc_n_g)

    outgoing_result = _cached_run_color_wf(out_n_q, out_n_qb, out_n_g)
    gm_coeffs_out, gm_labels_out = get_singlet_array(outgoing_result, basis="GM", singlet_index=singlet_out)
    outgoing_wf = build_wavefunction(gm_coeffs_out, gm_labels_out, out_n_q, out_n_qb, out_n_g)

    # --- delta-free spectator map ---
    # Helper: extract integer ID from (id, type) tuple or plain int
    def _pid_int(pid):
        return pid[0] if isinstance(pid, (tuple, list)) else pid

    # Keys/values are integer IDs only (ignore type tags from plan)
    spectator_map = {}
    for v in plan['vertices']:
        if v['type'] == 'spectator':
            out_id = _pid_int(v['outgoing'][0])
            inc_id = _pid_int(v['incoming'][0])
            spectator_map[out_id] = inc_id

    # --- einsum index mapping ---
    # pid = particle ID (may be (id, type) tuple); incoming = True/False
    def idx(pid, incoming):
        pid_int = _pid_int(pid)
        if incoming:
            return pid_int
        else:
            if pid_int in spectator_map:
                return spectator_map[pid_int]
            return 25 + pid_int
    # --- operands list for np.einsum ---
    operands = []

    # incoming wavefunction: quarks -> antiquarks -> gluons
    inc_idx = []
    for i in range(1, inc_n_q + 1): inc_idx.append(idx(i, True))
    for i in range(1, inc_n_qb + 1): inc_idx.append(idx(inc_n_q + i, True))
    for i in range(1, inc_n_g + 1): inc_idx.append(idx(inc_n_q + inc_n_qb + i, True))
    operands.extend([incoming_wf, inc_idx])

    # outgoing wavefunction
    out_idx = []
    for i in range(1, out_n_q + 1): out_idx.append(idx(i, False))
    for i in range(1, out_n_qb + 1): out_idx.append(idx(out_n_q + i, False))
    for i in range(1, out_n_g + 1): out_idx.append(idx(out_n_q + out_n_qb + i, False))
    operands.extend([np.conj(outgoing_wf), out_idx])

    # vertices
    for v in plan['vertices']:
        inc_ids = v['incoming']
        out_ids = v['outgoing']
        vtype = v['type']

        if vtype == 'spectator':
            continue  # delta-free: indices already merged via spectator_map

        # Build index list: incoming， outgoing
        v_idx = []
        for pid in inc_ids:
            v_idx.append(idx(pid, True))
        for pid in out_ids:
            v_idx.append(idx(pid, False))

        if vtype == '3-gluon':
            v_tensor = three_gluon_vertex()
        elif vtype == '4-gluon':
            v_tensor = four_gluon_vertex()
        elif vtype == 'quark-gluon':
            v_tensor = quark_gluon_vertex()
            v_idx = [v_idx[2], v_idx[1], v_idx[0]]
        elif vtype == 'antiquark-gluon':
            v_tensor = antiquark_gluon_vertex()
            v_idx = [v_idx[2], v_idx[1], v_idx[0]]
        elif vtype == 'gluon-splitting':
            v_tensor = gluon_splitting_vertex()
        else:
            raise ValueError(f"Unknown vertex type: {vtype}")
        operands.extend([v_tensor, v_idx])

    # Result is a scalar
    operands.append([])
    # Execute contraction
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
result = run_color_wf_py(n=0, m=0, k=2)
gm_coeffs, gm_labels = get_singlet_array(result, basis="GM", singlet_index=1)
print(gm_coeffs)
print(gm_labels)
num = int(gm_labels[0][3:4])
print(num)
'''
