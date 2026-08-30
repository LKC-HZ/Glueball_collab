# main

import numpy as np
import allvertices as allvertices
import gluon_to_gluon_color_factor as gluon_to_gluon_color_factor
from color_wf_py import run_color_wf_py
from colorwf_runner import ColorWfResult, parse_color_wf_output
from pathlib import Path
import time

import sys
import os


start = time.perf_counter()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

script_dir = Path(__file__).parent



def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value < 0:
                print("Please enter a non-negative integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

# incoming / outgoing particle counts
inc_qrk_n = get_positive_int("# of incoming quarks: ")
out_qrk_n = get_positive_int("# of outgoing quarks: ")
inc_antiqrk_n = get_positive_int("# of incoming anti-quarks: ")
out_antiqrk_n = get_positive_int("# of outgoing anti-quarks: ")
inc_gl_n = get_positive_int("# of incoming gluons: ")
out_gl_n = get_positive_int("# of outgoing gluons: ")


# color indices for each particle type
inc_qrk_cl_index = np.zeros((inc_qrk_n, 3), dtype = np.int8)
out_qrk_cl_index = np.zeros((out_qrk_n, 3), dtype = np.int8)
inc_antiqrk_cl_index = np.zeros((inc_antiqrk_n, 3), dtype = np.int8)
out_antiqrk_cl_index = np.zeros((out_antiqrk_n, 3), dtype = np.int8)
inc_gl_cl_index = np.zeros((inc_gl_n, 8), dtype = np.int8)
out_gl_cl_index = np.zeros((out_gl_n, 8), dtype = np.int8)
for i in range (8):
    inc_gl_cl_index[:, i] = i + 1
    out_gl_cl_index[:, i] = i + 1

# --- color singlets from pure-Python color_wf_py ---
inc_stdout = run_color_wf_py(inc_qrk_n, inc_antiqrk_n, inc_gl_n)
inc_result = ColorWfResult(n=inc_qrk_n, m=inc_antiqrk_n, k=inc_gl_n)
parse_color_wf_output(inc_stdout, inc_result)
inc_num_singlets = inc_result.actual_singlets

out_stdout = run_color_wf_py(out_qrk_n, out_antiqrk_n, out_gl_n)
out_result = ColorWfResult(n=out_qrk_n, m=out_antiqrk_n, k=out_gl_n)
parse_color_wf_output(out_stdout, out_result)
out_num_singlets = out_result.actual_singlets


# --- build tagged particle lists ---
inc_particles = []
for i in range(1, inc_qrk_n + 1): inc_particles.append((i, 'q'))
for i in range(1, inc_antiqrk_n + 1): inc_particles.append((inc_qrk_n + i, 'qb'))
for i in range(1, inc_gl_n + 1): inc_particles.append((inc_qrk_n + inc_antiqrk_n + i, 'g'))

out_particles = []
for i in range(1, out_qrk_n + 1): out_particles.append((i, 'q'))
for i in range(1, out_antiqrk_n + 1): out_particles.append((out_qrk_n + i, 'qb'))
for i in range(1, out_gl_n + 1): out_particles.append((out_qrk_n + out_antiqrk_n + i, 'g'))

inc_total = inc_qrk_n + inc_antiqrk_n + inc_gl_n
out_total = out_qrk_n + out_antiqrk_n + out_gl_n













#  GENERATE ALL VERTEX CONFIGURATIONS
print(f"\n Generating all valid vertex configurations for {inc_total} incoming and {out_total} outgoing particles...\n")
vtx_plans = allvertices.generate_vertex_plans(inc_particles, out_particles)



print(f"\n ====================")
print(f"  Incoming quarks: {inc_qrk_n}")
print(f"  Incoming anti-quarks: {inc_antiqrk_n}")
print(f"  Incoming gluons: {inc_gl_n}")
print(f"  Outgoing quarks: {out_qrk_n}")
print(f"  Outgoing anti-quarks: {out_antiqrk_n}")
print(f"  Outgoing gluons: {out_gl_n}")
print(f"  Valid vertex configurations found: {len(vtx_plans)}")
print(f"==================== \n")



filename = script_dir / 'output' / f'{inc_qrk_n}q_{inc_antiqrk_n}qb_{inc_gl_n}g_to_{out_qrk_n}q_{out_antiqrk_n}qb_{out_gl_n}g_Color_Factor.txt'

# CALCULATING COLOR FACTOR 

with open(filename, 'w', encoding='utf-8') as f:
    for s_in in range(1, inc_num_singlets + 1):
        for s_out in range(1, out_num_singlets + 1):
            for i in range(len(vtx_plans)):
                cf = gluon_to_gluon_color_factor.gluon_contraction_by_plan(
                    inc_gl_n, out_gl_n, vtx_plans[i], s_in, s_out,
                    inc_qrk_n, out_qrk_n, inc_antiqrk_n, out_antiqrk_n
                )
                print(f"\n -------------------- \n COLOR FACTOR: {cf['contraction_result']}; "
                  f"FEYNMAN DIAGRAM CONFIGURATION: {cf['plan']}; "
                  f"INCOMING SINGLET ID: {cf['singlet_in_index']}({cf['singlet_in_type']}); "
                  f"OUTGOING SINGLET ID: {cf['singlet_out_index']}({cf['singlet_out_type']})\n", file=f)


# FOR HAMILTONIAN CALCULATION: SAVE TO .dat
# Vertex type encoding: 0=spectator, 3=3-gluon, 4=4-gluon,
#   5=quark-gluon, 6=antiquark-gluon, 7=gluon-splitting
# Format: cf_re cf_im s_in s_out plan_id num_vertices [vtype num_in in1... num_out out1...]...
dat_filename = script_dir / 'output' / f'{inc_qrk_n}q_{inc_antiqrk_n}qb_{inc_gl_n}g_to_{out_qrk_n}q_{out_antiqrk_n}qb_{out_gl_n}g_Color_Factor.dat'
CF_TOL = 1e-14
VTYPE_CODE = {'spectator': 0, '3-gluon': 3, '4-gluon': 4,
              'quark-gluon': 5, 'antiquark-gluon': 6, 'gluon-splitting': 7}

def _id_str(elem):
    return str(elem[0]) if isinstance(elem, (tuple, list)) else str(elem)

with open(dat_filename, 'w', encoding='utf-8') as f_dat:
    for s_in in range(1, inc_num_singlets + 1):
        for s_out in range(1, out_num_singlets + 1):
            for i in range(len(vtx_plans)):
                cf = gluon_to_gluon_color_factor.gluon_contraction_by_plan(
                    inc_gl_n, out_gl_n, vtx_plans[i], s_in, s_out,
                    inc_qrk_n, out_qrk_n, inc_antiqrk_n, out_antiqrk_n
                )
                cf_val = cf['contraction_result']
                if abs(cf_val) > CF_TOL:
                    plan = vtx_plans[i]
                    parts = []
                    parts.append(f"{cf_val.real:.12f}")
                    parts.append(f"{cf_val.imag:.12f}")
                    parts.append(str(s_in))
                    parts.append(str(s_out))
                    parts.append(str(plan['id']))
                    parts.append(str(len(plan['vertices'])))
                    for v in plan['vertices']:
                        parts.append(str(VTYPE_CODE[v['type']]))
                        parts.append(str(len(v['incoming'])))
                        parts.extend(_id_str(x) for x in v['incoming'])
                        parts.append(str(len(v['outgoing'])))
                        parts.extend(_id_str(x) for x in v['outgoing'])
                    f_dat.write(' '.join(parts) + '\n')

print(f"non-zero CF's saved to: {dat_filename}")
end = time.perf_counter()
print(f"Wall time: {end - start:.4f} seconds")
