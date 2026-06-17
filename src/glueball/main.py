# main

import numpy as np
import allvertices
import gluon_to_gluon_color_factor
from colorwf_runner import run_color_wf
import colorwf_runner
from pathlib import Path

script_dir = Path(__file__).parent



#  GET # OF INCOMING/OUTGOING GLUONS

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

# incoming / outgoing gluon num.
inc_gl_n = get_positive_int("# of incoming gluons: ")
out_gl_n = get_positive_int("# of outgoing gluons: ")


# color indices for each of the gluons
inc_gl_cl_index = np.zeros((inc_gl_n, 8), dtype = np.int8)
out_gl_cl_index = np.zeros((out_gl_n, 8), dtype = np.int8)
for i in range (8):
    inc_gl_cl_index[:, i] = i + 1
    out_gl_cl_index[:, i] = i + 1

inc_result = colorwf_runner.run_color_wf(0, 0, inc_gl_n)
inc_num_singlets = inc_result.actual_singlets

out_result = colorwf_runner.run_color_wf(0, 0, out_gl_n)
out_num_singlets = out_result.actual_singlets













#  GENERATE ALL VERTEX CONFIGURATIONS
print(f"\n Generating all valid vertex configurations for {inc_gl_n} incoming and {out_gl_n} outgoing gluons...\n")
vtx_plans = allvertices.generate_vertex_plans(inc_gl_n, out_gl_n)




print(f"\n ====================")
print(f"  Incoming gluons: {inc_gl_n}")
print(f"  Outgoing gluons: {out_gl_n}")
print(f"  Valid vertex configurations found: {len(vtx_plans)}")
print(f"==================== \n")



filename = script_dir / f'{inc_gl_n}_gluon_to_{out_gl_n}_gluon_Color_Factor.txt'
# CALCULATING COLOR FACTOR 
with open(filename, 'w', encoding='utf-8') as f:
    for s_in in range(1, inc_num_singlets + 1):
        for s_out in range(1, out_num_singlets + 1):
            for i in range(len(vtx_plans)):
                cf = gluon_to_gluon_color_factor.gluon_contraction_by_plan(inc_gl_n, out_gl_n, vtx_plans[i], s_in, s_out)
                '''
                print(f"\n -------------------- \n COLOR FACTOR: {cf['contraction_result']}; "
                    f"FEYNMAN DIAGRAM CONFIGURATION: {cf['plan']}; "
                    f"INCOMING SINGLET ID: {cf['singlet_in_index']}({cf['singlet_in_type']}); "
                    f"OUTGOING SINGLET ID: {cf['singlet_out_index']}({cf['singlet_out_type']})\n")
                '''
                print(f"\n -------------------- \n COLOR FACTOR: {cf['contraction_result']}; "
                  f"FEYNMAN DIAGRAM CONFIGURATION: {cf['plan']}; "
                  f"INCOMING SINGLET ID: {cf['singlet_in_index']}({cf['singlet_in_type']}); "
                  f"OUTGOING SINGLET ID: {cf['singlet_out_index']}({cf['singlet_out_type']})\n", file=f)
                