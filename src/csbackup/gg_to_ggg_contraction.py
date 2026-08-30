import numpy as np
from pathlib import Path
from colorwf_runner import run_color_wf, get_singlet_array, get_singlet_coefficients
from su3constants import SU3Constants

'''
TWO GLUON TO THREE GLUON COLOR FACTOR CONTRACTION BY NP.EISUM -- A MORE GENERAL METHOD THAN PREVIOUS NOTEBOOK.
THE RESULT IS CORRECT.
'''


su3 = SU3Constants()

def gg_to_ggg_mtx_ele(): # fabc
    print("f_abc:", su3.fabc)
    return 1j * su3.fabc



def gg_to_ggg_contraction():
    # read from color_wf

    # incoming wf
    incoming_result = run_color_wf(n=0, m=0, k=2)
    gm_coeffs_inc, gm_labels_inc = get_singlet_array(incoming_result, basis="GM", singlet_index=1)

    incoming_wf = np.zeros((8, 8), dtype = complex)

    for k in range(len(gm_labels_inc)):
        i_basis = int(gm_labels_inc[k][3:4])
        j_basis = int(gm_labels_inc[k][8:9])
        incoming_wf[i_basis - 1, j_basis - 1] = gm_coeffs_inc[k]


    
    # outgoing coeff
    outgoing_result = run_color_wf(n=0, m=0, k=3)
    gm_coeffs_out, gm_labels_out = get_singlet_array(outgoing_result, basis="GM", singlet_index=2)

    outgoing_wf = np.zeros((8, 8, 8), dtype = complex)

    for k in range(len(gm_labels_out)):
        q_basis = int(gm_labels_out[k][3:4])
        r_basis = int(gm_labels_out[k][8:9])
        p_basis = int(gm_labels_out[k][13:14])
        #print(q_basis, r_basis, p_basis)
        outgoing_wf[q_basis - 1, r_basis - 1, p_basis - 1] = (gm_coeffs_out[k])
    
    print("1:", gm_coeffs_out)
    print("1.5:", gm_labels_out)
    print("2:", outgoing_wf)

    
    mtx_ele = gg_to_ggg_mtx_ele()

    # contraction:  einsum
    gg_to_ggg_cf_contraction_result = np.einsum('dcb, acd ,ab -> ', outgoing_wf, mtx_ele, incoming_wf)
    # return result
    return gg_to_ggg_cf_contraction_result

'''
result = run_color_wf(n=0, m=0, k=2)
gm_coeffs, gm_labels = get_singlet_array(result, basis="GM", singlet_index=1)
print(gm_coeffs)
print(gm_labels)
num = int(gm_labels[0][3:4])
print(num)
'''

if __name__ == "__main__":
    print(gg_to_ggg_contraction())