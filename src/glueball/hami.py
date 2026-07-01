from math import sqrt
from collections import defaultdict
from TMC import TMC
from tool import *
from functools import lru_cache
from pathlib import Path
import sys


def build_sparse_hamiltonian(global_basis, params):
    H = defaultdict(float)
    dim = len(global_basis)
    for i, (si, bra) in enumerate(global_basis):
        for j, (sj, ket) in enumerate(global_basis):
            val = matrix_element(si, sj, bra, ket, params)
            if abs(val) > 1e-14:
                H[(i, j)] += val
    return H, dim
# =========================================================
# Matrix element dispatcher
# =========================================================
def matrix_element(sector_i, sector_j, bra, ket, params):

    # -------------------------
    # g sector
    # -------------------------
    if sector_i == "g" and sector_j == "g":
        return H_g_g(bra, ket, params)

    # -------------------------
    # gg sector
    # -------------------------
    if sector_i == "gg" and sector_j == "gg":
        return H_gg_gg(bra, ket, params)

    # -------------------------
    # ggg sector
    # -------------------------
    if sector_i == "ggg" and sector_j == "ggg":
        return H_ggg_ggg(bra, ket, params)
    
        # -------------------------
    # gggg sector
    # -------------------------
    if sector_i == "gggg" and sector_j == "gggg":
        return H_gggg_gggg(bra, ket, params)
    
    # -------------------------
    # interaction (Hermitian)
    # -------------------------
    if sector_i == "g" and sector_j == "gg":
        return H_g_gg(bra, ket, params)

    if sector_i == "gg" and sector_j == "g":
        return H_g_gg(bra, ket, params)

    if sector_i == "gg" and sector_j == "ggg":
        return H_gg_ggg(ket, bra, params)
    
    if sector_i == "ggg" and sector_j == "gg":
        return H_gg_ggg(bra, ket, params)
    '''
    if sector_i == "gg" and sector_j == "gggg":
        return H_gg_gggg(bra, ket, params)
    
    if sector_i == "gggg" and sector_j == "gg":
        return H_gg_gggg(bra, ket, params)
    
    if sector_i == "ggg" and sector_j == "gggg":
        return H_ggg_gggg(bra, ket, params)
    
    if sector_i == "gggg" and sector_j == "ggg":
        return H_ggg_gggg(bra, ket, params)
    '''

    return 0.0


# =========================================================
# Physics kernels
# =========================================================
def H_g_g(bra, ket, params):

    kp1 = bra.k
    sp1 = bra.s
    np1 = bra.n
    mp1 = bra.m
    
    
    kk1 = ket.k
    sk1 = ket.s
    nk1 = ket.n
    mk1 = ket.m

    # --------- selection rule (diagonal structure) ----------
    if not (kp1 == kk1 and sp1 == sk1 and np1 == nk1 and mp1 == mk1):
        return 0.0

    # --------- parameters ----------
    mass = params.mass_g
    b = params.b
    lag = 30.1

    Pplus = params.p_plus

    kt = kp1
    mj = 2
    
    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)


    kp1half = kp1 
    Pplus = kp1half  # 如果你有全局 Pplus，这里可以替换

    # --------- core matrix elements ----------
    Ememe = fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1)
    Ep1p1 = adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1)

    shift = 2 * Ememe

    kinetic1 = (mass**2 * Ememe +
                bee**2 * (kp1half / Pplus) * Ep1p1) / kp1half

    EnergySingle = kinetic1
    EnergyCM = bee**2 * (Ep1p1 * kp1half / Pplus) / Pplus

    fourierphasen = -np1 + nk1
    fourierphasem = -abs(mp1) + abs(mk1)

    fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)
    

    lagrangeterm = lag * bee**2 * (
        Ep1p1 * kp1half / Pplus +
        (Ep1p1 * kp1half / Pplus) * fourierphase -
        shift
    ) / Pplus

    hamiltonian = EnergySingle - EnergyCM + lagrangeterm

    return hamiltonian

def H_gg_gg(bra, ket, params):
    """
    two-gluon kinetic term (BLFQ-style)
    """


    g3, g4 = bra.particles

    g1, g2 = ket.particles

    kp1 = g3.k
    sp1 = g3.s
    np1 = g3.n
    mp1 = g3.m
    
    kp2 = g4.k
    sp2 = g4.s
    np2 = g4.n
    mp2 = g4.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m
    
    
    initialcolor = bra.color_state
    finalcolor   = ket.color_state

    if not (
        kp1 == kk1 and kp2 == kk2 and
        sp1 == sk1 and sp2 == sk2 
    ):
        return 0.0

    if initialcolor != finalcolor:
        return 0.0

    # -----------------------------
    # parameters
    # -----------------------------
    mass1 = params.mass_gg
    mass2 = params.mass_gg
    b = params.b
    
    lag = 30.1



    # -----------------------------
    # longitudinal momenta
    # -----------------------------
    kp1half = kp1 
    kp2half = kp2 
    kt = kp1 + kp2
    mj = 2 


##!!!!!!!!!!!!!!!!!!!!!!!总动量， 可能要调整

    Pplus = params.p_plus

    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)
    
    Pplus = kt
    
    

    # -----------------------------
    # Fourier phase (DIRECT translation)
    # -----------------------------
    # fourierphasen = -np1 - np2 + nk1 + nk2
    # fourierphasem = -abs(mp1) - abs(mp2) + abs(mk1) + abs(mk2)

    # fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    # -----------------------------
    # delta / matrix elements
    # -----------------------------
    Ememe = (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    Ep1p1 = (adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    Ep2p2 = (adotaq(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Es1s1 = (adotas(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    Es2s2 = (adotas(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Ep1p2 = adotbq(np1, mp1, nk1, mk1,
                   np2, mp2, nk2, mk2,
                   sp1, sk1, sp2, sk2,
                   kp1, kk1, kp2, kk2)

    shift = 2.0 * (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
                   fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    # -----------------------------
    # kinetic energy term
    # -----------------------------
    kinetic1 = ((mass1 ** 2) * Ememe +
                bee ** 2 * (kp1half / Pplus) * Ep1p1) / kp1half

    kinetic2 = ((mass2 ** 2) * Ememe +
                bee ** 2 * (kp2half / Pplus) * Ep2p2) / kp2half

    EnergySingle = kinetic1 + kinetic2

    EnergyCM = bee ** 2 * (
        Ep1p1 * kp1half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        Ep2p2 * kp2half / Pplus
    ) / Pplus

    fourierphasen = -np1-np2+nk1+nk2
    fourierphasem = -abs(mp1)-abs(mp2)+abs(mk1)+abs(mk2)

    fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    lagrangeterm = lag * bee ** 2 * (
        (Ep1p1 * kp1half / Pplus +
         2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
         Ep2p2 * kp2half / Pplus)
        +
        (Ep1p1 * kp1half / Pplus +
         2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
         Ep2p2 * kp2half / Pplus)*fourierphase 
        - shift
    ) / Pplus

    newlag = np.abs(lagrangeterm) if abs(lagrangeterm.imag) > 1e-12 else lagrangeterm.real

    # -----------------------------
    # Hamiltonian
    # -----------------------------
    hamiltonian = (
        EnergySingle
        - EnergyCM
        + newlag
    )
    
    # if(hamiltonian!=0):
    #     print("hamiltonian =", hamiltonian ,"EnergySingle = ", EnergySingle, "EnergyCM = ", EnergyCM, "lagrangeterm = ", lagrangeterm)

    return hamiltonian

def H_ggg_ggg(bra, ket, params):
    g4, g5, g6 = bra.particles #out

    g1, g2, g3 = ket.particles #in
# p-out k-in
    kp1 = g4.k
    sp1 = g4.s
    np1 = g4.n
    mp1 = g4.m
    
    kp2 = g5.k
    sp2 = g5.s
    np2 = g5.n
    mp2 = g5.m

    kp3 = g6.k
    sp3 = g6.s
    np3 = g6.n
    mp3 = g6.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m
    
    kk3 = g3.k
    sk3 = g3.s
    nk3 = g3.n
    mk3 = g3.m
    
    initialcolor = bra.color_state
    finalcolor   = ket.color_state

    if not (
        kp1 == kk1 and kp2 == kk2 and kp3 == kk3 and
        sp1 == sk1 and sp2 == sk2 and sp3 == sk3
    ):
        return 0.0

    #if initialcolor != finalcolor:
    #    return 0.0
    # -----------------------------
    # parameters
    # -----------------------------
    mass1 = params.mass_ggg
    mass2 = params.mass_ggg
    mass3 = params.mass_ggg
    b = params.b
    
    lag = 30.1
    # -----------------------------
    # longitudinal momenta
    # -----------------------------
    kp1half = kp1 
    kp2half = kp2 
    kp3half = kp3
    kt = kp1 + kp2 + kp3
    mj = 2 
##!!!!!!!!!!!!!!!!!!!!!!!总动量， 可能要调整

    Pplus = params.p_plus

    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)
    
    Pplus = kt

    # -----------------------------
    # Fourier phase (DIRECT translation)
    # -----------------------------
    # fourierphasen = -np1 - np2 + nk1 + nk2
    # fourierphasem = -abs(mp1) - abs(mp2) + abs(mk1) + abs(mk2)

    # fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    # -----------------------------
    # delta / matrix elements
    # -----------------------------
    Ememe = (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3)) 

    Ep1p1 = (adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3)) # conservation of the other two particles

    Ep2p2 = (adotaq(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) * fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3))
    
    Ep3p3 = (adotaq(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Es1s1 = (adotas(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3))

    Es2s2 = (adotas(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) * fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3))
    
    Es3s3 = (adotas(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Ep1p2 = adotbq(np1, mp1, nk1, mk1,
                   np2, mp2, nk2, mk2,
                   sp1, sk1, sp2, sk2,
                   kp1, kk1, kp2, kk2)
    
    Ep1p3 = adotbq(np1, mp1, nk1, mk1,
                   np3, mp3, nk3, mk3,
                   sp1, sk1, sp3, sk3,
                   kp1, kk1, kp3, kk3)
    
    Ep2p3 = adotbq(np2, mp2, nk2, mk2,
                   np3, mp3, nk3, mk3,
                   sp2, sk2, sp3, sk3,
                   kp2, kk2, kp3, kk3)

    shift = 3.0 * (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
                   fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
                   fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3))

    # -----------------------------
    # kinetic energy term
    # -----------------------------
    kinetic1 = ((mass1 ** 2) * Ememe +
                bee ** 2 * (kp1half / Pplus) * Ep1p1) / kp1half

    kinetic2 = ((mass2 ** 2) * Ememe +
                bee ** 2 * (kp2half / Pplus) * Ep2p2) / kp2half
    
    kinetic3 = ((mass3 ** 2) * Ememe +
                bee ** 2 * (kp3half / Pplus) * Ep3p3) / kp3half

    EnergySingle = kinetic1 + kinetic2 + kinetic3

    EnergyCM = bee ** 2 * (
        Ep1p1 * kp1half / Pplus +
        Ep2p2 * kp2half / Pplus+
        Ep3p3 * kp3half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        2 * Ep1p3 * np.sqrt(kp1half * kp3half / Pplus ** 2) + 
        2  * Ep2p3 * np.sqrt(kp2half * kp3half / Pplus ** 2)
    ) / Pplus

    fourierphasen = -np1-np2 - np3 +nk1+nk2 # (-1)^n e^{i m phi} of Transverse basis 
    fourierphasem = -abs(mp1)-abs(mp2) - abs(mp3) +abs(mk1)+abs(mk2) + abs(mk3)

    fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

# ????????????????????????????????????????????????????????????????????????
    lagrangeterm = lag * bee ** 2 * ((
        (Ep1p1 * kp1half / Pplus +
        Ep2p2 * kp2half / Pplus+
        Ep3p3 * kp3half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        2 * Ep1p3 * np.sqrt(kp1half * kp3half / Pplus ** 2) + 
        2  * Ep2p3 * np.sqrt(kp2half * kp3half / Pplus ** 2)) / Pplus) 
        + (Ep1p1 * kp1half / Pplus +
        Ep2p2 * kp2half / Pplus+
        Ep3p3 * kp3half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        2 * Ep1p3 * np.sqrt(kp1half * kp3half / Pplus ** 2) + 
        2  * Ep2p3 * np.sqrt(kp2half * kp3half / Pplus ** 2))*fourierphase 
        - shift
    ) / Pplus
# ???????????????????????????????????????????????????????????????????????????





    # lagrangeterm = lag * bee ** 2 * (
    #     (Ep1p1 * kp1half / Pplus +
    #      2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
    #      Ep2p2 * kp2half / Pplus)
    #     +
    #     (Ep1p1 * kp1half / Pplus +
    #      2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
    #      Ep2p2 * kp2half / Pplus)
    #     - shift
    #     ) / Pplus
    
    # print("lag = ", lag, "b2 = ", bee**2, "Pplus = ", Pplus)
    




    newlag = np.abs(lagrangeterm) if abs(lagrangeterm.imag) > 1e-12 else lagrangeterm.real
    # -----------------------------
    # Hamiltonian
    # -----------------------------
    hamiltonian = (
        EnergySingle
        - EnergyCM
        + newlag
    )
    
    # if(hamiltonian!=0):
    #     print("hamiltonian =", hamiltonian ,"EnergySingle = ", EnergySingle, "EnergyCM = ", EnergyCM, "lagrangeterm = ", lagrangeterm)

    return hamiltonian




def H_gggg_gggg(bra, ket, params):
    g5, g6, g7, g8 = bra.particles #out

    g1, g2, g3, g4 = ket.particles #in
# p-out k-in
    kp1 = g5.k
    sp1 = g5.s
    np1 = g5.n
    mp1 = g5.m
    
    kp2 = g6.k
    sp2 = g6.s
    np2 = g6.n
    mp2 = g6.m

    kp3 = g7.k
    sp3 = g7.s
    np3 = g7.n
    mp3 = g7.m

    kp4 = g8.k
    sp4 = g8.s
    np4 = g8.n
    mp4 = g8.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m
    
    kk3 = g3.k
    sk3 = g3.s
    nk3 = g3.n
    mk3 = g3.m

    kk4 = g4.k
    sk4 = g4.s
    nk4 = g4.n
    mk4 = g4.m
    
    initialcolor = bra.color_state
    finalcolor   = ket.color_state

    if not (
        kp1 == kk1 and kp2 == kk2 and kp3 == kk3 and kp4 == kk4 and
        sp1 == sk1 and sp2 == sk2 and sp3 == sk3 and sp4 == sk4
    ):
        return 0.0

    #if initialcolor != finalcolor:
    #    return 0.0

    # -----------------------------
    # parameters
    # -----------------------------
    mass1 = params.mass_gggg
    mass2 = params.mass_gggg
    mass3 = params.mass_gggg
    mass4 = params.mass_gggg
    b = params.b
    
    lag = 30.1



    # -----------------------------
    # longitudinal momenta
    # -----------------------------
    kp1half = kp1 
    kp2half = kp2 
    kp3half = kp3
    kp4half = kp4
    kt = kp1 + kp2 + kp3 + kp4
    mj = 2 


##!!!!!!!!!!!!!!!!!!!!!!!总动量， 可能要调整

    Pplus = params.p_plus

    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)
    
    Pplus = kt
    
    

    # -----------------------------
    # Fourier phase (DIRECT translation)
    # -----------------------------
    # fourierphasen = -np1 - np2 - np3 - np4 + nk1 + nk2 + nk3 + nk4
    # fourierphasem = -abs(mp1) - abs(mp2) - abs(mp3) - abs(mp4) + abs(mk1) + abs(mk2) + abs(mk3) + abs(mk4)

    # fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    # -----------------------------
    # delta / matrix elements
    # -----------------------------
    Ememe = (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4)) 

    Ep1p1 = (adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))

    Ep2p2 = (adotaq(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))
    
    Ep3p3 = (adotaq(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))

    Ep4p4 = (adotaq(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Es1s1 = (adotas(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))

    Es2s2 = (adotas(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))
    
    Es3s3 = (adotas(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))

    Es4s4 = (adotas(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
             fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Ep1p2 = adotbq(np1, mp1, nk1, mk1,
                   np2, mp2, nk2, mk2,
                   sp1, sk1, sp2, sk2,
                   kp1, kk1, kp2, kk2)
    
    Ep1p3 = adotbq(np1, mp1, nk1, mk1,
                   np3, mp3, nk3, mk3,
                   sp1, sk1, sp3, sk3,
                   kp1, kk1, kp3, kk3)
    
    Ep1p4 = adotbq(np1, mp1, nk1, mk1,
                   np4, mp4, nk4, mk4,
                   sp1, sk1, sp4, sk4,
                   kp1, kk1, kp4, kk4)

    Ep2p3 = adotbq(np2, mp2, nk2, mk2,
                   np3, mp3, nk3, mk3,
                   sp2, sk2, sp3, sk3,
                   kp2, kk2, kp3, kk3)

    Ep2p4 = adotbq(np2, mp2, nk2, mk2,
                   np4, mp4, nk4, mk4,
                   sp2, sk2, sp4, sk4,
                   kp2, kk2, kp4, kk4)

    Ep3p4 = adotbq(np3, mp3, nk3, mk3,
                   np4, mp4, nk4, mk4,
                   sp3, sk3, sp4, sk4,
                   kp3, kk3, kp4, kk4)

    shift = 4.0 * (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
                   fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) * 
                   fidelta(np3, mp3, nk3, mk3, sp3, sk3, kp3, kk3) *
                   fidelta(np4, mp4, nk4, mk4, sp4, sk4, kp4, kk4))

    # -----------------------------
    # kinetic energy term
    # -----------------------------
    kinetic1 = ((mass1 ** 2) * Ememe +
                bee ** 2 * (kp1half / Pplus) * Ep1p1) / kp1half

    kinetic2 = ((mass2 ** 2) * Ememe +
                bee ** 2 * (kp2half / Pplus) * Ep2p2) / kp2half
    
    kinetic3 = ((mass3 ** 2) * Ememe +
                bee ** 2 * (kp3half / Pplus) * Ep3p3) / kp3half

    kinetic4 = ((mass4 ** 2) * Ememe +
                bee ** 2 * (kp4half / Pplus) * Ep4p4) / kp4half

    EnergySingle = kinetic1 + kinetic2 + kinetic3 + kinetic4

    EnergyCM = bee ** 2 * (
        Ep1p1 * kp1half / Pplus +
        Ep2p2 * kp2half / Pplus+
        Ep3p3 * kp3half / Pplus +
        Ep4p4 * kp4half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        2 * Ep1p3 * np.sqrt(kp1half * kp3half / Pplus ** 2) + 
        2 * Ep1p4 * np.sqrt(kp1half * kp4half / Pplus ** 2) +
        2 * Ep2p3 * np.sqrt(kp2half * kp3half / Pplus ** 2) +
        2 * Ep2p4 * np.sqrt(kp2half * kp4half / Pplus ** 2) +
        2 * Ep3p4 * np.sqrt(kp3half * kp4half / Pplus ** 2)
    ) / Pplus

    fourierphasen = -np1-np2 - np3 - np4 + nk1 + nk2 + nk3 + nk4
    fourierphasem = -abs(mp1)-abs(mp2) - abs(mp3) - abs(mp4) + abs(mk1) + abs(mk2) + abs(mk3) + abs(mk4)

    fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    lagrangeterm = lag * bee ** 2 * ((
        (Ep1p1 * kp1half / Pplus +
        Ep2p2 * kp2half / Pplus+
        Ep3p3 * kp3half / Pplus +
        Ep4p4 * kp4half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        2 * Ep1p3 * np.sqrt(kp1half * kp3half / Pplus ** 2) + 
        2 * Ep1p4 * np.sqrt(kp1half * kp4half / Pplus ** 2) +
        2 * Ep2p3 * np.sqrt(kp2half * kp3half / Pplus ** 2) +
        2 * Ep2p4 * np.sqrt(kp2half * kp4half / Pplus ** 2) +
        2 * Ep3p4 * np.sqrt(kp3half * kp4half / Pplus ** 2)) / Pplus) 
        + (Ep1p1 * kp1half / Pplus +
        Ep2p2 * kp2half / Pplus+
        Ep3p3 * kp3half / Pplus +
        Ep4p4 * kp4half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        2 * Ep1p3 * np.sqrt(kp1half * kp3half / Pplus ** 2) + 
        2 * Ep1p4 * np.sqrt(kp1half * kp4half / Pplus ** 2) +
        2 * Ep2p3 * np.sqrt(kp2half * kp3half / Pplus ** 2) +
        2 * Ep2p4 * np.sqrt(kp2half * kp4half / Pplus ** 2) +
        2 * Ep3p4 * np.sqrt(kp3half * kp4half / Pplus ** 2))*fourierphase 
        - shift
    ) / Pplus

    newlag = np.abs(lagrangeterm) if abs(lagrangeterm.imag) > 1e-12 else lagrangeterm.real
    # -----------------------------
    # Hamiltonian
    # -----------------------------
    hamiltonian = (
        EnergySingle
        - EnergyCM
        + newlag
    )
    
    # if(hamiltonian!=0):
    #     print("hamiltonian =", hamiltonian ,"EnergySingle = ", EnergySingle, "EnergyCM = ", EnergyCM, "lagrangeterm = ", lagrangeterm)

    return hamiltonian



def H_g_gg(bra, ket, params):


    # =====================================================
    # enforce interpretation:
    # bra = g
    # ket = gg
    # =====================================================

    g_state = bra
    g1, g2 = ket.particles

    # -------------------------
    # unpack
    # -------------------------
    kp1 = g_state.k
    sp1 = g_state.s
    np1 = g_state.n
    mp1 = g_state.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m


    finalcolor   = ket.color_state

    if finalcolor == 0:
        CF = -np.sqrt(3.0)

    elif finalcolor == 1:
        # CF = -np.sqrt(3.0)
        return 0.0

    else:
        raise ValueError(f"Unknown finalcolor = {finalcolor}")
    

    # -------------------------
    # params
    # -------------------------
    couplings = params.couplings
    b = params.b
    Pplus = params.p_plus
    mj = 2
    bee = b  
    coupling_eff = couplings * CF / sqrt(2.0)

    # -------------------------
    # selection rule
    # -------------------------
    if kp1 != kk1 + kk2:
        return 0.0

    # -------------------------
    # quantum number structure
    # -------------------------
    m = sp1 - (sk1 + sk2)

    n = nk1 + nk2 - np1 + (abs(mk1) + abs(mk2) - abs(mp1) - abs(m)) // 2
    
    if n < 0:
        return 0.0
    
    
    kp1h = float(kp1)
    kk1h = float(kk1)
    kk2h = float(kk2)

    constant = sqrt(2.0) * bee**2 * coupling_eff / (np.pi * Pplus)

    tandelta = sqrt(kk2h / kk1h)

    spinor = sqrt(n + 1.0) * ((-1.0) ** n)

    # -------------------------
    # TMC
    # -------------------------
    T = TMC(
        np1, mp1,
        n, m,
        nk1, mk1,
        nk2, mk2,
        tandelta
    )

    # -------------------------
    # terms
    # -------------------------
    c1 = c2 = c3 = 0.0

    if sp1 == sk2:
        longipart1 = sqrt(kk2h / (kk1h * kp1h))
        c1 = -longipart1 * T * spinor

    if sk1 == -sk2:
        longipart2 = sqrt(kk2h * kk1h / kp1h) / kp1h
        c2 = longipart2 * T * spinor

    if sp1 == sk1:
        longipart3 = sqrt(kk1h / (kk2h * kp1h))
        c3 = -longipart3 * T * spinor

    # -------------------------
    # final vertex
    # -------------------------
    interaction = constant * (c1 + c2 + c3)


    return interaction 



def H_gg_ggg(bra, ket, params):
    g3, g4, g5 = bra.particles
    g1, g2 = ket.particles

    kp1 = g3.k
    sp1 = g3.s
    np1 = g3.n
    mp1 = g3.m

    kp2 = g4.k
    sp2 = g4.s
    np2 = g4.n
    mp2 = g4.m

    kp3 = g5.k
    sp3 = g5.s
    np3 = g5.n
    mp3 = g5.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m

    s_in = ket.color_singlet
    s_out = bra.color_singlet

    incoming = [g1, g2]
    outgoing = [g3, g4, g5]
    num_incoming = 2
    num_outgoing = 3

    # gCF======================================================================================
    dat = Path(__file__).parent / 'gCF' / 'output' / f'{num_incoming}_gluon_to_{num_outgoing}_gluon_Color_Factor.dat'

    total = 0.0
    with open(dat, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if int(parts[2]) != s_in or int(parts[3]) != s_out:
                continue

            cf = complex(float(parts[0]), float(parts[1]))
            idx = 5
            num_vertices = int(parts[idx])
            idx += 1 # vtype col.

            spectator = None
            three_gluon = None
            four_gluon = None

            for v in range(num_vertices): # loop over every vertex
                vtype = int(parts[idx])
                idx += 1
                num_in = int(parts[idx]) # getting # of gluons going into this vertex
                idx += 1
                vertex_incoming = [int(parts[idx + i]) for i in range(num_in)] #LIST of incoming id's 
                idx += num_in
                num_out = int(parts[idx])
                idx += 1
                vertex_outgoing = [int(parts[idx + i]) for i in range(num_out)]# LIST of outgoing id's
                idx += num_out

                if vtype == 0:
                    spectator = (vertex_incoming, vertex_outgoing)
                elif vtype == 3:
                    three_gluon = (vertex_incoming, vertex_outgoing)

                elif vtype == 4:
                    four_gluon = (vertex_incoming, vertex_outgoing)

            if spectator is None or three_gluon is None or four_gluon is None: # rule out cases without a vertex
                continue
            # ========== SPECTATOR
            # NOTE: NO NEED TO LOOP FOR ALL SPECTATORS HERE BECAUSE WE HAVE for line in f WHICH LOOPS ALL POSSIBLE CONFIGs. WE ARE JUST DEALING WITH ONE CONFIG HERE. ONLY ONE GOES IN TO THE SPEC., AND ONLY ONE COMES OUT FROM THE SPEC., so [0][0] and [1][0]
            incoming_spectator = incoming[spectator[0][0] - 1] # direct to the correct basis in incoming = [g1, g2] by using the id of spectator extracted from .dat. 
            outgoing_spectator = outgoing[spectator[1][0] - 1]
            # conservation check
            if not (incoming_spectator.k == outgoing_spectator.k and
                    incoming_spectator.s == outgoing_spectator.s and
                    incoming_spectator.n == outgoing_spectator.n and
                    incoming_spectator.m == outgoing_spectator.m):
                continue

            # ========== THREE-GLUON VERTEX
            three_in = incoming[three_gluon[0][0] - 1]
            three_out1 = outgoing[three_gluon[1][0] - 1]
            three_out2 = outgoing[three_gluon[1][1] - 1]
            #conservation check
            if three_in.k != three_out1.k + three_out2.k:
                continue

            # ========== FOUR-GLUON VERTEX
            four_in1 = incoming[four_gluon[0][0] - 1]
            four_in2 = incoming[four_gluon[0][1] - 1]
            four_out1 = outgoing[four_gluon[1][0] - 1]
            four_out2 = outgoing[four_gluon[1][1] - 1]
            #conservation check
            if four_in1.k + four_in2.k != four_out1.k + four_out2.k:
                continue
            # ================================================================================
            # -------------------------
            # parameters
            # --------------------------
            couplings = params.couplings
            b = params.b
            Pplus = params.p_plus
            mj = 2
            bee = b  
            coupling_eff = couplings / sqrt(2.0) # CF will be multiplied later.
            # -------------------------
            # quantum number structure
            # -------------------------
            m = (sp1 + sp2 + sp3) - (sk1 + sk2)

            n = nk1 + nk2 - (np1 + np2 + np3) + (abs(mk1) + abs(mk2) - (abs(mp1) + abs(mp2) + abs(mp3)) - abs(m)) // 2
            
            if n < 0:
                return 0.0
            
      
            constant = sqrt(2.0) * bee**2 * coupling_eff / (np.pi * Pplus)

            tandelta = sqrt(float(three_out2.k) / float(three_out1.k))

            spinor = sqrt(n + 1.0) * ((-1.0) ** n)

            # -------------------------
            # (for three-gluon vertex) TMC (N, M, n, m, n1, m1, n2, m2)
            # -------------------------
            T = TMC(
                three_in.n, three_in.m, # (by orthogonal condition)
                n, m,
                three_out1.n, three_out1.m,
                three_out2.n, three_out2.m,
                tandelta
            )


            # -------------------------
            # terms
            # -------------------------
            c1 = c2 = c3 = 0.0

            if three_in.s == three_out2.s:
                longipart1 = sqrt(float(three_out2.k) / (float(three_out1.k) * float(three_in.k)))
                c1 = -longipart1 * T * spinor

            if three_out1.s == -three_out2.s:
                longipart2 = sqrt(float(three_out2.k) * (float(three_out1.k) / float(three_in.k)) / float(three_in.k))
                c2 = longipart2 * T * spinor

            if three_in.s == three_out2.s:
                longipart3 = sqrt(float(three_out1.k) / (float(three_out2.k) * float(three_in.k)))
                c3 = -longipart3 * T * spinor

            # -------------------------
            # final vertex
            # -------------------------
            interaction = constant * (c1 + c2 + c3)
            total += cf * interaction

    return total
'''
def H_gg_gggg(bra, ket, params):
    pass

def H_ggg_gggg(bra, ket, params):
    pass
'''