import numpy as np

def kronecker(i, j):
    return 1 if i == j else 0


def fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1):
    return (kronecker(np1, nk1) *
            kronecker(mp1, mk1) *
            kronecker(sp1, sk1) *
            kronecker(kp1, kk1))


def adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1):
    if kronecker(mk1, mp1) == 0:
        return 0.0

    term1 = (1 + 2*np1 + abs(mp1)) * kronecker(nk1, np1)
    term2 = 0.0

    if max(nk1, np1) - min(nk1, np1) == 1:
        term2 = np.sqrt(max(nk1, np1) * (abs(mp1) + max(nk1, np1)))

    return (term1 - term2) * kronecker(sp1, sk1) * kronecker(kp1, kk1)

def adotas(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1):

    # KroneckerDelta(np1, nk1)
    term1 = (1 if np1 == nk1 else 0) * (2 * np1 + abs(mp1) + 1)

    # KroneckerDelta(|np1-nk1|, 1)
    term2 = 0.0
    if abs(np1 - nk1) == 1:
        term2 = np.sqrt(max(np1, nk1) * (max(np1, nk1) + abs(mp1)))

    # full structure
    result = (term1 + term2)

    result *= (1 if mp1 == mk1 else 0)
    result *= (1 if sp1 == sk1 else 0)
    result *= (1 if kp1 == kk1 else 0)

    return float(result)

def ifactor(n1, m1, n2, m2):

    def d(a, b):
        return 1 if a == b else 0

    # -----------------------------
    # Case 1: m1 >= 0 and m2 = m1 + 1
    # -----------------------------
    if (m1 >= 0) and (m2 == m1 + 1):

        term1 = -np.sqrt(float(n1)) * d(n1 - 1, n2)
        term2 = np.sqrt(float(1 + m1 + n1)) * d(n1, n2)

        return term1 + term2

    # -----------------------------
    # Case 2: m1 > 0 and m2 = m1 - 1
    # -----------------------------
    elif (m1 > 0) and (m2 == m1 - 1):

        term1 = np.sqrt(float(m1 + n1)) * d(n1, n2)
        term2 = -np.sqrt(float(1 + n1)) * d(n1 + 1, n2)

        return term1 + term2

    # -----------------------------
    # Case 3: m1 < 0 and m2 = m1 + 1
    # -----------------------------
    elif (m1 < 0) and (m2 == m1 + 1):

        term1 = np.sqrt(float(-m1 + n1)) * d(n1, n2)
        term2 = -np.sqrt(float(1 + n1)) * d(n1 + 1, n2)

        return term1 + term2

    # -----------------------------
    # Case 4: m1 <= 0 and m2 = m1 - 1
    # -----------------------------
    elif (m1 <= 0) and (m2 == m1 - 1):

        term1 = -np.sqrt(float(n1)) * d(n1 - 1, n2)
        term2 = np.sqrt(float(1 - m1 + n1)) * d(n1, n2)

        return term1 + term2

    # -----------------------------
    # default
    # -----------------------------
    else:
        return 0.0
    
def adotbq(np1, mp1, nk1, mk1,
           np2, mp2, nk2, mk2,
           sp1, sk1, sp2, sk2,
           kp1, kk1, kp2, kk2):

    def d(a, b):
        return 1 if a == b else 0

    # -------------------------
    # HO ladder factors
    # -------------------------
    term1 = ifactor(np1, mp1, nk1, mk1)
    term2 = ifactor(np2, mp2, nk2, mk2)

    # -------------------------
    # m coupling structure
    # -------------------------
    m_coupling = (
        d(mk1, 1 + mp1) * d(mk2, -1 + mp2)
        +
        d(mk1, -1 + mp1) * d(mk2, 1 + mp2)
    )

    # -------------------------
    # spin + longitudinal deltas
    # -------------------------
    spin = d(sp1, sk1) * d(sp2, sk2)
    kdelta = d(kp1, kk1) * d(kp2, kk2)

    # -------------------------
    # final expression (exact Fortran structure)
    # -------------------------
    return 0.5 * term1 * term2 * m_coupling * spin * kdelta