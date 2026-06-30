import numpy as np
from math import log


# =========================================================
# binomial
# =========================================================

def binomial(n, m):

    if m < 0 or m > n:
        return 0.0

    m = min(m, n - m)

    res = 1.0
    for i in range(1, m + 1):
        res *= (n - m + i) / i

    return res


# =========================================================
# lognm
# =========================================================

def lognm(n, m):

    logn = 0.0
    for i in range(2, n + 1):
        logn += log(i)

    lognm_val = logn

    for i in range(n + 1, n + abs(m) + 1):
        lognm_val += log(i)

    return 0.5 * (lognm_val + logn)


# =========================================================
# TMC kernel
# =========================================================

def TMC(nn, mm, n, m, n1, m1, n2, m2, tandelta):# tan delta = sqrt(x2/x1)

    # kronecker 
    lhs = 2*nn + abs(mm) + 2*n + abs(m)
    rhs = 2*n1 + abs(m1) + 2*n2 + abs(m2)
    if lhs != rhs or (mm + m != m1 + m2):
        return 0.0
    
    
    sindelta = tandelta / np.sqrt(1.0 + tandelta**2)
    cosdelta = 1.0 / np.sqrt(1.0 + tandelta**2)

    t = -1.0 / (tandelta * tandelta)

    EEp = nn + (abs(mm) + mm) // 2
    EEm = nn + (abs(mm) - mm) // 2
    Ep  = n  + (abs(m) + m) // 2
    Em  = n  + (abs(m) - m) // 2
    E1p = n1 + (abs(m1) + m1) // 2
    E1m = n1 + (abs(m1) - m1) // 2

    s1 = 0.0
    for a in range(max(0, E1m - Em), min(E1m, EEm) + 1):
        s1 += (t**a) * binomial(EEm, a) * binomial(Em, E1m - a)

    s2 = 0.0
    for b in range(max(0, E1p - Ep), min(E1p, EEp) + 1):
        s2 += (t**b) * binomial(EEp, b) * binomial(Ep, E1p - b)

    phase = 1 - 2 * ((abs(nn + n + n1 + n2 + m + m1)) % 2)

    return (
        phase
        * s1
        * s2
        * (tandelta ** (2*n1 + abs(m1)))
        * (sindelta ** (2*nn + abs(mm)))
        * (cosdelta ** (2*n + abs(m)))
        * np.exp(
            lognm(n1, m1)
            + lognm(n2, m2)
            - lognm(nn, mm)
            - lognm(n, m)
        )
    )