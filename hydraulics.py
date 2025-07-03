"""
this file contains all the equations I'll use
to calculate the hydraulic characteristics at
the cross-sections of the low-head dams.
"""
import numpy as np
from scipy.optimize import fsolve

g = 9.81 # grav. const.


def weir_eq(H, P, q):
    """
    H = head (m)
    P = dam height (m)
    q = unit discharge (m**2/s)
    """
    return (2 / 3) * (0.611 + 0.075 * (H / P)) * np.sqrt(2 * g) * H ** (3/2) - q


def Fr_eq(Fr, x):
    # x = H/P
    A = (9 / (4 * (0.611 + 0.075 * x)**2)) * 0.5 * Fr**2
    term1 = A**(1/3) * (1 + 1/x)
    term2 = 0.5 * Fr**2 * (1 + 0.1/x)
    return 1 - term1 + term2


# noinspection PyTypeChecker
def compute_flip_and_conjugate(Q, L, P):
    # print(Q, L, P)
    q = Q / L
    H = fsolve(weir_eq, x0=1.0, args=(P, q))[0] # x0 is initial guess for H
    y_flip = (H + P) / 1.1

    # on leuthusser's graphs, the x-axis is H/P
    x = H / P
    coeffs = [1, -(1 + 1 / x), 0, (4 / 9) * (0.611 + 0.075 * x) ** 2 * (1 + 0.1 / x)]
    y_1 = min([r.real for r in np.roots(coeffs) if np.isreal(r) and r.real > 0]) * H
    Fr_1 = fsolve(Fr_eq, x0=100, args=(x,))[0] # x0 is initial guess for Fr_1
    y_2 = y_1 * 0.5 * (-1 + np.sqrt(1 + 8 * Fr_1 ** 2))
    return y_flip, y_2


def dam_height(Q, b, delta_wse, y_t, delta_z=0):
    """
        eq.1: P = delta_wse - H + y_t + dela_z
        ... but we need to find H
        also—all these calcs are in metric units

        eq.2: q = (2/3) * C_w * np.sqrt(2 * g) * H**(3/2)
              where,
              C_w = 0.611 + 0.075 * H/P
    """

    # constant terms
    q = Q/b  # discharge (m^2/s)
    ## derived constants
    A = (2 / 3) * np.sqrt(2 * g)
    D = delta_wse + y_t + delta_z  # total pressure head + elevation

    # solve for q in terms of H
    def func(H):
        if H >= D:  # avoid division by zero or negative P
            return 1e6
        lhs = q / A
        rhs = 0.611 * H ** (3 / 2) + 0.075 * H ** (5 / 2) / (D - H)
        return lhs - rhs

    # Initial guess for H (must be less than D)
    H_0 = D * 0.8
    # Solve for H
    H_sol = fsolve(func, H_0)[0]

    # plug H into eq.1
    P = D - H_sol
    return round(P, 3)
