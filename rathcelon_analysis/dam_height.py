import numpy as np
"""
hopefully we can get dam height from rating curves?

basically, the goal is to estimate dam height, P, from the RathCelon rating curves
"""
g = 32.2 #ft/s**2

# equation 4
# C_W = 0.611 + 0.75 * H/P
# equation 3
# q = 2/3 * C_W * np.sqrt(2 * g) * H^(3/2)

# equation 6
# C_L = 0.1 * P/H
# DE_p = (P + H) - (Y_1 + q**2/(Y_1**2 * 2 * g))

"""
after reading leutheusser and fan (2001), here's what I'm thinking:

H + P is the total head upstream... let's call it Y_u

upstream:
    Q = b * 2/3 * (0.611 + 0.75 * H/P) * np.sqrt(2 * g) * H^(3/2)
    
    we know Q, b, & g
    
    
downstream:
    

"""
