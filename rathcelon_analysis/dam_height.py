import numpy as np
import pandas as pd

def weir_height(Q, b, y_u, tol=0.001):
    """
    Q = flow in river (cms)
    b = bank width (m)
    y_u = upstream depth (m)
    """
    q = Q/b # unit flow
    g = 9.81 # gravitational constant
    # left-hand side
    A = 3 * q / (2 * np.sqrt(2 * g))
    # initial weir height estimate
    p = 0.5 * y_u # we want to start with a positive number < y_u
    # right-hand side
    B = 0.611 * (y_u - p)**(3/2) + 0.075 * ((y_u - p)**(5/2))/p
    counter = 0 # to avoid infinite loop
    while abs(A - B) > tol:
        counter += 1
        if A < B:
            p += 0.001
        else:
            p -= 0.001
        # recalculate B after adjusting height
        B = 0.611 * (y_u - p) ** (3 / 2) + 0.075 * ((y_u - p) ** (5 / 2)) / p
        if counter > 10000:
            break
    return round(p, 3)


dam_info = pd.read_csv("C:/Users/ki87ujmn/Downloads/height_test.csv")
dam_info['p'] = 0
for index, row in dam_info.iterrows():
    dam_info.at[index, 'p'] = weir_height(row['Q'], row['b'], row['y_u'])
print(dam_info)
