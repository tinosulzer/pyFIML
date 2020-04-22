#
# Fit a single case using scipy.minimize
#

import numpy as np
import pybamm
from pybamm_model import RHT
from scipy.optimize import minimize

npoints = 129
rht = RHT(T_inf=10, savesol=False, plot=False, lambda_reg=1, npoints=npoints)
x = np.linspace(0,1,129)
rht.beta = 1 + 0.1 * np.sin(x)
rht.direct_solve()
data = rht.direct_solve()

def objective(x):
    rht.beta = x
    return rht.getObj(data)

def jac(x):
    rht.beta = x
    return rht.adjoint_solve(data)

timer = pybamm.Timer()
sol = minimize(objective, [1.1] * 129)
print("Without jac: ", timer.time())
timer.reset()
sol = minimize(objective, [1.1] * 129, jac=jac)
print("With jac: ", timer.time())
# print(sol)
