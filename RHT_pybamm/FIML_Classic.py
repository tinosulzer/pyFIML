import numpy as np
# from pyML.NN import NN
from pybamm_model import RHT
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def FIML_Classic(cases=[], data=[]):

    # for iOptimIter in range(nOptimIter):

    #     obj_total = 0.0

    #     for case, case_data in zip(cases, data):

    #         case.direct_solve()
    #         obj, beta_sens = case.adjoint_solve(case_data)
    #         obj_total += obj
    #         case.beta -= beta_sens / np.max(np.abs(beta_sens)) * step

    #     print("Iteration %6d    Objective Function %.10le"%(iOptimIter, obj_total))
    for case, case_data in zip(cases, data):

        def objective(x):
            case.beta = x
            obj = case.getObj(case_data)
            print(obj)
            return obj

        def jac(x):
            case.beta = x
            return case.adjoint_solve(case_data)

        x0 = np.ones_like(case_data)
        bounds = [(0, None)] * len(x0)
        sol = minimize(objective, x0, jac=jac, bounds=bounds)
        case.beta = sol.x

        plt.figure()
        plt.plot(case.direct_solve())
        plt.plot(case_data)
        plt.figure()
        plt.plot(case.beta)
        plt.show()

    features = []
    beta     = []

    for case in cases:

        case.direct_solve()
        features.append(case.getFeatures())
        beta.append(case.getBeta())

    features = np.hstack(features)
    beta     = np.hstack(beta)

    nn = NN({"shape":[ features.shape[0] , 7 , 7 , beta.shape[0] ], "actfn":[0,2,2,0], "vars":None})

    nn.Train(features, beta, nEpochs=2000)

    for case in cases:

        case.nn = nn
        case.plot = True
        case.direct_solve()

if __name__=="__main__":

    FIML_Classic(cases=[RHT(T_inf=50, npoints=129)],
                data=[np.loadtxt("RHT_pybamm/True_solutions/solution_50")])
