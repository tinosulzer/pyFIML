#
# PyBaMM implementation of the RHT model, and wrapper class for FIML
#
import casadi
from os import path
import os
import pybamm
import matplotlib.pyplot as plt
import numpy as np


class RHT:
    "Wrapper for the model"
    def __init__(
        self,
        T_inf=5.0,
        npoints=129,
        tol=1e-8,
        lambda_reg=1e-5,
        plot=True,
        savesol=False,
    ):

        self.T_inf = T_inf  # Temperature of the one-dimensional body
        # Boolean flag whether to plot the solution at the end of simulation
        self.plot = plot
        # Boolean flag whether to save the converged temperatures
        self.savesol = savesol
        self.lambda_reg = lambda_reg  # Regularization constant for objective function
        # Initialize beta
        self.beta = np.ones(npoints)
        # Initialize flags
        self.has_obj_and_jac_funs = False

        # Define model
        model = Model()

        # Define settings
        parameter_values = model.default_parameter_values
        parameter_values.update({"T_inf": T_inf, "beta": pybamm.InputParameter("beta", domain="line")})
        var_pts = {model.x: npoints}
        solver = pybamm.CasadiAlgebraicSolver(tol=tol)

        # Create simulation
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, solver=solver, var_pts=var_pts
        )
        t_eval = [0]

        sim.solve(t_eval)
        self.sol = sim.solution
        self.T = self.sol["Temperature"]
        self.x = self.T.x_sol

        fig, self.ax = plt.subplots()

    # ----------------------------------------------------------------------------------

    def direct_solve(self):

        T_eval = self.T.value({"beta": self.beta}).full()

        if self.savesol is True:
            save_folder = "RHT_pybamm/Model_solutions"
            os.makedirs(save_folder, exist_ok=True)
            print("Saving solution to file")
            np.savetxt("{}/solution_{}".format(save_folder, self.T_inf), T_eval)

        # Once the simulation is terminated, show the results if plot is True

        if self.plot is True:
            ax = self.ax
            ax.plot(self.x, T_eval)
            ax.set_xlabel("x")
            ax.set_ylabel("Temperature")
            plt.show()

        self.T_eval = T_eval
        return T_eval

    # ----------------------------------------------------------------------------------

    def adjoint_solve(self, data):
        if not self.has_obj_and_jac_funs:
            self.create_obj_and_jac_funs(data)

        return self.jac_fun(self.beta).full().flatten()

    # ----------------------------------------------------------------------------------

    def getObj(self, data):
        if not self.has_obj_and_jac_funs:
            self.create_obj_and_jac_funs(data)

        return self.obj_fun(self.beta).full()

    def create_obj_and_jac_funs(self, data):
        # Create objective function and derivative
        beta = self.sol.inputs["beta"]
        T = self.T.value({"beta": beta})
        obj = casadi.sum1((T - data) ** 2) + self.lambda_reg * casadi.sum1(
            (beta - 1.0) ** 2
        )
        self.obj_fun = casadi.Function("obj", [beta], [obj])

        jac = casadi.jacobian(obj, beta)
        self.jac_fun = casadi.Function("jacfn", [beta], [jac])

        self.has_obj_and_jac_funs = True

class Model(pybamm.BaseModel):
    "Model with equations"
    def __init__(self, options=None):
        super().__init__()

        self.name = "RHT"
        T = pybamm.Variable("Temperature", domain="line")
        self.x = pybamm.SpatialVariable("x", domain="line")

        # Define parameters beta
        T_inf = pybamm.FunctionParameter("T_inf", {"x": self.x})
        # beta = pybamm.Parameter("beta")
        eps0 = pybamm.Parameter("eps0")

        beta = pybamm.FunctionParameter("beta", {"Temperature": T})
        # Also make a beta_decoupled, for plotting
        beta_decoupled = pybamm.FunctionParameter("beta_decoupled", {"Temperature": T})

        # Define model
        dTdx = pybamm.grad(T)
        source = beta * eps0 * (T_inf ** 4 - T ** 4)

        self.algebraic = {T: pybamm.div(dTdx) + source}
        # Careful initialization as solution is not unique for large T_inf
        self.initial_conditions = {T: 0.7 * T_inf}
        self.boundary_conditions = {
            T: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        self.variables = {
            "Temperature": T,
            "beta": beta,
            "x": self.x,
            "x [m]": self.x,
            "beta_decoupled": beta_decoupled,
        }

    @property
    def default_parameter_values(self):
        # Define default parameter values with true beta
        def beta(T):
            T_inf = pybamm.FunctionParameter("T_inf", {"x": self.x})
            h = pybamm.Parameter("h")
            eps0 = pybamm.Parameter("eps0")
            return (
                1e-4
                * (1.0 + 5.0 * pybamm.sin(3 * np.pi * T / 200.0) + pybamm.exp(0.02 * T))
                / eps0
                + h * (T_inf - T) / (T_inf ** 4 - T ** 4) / eps0
            )

        return pybamm.ParameterValues(
            {
                "T_inf": 30,
                "beta": beta,
                "beta_decoupled": beta,
                "eps0": 5e-4,
                "h": 0.5,
            }
        )

    @property
    def default_geometry(self):
        geometry = {
            "line": {
                "primary": {self.x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            },
        }
        return geometry

    @property
    def default_var_pts(self):
        var_pts = {self.x: 50}
        return var_pts

    @property
    def default_submesh_types(self):
        submesh_types = {"line": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
        return submesh_types

    @property
    def default_spatial_methods(self):
        spatial_methods = {"line": pybamm.FiniteVolume()}
        return spatial_methods

    @property
    def default_solver(self):
        return pybamm.CasadiAlgebraicSolver()


if __name__ == "__main__":
    pybamm.set_logging_level("INFO")

    true_model = Model()
    true_model.name = "True RHT"
    parameter_values = true_model.default_parameter_values
    parameter_values["T_inf"] = 45
    sim_true = pybamm.Simulation(true_model, parameter_values=parameter_values)

    base_model = Model()
    base_model.name = "Base RHT"
    parameter_values = base_model.default_parameter_values
    parameter_values["T_inf"] = 45
    parameter_values["beta"] = 1
    sim_base = pybamm.Simulation(base_model, parameter_values=parameter_values)

    t_eval = np.array([0, 1])
    sims = []
    for sim in [sim_true, sim_base]:
        sim.solve(t_eval)
        sims.append(sim)

    pybamm.dynamic_plot(
        sims, ["Temperature", "beta", "beta_decoupled"], spatial_unit="m"
    )
