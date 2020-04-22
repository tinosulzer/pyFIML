#
# Solve RHT model and save solutions
#
import numpy as np
import pybamm
import os
import matplotlib.pyplot as plt
from pybamm_model import Model

plot = True
tol = 1e-9
n_points = 129


# Prepare plotting and saving
save_folder = "RHT_pybamm/True_solutions"
os.makedirs(save_folder, exist_ok=True)
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("Temperature")

# Solve for various T_inf
for T_inf in np.linspace(5.0, 50, 10):
    # Define model
    model = Model()

    # Define settings
    parameter_values = model.default_parameter_values
    parameter_values.update({"T_inf": T_inf})
    var_pts = {model.x: n_points}
    solver = pybamm.CasadiAlgebraicSolver(tol=tol)
    t_eval = np.array([0, 1])

    # Create simulation
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=solver, var_pts=var_pts
    )
    sim.solve(t_eval, inputs={"T_inf": T_inf})
    sol = sim.solution

    x = sol["x"].data[:, -1]
    T_final = sol["Temperature"].data[:, -1]
    beta_decoupled_final = sol["beta_decoupled"].data[:, -1]

    np.savetxt("{}/solution_{}".format(save_folder, int(T_inf)), T_final)
    ax.plot(x, T_final, label=T_inf)
    # ax.plot(x, beta_decoupled_final, label=T_inf)

# Once the simulation is terminated, show the results if plot is True
if plot is True:
    ax.legend()
    plt.show()
