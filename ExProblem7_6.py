import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
SurfaceChargeDensity = 1
d = 0.4
L = 0.5

# Grid constants
N = np.array([99, 1, 99], dtype = int)
delta_x = np.array([2, 2, 2])
x0 = np.array([-1, -1, -1])
Boundaries = [["flat", "closed"], "periodic", ["closed", "closed"]]

# Evaluation constants
Exact = True
Progress = 5
approx_n = 0.1

# File names
Name_E_2D = "ExProblem7_6_E.png"
Name_V_2D = "ExProblem7_6_V.png"
Name_Rho_2D = "ExProblem7_6_Rho.png"
Name_V_loop = "ExProblem7_6_loopV.png"
Name_E_loop = "ExProblem7_6_loopE.png"
Save = True

# Define the current
def J(dx, N, x0, c, mu0):
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current, normalising so the current is the same no matter the grid size
    Grid[:int(N[0] * L), :, int(N[2] * (1 + d) / 2), 0] = -c * SurfaceChargeDensity / dx[2]
    Grid[:int(N[0] * L), :, int(N[2] * (1 - d) / 2), 0] = c * SurfaceChargeDensity / dx[2]
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return a sin times this vector
    def get_J(t):
        return J_Vector
    
    return get_J

# Constants for the loop
L_loop = 0.8
h_loop = 0.15

# Create corner points of loop
p = np.array([[-delta_x[0] * L_loop / 2, 0, -delta_x[2] * h_loop / 2], [-delta_x[0] * L_loop / 2, 0, delta_x[2] * h_loop / 2], [delta_x[0] * L_loop / 2, 0, delta_x[2] * h_loop / 2], [delta_x[0] * L_loop / 2, 0, -delta_x[2] * h_loop / 2]])
p_next = np.roll(p, -1, axis = 0)

# Get the points to draw the drawing the lines
Line_x = np.array([[Pos[0], Pos_next[0]] for Pos, Pos_next in zip(p, p_next)]).transpose()
Line_y = np.array([[Pos[2], Pos_next[2]] for Pos, Pos_next in zip(p, p_next)]).transpose()

# Setup the simulation
Sim = EM.sim(N, delta_x = delta_x, x0 = x0, approx_n = approx_n, J = J, boundaries = Boundaries)

# Define the resolutions
Res_scalar = 1000
Res_vector = 30
Res_line = 1000

# Define hat vectors
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 0, 1])
hat_loop = np.array([(Pos_next - Pos) / np.sqrt(np.sum((Pos_next - Pos) ** 2)) for Pos, Pos_next in zip(p, p_next)])
hat_loop = np.repeat(hat_loop.reshape((1,) + hat_loop.shape), Res_line, axis = 0)

# Define extents
extent = [x0[0], x0[0] + delta_x[0], x0[2], x0[2] + delta_x[2]]
PointsSize = np.array([delta_x[0], delta_x[2]])

# Get grid points
Points_scalar = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_scalar, Res_scalar]))
Points_vector = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_vector, Res_vector]))
Points_loop = np.array([EM.sample_points_line(Pos, Pos_next, Res_line) for Pos, Pos_next in zip(p, p_next)]).transpose((1, 2, 0))
x0_loop = 0
x_loop = []
for p1, p2 in zip(p, p_next):
    x1_loop = x0_loop + np.sqrt(np.sum((p2 - p1) ** 2))
    x_loop.append(np.linspace(x0_loop, x1_loop, Res_line))
    x0_loop = x1_loop
x_loop = np.array(x_loop).transpose((1, 0))

# Setup samplers
Sampler_E_2D = EM.sampler_E_vector(Sim, Points_vector, x_hat, y_hat)
Sampler_V_2D = EM.sampler_V_scalar(Sim, Points_scalar)
Sampler_Rho_2D = EM.sampler_Rho_scalar(Sim, Points_scalar)
Sampler_V_loop = EM.sampler_V_line(Sim, Points_loop, x = x_loop)
Sampler_E_loop = EM.sampler_E_line(Sim, Points_loop, hat_loop, x = x_loop)

# Solve the statics problem
print("Solving")
StaticTime = Sim.solve(exact = Exact, progress = Progress)
print(f"Solved starting conditions in {StaticTime:.2g} s")

# Create the images
fig_E_2D, ax_E_2D, _ = Sampler_E_2D.plot(0, extent = extent)
ax_E_2D.plot(Line_x, Line_y)
if Save is True:
    fig_E_2D.savefig(Name_E_2D)

fig_V_2D, _, _ = Sampler_V_2D.plot(0, extent = extent, use_contour = True, levels = 15)
if Save is True:
    fig_V_2D.savefig(Name_V_2D)

fig_Rho_2D, _, _ = Sampler_Rho_2D.plot(0, extent = extent)
if Save is True:
    fig_Rho_2D.savefig(Name_Rho_2D)

fig_V_loop, ax_V_loop, _ = Sampler_V_loop.plot(0)
ax_V_loop.set_title("Potential on rectangular wire")
ax_V_loop.set_xlabel("Distance")
ax_V_loop.set_ylabel("Electric potential")
if Save is True:
    fig_V_loop.savefig(Name_V_loop)

fig_E_loop, ax_E_loop, _ = Sampler_E_loop.plot(0)
ax_E_loop.set_title("Electromotive force per unit length on rectangular wire")
ax_E_loop.set_xlabel("Distance")
ax_E_loop.set_ylabel("Electromotive for per unit length (Strength of E along wire)")
if Save is True:
    fig_E_loop.savefig(Name_E_loop)