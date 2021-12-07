import sys
import os
sys.path.append('..')
import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
Charge = 1
d = 0.4

# Grid constants
N = np.array([49, 49, 49], dtype = int)
delta_x = np.array([2, 2, 2])
x0 = np.array([-1, -1, -1])
Boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]

# Evaluation constants
Exact = False
Progress = 5
approx_n = 0.1

# Plotting settings
PlotScalar = True
PlotContour = True
PlotVector = False
PlotStreams = True

StreamDensity = 2
StreamLength = 1
ContourLevels = 10
ContourLim = (-0.4, 0.4)

# File names
FilePos = "ElectricDipole/"
Name_E_2D = "ExElectricDipoleE.png"
Name_V_2D = "ExElectricDipoleV.png"
Name_Rho_2D = "ExElectricDipoleRho.png"
Save = True

# Define the current
def J(dx, N, x0, c, mu0):
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current, normalising so the current is the same no matter the grid size
    Grid[int(N[0] / 2), int(N[1] / 2), int(N[2] * (1 - d) / 2), 0] = c * Charge / np.prod(dx)
    Grid[int(N[0] / 2), int(N[1] / 2), int(N[2] * (1 + d) / 2), 0] = -c * Charge / np.prod(dx)
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return a sin times this vector
    def get_J(t):
        return J_Vector
    
    return get_J

# Define the scale function
Scale_k = 0.04
def Scale(x):
    k = (np.max(x) - np.min(x)) / 2 * Scale_k
    m = (np.max(x) + np.min(x)) / 2
    return np.tanh((x - m) / k)

# Setup the simulation
Sim = EM.sim(N, delta_x = delta_x, x0 = x0, approx_n = approx_n, J = J, boundaries = Boundaries)

# Define hat vectors
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 0, 1])

# Define the resolutions
Res_scalar = 1000
Res_vector = 30

# Define extents
extent = [0, delta_x[0], 0, delta_x[2]]
PointsSize = np.array([delta_x[0], delta_x[2]])

# Get grid points
Points_scalar = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_scalar, Res_scalar]))
Points_vector = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_vector, Res_vector]))

# Setup samplers
Sampler_E_2D = EM.sampler_E_vector(Sim, Points_vector, x_hat, y_hat)
Sampler_V_2D = EM.sampler_V_scalar(Sim, Points_scalar)
Sampler_Rho_2D = EM.sampler_Rho_scalar(Sim, Points_scalar)

# Solve the statics problem
print("Solving")
StaticTime = Sim.solve(exact = Exact, progress = Progress)
print(f"Solved starting conditions in {StaticTime:.2g} s")

# Create the images
if Save is True:
    os.mkdir(FilePos)

fig_E_2D, _, _ = Sampler_E_2D.plot(0, extent = extent, scale = np.log, use_vector = PlotVector, use_streams = PlotStreams, density = StreamDensity, length = StreamLength)
if Save is True:
    fig_E_2D.savefig(FilePos + Name_E_2D)

fig_V_2D, _, _ = Sampler_V_2D.plot(0, extent = extent, scale = Scale, contour_lim = ContourLim, use_scalar = PlotScalar, use_contour = PlotContour)
if Save is True:
    fig_V_2D.savefig(FilePos + Name_V_2D)

fig_Rho_2D, _, _ = Sampler_Rho_2D.plot(0, extent = extent)
if Save is True:
    fig_Rho_2D.savefig(FilePos + Name_Rho_2D)