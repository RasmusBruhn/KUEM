import sys
import os
sys.path.append('..')
import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
Current = 1

# Grid constants
N = np.array([49, 49, 1], dtype = int)
delta_x = np.array([2, 2, 2])
x0 = np.array([-1, -1, -1])
Boundaries = [["closed", "closed"], ["closed", "closed"], "periodic"]

# Evaluation constants
Exact = True
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
ContourLim = (0, 0.15)


# File names
FilePos = "LineCurrent/"
Name_B_2D = "ExLineCurrentB.png"
Name_A_2D = "ExLineCurrentA.png"
Name_J_2D = "ExLineCurrentJ.png"
Save = True

# Define the current
def J(dx, N, x0, c, mu0):
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current, normalising so the current is the same no matter the grid size
    Grid[int(N[0] / 2), int(N[1] / 2), :, 3] = Current / np.prod(dx)
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return the vector
    def get_J(t):
        return J_Vector
    
    return get_J

# Setup the simulation
Sim = EM.sim(N, delta_x = delta_x, x0 = x0, approx_n = approx_n, J = J, boundaries = Boundaries)

# Define hat vectors
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 1, 0])
hat = np.array([0, 0, 1])

# Define the resolutions
Res_scalar = 1000
Res_vector = 30

# Define extents
extent = [0, delta_x[0], 0, delta_x[1]]
PointsSize = np.array([delta_x[0], delta_x[1]])

# Get grid points
Points_scalar = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_scalar, Res_scalar]))
Points_vector = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_vector, Res_vector]))

# Setup samplers
Sampler_B_2D = EM.sampler_B_vector(Sim, Points_vector, x_hat, y_hat)
Sampler_A_2D = EM.sampler_A_scalar(Sim, Points_scalar, hat = hat)
Sampler_J_2D = EM.sampler_J_scalar(Sim, Points_scalar, hat = hat)

# Solve the statics problem
print("Solving")
StaticTime = Sim.solve(exact = Exact, progress = Progress)
print(f"Solved in {StaticTime:.2g} s")

# Create the images
if Save is True and not os.path.exists(FilePos):
    os.mkdir(FilePos)

fig_B_2D, _, _ = Sampler_B_2D.plot(0, extent = extent, scale = np.log, density = StreamDensity, length = StreamLength, use_vector = PlotVector, use_streams = PlotStreams)
if Save is True:
    fig_B_2D.savefig(FilePos + Name_B_2D)

fig_A_2D, _, _ = Sampler_A_2D.plot(0, extent = extent, scale = np.log, contour_lim = ContourLim, levels = ContourLevels, use_scalar = PlotScalar, use_contour = PlotContour)
if Save is True:
    fig_A_2D.savefig(FilePos + Name_A_2D)

fig_J_2D, _, _ = Sampler_J_2D.plot(0, extent = extent)
if Save is True:
    fig_J_2D.savefig(FilePos + Name_J_2D)