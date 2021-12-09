import os
import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
Charge = 1
Frequency = 2
d = 0.1

# Grid constants
N = np.array([49, 49, 49], dtype = int)
delta_x = np.array([2, 2, 2])
x0 = np.array([-1, -1, -1])
Boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]

# Evaluation constants
StaticsExact = False
DynamicsExact = False
Progress = 5
approx_n = 0.1

# Video constants
FPS = 30
Speed = 0.1
Delta_t = 2
TimeConstant = 5
Steps = int(FPS / Speed * Delta_t)
SubSteps = int(np.ceil(TimeConstant * Delta_t * np.max(N / delta_x) / Steps))
dt = Delta_t / (Steps * SubSteps)

# Plotting settings
PlotScalar = True
PlotContour = False

ContourLevels = 10
ContourLim = (0, 0.15)


# File names
FilePos = "OscillatingDipole/"
Name_u_2D = "ExOscillatingDipole_u_2D.avi"
Save = True

# Define the current
def J(dx, N, x0, c, mu0):
    # Find the position of the 2 charges
    Pos1 = int(N[2] * (1 + d) / 2)
    Pos2 = int(N[2] * (1 - d) / 2)
    
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current, normalising so the current is the same no matter the grid size
    Grid[int(N[0] / 2), int(N[1] / 2), Pos1 + 1:Pos2, 3] = -Charge * 2 * np.pi * Frequency / (dx[0] * dx[1])
    
    # Add in the charges
    Grid[int(N[0] / 2), int(N[1] / 2), Pos1, 0] = c * Charge / np.prod(dx)
    Grid[int(N[0] / 2), int(N[1] / 2), Pos2, 0] = -c * Charge / np.prod(dx)
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return the vector
    def get_J(t):
        return J_Vector * np.array([np.cos(2 * np.pi * Frequency * t), 0, 0, np.sin(2 * np.pi * Frequency * t)]).reshape((1, -1))
    
    return get_J

# Setup the simulation
Sim = EM.sim(N, delta_x = delta_x, x0 = x0, approx_n = approx_n, dt = dt, J = J, boundaries = Boundaries)

# Set clim
max_val_u = 5
clim_u = np.array([0, max_val_u])

# Define hat vectors
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 0, 1])

# Define the resolutions
Res_scalar = 1000

# Define extents
extent = [x0[0],x0[0] + delta_x[0], x0[2], x0[2] + delta_x[2]]
PointsSize = np.array([delta_x[0], delta_x[2]])

# Get grid points
Points_scalar = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_scalar, Res_scalar]))

# Setup samplers
Sampler_u_2D = EM.sampler_u_scalar(Sim, Points_scalar)

# Solve the statics problem
print("Solving starting conditions")
StaticTime = Sim.solve(exact = StaticsExact, progress = Progress)
print(f"Solved starting conditions in {StaticTime:.2g} s")

# Solve the dynamics
print("Solving dynamics")
DynamicTime = Sim.dynamics(Steps, SubSteps, exact = DynamicsExact, progress = Progress)
print(f"Solved dynamics in {DynamicTime:.2g} s")

# Create the videos
if Save is True and not os.path.exists(FilePos):
    os.mkdir(FilePos)

if Save is True:
    print("Creating videos, this may take a while")
    Sampler_u_2D.make_video(FilePos + Name_u_2D, FPS = FPS, extent = extent, clim = clim_u, contour_lim = ContourLim, levels = ContourLevels, use_scalar = PlotScalar, use_contour = PlotContour)
    print(f"Created video {Name_u_2D}")
