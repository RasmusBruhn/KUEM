import os
import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
Current = 1
Frequency = 2

# Grid constants
N = np.array([49, 49, 1], dtype = int)
delta_x = np.array([2, 2, 2])
x0 = np.array([-1, -1, -1])
Boundaries = [["closed", "closed"], ["closed", "closed"], "periodic"]

# Evaluation constants
StaticsExact = True
DynamicsExact = True
Progress = 5
approx_n = 0.1

# Video constants
FPS = 30
Speed = 0.2
Delta_t = 5
TimeConstant = 5
Steps = int(FPS / Speed * Delta_t)
SubSteps = int(np.ceil(TimeConstant * Delta_t * np.max(N / delta_x) / Steps))
dt = Delta_t / (Steps * SubSteps)

# Plotting settings
PlotScalar = True
PlotContour = False
PlotVector = True
PlotStreams = False

StreamDensity = 2
StreamLength = 1
ContourLevels = 10
ContourLim = (0, 0.15)


# File names
FilePos = "AlternatingLineCurrent/"
Name_B_2D = "ExAlternatingLineCurrentB_2D.avi"
Name_A_2D = "ExAlternatingLineCurrentA_2D.avi"
Name_B_1D = "ExAlternatingLineCurrentB_1D.avi"
Name_A_1D = "ExAlternatingLineCurrentA_1D.avi"
Save = True

# Define the current
def J(dx, N, x0, c, mu0):
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current, normalising so the current is the same no matter the grid size
    Grid[int(N[0] / 2), int(N[1] / 2), :, 3] = Current / (dx[0] * dx[1])
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return the vector
    def get_J(t):
        return J_Vector * np.sin(2 * np.pi * Frequency * t)
    
    return get_J

# Setup the simulation
Sim = EM.sim(N, delta_x = delta_x, x0 = x0, approx_n = approx_n, dt = dt, J = J, boundaries = Boundaries)

# Set clim
max_val_A = 0.3
max_val_B = 2.5
clim_A = np.array([-max_val_A, max_val_A])
clim_B = np.array([-max_val_B, max_val_B])

# Define hat vectors
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 1, 0])
hat = np.array([0, 0, 1])
B_hat = np.array([0, -1, 0])

# Define the resolutions
Res_scalar = 1000
Res_vector = 30
Res_line = 1000

# Define extents
extent = [0, delta_x[0], 0, delta_x[1]]
PointsSize = np.array([delta_x[0], delta_x[1]])
x_vals = np.linspace(0, delta_x[0] / 2, Res_line)

# Get grid points
Points_scalar = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_scalar, Res_scalar]))
Points_vector = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), PointsSize, np.array([Res_vector, Res_vector]))
Points_line = EM.sample_points_line(np.array([0, 0, 0]), np.array([delta_x[0] / 2, 0, 0]), Res_line)

# Setup samplers
Sampler_B_2D = EM.sampler_B_vector(Sim, Points_vector, x_hat, y_hat)
Sampler_A_2D = EM.sampler_A_scalar(Sim, Points_scalar, hat = hat)
Sampler_B_1D = EM.sampler_B_line(Sim, Points_line, x = x_vals, hat = B_hat)
Sampler_A_1D = EM.sampler_A_line(Sim, Points_line, x = x_vals, hat = hat)

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
    Sampler_B_2D.make_video(FilePos + Name_B_2D, FPS = FPS, extent = extent, clim = clim_B, density = StreamDensity, length = StreamLength, use_vector = PlotVector, use_streams = PlotStreams)
    print(f"Created video {Name_B_2D}")
    Sampler_A_2D.make_video(FilePos + Name_A_2D, FPS = FPS, extent = extent, clim = clim_A, contour_lim = ContourLim, levels = ContourLevels, use_scalar = PlotScalar, use_contour = PlotContour)
    print(f"Created video {Name_A_2D}")
    Sampler_B_1D.make_video(FilePos + Name_B_1D, FPS = FPS, ylim = clim_B)
    print(f"Created video {Name_B_1D}")
    Sampler_A_1D.make_video(FilePos + Name_A_1D, FPS = FPS, ylim = clim_A)
    print(f"Created video {Name_A_1D}")