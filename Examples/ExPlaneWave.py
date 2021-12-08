import sys
import os
sys.path.append('..')
import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
Amplitude = 1
Frequency = 2

# Grid constants
N = np.array([1, 1, 1000], dtype = int)
delta_x = np.array([2, 2, 1])
x0 = np.array([-1, -1, 0])
Boundaries = ["periodic", "periodic", ["closed", "closed"]]

# Evaluation constants
StaticsExact = True
DynamicsExact = True
Progress = 5
approx_n = 0.1

# Video constants
FPS = 30
Speed = 0.2
Delta_t = 10
TimeConstant = 5
Steps = int(FPS / Speed * Delta_t)
SubSteps = int(np.ceil(TimeConstant * Delta_t * np.max(N / delta_x) / Steps))
dt = Delta_t / (Steps * SubSteps)

# Plotting settings
PlotScalar = True
PlotContour = False
PlotVector = False
PlotStreams = True

StreamDensity = 2
StreamLength = 1
ContourLevels = 10
ContourLim = (0, 0.15)


# File names
FilePos = "PlaneWave/"
Name_B_1D = "ExPlaneWaveB.avi"
Name_A_1D = "ExPlaneWaveA.avi"
Save = True

# Create the plane wave boundary condition
def C(dx, N, x0, c, mu0):
    # Create an empty C
    C = np.zeros(tuple(N) + (4, 3, 2), dtype = float)

    # Add the plane wave
    C[:, :, 0, 1, 2, 0] = Amplitude
    
    # Turn into vector
    C = EM.to_vector(C, N)

    # Create the function to return the conditions
    # t:        The time
    def GetC(t):
        return C * np.cos(2 * np.pi * Frequency * t)
    
    return GetC


# Define the vacuum
def J(dx, N, x0, c, mu0):
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return the vector
    def get_J(t):
        return J_Vector
    
    return get_J

# Setup the simulation
Sim = EM.sim(N, delta_x = delta_x, x0 = x0, approx_n = approx_n, dt = dt, J = J, boundaries = Boundaries, C = C)

# Set clim
max_val_A = 1.2
max_val_B = 20
clim_A = np.array([-max_val_A, max_val_A])
clim_B = np.array([-max_val_B, max_val_B])

# Define hat vectors
hat = np.array([1, 0, 0])
B_hat = np.array([0, -1, 0])

# Define the resolutions
Res_line = 1000

# Define extents
x_vals = np.linspace(0, delta_x[2], Res_line)

# Get grid points
Points_line = EM.sample_points_line(np.array([0, 0, 0]), np.array([0, 0, delta_x[2]]), Res_line)

# Setup samplers
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
    Sampler_B_1D.make_video(FilePos + Name_B_1D, FPS = FPS, ylim = clim_B)
    print(f"Created video {Name_B_1D}")
    Sampler_A_1D.make_video(FilePos + Name_A_1D, FPS = FPS, ylim = clim_A)
    print(f"Created video {Name_A_1D}")