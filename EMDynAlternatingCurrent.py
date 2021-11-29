import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt

plt.close("all")

# Constants for J
MaxCurrent = 1
Frequency = 0.2

# Grid constants
N = np.array([99, 99, 1], dtype = int)
delta_x = np.array([25, 25, 1])
x0 = np.array([-12.5, -12.5, 0])

# Video constants
FPS = 30
Speed = 2
Delta_t = 25
TimeConstant = 5
Steps = int(FPS / Speed * Delta_t)
SubSteps = int(TimeConstant * np.max(N) / Steps) + 1
dt = Delta_t / (Steps * SubSteps)

# Evaluation constants
Exact = True
Progress = 5

# File names
Name_A_2D = "ACRadiation_A_2D.avi"
Name_A_1D = "ACRadiation_A_1D.avi"
Name_B_2D = "ACRadiation_B_2D.avi"
Name_B_1D = "ACRadiation_B_1D.avi"

# Define the current
def J(dx, N, x0, c, mu0):
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current, normalising so the current is the same no matter the grid size
    Grid[int(N[0] / 2), int(N[1] / 2), :, 3] = MaxCurrent / (dx[0] * dx[1])
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return a sin times this vector
    def get_J(t):
        return J_Vector * np.sin(2 * np.pi * Frequency * t)
    
    return get_J

# Setup the simulation
Sim = EM.sim(N, dt = dt, delta_x = delta_x, x0 = x0, J = J, boundaries = [["closed", "closed"], ["closed", "closed"], "periodic"])

# Set clim
max_val_A = 0.5
max_val_B = 0.2
clim_A = np.array([-max_val_A, max_val_A])
clim_B = np.array([-max_val_B, max_val_B])

# Define hat vectors
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 1, 0])
hat = np.array([0, 0, 1])
B_hat = np.array([0, -1, 0])

# Define the resolutions
Res_scalar = 100
Res_vector = 30
Res_line = 1000

# Define extents
extent_2D = [x0[0], x0[0] + delta_x[0], x0[1], x0[1] + delta_x[1]]
extent_1D = [0, x0[0] + delta_x[0], -1, 1]

# Get grid points
Points_scalar = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), delta_x[:2], np.array([Res_scalar, Res_scalar]))
Points_vector = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), delta_x[:2], np.array([Res_vector, Res_vector]))
Points_line = EM.sample_points_line(np.array([extent_1D[0], 0, 0]), np.array([extent_1D[1], 0, 0]), Res_line)

# Setup samplers
Sampler_A_2D = EM.sampler_A_scalar(Sim, Points_scalar, hat = hat)
Sampler_A_1D = EM.sampler_A_line(Sim, Points_line, hat)
Sampler_B_2D = EM.sampler_B_vector(Sim, Points_vector, x_hat, y_hat)
Sampler_B_1D = EM.sampler_B_line(Sim, Points_line, B_hat)

# Solve the statics problem
print("Solving starting conditions")
StaticTime = Sim.solve(exact = True, progress = Progress)
print(f"Solved starting conditions in {StaticTime:.2g} s")

# Solve the dynamics
print("Simulating dynamics")
DynamicTime = Sim.dynamics(Steps, SubSteps, exact = Exact, progress = Progress)
print(f"Solved dynamics in {DynamicTime:.2g} s")

# Create the videos
Sampler_A_2D.make_video(Name_A_2D, FPS = FPS, extent = extent_2D, clim = clim_A)
print(f"Finished video {Name_A_2D:s}")
Sampler_A_1D.make_video(Name_A_1D, FPS = FPS, extent = extent_1D)
print(f"Finished video {Name_A_1D:s}")
Sampler_B_2D.make_video(Name_B_2D, FPS = FPS, extent = extent_2D, clim = clim_B)
print(f"Finished video {Name_B_2D:s}")
Sampler_B_1D.make_video(Name_B_1D, FPS = FPS, extent = extent_1D)
print(f"Finished video {Name_B_1D:s}")