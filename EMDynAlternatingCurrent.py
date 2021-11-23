import numpy as np
import KUEM as EM
import matplotlib.pyplot as plt
import time

plt.close("all")

# Define some constants
MaxCurrent = 1
Frequency = 0.2
N = np.array([199, 199, 1], dtype = int)
delta_x = np.array([25, 25, 1])
x0 = np.array([-12.5, -12.5, 0])
FPS = 30
Speed = 2
Delta_t = 50
TimeConstant = 10
Steps = int(FPS / Speed * Delta_t)
SubSteps = int(TimeConstant * np.max(N) / Steps) + 1
dt = Delta_t / (Steps * SubSteps)
Exact = True
Progress = 5
Name = "AlternatingCurrentA"

# Define the current
def J(dx, N, x0, c, mu0):
    global Omega
    
    # Create grid
    Grid = np.zeros(tuple(N) + (4,))
    
    # Add in the current
    Grid[int(N[0] / 2), int(N[1] / 2), :, 3] = MaxCurrent / (dx[0] * dx[1])
    
    # Turn into a vector
    J_Vector = EM.to_vector(Grid, N)
    
    # Return a sin times this vector
    def get_J(t):
        return J_Vector * np.sin(2 * np.pi * Frequency * t)
    
    return get_J

# Setup the simulation
Sim = EM.sim(N, dt = dt, delta_x = delta_x, x0 = x0, J = J, boundaries = [["flat", "flat"], ["flat", "flat"], "periodic"])

# Solve the statics problem
#print("Solving starting conditions")
#print("Solved starting conditions in %.2g s" %(Sim.solve(exact = True, progress = Progress)))

# Get points for viewing
Res = 100
x_hat = np.array([1, 0, 0])
y_hat = np.array([0, 1, 0])
extent = [x0[0], x0[0] + delta_x[0], x0[1], x0[1] + delta_x[1]]
Points = EM.sample_points_plane(x_hat, y_hat, np.array([0, 0, 0]), delta_x[:2], np.array([Res, Res]))
max_val = 0.3

# Plot starting conditions
#vx, vy = Sim.sample_vectors(Sim.get_B(), Points, x_hat, y_hat)
#EM.plot_vector(vx, vy, extent = [x0[0], x0[0] + delta_x[0], x0[1], x0[1] + delta_x[1]])
#plt.show()

# Create the video
Video = EM.video(Name, FPS = FPS)
Video.plot_scalar(Sim.sample_values(Sim.get_A()[:, 2], Points),  clim = np.array([-max_val, max_val]))
Video.update()

# Loop
print("Simulating dynamics")

Time = time.time()
OriginalTime = Time

for n in range(Steps):
    NewTime = time.time()
    if NewTime > Time + Progress:
        Time = NewTime
        print("%.2g s remaining" %((Time - OriginalTime) / n * (Steps - n)))

    # Update simulation        
    for _ in range(SubSteps):
        Sim.step(exact = Exact)
        
    # Update plot
    Video.update_scalar(Sim.sample_values(Sim.get_A()[:, 2], Points))
    
    # Update video
    Video.update()
    
# Finish video
Video.finish()