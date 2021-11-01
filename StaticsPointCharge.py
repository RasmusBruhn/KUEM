import numpy as np
import matplotlib.pyplot as plt
import EMStatics as EM

plt.close("all")

Charge = 1
Size = np.array([50, 50, 50], dtype = int)
approx_n = 100
exact = False
x0 = np.array([-1, -1, -1], dtype = float)
delta_x = np.array([2, 2, 2], dtype = float)

# Create J function
def J(dx, N, x0, c, mu0):
    Pos = np.array(N / 2, dtype = int)
    RealPos = EM.get_vector_index(Pos, N)
    J = np.zeros((np.prod(N), 4))
    J[RealPos, 0] = Charge

    def GetJ():
        return J
    
    return GetJ

# Create simulation class
Sim = EM.sim(Size, delta_x = delta_x, x0 = x0, approx_n = approx_n, J = J)

# Estimate the run time
print("Expected run time: %.2g s" %(Sim.estimate_solve_time(exact = exact)))

# Solve the system
print("Solve time = %.2g s" %(Sim.solve(exact = exact)))

# Get points to sample over
Points = EM.sample_points_plane(np.array([1, 0, 0], dtype = float), np.array([0, 1, 0], dtype = float), np.array([0, 0, 0], dtype = float), np.array([1, 1], dtype = float), np.array([1000, 1000]))

def scale(x):
    return np.log(x)

# Plot
fig, _, _ = EM.plot_scalar(Sim.get_V(), Points, delta_x / (Size + 1), Size, x0, extent = [-0.5, 0.5, -0.5, 0.5], scale = scale)
fig.show()

# Get points to sample over
Points2 = EM.sample_points_line(np.array([0, 0, 0], dtype = float), np.array([0.8, 0, 0], dtype = float), 1000)

# Plot
fig2, ax2, _ = EM.plot_1D(Sim.get_V(), Points2, delta_x / (Size + 1), Size, x0, extent = [0, 0.8])
fig2.show()

