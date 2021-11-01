import numpy as np
import matplotlib.pyplot as plt
import EMStatics as EM

plt.close("all")

Charge = 1
Size = np.array([100, 100, 100], dtype = int)
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

# Get points to sample over
#x = np.linspace(0.1, 0.9, 1000)
#X, Y = np.meshgrid(x, x)
#Points = np.empty((3,) + X.shape)
#Points[0] = X
#Points[1] = Y
#Points[2] = 0.5

def scale(x):
    return np.log(x)

# Plot
EM.plot_scalar(Sim.get_V(), Points, delta_x / (Size + 1), Size, np.array([-1, -1, -1], dtype = float), extent = [-0.5, 0.5, -0.5, 0.5], scale = scale)
plt.show()