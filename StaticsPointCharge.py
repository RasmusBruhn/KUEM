import numpy as np
import matplotlib.pyplot as plt
import EMStatics as EM

plt.close("all")

Charge = 1
Size = np.array([49, 49, 49], dtype = int)
approx_n = 100
exact = False
x0 = np.array([-1, -1, -1], dtype = float)
delta_x = np.array([2, 2, 2], dtype = float)

# Create J function
def J(dx, N, x0, c, mu0):
    Pos = np.array(N / 2, dtype = int)
    RealPos = EM.get_vector_index(Pos, N)
    J = np.zeros((np.prod(N), 4))
    J[RealPos, 0] = Charge / np.prod(dx)

    def GetJ():
        return J
    
    return GetJ

# Create simulation class
Sim = EM.sim(Size, delta_x = delta_x, x0 = x0, approx_n = approx_n, J = J)

# Solve the system
print("Solve time = %.2g s" %(Sim.solve(exact = exact, progress = 1)))

def scale(x):
    return np.log(x)

# Get points to sample over
Width = np.array([2, 2], dtype = float)
Center = np.array([0, 0, 0], dtype = float)
Points = EM.sample_points_plane(np.array([1, 0, 0], dtype = float), np.array([0, 1, 0], dtype = float), Center, Width, np.array([1000, 1000]))
Values = Sim.sample_values(Sim.get_V(), Points)

# Plot V in xy plane
fig, _, _ = EM.plot_scalar(Values, extent = [Center[0] - Width[0] / 2, Center[0] + Width[0] / 2, Center[1] - Width[1] / 2, Center[1] + Width[1] / 2], scale = scale)
fig.show()

# Get points to sample over
Start = np.array([0, 0, 0], dtype = float)
End = np.array([1, 0, 0], dtype = float)
Points2 = EM.sample_points_line(Start, End, 1000)
Values2 = Sim.sample_values(Sim.get_V(), Points2)

# Plot V along x-axis
fig2, ax2, _ = EM.plot_1D(Values2, extent = [Start[0], End[0]])
ax2.plot(np.linspace(Start[0], End[0], 1000), 1 / (4 * np.pi * np.linspace(0.1, 1, 1000)), "-")
fig2.show()

# Get points to sample over
Points3 = EM.sample_points_plane(np.array([1, 0, 0], dtype = float), np.array([0, 1, 0], dtype = float), Center, Width, np.array([30, 30]))
vx3, vy3 = Sim.sample_vectors(Sim.get_E(), Points3, np.array([1, 0, 0], dtype = float), np.array([0, 1, 0], dtype = float))

# Plot E in xy plane
fig3, _, _ = EM.plot_vector(vx3, vy3, scale = scale, extent = [Center[0] - Width[0] / 2, Center[0] + Width[0] / 2, Center[1] - Width[1] / 2, Center[1] + Width[1] / 2])
fig3.show()
