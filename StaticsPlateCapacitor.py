import numpy as np
import matplotlib.pyplot as plt
import EMStatics as EM

plt.close("all")

SurfaceCharge = 1
Size = np.array([50, 50, 50], dtype = int)
approx_n = 0.1
exact = False
x0 = np.array([-1, -1, -1], dtype = float)
delta_x = np.array([2, 2, 2], dtype = float)

# Create J function
Length = 0.8
d = 0.2
def J(dx, N, x0, c, mu0):
    J_Array = np.zeros(tuple(N) + (4,))
    J_Array[int(N[0] * (1 - Length) / 2):int(N[0] * (1 + Length) / 2), int(N[0] * (1 - Length) / 2):int(N[0] * (1 + Length) / 2), int(N[2] * (1 - d) / 2), 0] = SurfaceCharge / dx[2]
    J_Array[int(N[0] * (1 - Length) / 2):int(N[0] * (1 + Length) / 2), int(N[0] * (1 - Length) / 2):int(N[0] * (1 + Length) / 2), int(N[2] * (1 + d) / 2), 0] = -SurfaceCharge / dx[2]
    J = EM.to_vector(J_Array, N)

    def GetJ(t):
        return J
    
    return GetJ

# Create simulation class
Sim = EM.sim(Size, delta_x = delta_x, x0 = x0, approx_n = approx_n, approx_k = 1, J = J, boundaries = [["open", "open"], ["open", "open"], ["open", "closed"]])

# Solve the system
print("Solve time = %.2g s" %(Sim.solve(exact = exact, progress = 1)))

def scale(x):
    return np.log(x)

# Get points to sample over
Width = np.array([2, 2], dtype = float)
Center = np.array([0, 0, 0], dtype = float)
Points = EM.sample_points_plane(np.array([1, 0, 0], dtype = float), np.array([0, 0, 1], dtype = float), Center, Width, np.array([1000, 1000]))
Values = Sim.sample_values(Sim.get_V(), Points)

# Plot V in xy plane
EM.plot_scalar(Values, extent = [Center[0] - Width[0] / 2, Center[0] + Width[0] / 2, Center[1] - Width[1] / 2, Center[1] + Width[1] / 2])

# Get points to sample over
Length = 2
Start = np.array([0, 0, -Length / 2], dtype = float)
End = np.array([0, 0, Length / 2], dtype = float)
Points2 = EM.sample_points_line(Start, End, 1000)
Values2 = Sim.sample_values(Sim.get_V(), Points2)

# Plot V along x-axis
EM.plot_1D(Values2, extent = [Start[2], End[2]])

# Get points to sample over
Points3 = EM.sample_points_plane(np.array([1, 0, 0], dtype = float), np.array([0, 0, 1], dtype = float), Center, Width, np.array([30, 30]))
vx3, vy3 = Sim.sample_vectors(Sim.get_E(), Points3, np.array([1, 0, 0], dtype = float), np.array([0, 0, 1], dtype = float))

# Plot E in xy plane
EM.plot_vector(vx3, vy3, extent = [Center[0] - Width[0] / 2, Center[0] + Width[0] / 2, Center[1] - Width[1] / 2, Center[1] + Width[1] / 2], cutoff = 0.01)

plt.show()