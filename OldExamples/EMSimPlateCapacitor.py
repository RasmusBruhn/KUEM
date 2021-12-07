import EMSimClass as EM
import numpy as np
import time as t

Size = 50
Exact = False

ShowCount = 25
base_dx = np.array([25, 25, 25], dtype = float)
base_N = np.array([1, 1, 1], dtype = int)
base_approx_n = 25

def J_PlateCapacitor(dx, dt, N, x0, c, mu0):
    Dist = 1 / 8
    Dist2 = 1 / 8
    
    J = np.zeros((np.prod(N), 4))
    
    xIndex = np.linspace(int(N[0] * Dist), int(N[0] * (1 - Dist)), int(N[0] * (1 - Dist)) - int(N[0] * Dist) + 1)
    yIndex = np.linspace(int(N[1] * Dist), int(N[1] * (1 - Dist)), int(N[1] * (1 - Dist)) - int(N[1] * Dist) + 1)
    xIndex, yIndex = np.meshgrid(xIndex, yIndex)
    n = np.empty(xIndex.shape + (3,), dtype = int)
    n[:, :, 0] = xIndex
    n[:, :, 1] = yIndex
    n[:, :, 2] = int(N[2] / 2) - int(N[2] * Dist2)
    J[EM.get_vector_index(n, N)] = -1 / dx[2]
    n[:, :, 2] = int(N[2] / 2) + int(N[2] * Dist2)
    J[EM.get_vector_index(n, N)] = 1 / dx[2]
    
    # Create function to return J
    # t: The time
    def GetJ(t):
        return J
    
    # Return the function
    return GetJ    

dx = base_dx / Size
N = base_N * Size
x0 = np.array(- (N - 1) / 2, dtype = float) * dx

Index1 = np.floor(np.linspace(0, 1, ShowCount, endpoint = False) * N[0])
Index2 = np.floor(np.linspace(0, 1, ShowCount, endpoint = False) * N[2])
Index1, Index2 = np.meshgrid(Index1, Index2)
x = np.empty((ShowCount, ShowCount, 3))
x[:, :, 0] = x0[0] + Index1 * dx[0]
x[:, :, 1] = 0
x[:, :, 2] = x0[2] + Index2 * dx[2]

Sim = EM.sim(dx, 1, N, J = J_PlateCapacitor, approx_n = base_approx_n * Size, x0 = x0)

print("Estimated time: " + str(Sim.estimate_static(exact = Exact)))

time1 = t.time()

Sim.static(exact = Exact)

time2 = t.time()

print("Time: " + str(time2 - time1))

ShowCount = 25

Sim.plot_E(x, axis1 = x[:, :, 0], axis2 = x[:, :, 2], dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 0, 1], dtype = float), clim = np.array([0.01, 0.6]), scale = EM.optional_scale_id)
Sim.plot_V(x, scale = EM.optional_scale_id)