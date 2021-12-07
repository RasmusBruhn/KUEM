import EMSimClass as EM
import numpy as np
import time as t

Size = 25
Exact = False

ShowCount = 25
base_dx = np.array([25, 25, 25], dtype = float)
base_N = np.array([1, 1, 1], dtype = int)
base_approx_n = 25

def J_PointCharge(dx, dt, N, x0, c, mu0):
    J = np.zeros((np.prod(N), 4))
    J[int(N[0] / 2) + int(N[1] / 2) * N[0] + int(N[2] / 2) * N[0] * N[1], 0] = 1 / np.prod(dx)
    
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
Index2 = np.floor(np.linspace(0, 1, ShowCount, endpoint = False) * N[1])
Index1, Index2 = np.meshgrid(Index1, Index2)
x = np.empty((ShowCount, ShowCount, 3))
x[:, :, 0] = x0[0] + Index1 * dx[0]
x[:, :, 1] = x0[1] + Index2 * dx[1]
x[:, :, 2] = 0

Sim = EM.sim(dx, 1, N, J = J_PointCharge, approx_n = base_approx_n * Size, x0 = x0)

print("Estimated time: " + str(Sim.estimate_static(exact = Exact)))

time1 = t.time()

Sim.static(exact = Exact)

time2 = t.time()

print("Time: " + str(time2 - time1))

ShowCount = 25

Sim.plot_E(x, axis1 = x[:, :, 0], axis2 = x[:, :, 1], clim = (0.0001, 0.1))
Sim.plot_V(x, scale = np.log)