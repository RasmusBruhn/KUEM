import KUEM as EM
import numpy as np
import time as t

# The size of the system
Size = 49

ShowCount = 25
base_dx = np.array([25, 25, 25], dtype = float)
base_N = np.array([1, 1, 1], dtype = int)
base_dt = 2.5
Omega = 2 * np.pi / 5
approx_n = 100
T = 10
TRes = 0.1

# -------------------------------------------------------------------------------

def J_LineCurrent(dx, N, x0, c, mu0):
    J = np.zeros((np.prod(N), 4))
    J[int(N[0] / 2) + 30 + (int(N[1] / 2) + 15) * N[0] :: N[0] * N[1], 3] = 1
    
    # Create function to return J
    # t: The time
    def GetJ(t):
        if Omega * t < 2 * np.pi:
            return J * np.sin(Omega * t)
        
        else:
            return J * 0
    
    # Return the function
    return GetJ    

dx = base_dx / (Size - 1)
dt = base_dt / (Size - 1)
N = base_N * Size
N[2] = 1
x0 = np.array(-base_dx / 2, dtype = float)
EvalCount = int(T / dt)
IntEvalCount = int(np.ceil(TRes / dt))

Index1 = np.floor(np.linspace(0, 1, ShowCount, endpoint = False) * N[0])
Index2 = np.floor(np.linspace(0, 1, ShowCount, endpoint = False) * N[1])
Index1, Index2 = np.meshgrid(Index1, Index2)
x = np.empty((ShowCount, ShowCount, 3))
x[:, :, 0] = x0[0] + Index1 * dx[0] * 1.0001
x[:, :, 1] = x0[1] + Index2 * dx[1] * 1.0001
x[:, :, 2] = 0

Sim = EM.sim(N, delta_x = base_dx, dt = dt, J = J_LineCurrent, approx_n = approx_n, x0 = x0, boundaries = [["flat", "flat"], ["flat", "flat"], "periodic"])

def LinScale(x):
    return x

#print("Estimated time: " + str(Sim.estimate_dynamics(exact = False) * EvalCount))

# Create video
Points = EM.sample_points_plane(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0]), base_dx, np.array([100, 100], dtype = int))
Video = EM.video("DynTest", FPS = 30)
Video.plot_scalar(Sim.sample_values(Sim.get_A()[:, 2], Points),  clim = np.array([-0.08, 0.08]))
#figB, axB, VideoB = EM.video_init("LineCurrentAlternatingB", FPS = 30)
#figS, axS, VideoS = EM.video_init("LineCurrentAlternatingS", FPS = 30)
#figA, axA, VideoA = EM.video_init("LineCurrentAlternatingA", FPS = 30)

#_, plotB = Sim.plot_B(x, axis1 = x[:, :, 0], axis2 = x[:, :, 1], clim = np.array([0.0005, 0.3]), ax = axB, video = True)
#_, plotS = Sim.plot_S(x, axis1 = x[:, :, 0], axis2 = x[:, :, 1], clim = np.array([0.00001, 0.01]), ax = axS, video = True)
#_, plotA = Sim.plot_AComp(x, 3, clim = np.array([-0.05, 0.05]), ax = axA, scale = LinScale)

time1 = t.time()

for i in range(int(EvalCount / IntEvalCount)):
    # Update potential
    for _ in range(IntEvalCount):
        Sim.step(exact = True)
    print(i / int(EvalCount / IntEvalCount))
    # Draw B field
    Video.update_scalar(Sim.sample_values(Sim.get_A()[:, 2], Points))
    #Sim.update_plot_B(plotB, x)
    #Sim.update_plot_S(plotS, x)
    #Sim.update_plot_AComp(plotA, x, 3, scale = LinScale)
    Video.update()
    # Update video
    #EM.video_update(figB, VideoB)
    #EM.video_update(figS, VideoS)
    #EM.video_update(figA, VideoA)

time2 = t.time()

print("Time: " + str(time2 - time1))

#Test.plot_B(x, axis1 = x[:, :, 0], axis2 = x[:, :, 1], clim = np.array([0.001, 0.1]), scale = scale)

Video.finish()
# Close video
#EM.video_finish(figB, VideoB)
#EM.video_finish(figS, VideoS)
#EM.video_finish(figA, VideoA)