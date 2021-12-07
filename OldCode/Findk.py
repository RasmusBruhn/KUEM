import EMStatics as EM
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
plt.close("all")

def ConvergenceTest(CorrectA, J, Laplacian, k, n_vals):
    # Make space for the result to be saved
    Values = np.empty(len(n_vals))
    
    # Create the initial conditions
    A = np.zeros_like(J)
    Lastn = 0
    
    # Loop through all values of n
    for i, n in enumerate(n_vals):
        # Solve the system
        A = EM.solve_approx(J, Laplacian, 1, A, n - Lastn, k)
        Lastn = n
        
        # Find the error
        Values[i] = np.sum((A - CorrectA) ** 2)
        
    return Values

def ConvergenceVolumeTest(J, k, N_vals, Threshold, n_max = 1000):
    Values = np.empty(N_vals.shape[0])
    delta_x = np.array([1, 1, 1], dtype = float)
    
    for i in range(N_vals.shape[0]):
        N_use = N_vals[i, :]
        print(N_use)
        dx_use = delta_x / (N_use + 1)
        J_use = J(dx_use, N_use, 1, 1, 1)()
        lapl_use = EM.get_lapl(dx_use, N_use)
        k_use = -1 / lapl_use[0, 0] * k
        Exact = EM.solve_exact(J_use, lapl_use, 1)
        A = np.ones_like(J_use)
        n_use = n_max
        
        for n in range(n_max):
            if np.sum((A - Exact) ** 2) / np.prod(N_use) <= Threshold:
                n_use = n
                break
            
            A = EM.solve_approx(J_use, lapl_use, 1, A, 1, k_use)
            
        Values[i] = n_use
        
    return Values

def ConvergenceKTest(CorrectA, J, Laplacian, k_vals, n):
    # Make space for the result to be saved
    Values = np.empty(len(k_vals))
    
    # Loop through all values of n
    for i, k in enumerate(k_vals):
        # Create the initial conditions
        A = np.zeros_like(J)
    
        print(i)
        # Solve the system
        A = EM.solve_approx(J, Laplacian, 1, A, n, k)
        
        # Find the error
        Values[i] = np.sum((A - CorrectA) ** 2)
        
    return Values


N = np.array([1, 1, 1], dtype = int) * 30
delta_x = np.array([1, 1, 1], dtype = float)
dx = delta_x / (N + 1)
Charge = 1

def J(dx, N, x0, c, mu0):
    Pos = np.array(N / 2, dtype = int)
    RealPos = EM.get_vector_index(Pos, N)
    J = np.zeros((np.prod(N), 4))
    J[RealPos, 0] = Charge / np.prod(dx)

    def GetJ():
        return J
    
    return GetJ

def Fit(x, a, b):
    return a * np.exp(-b * x)

def Fit2(x, a0, a, b, c, aa, bb, cc, ab, bc, ca):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return a0 + a * x1 + b * x2 + c * x3 + aa * x1 ** 2 + bb * x2 ** 2 + cc * x3 ** 2 + ab * x1 * x2 + bc * x2 * x3 + ca * x3 * x1

def Fit3(x, a0, a):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return a0 + a * np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)


Laplacian = EM.get_lapl(dx, N)
JVector = J(dx, N, 0, 0, 0)()
CorrectA = EM.solve_exact(JVector, Laplacian, 1)
k = -1 / Laplacian[0, 0]
n = np.linspace(1, 1000, 1000, dtype = int)
"""
Values = ConvergenceTest(CorrectA, JVector, Laplacian, k, n) / np.prod(N)

Vals, Cov = opt.curve_fit(Fit, n, Values)

plt.figure()
plt.plot(n, Values, "-")
plt.plot(n, Fit(n, *Vals), "-")
"""

"""
N_list = np.linspace(5, 15, 11, dtype = int)
N1, N2, N3 = np.meshgrid(N_list, N_list, N_list, indexing = "ij")
N1, N2, N3 = N1.reshape((-1, 1)), N2.reshape((-1, 1)), N3.reshape((-1, 1))
N_finallist = np.concatenate((N1, N2, N3), axis = -1)

Values2 = ConvergenceVolumeTest(J, 1, N_finallist, 0.01)
Vals2, Cov2 = opt.curve_fit(Fit2, N_finallist, Values2, maxfev = 5000)

plt.figure()
plt.plot(N_finallist[(N_finallist[:, 0] == 5) & (N_finallist[:, 1] == 5), 2], Values2[(N_finallist[:, 0] == 5) & (N_finallist[:, 1] == 5)], "-", label = "Data")
plt.plot(N_finallist[(N_finallist[:, 0] == 5) & (N_finallist[:, 1] == 5), 2], Fit2(N_finallist[(N_finallist[:, 0] == 5) & (N_finallist[:, 1] == 5), :], *Vals2), "-", label = "Fit")
plt.legend()
plt.figure()
plt.plot(N_finallist[(N_finallist[:, 0] == 10) & (N_finallist[:, 1] == 10), 2], Values2[(N_finallist[:, 0] == 10) & (N_finallist[:, 1] == 10)], "-", label = "Data")
plt.plot(N_finallist[(N_finallist[:, 0] == 10) & (N_finallist[:, 1] == 10), 2], Fit2(N_finallist[(N_finallist[:, 0] == 10) & (N_finallist[:, 1] == 10), :], *Vals2), "-", label = "Fit")
plt.legend()
plt.figure()
plt.plot(N_finallist[(N_finallist[:, 0] == 15) & (N_finallist[:, 1] == 15), 2], Values2[(N_finallist[:, 0] == 15) & (N_finallist[:, 1] == 15)], "-", label = "Data")
plt.plot(N_finallist[(N_finallist[:, 0] == 15) & (N_finallist[:, 1] == 15), 2], Fit2(N_finallist[(N_finallist[:, 0] == 15) & (N_finallist[:, 1] == 15), :], *Vals2), "-", label = "Fit")
plt.legend()
#plt.figure()
#plt.plot(N_finallist[(N_finallist[:, 0] == 20) & (N_finallist[:, 1] == 20), 2], Values2[(N_finallist[:, 0] == 20) & (N_finallist[:, 1] == 20)], "-", label = "Data")
#plt.plot(N_finallist[(N_finallist[:, 0] == 20) & (N_finallist[:, 1] == 20), 2], Fit2(N_finallist[(N_finallist[:, 0] == 20) & (N_finallist[:, 1] == 20), :], *Vals2), "-", label = "Fit")
#plt.legend()
#plt.figure()
#plt.plot(N_finallist[(N_finallist[:, 0] == 25) & (N_finallist[:, 1] == 25), 2], Values2[(N_finallist[:, 0] == 25) & (N_finallist[:, 1] == 25)], "-", label = "Data")
#plt.plot(N_finallist[(N_finallist[:, 0] == 25) & (N_finallist[:, 1] == 25), 2], Fit2(N_finallist[(N_finallist[:, 0] == 25) & (N_finallist[:, 1] == 25), :], *Vals2), "-", label = "Fit")
#plt.legend()
print(Vals2)
"""


Range = (0.1, 1.1)
RangeVals = np.linspace(Range[0], Range[1], 200)
k_vals = - 1 / Laplacian[0, 0] * RangeVals
ValuesForK = ConvergenceKTest(CorrectA, JVector, Laplacian, k_vals, 100)

plt.figure()
plt.plot(RangeVals, np.log(ValuesForK), "-")
plt.ylim(4, 6)
plt.xlabel("$k_r$")
plt.ylabel("log(Error)")
plt.title("The error as a function of k")
plt.savefig("FindK.png")

print(RangeVals[np.argmin(ValuesForK)])

plt.show()