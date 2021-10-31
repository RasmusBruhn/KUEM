import numpy as np
import matplotlib.pyplot as plt
import EMStatics as EM

plt.close("all")

Charge = 1
Size = np.array([149, 149, 149], dtype = int)
approx_n = 100
exact = False

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
Sim = EM.sim(Size, approx_n = approx_n, J = J)

# Estimate the run time
print("Expected run time: %.2g s" %(Sim.estimate_solve_time(exact = exact)))

# Solve the system
print("Solve time = %.2g s" %(Sim.solve(exact = exact)))