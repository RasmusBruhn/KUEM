import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import cv2 as cv2
import scipy.sparse.linalg as slinalg
import time as time
import scipy.optimize as opt

# Default arguments:
# dx:           Numpy array of length 3 containing the difference of coordinates
#               between grid points
# dt:           The time interval used in the simulation
# N:            Numpy array of length 3 containing the number of grid points
# x0:           Numpy array of length 3 containing the minimum value of each coordinate
# c:            The speed of light
# mu0:          The permeability of free space

# Default vector arguments:
# grad:         Function to calculate gradient of scalar field
# div:          Function to calculate divergence of vector field
# curl:         Function to calculate curl of vector field
# laplacian:    Matrix to calculate laplacian
# c:            The speed of light
# mu0:          The permeability of free space

def get_vector_index(n, N):
    Factor = np.array([1, N[0], N[0] * N[1]])
    return np.sum(n * Factor, axis = -1)

def cartesian_get_grid_index(x, x0, dx):
    return np.array((x - x0 + dx / 2) / dx, dtype = int)

# Creates matrices for differentiating once, 
# returns a numpy array containing all 3 matrices
# Uses default arguments
def get_ddx(dx, dt, N, x0, c, mu0):
    # Get function values from neighbohring points
    df = np.empty(3, dtype = np.ndarray)
    for i in range(3):
        df[i] = np.ones(np.prod(N[:i + 1]))
        df[i][-np.prod(N[:i]):] = 0
        df[i] = np.tile(df[i], np.prod(N[i + 1:]))[:-np.prod(N[:i])]
    
    # Get derivatives
    ddx = np.empty(3, dtype = sparse.csr_matrix)
    for i in range(3):
        ddx[i] = sparse.diags([df[i], -df[i]], [np.prod(N[:i]), -np.prod(N[:i])], format = "csr") / (2 * dx[i])

    return ddx

    
# Creates matrices for differentiating twice, 
# returns a numpy array containing all 3 matrices
# Uses default arguments
def get_ddx2(dx, dt, N, x0, c, mu0):
    # Get function values from neighbohring points
    df = np.empty(3, dtype = np.ndarray)
    for i in range(3):
        df[i] = np.ones(np.prod(N[:i + 1]))
        df[i][-np.prod(N[:i]):] = 0
        df[i] = np.tile(df[i], np.prod(N[i + 1:]))[:-np.prod(N[:i])]
    
    # Get derivatives
    ddx2 = np.empty(3, dtype = sparse.csr_matrix)
    for i in range(3):
        ddx2[i] = sparse.diags([-2 * np.ones(np.prod(N)), df[i], df[i]], [0, np.prod(N[:i]), -np.prod(N[:i])], format = "csr") / (dx[i] ** 2)

    return ddx2

# Creates a function to take the gradient in cartesian coordinates
# Uses default arguments
def cartesian_grad(dx, dt, N, x0, c, mu0):
    # Get the diff matrices
    ddx = get_ddx(dx, dt, N, x0, c, mu0)
    
    # Calculate the gradient
    # Scalar: A scalar field of shape (N1 * N2 * N3)
    def calcGrad(Scalar):
        # Create empty vector field
        Result = np.empty((np.prod(N), 3))
        
        # Calculate result
        for i in range(3):
            Result[:, i] = ddx[i].dot(Scalar)
        
        # Return result
        return Result
    
    return calcGrad

# Creates a function to take the divergence in cartesian coordinates
# Uses default arguments
def cartesian_div(dx, dt, N, x0, c, mu0):
    # Get the diff matrices
    ddx = get_ddx(dx, dt, N, x0, c, mu0)
    
    # Calculate the divergence
    # Vector: A vector field to take the divergence of
    def calcDiv(Vector):
        # Calculate result
        return ddx[0].dot(Vector[:, 0]) + ddx[1].dot(Vector[:, 1]) + ddx[2].dot(Vector[:, 2])
    
    return calcDiv

# Creates a function to take the curl in cartesian coordinates
# Uses default arguments
def cartesian_curl(dx, dt, N, x0, c, mu0):
    # Get diff matrices
    ddx = get_ddx(dx, dt, N, x0, c, mu0)
    
    # Calculate the curl
    # Vector: A vector fied to take the curl of
    def calcCurl(Vector):
        # Create result array
        Result = np.empty((np.prod(N), 3))
        
        # Calculate curl
        for i in range(3):
            Result[:, i] = ddx[(i + 1) % 3].dot(Vector[:, (i + 2) % 3]) - ddx[(i + 2) % 3].dot(Vector[:, (i + 1) % 3])
            
        return Result
    
    return calcCurl

# Creates the laplacian matrix in cartesian coordinates
# Uses default arguments
def cartesian_laplacian(dx, dt, N, x0, c, mu0):
    # Get ddx2
    ddx2 = get_ddx2(dx, dt, N, x0, c, mu0)
    
    # Create laplacian
    return np.sum(ddx2)

# Creates a function which returns a current/charge density which is 0 everywhere
# Uses default arguments
def default_J(dx, dt, N, x0, c, mu0):
    # Create current which is 0 everywhere
    J = np.zeros((np.prod(N), 4))
    
    # Create function to return J
    # t: The time
    def GetJ(t):
        return J
    
    # Return the function
    return GetJ

# Solves non-responsive static exactly
#
# J:            The current/charge density
# Laplacian:    The laplacian matrix
# mu0:          Value for mu0
def solve_static_non_exact(J, laplacian, mu0):
    return slinalg.spsolve(laplacian, -mu0 * J)

# Solves non-responsive static approximately
#
# J:            The current/charge density
# Laplacian:    The laplacian matrix
# mu0:          Value for mu0
# A0:           The starting guess for the potential
# n:            The number of iterations for the approximation
# k:            The approximation parameter to make sure it does not diverge
def solve_static_non_approx(J, Laplacian, mu0, A0, n, k):
    A = A0
    Matrix = sparse.identity(J.shape[0], format = "csr") + k * Laplacian
    
    for _ in range(n):
        A = Matrix.dot(A) + mu0 * k * J
        
    return A

# Calculate R given A(t0), dAdt(t0) and J
def calc_R(A, dAdt, J, dt, c, mu0):
    return (c * dt) ** 2 / 8. * mu0 * J + dt / 2. * dAdt + A

# Get the next value of A
def solve_dynamics_non(A, dAdt, J, laplacian, dt, c, mu0, n, exact):
    # Get R
    R = calc_R(A, dAdt, J, dt, c, mu0)
    
    # Calculate matrix
    M = (c * dt) ** 2 / 8 * laplacian
    
    # Calculate A(t0 + 1/2 dt)
    if exact is True:
        HalfA = slinalg.spsolve(sparse.identity(A.shape[0], format = "csr") - M, R)
        
    else:
        HalfA = A + dt / 2. * dAdt
        
        for _ in range(n):
            HalfA = M.dot(HalfA) + R
    
    # Calculate the new values for the potential
    NewA = 4 * HalfA - dt * dAdt - 3 * A
    NewdAdt = (8 / dt * (HalfA - A) - 3 * dAdt)
    return (NewA, NewdAdt)

# Finds the E-field
#
# V:            The electric potential: A[:, 0] * c
# dAdt:         The vector potential differentiated with respect to time: dAdt[:, 1:]
# grad:         Function to calculate gradient
def calc_E(V, dAdt, grad):
    return -grad(V) - dAdt

# Finds the B-field
#
# A:            The vector potential: A[:, 1:]
# curl:         Function to calculate curl
def calc_B(A, curl):
    return curl(A)

# Finds the Poynting vector field
#
# E:            The electric field
# B:            The magnetic field
# And default vector arguments
def calc_S(E, B, mu0 = 1):
    return np.cross(E, B) / mu0

# Plot a vector field
#
# Field:        The vector field to plot from, should be a 2D array where axis 1 is the
#               different vector coponents
# x:            3D array, the first 2 axes are the different vectors to plot
#               The last axis is the different coordinates for the vector
# N:            The size of the field
# x0:           The starting position of the field
# dx:           The distance between points in the field
# axis1:        2D vector, the coordinate along the x-axis on the plot for each vector
# axis2:        2D vector, the coordinate along the y-axis on the plot for each vector
# dir1:         The unit vector which corresponds to the plot x-direction
# dir2:         The unit vector which corresponds to the plot y-direction
# scale:        Function to scale the length of the vectors
# ax:           The axes to draw the plot inside
# figsize:      The size of the figure if fig is not given
# dpi:          The resolution of the figure if fig is not given
# cmap:         The colour map to plot the vector field with
# clim:         Array containing the lengths of the vectors corresponding to the
#               lowest and highest values on the colour scale
def plot_vector(Field, x, N, x0 = np.array([0, 0, 0], dtype = float), dx = np.array([1, 1, 1], dtype = float), axis1 = None, axis2 = None, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None, video = False):
    # Get the indices for the vectors
    GridIndex = cartesian_get_grid_index(x, x0, dx)
    Index = get_vector_index(GridIndex, N)
    
    if axis1 is None:
        axis1 = np.linspace(0, 1, x.shape[0])
    
    if axis2 is None:
        axis2 = np.linspace(0, 1, x.shape[1])
        
    # Vector values
    VectorField = Field[Index]
    
    # Vector components
    vx = np.dot(VectorField, dir1)
    vy = np.dot(VectorField, dir2)
    
    # Calculate length of vectors
    vAbs = np.sqrt(vx ** 2 + vy ** 2)    

    # Caluclate clim
    if clim is None:
        clim = np.array([np.min(vAbs[vAbs != 0]), np.max(vAbs[vAbs != 0])], dtype = float)
        
    clim = scale(clim)

    if video is True:
        if np.all(vAbs == 0):
            vAbs[0, 0] = 1
            vx[0, 0] = 1

    vAbs[vAbs == 0] = np.nan
    
    # Normalize lengths
    vx /= vAbs
    vy /= vAbs
    
    # Create figure
    if ax is None:
        _, ax = plt.subplots(figsize = figsize, dpi = dpi)
        
    # Plot vectors
    Plot = ax.quiver(axis1, axis2, vx, vy, scale(vAbs), cmap = cmap, pivot = "middle", clim = clim)
    
    return (ax, Plot)

def update_plot_vector(plot, Field, x, N, x0 = np.array([0, 0, 0], dtype = float), dx = np.array([1, 1, 1], dtype = float), dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log):
    # Get the indices for the vectors
    GridIndex = cartesian_get_grid_index(x, x0, dx)
    Index = get_vector_index(GridIndex, N)
  
    # Vector values
    VectorField = Field[Index]
    
    # Vector components
    vx = np.dot(VectorField, dir1)
    vy = np.dot(VectorField, dir2)

    # Calculate length of vectors
    vAbs = np.sqrt(vx ** 2 + vy ** 2)
    
    # Abort if only 0 lengths
    if np.all(vAbs == 0):
        return

    vAbs[vAbs == 0] = np.nan

    # Normalize lengths
    vx /= vAbs
    vy /= vAbs

    # Update plot
    plot.set_UVC(vx, vy, scale(vAbs))


def default_scalar_scale(x):
    k = (np.max(x) - np.min(x)) / 2 * 0.1
    m = (np.max(x) + np.min(x)) / 2
    return np.tanh((x - m) / k)

def optional_scale_id(x):
    return x

# Plot a scalar field
#
# Field:        The scalar field to plot from, should be a 1D array containing the values
# x:            3D array, the first 2 axes are the different vectors to plot
#               The last axis is the different coordinates for the vector
# N:            The size of the field
# x0:           The starting position of the field
# dx:           The distance between points in the field
# scale:        Function to scale the length of the vectors
# ax:           The axes to draw the plot inside
# figsize:      The size of the figure if fig is not given
# dpi:          The resolution of the figure if fig is not given
# cmap:         The colour map to plot the vector field with
# clim:         Array containing the lengths of the vectors corresponding to the
#               lowest and highest values on the colour scale
def plot_scalar(Field, x, N, x0 = np.array([0, 0, 0], dtype = float), dx = np.array([1, 1, 1], dtype = float), scale = default_scalar_scale, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None):
    # Get the indices for the scalars
    GridIndex = cartesian_get_grid_index(x, x0, dx)
    Index = get_vector_index(GridIndex, N)
            
    # Scalar values
    ScalarField = Field[Index]
        
    # Caluclate clim
    if clim is None:
        clim = np.array([np.min(ScalarField), np.max(ScalarField)], dtype = float)
    
    clim = scale(clim)
    
    # Create figure
    if ax is None:
        _, ax = plt.subplots(figsize = figsize, dpi = dpi)
        
    # Plot vectors
    Plot = plt.imshow(scale(ScalarField), cmap = cmap, clim = clim, origin = "lower")
    
    return ax, Plot

def update_plot_scalar(plot, Field, x, N, x0 = np.array([0, 0, 0], dtype = float), dx = np.array([1, 1, 1], dtype = float), scale = default_scalar_scale):
    # Get the indices for the scalars
    GridIndex = cartesian_get_grid_index(x, x0, dx)
    Index = get_vector_index(GridIndex, N)
            
    # Scalar values
    ScalarField = Field[Index]
            
    # Plot vectors
    plot.set_array(scale(ScalarField))


# The simulation class, create this class to define your simulation,
# Then use it's methods to run the simulation and diagnistics/plotting
#
# Initialize:
# dx:           Array with the distance between grid points
# dt:           The time interval used in the simulation
# N:            Array with the number of grid points in each direction
# x0:           Array with the minimum value of the coordinates
# t0:           The starting time
# c:            The value of c, the speed of light
# mu0:          The value of mu0, the permeability of free space
# approx_n:     The number of times to run the approximation algorithm
# approx_k:     Approximation parameter to make sure it does not diverge
# init:         If true then the potentials will initialize to be 0 everywhere
#               If false it will not initialize
#               If it is a numpy array of the correct size then it will
#               use that as the starting condition, the shape should either be
#               (N3, N2, N1, 4) or (N1 * N2 * N3, 4) if it is another shape but
#               still size 4 * N1 * N2 * N3 then it will reshape it to (N1 * N2 * N3, 4)
#               but with no guarantee that it will format correctly 
# init_copy:    Only used if init is a numpy array, if true then it will copy
#               the array, if false then it will use and alter the original array
# init_ddt:     If true then the potentials differentiated with respect to t
#               will initialize to be 0 everywhere. If false it will not initialize
#               If it is a numpy array of the correct size then it will
#               use that as the starting condition, the shape should either be
#               (N3, N2, N1, 4) or (N1 * N2 * N3, 4) if it is another shape but
#               still size 4 * N1 * N2 * N3 then it will reshape it to (N1 * N2 * N3, 4)
#               but with no guarantee that it will format correctly 
# init_ddt_copy:Only used if init_ddt is a numpy array, if true then it will copy
#               the array, if false then it will use and alter the original array
# J:            The current and charge distribution, must be a function with default
#               arguments which returns a function to give the current and charge
#               densit at any time
# grad:         A function to return a function to calculate the gradient in
#               the coordinate system used
# div:          A function to return a function to calculate the divergence in
#               the coordinate system used
# curl:         A function to return a function to calculate the curl in
#               the coordinate system used
# laplacian:    A function to calculate the laplacian in the coordinate system used
class sim:
    def __init__(self, dx, dt, N, x0 = np.array([0, 0, 0], dtype = float), t0 = 0, c = 1, mu0 = 1, approx_n = 100, approx_k = 1, init = True, init_copy = False, init_ddt = True, init_ddt_copy = False, J = default_J, grad = cartesian_grad, div = cartesian_div, curl = cartesian_curl, laplacian = cartesian_laplacian):
        # Store basic information
        self.__dx = dx.copy()
        self.__dt = float(dt)
        self.__N = N.copy()
        self.__x0 = x0.copy()
        self.__c = float(c)
        self.__mu0 = float(mu0)
        self.__cdt = self.__c * self.__dt
        self.__V = np.prod(self.__N)
        self.__n = int(approx_n)
        self.__k = float(approx_k)
        self.__t = float(t0)
        
        # Create starting condition for potential
        # Initialize to 0
        if init is True:
            self.__A = np.zeros((self.__V, 4))
            
        # Don't initialize
        elif init is False:
            self.__A = np.empty((self.__V, 4))
            
        # Use given starting conditions
        elif isinstance(init, np.ndarray):
            if init_copy is True:
                self.__A = init.copy().reshape((self.__V, 4))
                
            elif init_copy is False:
                self.__A = init.reshape((self.__V, 4))
                
            else:
                raise Exception("init_copy has wrong type, is " + str(type(init_copy)) + " but should be " + str(bool))
            
        else:
            raise Exception("init has wrong type, is " + str(type(init)) + " but should be either " + str(bool) + " or " + str(np.ndarray))
        
        # Create starting condition for potential differentiated with respect to time
        # Initialize to 0
        if init_ddt is True:
            self.__dAdt = np.zeros((self.__V, 4))
            
        # Don't initialize
        elif init_ddt is False:
            self.__dAdt = np.empty((self.__V, 4))
            
        # Use given starting conditions
        elif isinstance(init_ddt, np.ndarray):
            if init_ddt_copy is True:
                self.__A = init_ddt.copy().reshape((self.__V, 4))
                
            elif init_ddt_copy is False:
                self.__A = init_ddt.reshape((self.__V, 4))
                
            else:
                raise Exception("init_ddt_copy has wrong type, is " + str(type(init_ddt_copy)) + " but should be " + str(bool))
            
        else:
            raise Exception("init_ddt has wrong type, is " + str(type(init_ddt)) + " but should be either " + str(bool) + " or " + str(np.ndarray))
        
        # Get the current/charge density function
        self.J = J(self.__dx, self.__dt, self.__N, self.__x0, self.__c, self.__mu0)
        
        if not callable(self.J):
            raise Exception("J has wrong type, is " + str(type(self.J)) + " but should be function")
        
        # Get vector calculus functions
        if grad is None:
            self.grad = None
            
        else:
            self.grad = grad(self.__dx, self.__dt, self.__N, self.__x0, self.__c, self.__mu0)
                
            if not callable(self.grad):
                raise Exception("grad has wrong type, is " + str(type(self.grad)) + " but should be function")

        if div is None:
            self.div = None
            
        else:
            self.div = div(self.__dx, self.__dt, self.__N, self.__x0, self.__c, self.__mu0)
                
            if not callable(self.div):
                raise Exception("div has wrong type, is " + str(type(self.div)) + " but should be function")
                
        if curl is None:
            self.curl = None
            
        else:
            self.curl = curl(self.__dx, self.__dt, self.__N, self.__x0, self.__c, self.__mu0)
                
            if not callable(self.curl):
                raise Exception("curl has wrong type, is " + str(type(self.curl)) + " but should be function")
                
        self.laplacian = laplacian(self.__dx, self.__dt, self.__N, self.__x0, self.__c, self.__mu0)
                
        if not isinstance(self.laplacian, sparse.csr_matrix):
            raise Exception("laplacian has wrong type, is " + str(type(self.laplacian)) + " but should be " + str(sparse.csr_matrix))
            
        self.__k /= -self.laplacian[0, 0]

    # Set the number of times to use the approximation algerithm per approximation
    def update_approx_n(self, approx_n):
        self.__n = int(approx_n)
        
    # Set the approximation parameter
    def update_approx_k(self, approx_k):
        self.__k = float(approx_k) / self.laplacian[0, 0]
    
    # Sets the time
    def update_t(self, t):
        self.__t = float(t)
        
    # Get the potential
    def get_pot(self):
        return self.__A
    
    # Get potential ddt
    def get_pot_ddt(self):
        return self.__dAdt
    
    # Get n
    def get_approx_n(self):
        return self.__n
    
    # Get k
    def get_approx_k(self):
        return self.__k
    
    # Get t
    def get_t(self):
        return self.__t
    
    # Get E-field
    def get_E(self):
        return calc_E(self.__A[:, 0] * self.__c, self.__dAdt[:, 1:], self.grad)
    
    # Get the B-field
    def get_B(self):
        return calc_B(self.__A[:, 1:], self.curl)
    
    # Get the Poynting vector field
    def get_S(self):
        return calc_S(self.get_E(), self.get_B(), self.__mu0)

    # Finds the electrostatics solution
    def static(self, exact = False, method = "non-responsive"):
        # Do non-responsive simulation
        if (method == "non-responsive"):
            # Do it exact
            if exact is True:
                self.__A = solve_static_non_exact(self.J(self.__t), self.laplacian, self.__mu0)
                
            # Do it approximatly
            else:
                self.__A = solve_static_non_approx(self.J(self.__t), self.laplacian, self.__mu0, self.__A, self.__n, self.__k)
            
        else:
            raise Exception("Uknown method: " + method)
            
        return self.__A
    
    def dynamics(self, exact = False, method = "non-responsive"):
        # Do non-responsive simulation
        if (method == "non-responsive"):
            self.__A, self.__dAdt = solve_dynamics_non(self.__A, self.__dAdt, self.J(self.__t), self.laplacian, self.__dt, self.__c, self.__mu0, self.__n, exact)        
            
        else:
            raise Exception("Uknown method: " + method)
            
        # Increment t
        self.__t += self.__dt
            
        return (self.__A, self.__dAdt)
    
    # Estimate the time to do the simulation
    def estimate_static(self, exact = False, method = "non-responsive", size_n = 11, size_min = 15, size_max = 25, approx_n = 10):
        # Setup data
        Size = np.linspace(size_min, size_max, size_n, dtype = int)
        Time = np.empty(size_n)
        
        for i in range(size_n):
            # Create class
            SizeTest = np.ones(3, dtype = int) * Size[i]
            Sim = sim(self.__dx, 1, SizeTest, approx_n = approx_n)
            
            # Simulate
            Time1 = time.time()
            Sim.static(exact = exact, method = method)
            Time2 = time.time()
            
            Time[i] = Time2 - Time1
            
        # Estimate
        def FitFunc(x, a, n):
            return a * x ** n
            
        Val, _ = opt.curve_fit(FitFunc, Size ** 3, Time)
        
        if exact is False:
            ApproxChange = self.__n / approx_n
            
        else:
            ApproxChange = 1
        
        return FitFunc(self.__V, *Val) * ApproxChange
    
    def estimate_dynamics(self, exact = False, method = "non-responsive", size_n = 11, size_min = 15, size_max = 25, approx_n = 10):
        # Setup data
        Size = np.linspace(size_min, size_max, size_n, dtype = int)
        Time = np.empty(size_n)
        
        for i in range(size_n):
            # Create class
            SizeTest = np.ones(3, dtype = int) * Size[i]
            Sim = sim(self.__dx, self.__dt, SizeTest, approx_n = approx_n)
            
            # Simulate
            Time1 = time.time()
            Sim.dynamics(exact = exact, method = method)
            Time2 = time.time()
            
            Time[i] = Time2 - Time1
            
        # Estimate
        def FitFunc(x, a, n):
            return a * x ** n
            
        Val, _ = opt.curve_fit(FitFunc, Size ** 3, Time)
        
        if exact is False:
            ApproxChange = self.__n / approx_n
            
        else:
            ApproxChange = 1
        
        return FitFunc(self.__V, *Val) * ApproxChange
        
    def plot_E(self, x, axis1 = None, axis2 = None, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None, video = False):
        return plot_vector(self.get_E(), x, self.__N, x0 = self.__x0, dx = self.__dx, axis1 = axis1, axis2 = axis2, dir1 = dir1, dir2 = dir2, scale = scale, ax = ax, figsize = figsize, dpi = dpi, cmap = cmap, clim = clim, video = video)
        
    def plot_B(self, x, axis1 = None, axis2 = None, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None, video = False):
        return plot_vector(self.get_B(), x, self.__N, x0 = self.__x0, dx = self.__dx, axis1 = axis1, axis2 = axis2, dir1 = dir1, dir2 = dir2, scale = scale, ax = ax, figsize = figsize, dpi = dpi, cmap = cmap, clim = clim, video = video)

    def plot_S(self, x, axis1 = None, axis2 = None, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None, video = False):
        return plot_vector(self.get_S(), x, self.__N, x0 = self.__x0, dx = self.__dx, axis1 = axis1, axis2 = axis2, dir1 = dir1, dir2 = dir2, scale = scale, ax = ax, figsize = figsize, dpi = dpi, cmap = cmap, clim = clim, video = video)

    def plot_V(self, x, scale = default_scalar_scale, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None):
        return plot_scalar(self.get_pot()[:, 0]  * self.__c, x, self.__N, x0 = self.__x0, dx = self.__dx, scale = scale, ax = ax, figsize = figsize, dpi = dpi, cmap = cmap, clim = clim)

    def plot_AComp(self, x, comp, scale = default_scalar_scale, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None):
        return plot_scalar(self.get_pot()[:, comp], x, self.__N, x0 = self.__x0, dx = self.__dx, scale = scale, ax = ax, figsize = figsize, dpi = dpi, cmap = cmap, clim = clim)

    def plot_A(self, x, axis1 = None, axis2 = None, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None, video = False):
        return plot_vector(self.get_pot()[:, 1:], x, self.__N, x0 = self.__x0, dx = self.__dx, axis1 = axis1, axis2 = axis2, dir1 = dir1, dir2 = dir2, scale = scale, ax = ax, figsize = figsize, dpi = dpi, cmap = cmap, clim = clim, video = video)

    def update_plot_E(self, plot, x, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log):
        update_plot_vector(plot, self.get_E(), x, self.__N, x0 = self.__x0, dx = self.__dx, dir1 = dir1, dir2 = dir2, scale = scale)
        
    def update_plot_B(self, plot, x, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log):
        update_plot_vector(plot, self.get_B(), x, self.__N, x0 = self.__x0, dx = self.__dx, dir1 = dir1, dir2 = dir2, scale = scale)

    def update_plot_S(self, plot, x, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log):
        update_plot_vector(plot, self.get_S(), x, self.__N, x0 = self.__x0, dx = self.__dx, dir1 = dir1, dir2 = dir2, scale = scale)

    def update_plot_V(self, plot, x, scale = default_scalar_scale):
        update_plot_scalar(plot, self.get_pot()[:, 0] * self.__c, x, self.__N, x0 = self.__x0, dx = self.__dx, scale = scale)

    def update_plot_AComp(self, plot, x, comp, scale = default_scalar_scale):
        update_plot_scalar(plot, self.get_pot()[:, comp], x, self.__N, x0 = self.__x0, dx = self.__dx, scale = scale)

    def update_plot_A(self, plot, x, dir1 = np.array([1, 0, 0], dtype = float), dir2 = np.array([0, 1, 0], dtype = float), scale = np.log):
        update_plot_vector(plot, self.get_pot()[:, 1:], x, self.__N, x0 = self.__x0, dx = self.__dx, dir1 = dir1, dir2 = dir2, scale = scale)

        
def video_init(Name, FPS = 30, figsize = np.array([10., 10.]), dpi = 100):
    # Create hidden figure
    plt.ioff()
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    plt.ion()
    
    # Create video
    Video = cv2.VideoWriter(Name + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), FPS, tuple((figsize * dpi).astype(int)))

    return (fig, ax, Video)

def video_update(fig, Video):
    # Convert figure to image
    # Draw figure to canvas
    fig.canvas.draw()
    
    # Get shape
    Width, Height = fig.canvas.get_width_height()
    
    # Convert to numpy array
    Buffer = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8)
    
    # Convert buffer to image
    Image = Buffer.reshape((Height, Width, 3))

    # Convert to cv2 colours
    Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
    
    # Draw to video
    Video.write(Image)
    
def video_finish(fig, Video):
    # Close figure
    plt.close(fig)
    
    # Close video
    Video.release()

        
        
        
        