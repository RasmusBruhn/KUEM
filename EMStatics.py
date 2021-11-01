import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
#import cv2 as cv2
import scipy.sparse.linalg as slinalg
import time as time
import scipy.optimize as opt

# Default arguments:
# delta_x       Numpy array of length 3 containing the size of the grid
# dx:           Numpy array of length 3 containing the difference of coordinates between grid points
# N:            Numpy array of length 3 containing the number of grid points
# x0:           Numpy array of length 3 containing the minimum value of each coordinate
# c:            The speed of light
# mu0:          The permeability of free space

# Default vector arguments:
# grad:         Function to calculate gradient of scalar field
# div:          Function to calculate divergence of vector field
# curl:         Function to calculate curl of vector field
# lapl:         Matrix to calculate laplacian
# c:            The speed of light
# mu0:          The permeability of free space

# Function to get the vector indices for the 1 dimensional vector given the indices for the 3 dimensional vector:
# Returns a numpy array or number with the shape n.shape[1:] with the indices for the 1 dimensional vector
#
# N is default argument
# n:            A numpy numpy array with the indices for the 3 dimensional vector, 
#               more than one point can be specified but the first axis must be the indices for one point
def get_vector_index(n, N):
    Factor = np.array([1, N[0], N[0] * N[1]]).reshape((3,) + (1,) * (len(n.shape) - 1))
    return np.array(np.sum(n * Factor, axis = 0), dtype = int)

# Calculates the total weights for each of the 8 corners in the interpolation
# Returns an array with the shape (8,) + W.shape[1:] containing the weights for the 8 corners
# for each point. For (x,y,z)  the points are sorted as (-, -, -), (-, -, +), (-, +, -), (-, +, +),
# (+, -, -), (+, -, +), (+, +, -), (+, +, +)
#
# W:    The weights for each point, the shape can be any length but the first axis
#       should be of length 3 and contain the weights for each of the coordinates,
#       where 0 is for the - corner and 1 is for +
def total_weight(W):
    # Initialise the total weight array
    Result = np.ones((8,) + W.shape[1:])
    
    # Go through all of the coordinates
    for i in range(3):
        # Go through and multiply the weights v=1-w or w in the pattern:
        #       | | | | | | | | |
        # i = 0: v v v v w w w w
        # i = 1: v v w w v v w w
        # i = 2: v w v w v w v w
        for j in range(int(2 ** i)):
            # Do the v's
            Result[int(j * 2 ** (3 - i)):int((j + 1 / 2) * 2 ** (3 - i))] *= (1 - W[i]).reshape((1,) + W.shape[1:])
            # Do the w's
            Result[int((j + 1 / 2) * 2 ** (3 - i)):int((j + 1) * 2 ** (3 - i))] *= W[i].reshape((1,) + W.shape[1:])
            
    # Return the result
    return Result

# A function to interpolate the values of 8 corners in the middle 
#
# W:        The weights for each point for the different directions,
#           the first axis should be length 3 and contain the weights for the same point
# Values:   The values at the corners, the first axis defines the corner sorted by
#           (-, -, -), (-, -, +), (-, +, -), (-, +, +), (+, -, -), (+, -, +), (+, +, -), (+, +, +)
#           The shape of Values and W should be the same except for the first axis
def interpolation(W, Values):
    # Get the total weights
    TotalW = total_weight(W)
    
    # Calculate the lerp
    return np.sum(TotalW * Values, axis = 0)

def sample_values(Field, Points, dx, N, x0):
    # Find the new shape of dx and x0
    ShapeX = (3,) + (1,) * len(Points.shape)
    ShapeP = (3, 1) + Points.shape[1:]

    # Find the indices for the corners
    CornerIndex = np.repeat(np.array((Points.reshape(ShapeP) - x0.reshape(ShapeX)) / dx.reshape(ShapeX), dtype = int), 8, axis = 1)

    # Go through all of the coordinates
    for i in range(3):
        # Go through and multiply the weights v=1-w or w in the pattern:
        #       | | | | | | | | |
        # i = 0: - - - - + + + +
        # i = 1: - - + + - - + +
        # i = 2: - + - + - + - +
        for j in range(int(2 ** i)):
            # Add to the coordinates
            CornerIndex[i, int((j + 1 / 2) * 2 ** (3 - i)):int((j + 1) * 2 ** (3 - i))] += 1
            
    # Get the vector index for each point
    CornerVectorIndex = get_vector_index(CornerIndex, N)

    # Get values from field
    FieldValues = Field[CornerVectorIndex]
    
    # Find new shapes
    ShapeX = (3,) + (1,) * (len(Points.shape) - 1)
    
    # Find weights
    Weights = np.mod((Points - x0.reshape(ShapeX)) / dx.reshape(ShapeX), 1)

    # Interpolate
    return interpolation(Weights, FieldValues)

# A default scale used in the scalar plotter
# Returns a numpy array filled with the values in x scaled
# 
# x:    This is a numpy array of floating points
def default_scalar_scale(x):
    return x

# Plot a scalar field
# dx, N, x0 are default arguments
# Field:        A scalar field on vector form from which the values will be taken
# Points:       The points from where the values should be taken, length 3 along first axis containing the coordinates
#               the rest of the array can be anything
# extent:       Used to label the axis must be given as [x_min, x_max, y_min, y_max]
# scale:        Function to scale the values of the field
# ax:           The axes to draw the plot inside
# figsize:      The size of the figure if ax is not given
# dpi:          The resolution of the figure if ax is not given
# cmap:         The colour map to plot the scalar field with
# clim:         Array containing the (min, max) values in the colour map, these are the raw values of the field,
#               not the scaled values, if None then it will find the scale automatially by the minimum and maximum
#               values in the field
def plot_scalar(Field, Points, dx, N, x0, extent = [0, 1, 0, 1] , scale = default_scalar_scale, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None):
    # Get values
    Values = sample_values(Field, Points, dx, N, x0)
        
    # Calculate clim
    if clim is None:
        clim = np.array([np.min(Values), np.max(Values)], dtype = float)
    
    clim = scale(clim)
    
    # Create figure
    if ax is None:
        _, ax = plt.subplots(figsize = figsize, dpi = dpi)
        
    # Plot vectors
    Plot = ax.imshow(scale(Values), cmap = cmap, clim = clim, origin = "lower", extent = extent)
    
    return ax, Plot

# Creates an array of points sampled from a plane spanned by x_hat and y_hat and centered on x_c
def sample_points_plane(x_hat, y_hat, x_c, Size, Resolution):
    # Make sure x_hat and y_hat is normalised
    x_hat /= np.sqrt(np.sum(x_hat ** 2))
    y_hat /= np.sqrt(np.sum(y_hat ** 2))
    
    # Find the points in the x_hat, y_hat system
    x = np.linspace(-Size[0] / 2, Size[0] / 2, Resolution[0])
    y = np.linspace(-Size[1] / 2, Size[1] / 2, Resolution[1])
    X, Y = np.meshgrid(x, y)
    
    # Calculate the real positions
    return x_c.reshape((3, 1, 1)) + x_hat.reshape((3, 1, 1)) * X.reshape((1,) + X.shape) + y_hat.reshape((3, 1, 1)) * Y.reshape((1,) + Y.shape)
    

# Creates matrices for differentiating once, 
# dx and N are default arguments
def get_ddx(dx, N):
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
# dx and N are default arguments
def get_ddx2(dx, N):
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
# dx and N are default arguments
def get_grad(dx, N):
    # Get the diff matrices
    ddx = get_ddx(dx, N)
    
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
# dx and N are default arguments
def get_div(dx, N):
    # Get the diff matrices
    ddx = get_ddx(dx, N)
    
    # Calculate the divergence
    # Vector: A vector field to take the divergence of
    def calcDiv(Vector):
        # Calculate result
        return ddx[0].dot(Vector[:, 0]) + ddx[1].dot(Vector[:, 1]) + ddx[2].dot(Vector[:, 2])
    
    return calcDiv

# Creates a function to take the curl in cartesian coordinates
# dx and N are default arguments
def get_curl(dx, N):
    # Get diff matrices
    ddx = get_ddx(dx, N)
    
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
# dx and N are default arguments
def get_lapl(dx, N):
    # Get ddx2
    ddx2 = get_ddx2(dx, N)
    
    # Create laplacian
    return np.sum(ddx2)

# Finds the E-field
#
# V:            The electric potential: A[:, 0] * c
# grad:         Function to calculate gradient
def calc_E(V, grad):
    return -grad(V)

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

# Creates a function which returns a current/charge density which is 0 everywhere
# Uses default arguments
def default_J(dx, N, x0, c, mu0):
    # Create current which is 0 everywhere
    J = np.zeros((np.prod(N), 4))
    
    # Create function to return J
    # t: The time
    def GetJ():
        return J
    
    # Return the function
    return GetJ

# Solves non-responsive EM-statics exactly
#
# J:            The current/charge density
# Laplacian:    The laplacian matrix
# mu0:          Value for mu0
def solve_exact(J, laplacian, mu0):
    return slinalg.spsolve(laplacian, -mu0 * J)

# Solves non-responsive EM-statics approximately
#
# J:            The current/charge density
# Laplacian:    The laplacian matrix
# mu0:          Value for mu0
# A0:           The starting guess for the potential
# n:            The number of iterations for the approximation
# k:            The approximation parameter to make sure it does not diverge
def solve_approx(J, Laplacian, mu0, A0, n, k):
    A = A0
    Matrix = sparse.identity(J.shape[0], format = "csr") + k * Laplacian
    
    for _ in range(n):
        A = Matrix.dot(A) + mu0 * k * J
        
    return A

# The simulation class, create this class to define your simulation,
# Then use it's methods to run the simulation and diagnistics/plotting
#
# Initialize:
# N:            Array with the number of grid points in each direction
# delta_x:      Array with the size of the grid
# x0:           Array with the minimum value of the coordinates
# c:            The value of c, the speed of light
# mu0:          The value of mu0, the permeability of free space
# approx_n:     The number of times to run the approximation algorithm
# approx_k:     Approximation parameter to make sure it does not diverge
# init:         If true then the potentials will initialize to be 0 everywhere
#               If it is a numpy array of the correct size then it will
#               use that as the starting condition, the shape should either be
#               (N3, N2, N1, 4) or (N1 * N2 * N3, 4) if it is another shape but
#               still size 4 * N1 * N2 * N3 then it will reshape it to (N1 * N2 * N3, 4)
#               but with no guarantee that it will format correctly 
# init_copy:    Only used if init is a numpy array, if true then it will copy
#               the array, if false then it will use and alter the original array
# J:            The current and charge distribution, must be a function with default
#               arguments which returns a function to give the current and charge densit at any time
# grad:         A function to return a function to calculate the gradient in the coordinate system used
# div:          A function to return a function to calculate the divergence in the coordinate system used
# curl:         A function to return a function to calculate the curl in the coordinate system used
# lapl:    A function to calculate the laplacian in the coordinate system used
class sim:
    def __init__(self, N, delta_x = np.array([1, 1, 1]), x0 = np.array([0, 0, 0]), c = 1, mu0 = 1, approx_n = 100, approx_k = 1, init = True, init_copy = False, J = default_J, grad = get_grad, div = get_div, curl = get_curl, lapl = get_lapl):
        # Test for type errors
        if not isinstance(N, np.ndarray):
            raise Exception("N has wrong type, it is " + str(type(N)) + " but it should be " + str(np.ndarray))
        
        if not issubclass(N.dtype.type, np.integer):
            raise Exception("N has wrong dtype, it is " + str(N.dtype.type) + " but it should be " + str(np.integer))
        
        if not isinstance(delta_x, np.ndarray):
            raise Exception("delta_x has wrong type, it is " + str(type(delta_x)) + " but it should be " + str(np.ndarray))
        
        if not (issubclass(delta_x.dtype.type, np.floating) or issubclass(delta_x.dtype.type, np.integer)):
            raise Exception("delta_x has wrong dtype, it is " + str(delta_x.dtype.type) + " but it should be " + str(np.integer) + " or " + str(np.floating))
        
        if not isinstance(x0, np.ndarray):
            raise Exception("x0 has wrong type, it is " + str(type(x0)) + " but it should be " + str(np.ndarray))
        
        if not (issubclass(x0.dtype.type, np.floating) or issubclass(x0.dtype.type, np.integer)):
            raise Exception("x0 has wrong dtype, it is " + str(x0.dtype.type) + " but it should be " + str(np.integer) + " or " + str(np.floating))

        if not (isinstance(c, int) or isinstance(c, float)):
            raise Exception("c has wrong dtype, it is " + str(type(c)) + " but it should be " + str(int) + " or " + str(float))

        if not (isinstance(mu0, int) or isinstance(mu0, float)):
            raise Exception("mu0 has wrong dtype, it is " + str(type(mu0)) + " but it should be " + str(int) + " or " + str(float))

        if not isinstance(approx_n, int):
            raise Exception("approx_n has wrong dtype, it is " + str(type(approx_n)) + " but it should be " + str(int))
        
        if not (isinstance(approx_k, int) or isinstance(approx_k, float)):
            raise Exception("approx_k has wrong dtype, it is " + str(type(approx_k)) + " but it should be " + str(int) + " or " + str(float))
        
        # Store basic information
        self.__delta_x = np.array(delta_x.copy(), dtype = float)
        self.__N = np.array(N.copy(), dtype = int)
        self.__dx = self.__delta_x / (self.__N + 1)
        self.__x0 = np.array(x0.copy(), dtype = float)
        self.__c = float(c)
        self.__mu0 = float(mu0)
        self.__V = np.prod(self.__N)
        self.__n = int(approx_n)
        self.__k = float(approx_k)
        
        # Create starting condition for potential
        # Initialize to 0
        if init is True:
            self.__A = np.zeros((self.__V, 4))
            
        # Use given starting conditions
        elif isinstance(init, np.ndarray):
            if np.prod(np.array(init.shape)) != self.__V * 4:
                raise Exception("init has wrong length, it has length " + str(np.prod(np.array(init.shape))) + " but it should be " + str(self.__V * 4))
            
            if init_copy is True:
                self.__A = init.copy().reshape((self.__V, 4))
                
            elif init_copy is False:
                self.__A = init.reshape((self.__V, 4))
                
            else:
                raise Exception("init_copy has wrong type, it is " + str(type(init_copy)) + " but it should be " + str(bool))
            
        else:
            raise Exception("init has wrong type, it is " + str(type(init)) + " but it should be " + str(np.ndarray) + " or be True")
        
        # Save all the generators
        self.__genJ = J
        self.__genGrad = grad
        self.__genDiv = div
        self.__genCurl = curl
        self.__genLapl = lapl
        
        # Get the current/charge density function
        self.__J = J(self.__dx, self.__N, self.__x0, self.__c, self.__mu0)
        
        if not callable(self.__J):
            raise Exception("J has wrong type, it it is " + str(type(self.__J)) + " but it should be a function")
        
        # Get vector calculus functions
        self.__grad = grad(self.__dx, self.__N)
            
        if not callable(self.__grad):
            raise Exception("grad has wrong type, it is " + str(type(self.__grad)) + " but it should be a function")

        self.__div = div(self.__dx, self.__N)
            
        if not callable(self.__div):
            raise Exception("div has wrong type, it is " + str(type(self.__div)) + " but it should be a function")

        self.__curl = curl(self.__dx, self.__N)
            
        if not callable(self.__curl):
            raise Exception("curl has wrong type, it is " + str(type(self.__curl)) + " but it should be a function")
                
        self.__lapl = lapl(self.__dx, self.__N)
                
        if not isinstance(self.__lapl, sparse.csr_matrix):
            raise Exception("lapl has wrong type, is " + str(type(self.__lapl)) + " but it should be " + str(sparse.csr_matrix))
            
        # Update the value for k
        self.__k /= -self.__lapl[0, 0]
    
    # Get electric potential
    def get_V(self):
        return self.__A[:, 0] * self.__c
    
    # Get vector potential
    def get_A(self):
        return self.A[:, 1:]
    
    # Get E-field
    def get_E(self):
        return calc_E(self.__A[:, 0] * self.__c, self.__grad)
    
    # Get the B-field
    def get_B(self):
        return calc_B(self.__A[:, 1:], self.__curl)
    
    # Get the Poynting vector field
    def get_S(self):
        return calc_S(self.get_E(), self.get_B(), self.__mu0)

    # Finds the electrostatics solution
    # Returns the time it took
    #
    # exact:    Whether it should be solved exactly or approximatly, should be a boolean
    def solve(self, exact = False):
        Time1 = time.time()
        # Do it exact
        if exact is True:
            self.__A = solve_exact(self.__J(), self.__lapl, self.__mu0)
            
        # Do it approximatly
        else:
            self.__A = solve_approx(self.__J(), self.__lapl, self.__mu0, self.__A, self.__n, self.__k)
            
        Time2 = time.time()
        
        return Time2 - Time1
        
    # Estimate the time to do the simulation by solving smaller systems and extrapolating the time
    #
    # exact:    Whether it should be solved exactly or approximatly, should be a boolean
    # size_n:   The number of points
    # size_min: The minimum size of the system to test
    # size_max: The maximum size of the system to test
    def estimate_solve_time(self, exact = False, size_n = 11, size_min = 15, size_max = 25, approx_n = 10):
        # Setup data
        Size = np.linspace(size_min, size_max, size_n, dtype = int)
        Time = np.empty(size_n)
        
        for i in range(size_n):
            # Create class
            SizeTest = np.full(3, Size[i], dtype = int)
            Sim = sim(SizeTest, delta_x = self.__delta_x, approx_n = approx_n, approx_k = self.__k, J = self.__genJ, grad = self.__genGrad, div = self.__genDiv, curl = self.__genCurl, lapl = self.__genLapl)
            
            # Simulate
            Time1 = time.time()
            Sim.solve(exact = exact)
            Time2 = time.time()
            
            Time[i] = Time2 - Time1
            
        # Estimate
        def FitFunc(x, a, n):
            return a * x ** n
            
        Val, _ = opt.curve_fit(FitFunc, Size ** 3, Time)
        
        return FitFunc(self.__V, *Val) / approx_n * self.__n
            