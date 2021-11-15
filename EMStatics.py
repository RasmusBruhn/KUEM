import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
#import cv2 as cv2
import scipy.sparse.linalg as slinalg
import time as time

# Default arguments:
# delta_x       Numpy array of length 3 containing the size of the grid
# dx:           Numpy array of length 3 containing the difference of coordinates between grid points
# N:            Numpy array of length 3 containing the number of grid points
# x0:           Numpy array of length 3 containing the minimum value of each coordinate
# c:            The speed of light
# mu0:          The permeability of free space
# boundaries:   The boundary conditions, it is a 3 long list, one element for each coordinate
#               each element can either be "periodic" or a 2 long list with the boundary conditions
#               in the negative and the positive direction. These can either be "open", "closed" or a matrix for a custom boundary condition

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
    TotalW = TotalW.reshape(TotalW.shape + (1,) * (len(Values.shape) - len(W.shape)))

    # Calculate the lerp
    return np.sum(TotalW * Values, axis = 0)

# Samples values at specific points from a field, interpolate to get smooth values from field
# Return the values in the shape Points.shape[1:]
#
# dx, N and x0 are default arguments
# Field:        The field to sample values from in vector from, this can also be a vector field
# Points:       An array with the x,y,z coordinates in the first axis
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


# Samples vectors at specific points from a field projected onto a 2D plane, interpolate to get smooth values from field
# Return the values in the shape Points.shape[1:]
#
# dx, N and x0 are default arguments
# Field:        The field to sample values from in vector from, this has to be a vector field
# Points:       An array with the x,y,z coordinates in the first axis
# x_hat:        A 3 long array for the direction of the x axis
# y_hat:        A 3 long array for the direction of the y axis
def sample_vectors(Field, Points, x_hat, y_hat, dx, N, x0):
    # Get the 3D vectors
    FieldValues = sample_values(Field, Points, dx, N, x0)
    
    # Find the x and y axis components
    vx = np.sum(FieldValues * x_hat.reshape((1,) * (len(FieldValues.shape) - 1) + (3,)), axis = -1)
    vy = np.sum(FieldValues * y_hat.reshape((1,) * (len(FieldValues.shape) - 1) + (3,)), axis = -1)

    return vx, vy

# A default scale used in the scalar plotter
# Returns a numpy array filled with the values in x scaled
# 
# x:    This is a numpy array of floating points
def default_scalar_scale(x):
    return x

# Plot a scalar field
#
# Values:       The 2D array of values to plot
# extent:       Used to label the axis must be given as [x_min, x_max, y_min, y_max]
# scale:        Function to scale the values of the field
# ax:           The axes to draw the plot inside
# fig:          The figure to draw in, if given then ax must also be given
# figsize:      The size of the figure if ax is not given
# dpi:          The resolution of the figure if ax is not given
# cmap:         The colour map to plot the scalar field with
# clim:         Array containing the (min, max) values in the colour map, these are the raw values of the field,
#               not the scaled values, if None then it will find the scale automatially by the minimum and maximum
#               values in the field
def plot_scalar(Values, extent = [0, 1, 0, 1], scale = default_scalar_scale, fig = None, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None):
    # Calculate clim
    if clim is None:
        clim = np.array([np.min(Values), np.max(Values)], dtype = float)
    
    clim = scale(clim)
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        
    # Plot scalar field
    Plot = ax.imshow(np.transpose(scale(Values)), cmap = cmap, clim = clim, origin = "lower", extent = extent)
    
    return fig, ax, Plot

# Plot the values along a line in a scalar field
#
# Values:       The 1D array of values to plot
# extent:       Used to label the axis must be given as [x_min, x_max]
# scale:        Function to scale the values of the field
# ax:           The axes to draw the plot inside
# fig:          The figure to draw in, if given then ax must also be given
# figsize:      The size of the figure if ax is not given
# dpi:          The resolution of the figure if ax is not given
# fmt:          The fmt used for plotting
def plot_1D(Values, extent = [0, 1], scale = default_scalar_scale, fig = None, ax = None, figsize = np.array([10., 10.]), dpi = 100, fmt = "-", label = ""):
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        
    # Plot graph
    Plot = ax.plot(np.linspace(extent[0], extent[1], len(Values)), Values, fmt, label = label)
    
    return fig, ax, Plot

# Plot a vector field
#
# vx:           The vector components along the x axis to plot
# vy:           The vector components along the y axis to plot
# extent:       Used to label the axis must be given as [x_min, x_max, y_min, y_max]
# scale:        Function to scale the values of the field
# fig:          The figure to draw in, if given then ax must also be given
# ax:           The axes to draw the plot inside
# figsize:      The size of the figure if ax is not given
# dpi:          The resolution of the figure if ax is not given
# cmap:         The colour map to plot the vectors with
# clim:         Array containing the (min, max) values in the colour map, these are the raw values of the vector lengths,
#               not the scaled values, if None then it will find the scale automatially by the minimum and maximum
#               values of the lengths
def plot_vector(vx, vy, extent = [0, 1, 0, 1], scale = default_scalar_scale, fig = None, ax = None, figsize = np.array([10., 10.]), dpi = 100, cmap = "coolwarm", clim = None):  
    # Calculate the positions of the vectors
    x = np.linspace(extent[0], extent[1], vx.shape[0] + 1)[:-1]
    y = np.linspace(extent[2], extent[3], vy.shape[1] + 1)[:-1]
    X, Y = np.meshgrid(x, y, indexing = "ij")
    
    # Calculate length of vectors
    vAbs = np.sqrt(vx ** 2 + vy ** 2)    

    # Calculate clim
    if clim is None:
        clim = np.array([np.min(vAbs), np.max(vAbs)], dtype = float)
        
    clim = scale(clim)

    vAbs[vAbs == 0] = np.nan
    
    # Normalize lengths
    vx = vx / vAbs
    vy = vy / vAbs
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        
    # Plot vectors
    Plot = ax.quiver(X, Y, vx, vy, scale(vAbs), cmap = cmap, pivot = "middle", clim = clim)
    
    return fig, ax, Plot

# Creates an array of points sampled from a plane spanned by x_hat and y_hat and centered on x_c
# Returns the coordinates for all of the points in a 2D array form for imshow
#
# x_hat:        The direction for the x-axis
# y_hat:        The direction for the y-axis
# x_c:          The center point
# Size:         2 component array giving the width in the x_hat and y_hat direction
# Resolution:   2 component array giving the number of points in the x_hat and y_hat direction
def sample_points_plane(x_hat, y_hat, x_c, Size, Resolution):
    # Make sure x_hat and y_hat is normalised
    x_hat /= np.sqrt(np.sum(x_hat ** 2))
    y_hat /= np.sqrt(np.sum(y_hat ** 2))
    
    # Find the points in the x_hat, y_hat system
    x = np.linspace(-Size[0] / 2, Size[0] / 2, Resolution[0])
    y = np.linspace(-Size[1] / 2, Size[1] / 2, Resolution[1])
    
    # Calculate the real positions
    return x_c.reshape((3, 1, 1)) + x_hat.reshape((3, 1, 1)) * x.reshape((1, -1, 1)) + y_hat.reshape((3, 1, 1)) * y.reshape((1, 1, -1))
    
# Creates an array of points sampled from a line between x1 and x2
# Returns the coordinates for all points along a line to sample field values for in 1D plotting
#
# x1:           The starting point of the line
# x2:           The ending poing of the line
# Resolution:   The number of points to be sampled
def sample_points_line(x1, x2, Resolution):
    # Make an evenly distributed linspace between the 2 points
    return x1.reshape((3, 1)) + (x2.reshape((3, 1)) - x1.reshape((3, 1))) * np.linspace(0, 1, Resolution).reshape((1, -1))
    

# Creates boundary matrices, this is matrices with ones at the diagonal at points at each of the 6 boundaries
# the first dimension is the coordinate and the second dimension is the direction (positive or negative)
# N is a default argument
def get_boundaries_open(N):
    # Make list to save the boundaries (diagonals at the points at boundary)
    Bounds = np.empty((3, 2), dtype = sparse.csr_matrix)
    
    # Go through all 3 coordinates
    for Coordinate in range(3):
        # Go through each direction
        for Dir in range(2):
            # Calculate the position of the offdiagonal
            #OffPos = np.prod(N[:Coordinate])
            
            # Create a tile
            Tile = np.zeros(np.prod(N[:Coordinate + 1]))
            #OffTile = np.zeros(np.prod(N[:Coordinate + 1]))
            
            # Add the ones
            if (Dir == 0):
                Tile[:np.prod(N[:Coordinate])] = 1
                #OffTile[:np.prod(N[:Coordinate])] = -1
                #OffPos *= -1
                
            else:
                Tile[-np.prod(N[:Coordinate]):] = 1
                #OffTile[-np.prod(N[:Coordinate]):] = -1
                
            # Create the diagonal
            Diag = np.tile(Tile, np.prod(N[Coordinate + 1:]))
            #OffDiag = np.tile(OffTile, np.prod(N[Coordinate + 1:]))[:-np.prod(N[:Coordinate])]
            
            # Create the boundary
            #Bounds[Coordinate, Dir] = sparse.diags([Diag, OffDiag], [0, OffPos], format = "csr")
            Bounds[Coordinate, Dir] = sparse.diags([Diag], [0], format = "csr")
            
    return Bounds


# Creates matrices for differentiating once
# dx, N and boundaries are default arguments
def get_ddx(dx, N, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
    # Make a list for (ddx, ddy, ddz)
    ddx = np.empty(3, dtype = sparse.csr_matrix)
    
    # Get the open boundaries
    OpenBounds = get_boundaries_open(N)
    
    # Go through each coordinate
    for Coordinate in range(3):
        # Create a tile
        Tile = np.ones(np.prod(N[:Coordinate + 1]))
        
        # Add zeros at the boundaries
        Tile[-np.prod(N[:Coordinate]):] = 0
        
        # Create the diagonal
        Diag = np.tile(Tile, np.prod(N[Coordinate + 1:]))[:-np.prod(N[:Coordinate])]
        
        # Create the matrix
        ddx[Coordinate] = sparse.diags([Diag, -Diag], [np.prod(N[:Coordinate]), -np.prod(N[:Coordinate])], format = "csr")
        
        # Add the open boundaries
        if boundaries[Coordinate][0] == "open":
            ddx[Coordinate] -= OpenBounds[Coordinate, 0]
            
        elif isinstance(boundaries[Coordinate][0], sparse.csr_matrix):
            ddx[Coordinate] -= boundaries[Coordinate][0]
            
        if boundaries[Coordinate][1] == "open":
            ddx[Coordinate] += OpenBounds[Coordinate, 0]
            
        elif isinstance(boundaries[Coordinate][1], sparse.csr_matrix):
            ddx[Coordinate] += boundaries[Coordinate][1]
        
        # Correct the value
        ddx[Coordinate] /= (2 * dx[Coordinate])

    return ddx

    
# Creates matrices for differentiating twice, 
# dx, N and boundaries are default arguments
def get_ddx2(dx, N, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
    # Make a list for (ddx, ddy, ddz)
    ddx2 = np.empty(3, dtype = sparse.csr_matrix)

    # Get the open boundaries
    OpenBounds = get_boundaries_open(N)

    # Go through each coordinate
    for Coordinate in range(3):
        # Create a tile
        Tile = np.ones(np.prod(N[:Coordinate + 1]))
        
        # Add zeros at the edges
        Tile[-np.prod(N[:Coordinate]):] = 0
        
        # Create the diagonal
        Diag = np.tile(Tile, np.prod(N[Coordinate + 1:]))[:-np.prod(N[:Coordinate])]
        
        # Create the matrix
        ddx2[Coordinate] = sparse.diags([-2 * np.ones(np.prod(N)), Diag, Diag], [0, np.prod(N[:Coordinate]), -np.prod(N[:Coordinate])], format = "csr")

        # Add open boundary conditions
        if boundaries[Coordinate][0] == "open":
            ddx2[Coordinate] += OpenBounds[Coordinate, 0]
            
        elif isinstance(boundaries[Coordinate][0], sparse.csr_matrix):
            ddx2[Coordinate] += boundaries[Coordinate][0]
            
        if boundaries[Coordinate][1] == "open":
            ddx2[Coordinate] += OpenBounds[Coordinate, 0]

        elif isinstance(boundaries[Coordinate][1], sparse.csr_matrix):
            ddx2[Coordinate] += boundaries[Coordinate][1]

        # Correct the value
        ddx2[Coordinate] /= dx[Coordinate] ** 2

    return ddx2

# Creates a function to take the gradient in cartesian coordinates
# dx, N and boundaries are default arguments
def get_grad(dx, N, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
    # Get the diff matrices
    ddx = get_ddx(dx, N, boundaries = boundaries)
    
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
# dx, N and boundaries are default arguments
def get_div(dx, N, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
    # Get the diff matrices
    ddx = get_ddx(dx, N, boundaries = boundaries)
    
    # Calculate the divergence
    # Vector: A vector field to take the divergence of
    def calcDiv(Vector):
        # Calculate result
        return ddx[0].dot(Vector[:, 0]) + ddx[1].dot(Vector[:, 1]) + ddx[2].dot(Vector[:, 2])
    
    return calcDiv

# Creates a function to take the curl in cartesian coordinates
# dx, N and boundaries are default arguments
def get_curl(dx, N, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
    # Get diff matrices
    ddx = get_ddx(dx, N, boundaries = boundaries)
    
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
# dx, N and boundaries are default arguments
def get_lapl(dx, N, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
    # Get ddx2
    ddx2 = get_ddx2(dx, N, boundaries = boundaries)
    
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
# progress:     How to show how much time is remaining, False if it should be disabled
#               a number to show time remaining with a period >= progress
def solve_approx(J, Laplacian, mu0, A0, n, k, progress = False):
    # Setup the starting guess
    A = A0
    
    # Get the evolution matrix
    Matrix = sparse.identity(J.shape[0], format = "csr") + k * Laplacian
    
    # Setup time for writing time remaining
    Time0 = time.time()
    Time1 = Time0
    
    for i in range(n):
        # Evolve one step
        A = Matrix.dot(A) + mu0 * k * J
        
        # Write the time remaining
        Time2 = time.time()
        if progress is not False and (Time2 - Time1 >= progress):
            print("%.2g s remaining"%((Time2 - Time0) * (n - i - 1) / (i + 1)))
            Time1 = Time2
        
    return A

# The simulation class, create this class to define your simulation,
# Then use it's methods to run the simulation and diagnistics/plotting
# N, delta_x, x0, c, mu0 and boundaries are default arguments
#
# Initialize:
# approx_n:     The base number of times to run the approximation algorithm, this is multiplied by Nx^2+Ny^2+Nz^2
# approx_k:     The base approximation parameter to make sure it does not diverge, this is multiplied by -1 / lapl[0, 0]
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
    def __init__(self, N, delta_x = np.array([1, 1, 1]), x0 = np.array([0, 0, 0]), c = 1, mu0 = 1, approx_n = 0.1, approx_k = 1, init = True, init_copy = False, J = default_J, grad = get_grad, div = get_div, curl = get_curl, lapl = get_lapl, boundaries = [["closed", "closed"], ["closed", "closed"], ["closed", "closed"]]):
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

        if not (isinstance(approx_n, int) or isinstance(approx_n, float)):
            raise Exception("approx_n has wrong dtype, it is " + str(type(approx_n)) + " but it should be " + str(int) + " or " + str(float))
        
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
        self.__n = int(approx_n * np.sum(self.__N ** 2))
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
        self.__grad = grad(self.__dx, self.__N, boundaries = boundaries)
            
        if not callable(self.__grad):
            raise Exception("grad has wrong type, it is " + str(type(self.__grad)) + " but it should be a function")

        self.__div = div(self.__dx, self.__N, boundaries = boundaries)
            
        if not callable(self.__div):
            raise Exception("div has wrong type, it is " + str(type(self.__div)) + " but it should be a function")

        self.__curl = curl(self.__dx, self.__N, boundaries = boundaries)
            
        if not callable(self.__curl):
            raise Exception("curl has wrong type, it is " + str(type(self.__curl)) + " but it should be a function")
                
        self.__lapl = lapl(self.__dx, self.__N, boundaries = boundaries)
                
        if not isinstance(self.__lapl, sparse.csr_matrix):
            raise Exception("lapl has wrong type, is " + str(type(self.__lapl)) + " but it should be " + str(sparse.csr_matrix))
            
        # Update the value for k
        self.__k /= (2 * np.sum(1 / self.__dx ** 2))

    # Get electric potential
    def get_V(self):
        return self.__A[:, 0] * self.__c
    
    # Get vector potential
    def get_A(self):
        return self.__A[:, 1:]
    
    # Get E-field
    def get_E(self):
        return calc_E(self.__A[:, 0] * self.__c, self.__grad)
    
    # Get the B-field
    def get_B(self):
        return calc_B(self.__A[:, 1:], self.__curl)
    
    # Get the Poynting vector field
    def get_S(self):
        return calc_S(self.get_E(), self.get_B(), self.__mu0)

    # Get the charge density
    def get_Rho(self):
        return self.__J()[:, 0] / self.__c
    
    # Get the current density
    def get_J(self):
        return self.__J()[:, 1:]

    # Finds the electrostatics solution
    # Returns the time it took
    #
    # exact:        Whether it should be solved exactly or approximatly, should be a boolean
    # progress:     How to show how much time is remaining, False if it should be disabled
    #               a number to show time remaining with a period >= progress
    def solve(self, exact = False, progress = False):
        Time1 = time.time()
        # Do it exact
        if exact is True:
            self.__A = solve_exact(self.__J(), self.__lapl, self.__mu0)
            
        # Do it approximatly
        else:
            self.__A = solve_approx(self.__J(), self.__lapl, self.__mu0, self.__A, self.__n, self.__k, progress = progress)
            
        Time2 = time.time()
        
        return Time2 - Time1
    
    # Samples values at specific points from a field, interpolate to get smooth values from field
    # Return the values in the shape Points.shape[1:]
    #
    # Field:        The field to sample values from in vector from, this can also be a vector field
    # Points:       An array with the x,y,z coordinates in the first axis
    def sample_values(self, Field, Points):
        return sample_values(Field, Points, self.__dx, self.__N, self.__x0)
        
    # Samples vectors at specific points from a field projected onto a 2D plane, interpolate to get smooth values from field
    # Return the values in the shape Points.shape[1:]
    #
    # Field:        The field to sample values from in vector from, this has to be a vector field
    # Points:       An array with the x,y,z coordinates in the first axis
    # x_hat:        A 3 long array for the direction of the x axis
    # y_hat:        A 3 long array for the direction of the y axis
    def sample_vectors(self, Field, Points, x_hat, y_hat):
        return sample_vectors(Field, Points, x_hat, y_hat, self.__dx, self.__N, self.__x0)