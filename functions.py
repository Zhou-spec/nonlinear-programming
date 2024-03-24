import numpy as np





##############################################################################################################
# implementation of Rosenbrock_2 function
def Rosenbrock_2(x):
    # x is a numpy array with two elements
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# implementation of Rosenbrock_2 gradient
def Rosenbrock_2_grad(x):
    # x is a numpy array with two elements
    return np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)])

# implementation of Rosenbrock_2 hessian
def Rosenbrock_2_hess(x):
    # x is a numpy array with two elements
    return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])

##############################################################################################################
# implementation of Rosenbrock_100 function
def Rosenbrock_100(x):
    # x is a numpy array with 100 elements
   return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
 
# implementation of Rosenbrock_100 gradient
def Rosenbrock_100_grad(x):
    # x is a numpy array with 100 elements
    grad = np.zeros_like(x)
    grad[0] = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
    grad[-1] = 200*(x[-1] - x[-2]**2)
    grad[1:-1] = 200*(x[1:-1] - x[:-2]**2) - 400*x[1:-1]*(x[2:] - x[1:-1]**2) + 2*(x[1:-1] - 1)
    return grad

# implementation of Rosenbrock_100 hessian
def Rosenbrock_100_hess(x):
    # x is a numpy array with 100 elements
    hess = np.zeros((100, 100))
    hess[0, 0] = 1200*x[0]**2 - 400*x[1] + 2
    hess[0, 1] = -400*x[0]
    hess[-1, -1] = 200
    hess[-1, -2] = -400*x[-2]
    for i in range(1, 99):
        hess[i, i] = 202 + 1200*x[i]**2 - 400*x[i+1]
        hess[i, i+1] = -400*x[i]
        hess[i, i-1] = -400*x[i-1]
    return hess

##############################################################################################################


