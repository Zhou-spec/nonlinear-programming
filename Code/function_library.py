    
import numpy as np


def rosenbrock(x, dim = 100):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosenbrock_grad(x, dim = 100):
    g = np.zeros_like(x)
    for i in range(dim - 1):
        g[i] = g[i] + 400 * x[i] * (x[i] * x[i] - x[i+1]) + 2 * (x[i] - 1)
        g[i + 1] = g[i+1] + 200 * (x[i+1] - x[i] * x[i])
    return g

def rosenbrock_hessian(x, dim):
    h = np.zeros((dim, dim))
    for i in range(dim - 1):
        h[i,i] = h[i,i] + 1200 * x[i] * x[i] - 400 * x[i+1] + 2
        h[i, i + 1] = h[i, i + 1] - 400 * x[i]
        h[i+1, i] = h[i+1, i] - 400 * x[i]
        h[i+1, i+1] = h[i+1, i+1] + 200
    return h



def Beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def Beale_grad(x):
    grad = np.zeros(2)
    grad[0] = 2*(1.5 - x[0] + x[0]*x[1])*(x[1] - 1) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(x[1]**2 - 1) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(x[1]**3 - 1)
    grad[1] = 2*(1.5 - x[0] + x[0]*x[1])*x[0] + 4*(2.25 - x[0] + x[0]*x[1]**2)*x[0]*x[1] + 6*(2.625 - x[0] + x[0]*x[1]**3)*x[0]*x[1]**2
    return grad

def Beale_hessian(x):
    hessian = np.zeros((2,2))
    hessian[0,0] = 2*(x[1] - 1)**2 + 2*(x[1]**2 - 1)**2 + 2*(x[1]**3 - 1)**2
    hessian[0,1] = 2*(1.5 - x[0] + x[0]*x[1]) + 4*(2.25 - x[0] + x[0]*x[1]**2)*x[1] + 6*(2.625 - x[0] + x[0]*x[1]**3)*x[1]**2
    hessian[1,0] = hessian[0,1]
    hessian[1,1] = 30 * x[0] ** 2 * x[1] ** 4 + 12 * x[1] ** 2 * x[0] ** 2  - 12 * x[0] ** 2 *x[1] - 2 * x[1] * x[1] + 31.5 * x[0] * x[1] + 9 * x[1]
    return hessian


def problem10_main(x):
    n = 10
    f = x[0] * x[0]
    for i in range(9):
        f = f + (x[i+1] - x[i]) ** (2 * (i+1))
    
    return f

def problem10_grad(x):
    g = np.zeros_like(x)
    g[0] = x[0] * 2 + 2 * (x[0] - x[1])
    for i in range(9):
        g[i] = g[i] + 2 * (i + 1) * (x[i] - x[i+1]) ** (2 * i + 1)
        g[i + 1] = g[i + 1] + 2 * (i + 1) * (x[i+1] - x[i]) ** (2 * i + 1) 
    return g

def problem10_hessian(x):
    h = np.zeros((10,10))
    h[0, 0] = 2
    for i in range(9):
        h[i, i] = h[i, i] + (2 * i + 2) * (2 * i + 1) * (x[i] - x[i+1]) ** (2 * i)
        h[i, i + 1] = h[i, i+1] - 2 * (i + 1) * (2 * i + 1) * (x[i] - x[i + 1]) ** (2 * i)
        h[i +1, i] = h[i+1, i] - 2 * (i + 1) * (2 * i + 1) * (x[i] - x[i + 1]) ** (2 * i)
        h[i+1, i+1] = h[i+1, i+1] +  2 * (i + 1) * (2 * i + 1) * (x[i+1] - x[i]) ** (2 * i)
    
    return h
