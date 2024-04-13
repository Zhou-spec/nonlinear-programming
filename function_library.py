    
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


##############################################################################################

# the following implementation is for the project problems 1 - 4
def quad(x, Q, q):
    # dim is the dimension of the problem
    # x is numpy array
    return 0.5 * np.matmul(x, np.matmul(Q, x.transpose())) + np.dot(q, x)

def quad_grad(x, Q, q):
    return np.matmul(Q, x) + q

def quad_hessian(x, Q, q):
    return Q


##############################################################################################

# the following implementation is for the project problems 5 - 6
def special_quad(x, Q, sigma):
    val = 0.5 * np.matmul(x, x) + 0.25 * sigma * np.power(np.matmul(x, np.matmul(Q, x)), 2)
    return val

def special_quad_grad(x, Q, sigma):
    return x + 0.5 * sigma * np.matmul(Q, x) * np.matmul(x, np.matmul(Q, x))

def special_quad_hessian(x, Q, sigma):
    return np.eye(len(x)) + sigma * np.power(np.matmul(x, np.matmul(Q, x)), 2) + 0.5 * sigma * np.matmul(x, np.matmul(Q, x)) * Q

##############################################################################################

# the following implementation is for the project problems 9
def data_fit(x, y):
    return np.sum((y - x[0] * (1 - x[1]))**2)

def data_fit_grad(x, y):
    # parital derivative with respect to x[0], x[1]
    return np.array([2 * np.sum((y - x[0] * (1 - x[1])) * (x[1] - 1)), 2 * np.sum((y - x[0] * (1 - x[1])) * x[0])])

def data_fit_hessian(x, y):
    f_xx = 2 * np.sum((1 - x[1])**2)
    f_yy = 2 * np.sum(x[0]**2)
    f_xy = 2 * np.sum(y - 2 * x[0] + 2 * x[0] * x[1])
    return np.array([[f_xx, f_xy], [f_xy, f_yy]])

##############################################################################################

# the following implementation is for the project problems 10-11
def exponential(dim, x):
    # exponential function is defined as
    # frac{exp(x_1) - 1}{exp(x_1) + 1} + 0.1 exp{-x_1} + sum_{i=2}^d (x_i - 1)^4
    val = (np.exp(x[0]) - 1) / (np.exp(x[0]) + 1) + 0.1 * np.exp(-x[0]) + np.sum((x[1:] - 1)**4)
    return val

def exponential_grad(dim, x):
    # gradient of the exponential function
    grad = np.zeros(dim)
    grad[0] = (2 * np.exp(x[0])) / (np.exp(x[0]) + 1)**2 - 0.1 * np.exp(-x[0]) 
    grad[1:] = 4 * (x[1:] - 1)**3
    return grad

def exponential_hessian(dim, x):
    # hessian of the exponential function
    hessian = np.zeros((dim, dim))
    hessian[0, 0] = (2 * np.exp(x[0]) * np.power(np.exp(x[0]) + 1, 2) - 4 * np.exp(x[0]) * (np.exp(x[0] + 1))) / np.power(np.exp(x[0]) + 1, 4) + 0.1 * np.exp(-x[0])
    for i in range(1, dim):
        hessian[i, i] = 12 * np.power(x[i] - 1, 2)

    return hessian

##############################################################################################

# the following implementation is for the project problems 12
def genhump(dim, x):
    # sum of sin(2 * x_i)^2 * sin(2 * x_{i+1})^2 + 0.05 * (x_i^2 + x_{i+1}^2)
    val = 0
    for i in range(dim - 1):
        val += np.power(np.sin(2 * x[i]), 2) * np.power(np.sin(2 * x[i+1]), 2) + 0.05 * (x[i]**2 + x[i+1]**2)

    return val

def genhump_grad(dim, x):   
    grad = np.zeros(dim)
    grad[0] = 4 * np.sin(2 * x[0]) * np.cos(2 * x[0]) * np.power(np.sin(2 * x[1]), 2) + 0.1 * x[0]
    for i in range(1, dim - 1): 
        grad[i] = 4 * np.sin(2 * x[i]) * np.cos(2 * x[i]) * np.power(np.sin(2 * x[i+1]), 2) + 0.1 * x[i]
        grad[i] += 4 * np.sin(2 * x[i]) * np.cos(2 * x[i]) * np.power(np.sin(2 * x[i-1]), 2) + 0.1 * x[i]

    grad[-1] = 4 * np.sin(2 * x[-1]) * np.cos(2 * x[-1]) * np.power(np.sin(2 * x[-2]), 2) + 0.1 * x[-1]
    return grad

def genhump_hess(dim, x):
    hess = np.zeros((dim, dim))
    hess[0, 0] = 8 * np.cos(2 * x[0]) * np.power(np.sin(2 * x[1]), 2) - 8 * np.power(np.sin(2 * x[0]), 2) * np.sin(2 * x[1]) * np.cos(2 * x[1]) + 0.1
    hess[0, 1] = 8 * np.sin(2 * x[0]) * np.cos(2 * x[0]) * np.sin(2 * x[1]) * np.cos(2 * x[1])
    for i in range(1, dim - 1):
        hess[i, i] = 8 * np.cos(2 * x[i]) * np.power(np.sin(2 * x[i+1]), 2) - 8 * np.power(np.sin(2 * x[i]), 2) * np.sin(2 * x[i+1]) * np.cos(2 * x[i+1]) + 0.1
        hess[i, i+1] = 8 * np.sin(2 * x[i]) * np.cos(2 * x[i]) * np.sin(2 * x[i+1]) * np.cos(2 * x[i+1])
        hess[i, i-1] = 8 * np.sin(2 * x[i]) * np.cos(2 * x[i]) * np.sin(2 * x[i-1]) * np.cos(2 * x[i-1])

    hess[-1, -1] = 8 * np.cos(2 * x[-1]) * np.power(np.sin(2 * x[-2]), 2) - 8 * np.power(np.sin(2 * x[-1]), 2) * np.sin(2 * x[-2]) * np.cos(2 * x[-2]) + 0.1
    hess[-1, -2] = 8 * np.sin(2 * x[-1]) * np.cos(2 * x[-1]) * np.sin(2 * x[-2]) * np.cos(2 * x[-2])
    return hess

