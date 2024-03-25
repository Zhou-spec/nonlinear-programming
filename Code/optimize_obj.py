
import numpy as np
import time
import os


class linesearch_options():
    # Line search options
    # Method can be chosen from 'Armijo' or 'Wolfe'
    def __init__(self, c1 = 1e-4, alpha = 1, beta = 0.5, minstep = 1e-13, method = 'Armijo', c2 = 0.9, outputfile = None):
        
        self.method = method
        self.alpha = alpha          # Initial size of steplength
        self.beta = beta            # alpha^{k+1} = beta * alpha^k, beta will always be smaller than 1.
        self.c1 = c1                # Armijo or Wolfe condition constant
        self.c2 = c2                # Wolfe condition constant
        self.minstep = minstep        # Minimum step length
        self.outputfile = outputfile
        
    def printinfo(self):
        print("Line search options:")
        print("Line search method = ", self.method)
        print("alphainit = ", self.alpha)
        print("beta = ", self.beta)
        print("c1 = ", self.c1)
        if self.method == 'Wolfe':
            print("c2 = ", self.c2)
        print("minstep = ", self.minstep)
        
        if self.outputfile is not None:
            with open(self.outputfile, "a") as f:
                f.write("Line search options:\n")
                f.write("Line search method = " + self.method + "\n")
                f.write("alphainit = " + str(self.alpha) + "\n")
                f.write("beta = " + str(self.beta) + "\n")
                f.write("c1 = " + str(self.c1) + "\n")
                if self.method == 'Wolfe':
                    f.write("c2 = " + str(self.c2) + "\n")
                f.write("minstep = " + str(self.minstep) + "\n")
                f.write("\n")

class converge_options():
    def __init__(self, ftol = 1e-10, gtol = 1e-10, xtol = 1e-10, maxiter = 1000, outputfile = None):
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.maxiter = maxiter
        self.outputfile = outputfile
        
    def printinfo(self):
        print('Convergence options:')
        print('ftol = ', self.ftol)
        print('gtol = ', self.gtol)
        print('xtol = ', self.xtol)
        print('maxiter = ', self.maxiter)
        
        if self.outputfile is not None:
            with open(self.outputfile, "a") as f:
                f.write("Convergence options:\n")
                f.write("ftol = " + str(self.ftol) + "\n")
                f.write("gtol = " + str(self.gtol) + "\n")
                f.write("xtol = " + str(self.xtol) + "\n")
                f.write("maxiter = " + str(self.maxiter) + "\n")
                f.write("\n")
                
class Obj():
    def __init__(self, dim, f, grad, hessian, method = 'gradient_descend', outputfile = None):
        # method can be varied from {'gradient_descend', 'modified_newton', 'newton', 'Newton-CG', 'BFGS', 'L-BFGS'}
        self.dim = dim
        self.f = f
        self.grad = grad
        self.hessian = hessian
        self.outputfile = outputfile
        
        self.linesearch_options = linesearch_options()
        self.converge_options = converge_options()
        self.method = method
        
    def printinfo(self):
        print('Objective function info:')
        print('problem dim = ', self.dim)
        print("method = ", self.method)
        self.linesearch_options.printinfo()
        self.converge_options.printinfo()
        

from function_library import * 

def rosenbrock_obj(dim):
    rosenbrock_main = lambda x: rosenbrock(x, dim)
    rosenbrock_grad_main = lambda x: rosenbrock_grad(x, dim)
    rosenbrock_hessian_main = lambda x: rosenbrock_hessian(x, dim)
    
    return Obj(dim, rosenbrock_main, rosenbrock_grad_main, rosenbrock_hessian_main)


def Beale_obj():
    return Obj(2, Beale, Beale_grad, Beale_hessian)


def problem10_obj():
    return Obj(10, problem10_main, problem10_grad, problem10_hessian)




def print_output(iternum, fval, gradnorm, currentalpha, delta = None, outputfile = None):
    if delta is None:
        print("%d\t%.8f\t%.8f\t%.8f" % (iternum, fval, gradnorm, currentalpha))
    else:
        print("%d\t%.8f\t%.8f\t%.8f\t%.8f" % (iternum, fval, gradnorm, currentalpha, delta))
        
    if outputfile is not None:
        with open(outputfile, 'a') as f:
            if delta is None:
                f.write("%d\t%.8f\t%.8f\t%.8f\n" % (iternum, fval, gradnorm, currentalpha))
            else:
                f.write("%d\t%.8f\t%.8f\t%.8f\t%.8f\n" % (iternum, fval, gradnorm, currentalpha, delta))


class Optout():
    # Summarize the output of the optimization algorithm
    def __init__(self, method, x, fval, gradnorm, iternum, runtime, success = 1, converge_rate = None, converge_reason = '', message = '', outputfile = None, other = None):
        self.x = x
        self.method = method
        self.fval = fval
        self.gradnorm = gradnorm
        self.iternum = iternum
        self.message = message
        self.success = success
        self.runtime = runtime
        self.converge_rate = converge_rate
        self.converge_reason = converge_reason
        self.outputfile = outputfile
        self.other = other
        
        
    def printinfo(self):
        print("\n")
        print("Algorithm: ", self.method)
        print('Optimization output:')
        print("If success:", self.success)
        print('fval = ', self.fval)
        print('gradnorm = ', self.gradnorm)
        print('iternum = ', self.iternum)
        print("runtime = ", self.runtime, "s")
        print('converge info = ', self.message)

        if self.method == 'BFGS' or self.method == 'L-BFGS':
            print("skip time = ", self.other)
            
        if self.outputfile is not None:
            with open(self.outputfile, 'a') as f:
                f.write("\n")
                f.write('Optimization output:\n')
                f.write("If success: %d\n" % self.success)
                f.write('fval = %.8f\n' % self.fval)
                f.write('gradnorm = %.8f\n' % self.gradnorm)
                f.write('iternum = %d\n' % self.iternum)
                f.write("runtime = %.8f s\n" % self.runtime)
                f.write('Converge info = %s\n' % self.message)
                if self.method == 'BFGS' or self.method == 'L-BFGS':
                    f.write('skip time = %s\n' % self.other)
