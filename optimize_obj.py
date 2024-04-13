
import numpy as np
import time
import os
from function_library import * 


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
        print("  Line search method = ", self.method)
        print("  alphainit = ", self.alpha)
        print("  beta = ", self.beta)
        print("  c1 = ", self.c1)
        if self.method == 'Wolfe':
            print("c2 = ", self.c2)
        print("minstep = ", self.minstep)
        
        if self.outputfile is not None:
            with open(self.outputfile, "a") as f:
                f.write("Line search options:\n")
                f.write("  Line search method = " + self.method + "\n")
                f.write("  alphainit = " + str(self.alpha) + "\n")
                f.write("  beta = " + str(self.beta) + "\n")
                f.write("  c1 = " + str(self.c1) + "\n")
                if self.method == 'Wolfe':
                    f.write("  c2 = " + str(self.c2) + "\n")
                f.write("  minstep = " + str(self.minstep) + "\n")
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
        print('  ftol = ', self.ftol)
        print('  gtol = ', self.gtol)
        print('  xtol = ', self.xtol)
        print('  maxiter = ', self.maxiter)
        
        if self.outputfile is not None:
            with open(self.outputfile, "a") as f:
                f.write("Convergence options:\n")
                f.write("  ftol = " + str(self.ftol) + "\n")
                f.write("  gtol = " + str(self.gtol) + "\n")
                f.write("  xtol = " + str(self.xtol) + "\n")
                f.write("  maxiter = " + str(self.maxiter) + "\n")
                f.write("\n")
                
class method_options():
    def __init__(self, methodname = 'gradient_descend', eta_newton = None, outputfile = None, m_LBFGS = 10):
        self.methodname = methodname
        self.eta_newton = 0.01
        self.outputfile  = outputfile
        self.m_LBFGS = 10
        
        self.alpha_newton = 0.5
        self.gamma_newton = 0.5
        
        
        
    def printinfo(self):
        print("Optimization options:")
        print("  methodname = ", self.methodname)
        if self.methodname == 'Newton-CG':
            if self.eta_newton == 'nonlinear':
                print("  eta_newton updated rule: nonlinear")
                print("  nonlinear exponent alpha = ", self.alpha_newton)
            if self.eta_newton == 'Eisenstat-Walker':
                print("  eta_newton updated rule: Eisenstat-Walker")
                print("  gamma in Eisenstat-Walker = ", self.gamma_newton)
                print("  alpha in Eisenstat-Walker = ", self.alpha_newton)
            if type(self.eta_newton) is float or type(self.eta_newton) is int:
                print("  eta_newton = ", self.eta_newton)
                
        elif self.methodname == 'L-BFGS':
            print("  m = ", self.m_LBFGS)
            
        if self.outputfile is not None:
            with open(self.outputfile, "a") as f:
                f.write("Optimization options:\n")
                f.write("  methodname = " + self.methodname + "\n")
                if self.methodname == 'Newton-CG':
                    if self.eta_newton == 'nonlinear':
                        f.write("  eta_newton updated rule: nonlinear\n")
                        f.write("  nonlinear exponent alpha = " + str(self.alpha_newton) + "\n")
                    if self.eta_newton == 'Eisenstat-Walker':
                        f.write("  eta_newton updated rule: Eisenstat-Walker\n")
                        f.write("  gamma in Eisenstat-Walker = " + str(self.gamma_newton) + "\n")
                        f.write("  alpha in Eisenstat-Walker = " + str(self.alpha_newton) + "\n")
                    if type(self.eta_newton) is float or type(self.eta_newton) is int:
                        f.write("  eta_newton = " + str(self.eta_newton) + "\n")
                elif self.methodname == 'L-BFGS':
                    f.write("  m = " + str(self.m_LBFGS) + "\n")
                f.write("\n")
        
        
        
        
                
class Obj():
    def __init__(self, dim, f, grad, hessian, method = 'gradient_descend', outputfile = None, objname = None):
        # method can be varied from {'gradient_descend', 'modified_newton', 'newton', 'Newton-CG', 'BFGS', 'L-BFGS', 'DFP'}
        self.dim = dim
        #self.f = f
        #self.grad = grad
        #self.hessian = hessian
        self.__function__ = f
        self.__gradient__ = grad
        self.__hessian__ = hessian
        self.func_call = 0
        self.grad_call = 0
        self.hessian_call = 0
        self.outputfile = outputfile
        
        self.linesearch_options = linesearch_options()
        self.converge_options = converge_options()
        self.method_options = method_options()
        self.objname = objname
        
    def f(self, x):
        self.func_call += 1
        return self.__function__(x)

    def grad(self, x):
        self.grad_call += 1
        return self.__gradient__(x)
    
    def hessian(self, x):
        self.hessian_call += 1
        return self.__hessian__(x)
    
    def reset_call(self):
        self.func_call = 0
        self.grad_call = 0
        self.hessian_call = 0
        
    def printinfo(self):
        # Print the information of the objective function
        if self.objname is not None:
            print('Objective function: ', self.objname)
        print('Objective function info:')
        print('problem dim = ', self.dim)
        print("method = ", self.method_options.methodname)
        
        if self.outputfile is not None:
            with open(self.outputfile, "a") as f:
                if self.objname is not None:
                    f.write('Objective function: ' + self.objname + "\n")
                f.write('Objective function info:\n')
                f.write('problem dim = ' + str(self.dim) + "\n")
                f.write("method = " + self.method_options.methodname + "\n")
                f.write("\n")
        
        self.method_options.printinfo()
        self.linesearch_options.printinfo()
        self.converge_options.printinfo()
        

    def set_options(self, options = None):
        # Provide the options of the objective function as a dictionary
        
        if options is None:
            return 0
        
        elif not isinstance(options, dict):
            raise ValueError('The options should be a dictionary')
        
        # Otherwise, provide the dictionary of the options
        for key, value in options.items():
            if key == 'c1' or key == 'alpha' or key == 'beta' or key == 'minstep' or key == 'c2':
                if (type(value) is not int) and (type(value) is not float):
                    raise ValueError('The value of %s should be a number' % key)
                setattr(self.linesearch_options, key, value)
            if key == 'linesearch_method':
                if value == 'Armijo' or value == 'Wolfe':
                    self.linesearch_options.method = value
                else:
                    raise ValueError('The linesearch method should be either Armijo or Wolfe')
            
            if key == 'ftol' or key == 'gtol' or key == 'xtol' or key == 'maxiter':
                if (type(value) is not int) and (type(value) is not float):
                    raise ValueError('The value of %s should be a number' % key)
                setattr(self.converge_options, key, value)
            if key == 'optimization_method':
                if value in ['gradient_descend', 'modified_Newton', 'Newton', 'Newton-CG', 'BFGS', 'L-BFGS', 'DFP']:
                    #self.method = value
                    self.method_options.methodname = value
                else:
                    raise ValueError('The optimization method should be either gradient_descend, modified_newton, newton, Newton-CG, BFGS, L-BFGS, DFP')
            
            if key == 'eta_Newton':
                if value == 'nonlinear':
                    self.method_options.eta_newton = 'nonlinear'
                elif value == 'Eisenstat-Walker' or value == 'EW':
                    self.method_options.eta_newton = 'Eisenstat-Walker'
                elif type(value) is int or type(value) is float:
                    self.method_options.eta_newton = value
                else:
                    raise ValueError('The value of eta_Newton should be either \'nonlinear\', \'Eisenstat-Walker\' or a number')
            
            if key == 'm_LBFGS':
                if type(value) is not int:
                    raise ValueError('The value of m_LBFGS should be an integer')
                self.method_options.m_LBFGS = value
            if key == 'objname':
                self.objname = value
            if key == 'outputfile':
                self.outputfile = value
                self.linesearch_options.outputfile = value
                self.converge_options.outputfile = value
                self.method_options.outputfile = value
                
                
        
        



def rosenbrock_obj(dim):
    rosenbrock_main = lambda x: rosenbrock(x, dim)
    rosenbrock_grad_main = lambda x: rosenbrock_grad(x, dim)
    rosenbrock_hessian_main = lambda x: rosenbrock_hessian(x, dim)
    
    return Obj(dim, rosenbrock_main, rosenbrock_grad_main, rosenbrock_hessian_main)


def Beale_obj():
    return Obj(2, Beale, Beale_grad, Beale_hessian)


def problem10_obj():
    return Obj(10, problem10_main, problem10_grad, problem10_hessian)


##############################################################################################

# add some new functions to the object declare

# this quad object is for the project 1 - 4
def quad_obj(Q, q): 
    dim = len(q)
    quad_main = lambda x: quad(x, Q, q)
    quad_grad_main = lambda x: quad_grad(x, Q, q)
    quad_hessian_main = lambda x: quad_hessian(x, Q, q)
    
    return Obj(dim, quad_main, quad_grad_main, quad_hessian_main)



# this special_quad object for the project problems 5 - 6
def special_quad_obj(Q, sigma): 
    dim = len(Q[0])
    special_quad_main = lambda x: special_quad(x, Q, sigma)
    special_quad_grad_main = lambda x: special_quad_grad(x, Q, sigma)
    special_quad_hessian_main = lambda x: special_quad_hessian(x, Q, sigma)
    
    return Obj(dim, special_quad_main, special_quad_grad_main, special_quad_hessian_main)

# the data_fit_obj is for the project problem 9

def data_fit_obj(y):
    dim = 2
    data_fit_val = lambda x: data_fit(x, y)
    data_fit_grad_val = lambda x: data_fit_grad(x, y)
    data_fit_hessian_val = lambda x: data_fit_hessian(x, y)
    return Obj(dim, data_fit_val, data_fit_grad_val, data_fit_hessian_val)

# the exponential_obj is for the project problem 10-11
def exponential_obj(dim):
    exp_val = lambda x: exponential(dim, x)
    exp_grad = lambda x: exponential_grad(dim, x)
    exp_hessian = lambda x: exponential_hessian(dim, x)
    return Obj(dim, exp_val, exp_grad, exp_hessian)

# the genhumps_obj is for the project problem 12
def genhump_obj(dim):
    genhump_val = lambda x: genhump(dim, x)
    genhump_grad_val = lambda x: genhump_grad(dim, x)
    genhump_hess_val = lambda x: genhump_hess(dim, x)
    return Obj(dim, genhump_val, genhump_grad_val, genhump_hess_val)    



##############################################################################################

def print_output(iternum, fval, gradnorm, currentalpha, delta = None, outputfile = None, num_fcal = None, num_gradcal = None, num_hesscal = None):
    if delta is None:
        print("%d\t%.8f\t%.8f\t%.8f\t\t%d\t\t%d\t\t%d" % (iternum, fval, gradnorm, currentalpha, num_fcal, num_gradcal, num_hesscal))
    else:
        print("%d\t%.8f\t%.8f\t%.8f\t%.8f\t\t%d\t\t%d\t\t%d" % (iternum, fval, gradnorm, currentalpha, delta, num_fcal, num_gradcal, num_hesscal))
        
    if outputfile is not None:
        with open(outputfile, 'a') as f:
            if delta is None:
                f.write("%d\t%.8f\t%.8f\t%.8f\t\t%d\t\t%d\t\t%d\n" % (iternum, fval, gradnorm, currentalpha, num_fcal, num_gradcal, num_hesscal))
            else:
                f.write("%d\t%.8f\t%.8f\t%.8f\t%.8f\t\t%d\t\t%d\t\t%d\n" % (iternum, fval, gradnorm, currentalpha, delta, num_fcal, num_gradcal, num_hesscal))


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
