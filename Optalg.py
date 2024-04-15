import numpy as np
import time
import os

from optimize_obj import *



def ArmijoBacktracking(f, x, d, f0, grad, alphainit = 1.0,c1 = 1e-4, beta = 0.5, minstep = 1e-13):
    ''' 
    Armijo line search
    Search for t such that f(x + t*d) <= f(x) + c1*t*grad(x)*d
    '''
    t = alphainit
    while True:
#        print(grad)
#        print(d)
        if f(x + t*d) <= f0 + c1 * t * np.dot(grad, d):
            return t
        t *= beta
        if t < minstep:
            print('Warning: line search failed to find a good step length')
            break
    return t



def WolfeBacktracking(f, gradfunc, x, d, alphainit = 1.0, c1 = 1e-4, c2 = 0.9, beta = 0.5, maxiter = 200):
    '''
    Wolfe line search
    Search for t such that f(x + t*d) <= f(x) + c1*t*grad(x)*d
    and grad(x + t*d)*d >= c2*grad(x)*d
    '''
    alpha = alphainit
    alpha_lb = 0
    alpha_ub = np.inf
    
    f0 = f(x)
    g0 = gradfunc(x)
#    print(g0)
#    print(d)
    gp = np.dot(g0, d)
#    print(gp, c1, c2, beta)
    if gp >= 0:
        print('Wolfe Warning: the search direction is not a descent direction')
        return 0
    for i in range(maxiter):
#        print(alpha_lb, alpha_ub)
        if f(x +alpha * d) > f0 + c1 * alpha * gp:
            alpha_ub = alpha
        else:
            if np.dot(gradfunc(x + alpha * d), d) < c2 * gp:
                alpha_lb = alpha
            else:
                break
        
        if alpha_ub < np.inf:
            alpha = (alpha_ub + alpha_lb) / 2
        else:
            alpha = alpha / beta
    
    # Check if alpha satisfies Wolfe condition
    
    newx = x + alpha * d
    
    cond1 = ((f(newx) - f0 - c1 * alpha * gp) <= 0)
    cond2 = (np.dot(gradfunc(newx), d) >= c2 * gp)
    
    
    if not cond1:
        print("cond1", alpha)
        print(f(newx) - f0 - c1 * alpha * gp)
        print(np.linalg.norm(g0), gp)
    if not cond2:
        print("cond2", alpha)
        print(np.dot(gradfunc(newx), d), c2 * gp)
        print(np.linalg.norm(g0), gp)
    if cond1 and cond2:
        pass
    elif (not cond1) and cond2:
        print("Wolfe Warning: Descend condition fails")
    elif cond1 and not cond2:
        print("Wolfe Warning: Curvature condition fails")
    elif (not cond1) and (not cond2):
        print("Wolfe Warning: Both condition fails")
#    print(i)
        
    return alpha


def descend_direction(grad, hessian = None, method_options = 'gradient_descend', extrainfo = None):
    # Extrainfo contains the information we store in each step, like B_k in BFGS and (s_i, y_i) in L-BFGS
    extra_output = None
    beta = 1e-4
    methodname = method_options.methodname
    if methodname == 'gradient_descend':
#        return -quad_grad, extra_output
        return -grad, extra_output
    elif methodname == 'Newton':
        return -np.linalg.solve(hessian, grad), extra_output
    elif methodname == 'modified_Newton':
        # Add a small positive diagonal term to the hessian to make it positive definite
        hessian_diag = np.diag(hessian)
        
        if min(hessian_diag)>0:
            delta = 0
        else:
            delta = -min(hessian_diag) + beta
        while True:
            # Add a constant to the diagonal of the hessian, and judge if it is positive definite
            hessian_modified = hessian + delta * np.eye(hessian.shape[0])
            # Try to do Cholesky decomposition
            try:
                L = np.linalg.cholesky(hessian_modified)
                break
            except:
                delta = np.max([2*delta, beta])
            
            # Solve for the descend direction
        p =  -np.linalg.solve(hessian_modified, grad)   
        
        extra_output = delta
        return p, extra_output   
    
    elif methodname == 'BFGS' or methodname == 'DFP':
#        p = np.linalg.solve(extrainfo, -grad)
        p = -np.dot(extrainfo, grad)
        return p, extra_output
    
    elif methodname == 'L-BFGS':
        # Use two loop to compute the p
        m = len(extrainfo)
        if m == 0:
            gamma = 1
            p = -grad
            return p, extra_output
        
        sk = extrainfo[-1][0]
        yk = extrainfo[-1][1]
        gamma = np.dot(sk, yk) / np.dot(yk, yk)
        q = grad
        
        alphalist = np.zeros(m)
        rholist = np.zeros(m)
        
        for i in range(m - 1,-1,-1):
            
            sk = extrainfo[i][0]
            yk = extrainfo[i][1]
            
            rho = 1 / np.dot(sk, yk)
            alpha = rho * np.dot(sk, q)
            alphalist[i] = alpha
            rholist[i] = rho
            q = q - alpha * yk
        r = gamma * q
        for i in range(m):
            sk = extrainfo[i][0]
            yk = extrainfo[i][1]
            rho = rholist[i]
            alpha = alphalist[i]
            
            beta = rho * np.dot(yk, r)
            r = r + sk * (alpha - beta)
        p = -r
        
        return p, extra_output
    
    elif methodname == 'Newton-CG':
        d = -grad
        j = 0                       # j: inner loop iteration number
        stop = False      # inner loop stop flag for CG method  
        z = np.zeros_like(d)
        r = grad
        
        eta_Newton = method_options.eta_newton
        eta = eta_Newton
        if eta_Newton == 'nonlinear':
            # Criterion from O. Ghattas and K. Willcox Paper
            alpha = method_options.alpha_newton
            eta = np.min([np.linalg.norm(grad) ** alpha, 0.5])
        elif eta_Newton == 'Eisenstat-Walker':
            # Criterion from Eisenstat and Walker
            oldfk, fk = extrainfo
            alpha = method_options.alpha_newton
            gamma = method_options.gamma_newton
            eta = gamma * (np.abs(fk + 1e-8) ** alpha) / (np.abs(oldfk + 1e-8) ** alpha)
        
        while not stop:
            hd = hessian @ d
            dThd = np.dot(d, hd)
            
            if dThd <= 0:
                # If we meet a negative curvature direction
                if j == 0:
                    p = d
                    stop = True
                else:
                    p = z 
                    stop = True
                    
            else:
                j = j + 1
                rnorm = np.dot(r, r)
                alpha = rnorm / dThd
                z = z + alpha * d
                r = r + alpha * hd
                if np.linalg.norm(r) < eta * np.linalg.norm(grad):
                    p = z
                    stop = True
                
                beta = np.dot(r,r) / rnorm
                d = -r + beta * d 
        
        
        
        extra_output = eta
        return p, extra_output
    
    else:
        print('Error: method not supported')
        return None
    

def extrainfo_update(extrainfo, method_options, info_k):
    # in extrainfo, we store the extra information we need for doing optimization
    # in info_k object, we will store the new information we gain from current iteration
    
    # For DFP method, extrainfo refers to H_k, and info_k is {s_k, y_k}, which is saved as a 3-tuple
    # For BFGS method, extrainfo refers to H_k, and info_k is {s_k, y_k}, which is saved as a 3-tuple
    # For L-BFGS method, extra info refers to {(s_{k-m}..., s_{k-1}), (y_{k-m}, ..., y_{k-1})}, and info_k is {s_k, y_k}
    
    
    methodname = method_options.methodname
    
    if methodname == 'DFP':
        epsmin = 1e-8
        skip = False
        
        sk = info_k[0]; yk = info_k[1]
        n = sk.size
        
        # Judge if we need to skip the updates
        sy = np.dot(yk, sk)
        
        if sy <= epsmin * np.linalg.norm(yk) * np.linalg.norm(sk):
            newinfo = extrainfo
            skip = True
        else:
            # Do DFP update
            Hy = extrainfo @ yk
            yTHy = np.dot(yk, Hy)
            newinfo = extrainfo + np.outer(sk, sk) / sy - np.outer(Hy, Hy) / yTHy
            
        return newinfo, skip
            
    
    if methodname == 'BFGS':
        epsmin = 1e-8
        skip = False
    
        sk = info_k[0]
        yk = info_k[1]
        n = sk.size
        
        # Judge if we need to skip the updates
        sy = np.dot(yk, sk) 
        
        if sy <= epsmin * np.linalg.norm(yk) * np.linalg.norm(sk):
            newinfo = extrainfo
            skip = True
        else:
            # Do BFGS update
            Hy = extrainfo @ yk
            yTHy = np.dot(Hy, yk)
            newinfo = extrainfo + np.outer(sk, sk) * (sy + yTHy) / sy**2 - 1 / sy * (np.outer(Hy, sk) + np.outer(sk, Hy))
        
        return newinfo, skip
    
    if methodname == 'L-BFGS':
        m_LBFGS = method_options.m_LBFGS
        epsmin = 1e-8
        skip = False
    
        sk = info_k[0]
        yk = info_k[1]
        
        if np.dot(yk,sk) <= epsmin * np.linalg.norm(yk) * np.linalg.norm(sk):
            skip = True
            return extrainfo, skip
        else: 
            n = sk.size
            k = len(extrainfo)

            if k < m_LBFGS:
                extrainfo.append([sk, yk])
            else:
                extrainfo.append([sk, yk])
                extrainfo.pop(0)
            
            return extrainfo, skip

           

def OptAlg(Obj, xinit, outputfile = None):
    # The overall optimizing function.

    x = xinit
    f = Obj.f
    g = Obj.grad
    hessian = Obj.hessian
    dimension = x.size
    
    # Read converge_options
    maxiter = Obj.converge_options.maxiter
    ftol = Obj.converge_options.ftol
    gtol = Obj.converge_options.gtol
    xtol = Obj.converge_options.xtol
    
    # Read linesearch_options
    linesearch_method = Obj.linesearch_options.method
    beta = Obj.linesearch_options.beta
    c1 = Obj.linesearch_options.c1
    c2 = Obj.linesearch_options.c2
    minstep = Obj.linesearch_options.minstep
    
    # Read method options
    method_options = Obj.method_options
    methodname = method_options.methodname
    
    
    starttime = time.time()
    
    iter = 0
    fk = f(x)
    gk = g(x)
    oldfk = fk
    
    print("Gk:", gk)
    
    f0 = fk
    g0 = gk
    
    losslist = [fk]
    
    if methodname == 'BFGS' or methodname == 'L-BFGS' or methodname == 'DFP':
        skiptime = 0
    else:    
        skiptime = None
    
    if methodname == 'Newton-CG':
        CGitertime = 0
        
    
    if methodname == 'BFGS' or methodname == 'DFP':
        extrainfo = np.eye(dimension)
    elif methodname == 'L-BFGS':
        extrainfo = []    
    
    
    
    # Make output
    
    
    if methodname == 'modified_Newton':
        print("Iter\tfval\t\t||grad||\talpha\t\tdelta\t\tfcal\t\tgcal\t\thcal")
    elif methodname == 'Newton-CG':
        print("Iter\tfval\t\t||grad||\talpha\t\teta\t\tfcal\t\tgcal\t\thcal")
    else:
        print("Iter\tfval\t\t||grad||\talpha\t\tfcal\t\tgcal\t\thcal")
    
    # If outputfile is not none, create a new file and write the header
    if outputfile is not None:
        with open(outputfile, 'w') as file:
            if methodname == 'modified_Newton':
                file.write("Iter\tfval\t\t||grad||\talpha\t\tdelta\t\tfcal\t\tgcal\t\thcal\n")
            elif methodname == 'Newton-CG':
                file.write("Iter\tfval\t\t||grad||\talpha\t\tCGiter\t\tfcal\t\tgcal\t\thcal\n")
            else:
                file.write("Iter\tfval\t\t||grad||\talpha\t\tfcal\t\tgcal\t\thcal\n")
    
    
    num_fcal = Obj.func_call
    num_gcal = Obj.grad_call
    num_hcal = Obj.hessian_call
            
    print_output(iter, fk, np.linalg.norm(gk), 0, outputfile = outputfile, num_fcal = num_fcal, num_gradcal = num_gcal, num_hesscal = num_hcal)
    while True:
        
        iter = iter + 1
        # Find descend direction
        hk = None
        if methodname == 'modified_Newton':
            hk = hessian(x)
            pk, extra_output = descend_direction(gk, hk, method_options = method_options)
            delta = extra_output
        elif methodname == 'gradient_descend':
            pk = descend_direction(gk, hessian = None, method_options = method_options)
            pk = pk[0]
        
        elif methodname == 'Newton':
            hk = hessian(x)
            pk, extra_output = descend_direction(gk, hk, method_options)
            # Judge if pk is a descent direction
            
            if np.dot(pk, gk) > 0:
                converge_info = 'pk is not a descent direction.'
                converge_reason = '$p^Tg > 0$'
                success = 0
                break
#                pk = -gk
        elif methodname == 'Newton-CG':
            hk = hessian(x)
            newton_extrainfo = None
            if method_options.eta_newton == 'Eisenstat-Walker':
                newton_extrainfo = [oldfk, fk]
            
            
            pk, extra_output = descend_direction(gk, hk, method_options, extrainfo = newton_extrainfo)
            CGitertime = extra_output
            
        
        elif methodname == 'BFGS' or methodname == 'L-BFGS' or methodname == 'DFP':
            pk, extra_output = descend_direction(gk, hessian = None, extrainfo = extrainfo, method_options = method_options)
            
            
    
        # Find step length using Armijo or Wolfe linesearch
        if linesearch_method == 'Armijo':
            if iter == 1:
                alphainit = 1
            else:
                phi = np.dot(pk, gk)
                alphainit = 2 * (fk - oldfk) / phi
        elif linesearch_method == 'Wolfe':
            alphainit = Obj.linesearch_options.alpha
            
        if linesearch_method == 'Armijo':
#            print("Current line search method: ", linesearch_method)
            alpha = ArmijoBacktracking(f, x, pk, fk, gk, alphainit = alphainit, c1 = c1, beta = beta, minstep = minstep)
        elif linesearch_method == 'Wolfe':
#            print("Current line search method: ", linesearch_method)
            alpha = WolfeBacktracking(f, g, x, pk, alphainit = alphainit, c1 = c1, c2 = c2, maxiter = 200)
        
        # Update x
        oldx = x
        x = x + alpha * pk
        
        oldfk = fk
        oldgk = gk
    
        fk = f(x)
        gk = g(x)
        
        losslist.append(fk)
        
        if methodname == 'BFGS' or methodname == 'L-BFGS':
            sk = alpha * pk
            yk = gk - oldgk
            info_k = (sk, yk)
            extrainfo, skip = extrainfo_update(extrainfo, method_options, info_k)
            if skip:
                skiptime = skiptime + 1
        
        
        
        
        # Output
        num_fcal = Obj.func_call
        num_gcal = Obj.grad_call
        num_hcal = Obj.hessian_call
    
        if methodname == 'modified_Newton':
            print_output(iter, fk, np.linalg.norm(gk), alpha, delta, outputfile= outputfile, num_fcal = num_fcal, num_gradcal = num_gcal, num_hesscal = num_hcal)
        elif methodname == 'Newton-CG':
            print_output(iter, fk, np.linalg.norm(gk), alpha, CGitertime, outputfile= outputfile, num_fcal = num_fcal, num_gradcal = num_gcal, num_hesscal = num_hcal)
        else:
            print_output(iter, fk, np.linalg.norm(gk), alpha, outputfile= outputfile, num_fcal = num_fcal, num_gradcal = num_gcal, num_hesscal = num_hcal)
            
        # Convergence check
        if np.linalg.norm(gk) < gtol * np.min([np.linalg.norm(g0), 1.0]):
            success = 1
            converge_reason = 'gtol'
            converge_info = 'Converged since gradient tolerance'
            break
        elif np.abs(fk - oldfk) < ftol * np.min([1.0, np.abs(oldfk)]):
            success = 1
            converge_reason = 'ftol'
            converge_info = 'Converged since function value tolerance'
            break
        elif np.linalg.norm(x - oldx) < xtol:
            success = 1
            converge_reason = 'xtol'
            converge_info = 'Converged since x tolerance'
            break
        elif iter >= maxiter:
            success = 0
            converge_reason = 'max iter'
            converge_info = 'Converged since maximum iteration'
            break
    
#    print(converge_info)
    endtime = time.time()
    runtime = endtime - starttime

    
    # Use last 10 iterations to calculate the converge rate
    
    if len(losslist) > 10:
        losslist = losslist[-10:]
    else:
        losslist = losslist
        
    losslist = np.array(losslist)
    limit = losslist[-1]
    en = losslist - limit
    en = en[:-1] + 1e-10
    
    alpha = np.log(en[2:] / en[1:-1]) / np.log(en[1:-1] / en[:-2])
    converge_rate = np.mean(alpha)
    
    if methodname == 'BFGS' or methodname == 'L-BFGS' or 'DFP':
        other = skiptime
    elif methodname == 'Newton-CG':
        other = CGitertime
    else:
        other = None
        

    fcal = Obj.func_call
    gcal = Obj.grad_call
    hcal = Obj.hessian_call
    
    print("num of calls:", fcal, gcal, hcal)
    myout = Optout(method_options, x, fk, np.linalg.norm(gk), iter, runtime, success, converge_rate, converge_reason, converge_info, outputfile, other, linesearch_method = linesearch_method, fcal = fcal, gcal = gcal, hcal = hcal)
    #myout.printinfo()
    
    return x, myout



