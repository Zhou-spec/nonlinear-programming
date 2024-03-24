import numpy as np
import scipy

##############################################################################################################
# implementation of armijo and wolfe line search 
def armijo_line_search(f, f_grad, x_curr, tao, c):
    alpha = 1.0  # Starting with a default step size
    p_curr = -f_grad(x_curr)
    while f(x_curr + alpha * p_curr) > f(x_curr) + c * alpha * np.dot(f_grad(x_curr), p_curr):
        alpha *= tao
    return alpha

def wolfe_line_search(f, f_grad, x_curr, p_curr, c1, c2):
    alpha = 1  # Initial step size
    alpha_max = np.inf
    alpha_min = 0
    while True:
        if f(x_curr + alpha * p_curr) > f(x_curr) + c1 * alpha * np.dot(f_grad(x_curr), p_curr):
            alpha_max = alpha
            alpha = 0.5 * (alpha_min + alpha_max)
        elif np.dot(f_grad(x_curr + alpha * p_curr), p_curr) < c2 * np.dot(f_grad(x_curr), p_curr):
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha = 2 * alpha_min
            else:
                alpha = 0.5 * (alpha_min + alpha_max)
        else:
            break
    return alpha


##############################################################################################################
# implementation of gradient descent method
def gradient_descent(f, f_grad, x_init, tao, c, max_iter):
    x_curr = x_init
    for i in range(max_iter):
        p_curr = -f_grad(x_curr)
        alpha = armijo_line_search(f, f_grad, x_curr, tao, c)
        x_curr = x_curr + alpha * p_curr
    return x_curr


##############################################################################################################
# implementation of modified Newton method
def modified_Newton_method(f, f_grad, f_hess, x0, line_search_method='armijo', tau=0.5, c1=1e-4, c2=0.9, threshold=1e-5, max_iter=1000):
    x_curr = np.array(x0, dtype=float)

    for _ in range(max_iter):
        grad_curr = f_grad(x_curr)
        if np.linalg.norm(grad_curr) < threshold:
            break

        # Compute the Hessian and adjust it if necessary
        A = f_hess(x_curr)
        beta = 0.0001
        delta = max(0, -np.min(np.linalg.eigvalsh(A)) + beta)

        while True:
            try:
                # Try Cholesky decomposition to test positive definiteness
                scipy.linalg.cholesky(A + delta * np.eye(len(x_curr)))
                break
            except scipy.linalg.LinAlgError:
                delta = max(2 * delta, beta)
        
        # Adjusted Hessian
        Bk = A + delta * np.eye(len(x_curr))
        # Solve for the search direction
        p_curr = -scipy.linalg.solve(Bk, grad_curr)

        # Perform line search based on the specified method
        if line_search_method == 'armijo':
            alpha = armijo_line_search(f, f_grad, x_curr, tau, c1)
        elif line_search_method == 'wolfe':
            alpha = wolfe_line_search(f, f_grad, x_curr, p_curr, c1, c2)
        else:
            raise ValueError("Invalid line search method specified.")
        
        x_curr = x_curr + alpha * p_curr

        # if current gradient norm is less than threshold, break
        grad_curr = f_grad(x_curr)
        if np.linalg.norm(grad_curr) < threshold:
            break

    return x_curr, f(x_curr)

##############################################################################################################
# implementation of Newton-CG method

