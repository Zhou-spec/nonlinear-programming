
from function_library import *
from optimize_obj import *  
from Optalg import *

    

def rosenbrock_obj(dim):
    rosenbrock_main = lambda x: rosenbrock(x, dim)
    rosenbrock_grad_main = lambda x: rosenbrock_grad(x, dim)
    rosenbrock_hessian_main = lambda x: rosenbrock_hessian(x, dim)
    
    return Obj(dim, rosenbrock_main, rosenbrock_grad_main, rosenbrock_hessian_main)


def Beale_obj():
    return Obj(2, Beale, Beale_grad, Beale_hessian)


def problem10_obj():
    return Obj(10, problem10_main, problem10_grad, problem10_hessian)



from decimal import Decimal
    

# A list of possible options

gradient_descend_options = {"optimization_method": "gradient_descend", "line_search_method": "Armijo", "maxiter": 20000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4}
gradient_descend_W_options = {"optimization_method": "gradient_descend", "line_search_method": "Wolfe", "maxiter": 20000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4, "c2": 0.9}

modified_Newton_options = {"optimization_method": "modified_Newton", "line_search_method": "Armijo", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4}

modified_Newton_W_options = {"optimization_method": "modified_Newton", "line_search_method": "Wolfe", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4, "c2": 0.9}

Newton_CG_options = {"optimization_method": "Newton-CG", "line_search_method": "Armijo", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4}

Newton_CG_W_options = {"optimization_method": "Newton-CG", "line_search_method": "Wolfe", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4, "c2": 0.9}

Newton_CG_W_nonlinear_options = {"optimization_method": "Newton-CG", "line_search_method": "Wolfe", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4, "c2": 0.9, "eta_Newton":"nonlinear"}

BFGS_options = {"optimization_method": "BFGS", "line_search_method": "Armijo", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4}

BFGS_W_options = {"optimization_method": "BFGS", "line_search_method": "Wolfe", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4, "c2": 0.9}

L_BFGS_options = {"optimization_method": "L-BFGS", "line_search_method": "Armijo", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4}

L_BFGS_W_options = {"optimization_method": "L-BFGS", "line_search_method": "Wolfe", "maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6, "xtol":1e-6, "alpha": 1, "beta": 0.5, "c1": 1e-4, "c2": 0.9}


optionslist = [gradient_descend_options, gradient_descend_W_options, modified_Newton_options, modified_Newton_W_options, Newton_CG_options, Newton_CG_W_options, Newton_CG_W_nonlinear_options, BFGS_options, BFGS_W_options, L_BFGS_options, L_BFGS_W_options]



def runcase(Obj, xinit, Objname = 'OptObj', line_search_method = 'Armijo', optionslist = optionslist, make_table = True):
    # Run the optimization for one Obj using three methods: gradient descend, Newton, modified Newton
    
    # Create an outputfile with name Objname, in subdirectory '/result'
    
    # Gradient descend
    myobj = Obj

    outputfile_list = []    
    outinfo_list = []
    for options in optionslist:
        
#        myobj.set_options({'optimization_method': method})
        myobj.set_options(options)
        line_search_method = myobj.linesearch_options.method
        method = myobj.method_options.methodname
        #myobj.method = method
        #myobj.linesearch_options.method = line_search_method 
        
        if line_search_method == 'Armijo':
            outputfile = 'result/' + Objname + '_' + method + '_Armijo.txt'
        elif line_search_method == 'Wolfe':
            outputfile = 'result/' + Objname + '_' + method + '_Wolfe.txt'
        Obj.outputfile = outputfile
        myobj.printinfo()
        
        myobj.reset_call()
        
        xopt, outinfo = OptAlg(myobj, xinit, outputfile)
        
        outputfile_list.append(outputfile)
        outinfo_list.append(outinfo)
        
    if make_table:
        # Print the code for a LaTeX table, with caption Objname.
        # Save the number with .6 digit scientific notation
        # Print the converge rate with .3 digit scientific notation
        # Print the runtime with .3 digit scientific notation
        # Print the number of iterations
        # Print the reason for convergence
        # Print convergence rate


        print("\\begin{table}[htpb]")
        print("\\centering")
        print("\\caption{" + Objname + "}")
        print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}")
        print("\\hline")
        print(" Method & $f_{\\text{val}}$ & $||\\nabla f||$ &  Iter & Runtime & Converge Reason & Other info \\\\")
        print("\\hline")
        
        
        for outinfo in outinfo_list:
            method = outinfo.method
            fval = outinfo.fval
            gradnorm = outinfo.gradnorm
            iternum = outinfo.iternum
            runtime = outinfo.runtime
            converge_reason = outinfo.converge_reason
            otherinfo = outinfo.other
            if otherinfo is None:
                otherinfo = '/'
            else:
                otherinfo = str(otherinfo)
            
            print(method + " & " + "{:.5e}".format(fval) + " & " + "{:.5e}".format(gradnorm) + " & " + str(iternum) + " & " + "{:.3e}".format(runtime) + " & " + converge_reason + " & " + otherinfo + " \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
        
    
    # Write the latex table code in one txt file
    with open('result/' + Objname + ' ' + line_search_method + '_table.txt', 'w') as f:
        f.write("\\begin{table}[htpb]\n")
        f.write("\\centering\n")
        f.write("\\caption{" + Objname + "}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(" Method & $f_{\\text{val}}$ & $||\\nabla f||$ &  Iter & Runtime \\\\\n")
        f.write("\\hline\n")
        
        for outinfo in outinfo_list:
            method = outinfo.method
            fval = outinfo.fval
            gradnorm = outinfo.gradnorm
            iternum = outinfo.iternum
            runtime = outinfo.runtime
            converge_reason = outinfo.converge_reason
            otherinfo = outinfo.other
            if otherinfo is None:
                otherinfo = '/'
            else:
                otherinfo = str(otherinfo)
            
            # name = split the method string by _
            method = method.split('_')[0]
            f.write(method + ' ' + line_search_method + " & " + "{:.5e}".format(fval) + " & " + "{:.5e}".format(gradnorm) + " & " + str(iternum) + " & " + "{:.3e}".format(runtime) + " \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    