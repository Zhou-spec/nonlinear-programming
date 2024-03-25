
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

        
    

caseid_list = [1,2,3,4,5,6,7,8,9,10]
objname_list = ["Rosenbrock 1", "Rosenbrock 2", "Rosenbrock 3", "Rosenbrock 4", "Rosenbrock 5", "Rosenbrock 6", "Rosenbrock 7", "Beale Function 1", "Beale Function 2", "Problem 10"]
dim_list = [2, 10, 10, 100, 100, 1000, 10000, 2, 2, 10]
maxiter_list = [5000, 5000, 5000, 5000, 5000, 10000, 10000, 3000, 3000, 3000]
xinit_list = []
for i in range(10):
    if i == 0:
        xinit_list.append(np.array([-1.2, 1]))
    elif i == 1 or i == 3:
        dim = dim_list[i]
        xinit_list.append(-1 * np.ones(dim))
    elif i == 2 or i == 4 or i == 5 or i == 6:
        dim = dim_list[i]
        xinit_list.append(2 * np.ones(dim))
    elif i == 7:
        xinit_list.append(np.array([1, 1]))
    elif i == 8:
        xinit_list.append(np.array([0, 0]))
    elif i == 9:
        xinit_list.append(np.array([1,2,3,4,5,6,7,8,9,10]))


def runcase_with_id(i, methodlist = ['BFGS','L-BFGS','Newton-CG'], make_table = False):
    dim = dim_list[i]
    caseid = caseid_list[i]
    xinit = xinit_list[i]
    objname = objname_list[i]
    maxiter = maxiter_list[i]
        
    if i <= 6:
        myobj = rosenbrock_obj(dim)

    elif i == 7 or i == 8:
        myobj = Beale_obj()
    
    elif i == 9:
        myobj = problem10_obj()
        myobj.linesearch_options.alpha = 1e-4
    
    myobj.converge_options.maxiter = maxiter
    
    print("Current running problem:", objname)
    runcase(myobj, xinit, casenumber = caseid, Objname = objname, methodlist = methodlist, make_table = make_table)
    
    

def runcase(Obj, xinit, casenumber = 1, Objname = 'OptObj', methodlist = ['BFGS','L-BFGS','Newton-CG'], make_table = True):
    # Run the optimization for one Obj using three methods: gradient descend, Newton, modified Newton
    
    # Create an outputfile with name Objname, in subdirectory '/result'
    
    # Gradient descend
    myobj = Obj

    outputfile_list = []    
    outinfo_list = []
    for method in methodlist:
        if Objname == 'Rosenbrock 7' and (method == 'BFGS' or method == 'L-BFGS'):
            print("Skip rosenbrock 7 with " + str(method) + " .")
            continue
        
        myobj.method = method
        if method == 'Newton-CG' or method == 'BFGS' or method == 'L-BFGS':
            myobj.linesearch_options.method = 'Wolfe'
        
        elif method == 'gradient_descend' or method == 'Newton' or method == 'modified_Newton':
            myobj.linesearch_options.method = 'Armijo'
        outputfile = 'result/' + Objname + '_' + method + '.txt'
        Obj.outputfile = outputfile
        myobj.printinfo()
        
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
    with open('result/' + Objname + '_table.txt', 'w') as f:
        f.write("\\begin{table}[htpb]\n")
        f.write("\\centering\n")
        f.write("\\caption{" + Objname + "}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(" Method & $f_{\\text{val}}$ & $||\\nabla f||$ &  Iter & Runtime & Converge Reason & Other info \\\\\n")
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
            
            f.write(method + " & " + "{:.5e}".format(fval) + " & " + "{:.5e}".format(gradnorm) + " & " + str(iternum) + " & " + "{:.3e}".format(runtime) + " & " + converge_reason + " & " + otherinfo + " \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        